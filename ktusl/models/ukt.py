# ktusl/models/ukt.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Iterable
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from enum import IntEnum

from .base import KTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Low-level helpers (positional, Wasserstein, attention) ====================

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> return (1, T, D)
        return self.weight[:, :x.size(Dim.seq), :]


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[:, :x.size(Dim.seq), :]


def wasserstein_distance_matmul(
    mean1: torch.Tensor,
    cov1: torch.Tensor,
    mean2: torch.Tensor,
    cov2: torch.Tensor,
) -> torch.Tensor:
    """
    Equation (4) from the paper: approximate Wasserstein distance between two factored Gaussians.
    mean*, cov*: (B, H, T, D) -> return (B, H, T, T)
    """
    mean1_2 = torch.sum(mean1 ** 2, -1, keepdim=True)
    mean2_2 = torch.sum(mean2 ** 2, -1, keepdim=True)
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(
        -1, -2
    )

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    cov_ret = (
        -2
        * torch.matmul(
            torch.sqrt(torch.clamp(cov1, min=1e-24)),
            torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2),
        )
        + cov1_2
        + cov2_2.transpose(-1, -2)
    )

    return ret + cov_ret


def uattention(
    q_mean: torch.Tensor,
    q_cov: torch.Tensor,
    k_mean: torch.Tensor,
    k_cov: torch.Tensor,
    v_mean: torch.Tensor,
    v_cov: torch.Tensor,
    d_k: int,
    mask: torch.Tensor,
    dropout: nn.Dropout,
    zero_pad: bool,
    gamma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wasserstein-based attention (UKT paper version).
    q/k/v_*: (B, H, T, D), mask: (B, 1, T, T) bool
    """
    scores = (-wasserstein_distance_matmul(q_mean, q_cov, k_mean, k_cov)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()

    x1 = torch.arange(seqlen, device=device).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float()

        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)

        position_effect = torch.abs(x1 - x2)[None, None, :, :].float()  # (1,1,T,T)
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

    m = nn.Softplus()
    gamma = -1.0 * m(gamma).unsqueeze(0)  # (1,H,1,1)
    total_effect = torch.clamp(
        torch.clamp((dist_scores * gamma).exp(), min=1e-5),
        max=1e5,
    )

    scores = scores * total_effect
    scores = scores.masked_fill(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen, device=device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)

    out_mean = torch.matmul(scores, v_mean)
    out_cov = torch.matmul(scores ** 2, v_cov)
    return out_mean, out_cov


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_feature: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_cov_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_cov_linear = nn.Linear(d_model, d_model, bias=bias)

        if not kq_same:
            self.q_mean_linear = nn.Linear(d_model, d_model, bias=bias)
            self.q_cov_linear = nn.Linear(d_model, d_model, bias=bias)
        else:
            self.q_mean_linear = self.k_mean_linear
            self.q_cov_linear = self.k_cov_linear

        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_mean_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_cov_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_mean_linear.weight)
        xavier_uniform_(self.k_cov_linear.weight)
        xavier_uniform_(self.v_mean_linear.weight)
        xavier_uniform_(self.v_cov_linear.weight)

        if not self.kq_same:
            xavier_uniform_(self.q_mean_linear.weight)
            xavier_uniform_(self.q_cov_linear.weight)

        if self.proj_bias:
            constant_(self.k_mean_linear.bias, 0.0)
            constant_(self.k_cov_linear.bias, 0.0)
            constant_(self.v_mean_linear.bias, 0.0)
            constant_(self.v_cov_linear.bias, 0.0)
            if not self.kq_same:
                constant_(self.q_mean_linear.bias, 0.0)
                constant_(self.q_cov_linear.bias, 0.0)
            constant_(self.out_mean_proj.bias, 0.0)
            constant_(self.out_cov_proj.bias, 0.0)

    def forward(
        self,
        q_mean: torch.Tensor,
        q_cov: torch.Tensor,
        k_mean: torch.Tensor,
        k_cov: torch.Tensor,
        v_mean: torch.Tensor,
        v_cov: torch.Tensor,
        mask: torch.Tensor,
        atten_type: str,
        zero_pad: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q_*,k_*,v_*: (B,T,D), mask: (B,1,T,T)
        """
        bs, T, _ = q_mean.size()

        k_mean = self.k_mean_linear(k_mean).view(bs, T, self.h, self.d_k).transpose(1, 2)
        k_cov = self.k_cov_linear(k_cov).view(bs, T, self.h, self.d_k).transpose(1, 2)
        q_mean = self.q_mean_linear(q_mean).view(bs, T, self.h, self.d_k).transpose(1, 2)
        q_cov = self.q_cov_linear(q_cov).view(bs, T, self.h, self.d_k).transpose(1, 2)
        v_mean = self.v_mean_linear(v_mean).view(bs, T, self.h, self.d_k).transpose(1, 2)
        v_cov = self.v_cov_linear(v_cov).view(bs, T, self.h, self.d_k).transpose(1, 2)

        if atten_type == "w2":
            out_mean, out_cov = uattention(
                q_mean,
                q_cov,
                k_mean,
                k_cov,
                v_mean,
                v_cov,
                self.d_k,
                mask,
                self.dropout,
                zero_pad,
                self.gammas.to(device),
            )
        else:
            raise NotImplementedError("Only 'w2' is implemented.")

        concat_mean = out_mean.transpose(1, 2).contiguous().view(bs, T, self.d_model)
        concat_cov = out_cov.transpose(1, 2).contiguous().view(bs, T, self.d_model)

        output_mean = self.out_mean_proj(concat_mean)
        output_cov = self.out_cov_proj(concat_cov)
        return output_mean, output_cov


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_feature: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: int,
    ):
        super().__init__()
        kq_same_bool = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same_bool
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.mean_linear1 = nn.Linear(d_model, d_ff)
        self.cov_linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.mean_linear2 = nn.Linear(d_ff, d_model)
        self.cov_linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation2 = nn.ELU()

    def forward(
        self,
        mask: int,
        query_mean: torch.Tensor,
        query_cov: torch.Tensor,
        key_mean: torch.Tensor,
        key_cov: torch.Tensor,
        values_mean: torch.Tensor,
        values_cov: torch.Tensor,
        atten_type: str = "w2",
        apply_pos: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqlen = query_mean.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        if mask == 0:
            q2_mean, q2_cov = self.masked_attn_head(
                query_mean,
                query_cov,
                key_mean,
                key_cov,
                values_mean,
                values_cov,
                mask=src_mask,
                atten_type=atten_type,
                zero_pad=True,
            )
        else:
            q2_mean, q2_cov = self.masked_attn_head(
                query_mean,
                query_cov,
                key_mean,
                key_cov,
                values_mean,
                values_cov,
                mask=src_mask,
                atten_type=atten_type,
                zero_pad=False,
            )

        query_mean = query_mean + self.dropout1(q2_mean)
        query_cov = query_cov + self.dropout1(q2_cov)

        query_mean = self.layer_norm1(query_mean)
        query_cov = self.layer_norm1(self.activation2(query_cov) + 1)

        if apply_pos:
            qff_mean = self.mean_linear2(
                self.dropout(self.activation(self.mean_linear1(query_mean)))
            )
            qff_cov = self.cov_linear2(
                self.dropout(self.activation(self.cov_linear1(query_cov)))
            )
            query_mean = query_mean + self.dropout2(qff_mean)
            query_cov = query_cov + self.dropout2(qff_cov)
            query_mean = self.layer_norm2(qff_mean)
            query_cov = self.layer_norm2(self.activation2(qff_cov) + 1)

        return query_mean, query_cov


class Architecture(nn.Module):
    """
    UKT architecture: stack of TransformerLayer with cosine positional embeddings.
    """

    def __init__(
        self,
        n_question: int,
        n_blocks: int,
        d_model: int,
        d_feature: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: int,
        model_type: str,
        seq_len: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        self.position_mean_embeddings = CosinePositionalEmbedding(
            d_model=d_model, max_len=seq_len
        )
        self.position_cov_embeddings = CosinePositionalEmbedding(
            d_model=d_model, max_len=seq_len
        )

        assert model_type in {"ukt"}
        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    dropout=dropout,
                    kq_same=kq_same,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        q_mean_embed_data: torch.Tensor,
        q_cov_embed_data: torch.Tensor,
        qa_mean_embed_data: torch.Tensor,
        qa_cov_embed_data: torch.Tensor,
        atten_type: str = "w2",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q_* and qa_*: (B,T,D)
        Returns x_mean, x_cov: (B,T,D)
        """
        # seqlen, batch_size: not used, but kept just in case
        seqlen, batch_size = q_mean_embed_data.size(1), q_mean_embed_data.size(0)

        mean_q_posemb = self.position_mean_embeddings(q_mean_embed_data)
        cov_q_posemb = self.position_cov_embeddings(q_cov_embed_data)
        q_mean_embed_data = q_mean_embed_data + mean_q_posemb
        q_cov_embed_data = q_cov_embed_data + cov_q_posemb

        qa_mean_posemb = self.position_mean_embeddings(qa_mean_embed_data)
        qa_cov_posemb = self.position_cov_embeddings(qa_cov_embed_data)
        qa_mean_embed_data = qa_mean_embed_data + qa_mean_posemb
        qa_cov_embed_data = qa_cov_embed_data + qa_cov_posemb

        elu_act = nn.ELU()
        q_cov_embed_data = elu_act(q_cov_embed_data) + 1
        qa_cov_embed_data = elu_act(qa_cov_embed_data) + 1

        y_mean = qa_mean_embed_data
        y_cov = qa_cov_embed_data

        x_mean = q_mean_embed_data
        x_cov = q_cov_embed_data

        for block in self.blocks_2:
            x_mean, x_cov = block(
                mask=0,
                query_mean=x_mean,
                query_cov=x_cov,
                key_mean=x_mean,
                key_cov=x_cov,
                values_mean=y_mean,
                values_cov=y_cov,
                atten_type=atten_type,
                apply_pos=True,
            )

        return x_mean, x_cov


# ==================== UKT adapted to the KTModel framework ====================

class UKT(KTModel, nn.Module):
    """
    UKT adapted to these datasets:
      - vocab = concepts
      - stochastic embeddings (mean/cov) for concepts + answers (0/1)
      - core = Architecture (Wasserstein attention)
      - API: fit / predict_concept_proba / update compatible with KTModel.
    """

    def __init__(
        self,
        concept_ids: Iterable[int],
        question_ids: Optional[Iterable] = None,
        d_model: int = 128,
        n_blocks: int = 2,
        n_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.2,
        seq_len: int = 200,
        # Offline training
        epochs: int = 80,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[str] = None,
        seed: int = 42,
        # early stopping
        val_frac: float = 0.1,
        patience: int = 8,
        emb_type: str = "stoc_qid",
        **kwargs,
    ):
        nn.Module.__init__(self)
        KTModel.__init__(self)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- vocabs ---
        self.concept_ids = sorted(set(int(x) for x in concept_ids))
        self.id2cidx = {cid: i for i, cid in enumerate(self.concept_ids)}
        self.num_c = len(self.concept_ids)

        if question_ids is None:
            # Proxy: question index = concept index
            self.qid2qidx = {i: i for i in range(self.num_c)}
            self.num_q = self.num_c
        else:
            uq = sorted(set(str(q) for q in question_ids))
            self.qid2qidx = {q: i for i, q in enumerate(uq)}
            self.num_q = len(self.qid2qidx)

        # --- hyperparams ---
        self.d_model = int(d_model)
        self.n_blocks = int(n_blocks)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.dropout_p = float(dropout)
        self.seq_len = int(seq_len)

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.val_frac = float(val_frac)
        self.patience = int(patience)
        self.emb_type = emb_type

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # --- stochastic embeddings ---
        self.mean_q_embed = nn.Embedding(self.num_c, self.d_model)
        self.cov_q_embed = nn.Embedding(self.num_c, self.d_model)
        self.mean_qa_embed = nn.Embedding(2, self.d_model)   # answer 0/1
        self.cov_qa_embed = nn.Embedding(2, self.d_model)

        # --- core UKT (Architecture) ---
        self.core = Architecture(
            n_question=self.num_c,
            n_blocks=self.n_blocks,
            d_model=self.d_model,
            d_feature=self.d_model // self.n_heads,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            dropout=self.dropout_p,
            kq_same=1,
            model_type="ukt",
            seq_len=self.seq_len,
        )

        # --- output head ---
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(4 * self.d_model, 1)
        self.sigmoid = nn.Sigmoid()

        self.to(self.device)

        # --- online state ---
        # u -> {"qidx": [...], "cidx": [...], "y": [...]}
        self._hist: Dict[int, Dict[str, List[int]]] = {}

    # ===================== API online (KTModel) =====================

    def reset_state(self):
        self._hist.clear()

    def _first_cidx(self, concepts: List[int]) -> Optional[int]:
        for c in concepts:
            c = int(c)
            if c in self.id2cidx:
                return self.id2cidx[c]
        return None

    @torch.no_grad()
    def predict_concept_proba(self, user: int, concept: int) -> float:
        """
        p(correct | user_history, target_concept) with a virtual step for the target concept.
        """
        u = int(user)
        cidx = self.id2cidx.get(int(concept))
        if cidx is None:
            return 0.5

        hist = self._hist.get(u)
        if hist is None or len(hist.get("cidx", [])) == 0:
            return 0.5

        max_hist = max(1, self.seq_len - 1)
        qh = hist["qidx"][-max_hist:]
        ch = hist["cidx"][-max_hist:]
        rh = hist["y"][-max_hist:]

        pid_last = qh[-1]
        qseq = np.asarray(qh + [pid_last], dtype=np.int64)
        cseq = np.asarray(ch + [cidx], dtype=np.int64)
        rseq = np.asarray(rh + [0], dtype=np.int64)

        q = torch.tensor(qseq[:-1], device=self.device).unsqueeze(0)
        c = torch.tensor(cseq[:-1], device=self.device).unsqueeze(0)
        r = torch.tensor(rseq[:-1], device=self.device).unsqueeze(0)
        qsh = torch.tensor(qseq[1:], device=self.device).unsqueeze(0)
        csh = torch.tensor(cseq[1:], device=self.device).unsqueeze(0)
        rsh = torch.tensor(rseq[1:], device=self.device).unsqueeze(0)

        # Here we encode only by concept + response
        q_data = torch.cat([c[:, 0:1], csh], dim=1)
        target = torch.cat([r[:, 0:1], rsh], dim=1)

        if q_data.numel() > 0:
            qmax = int(q_data.max().item())
            qmin = int(q_data.min().item())
            if not (0 <= qmin <= qmax < self.num_c):
                return 0.5

        q_mean = self.mean_q_embed(q_data)
        q_cov = self.cov_q_embed(q_data)
        qa_mean = self.mean_qa_embed(target) + q_mean
        qa_cov = self.cov_qa_embed(target) + q_cov

        x_mean, x_cov = self.core(q_mean, q_cov, qa_mean, qa_cov, atten_type="w2")

        elu = nn.ELU()
        cov_act = elu(x_cov) + 1
        concat = torch.cat([x_mean, cov_act, q_mean, q_cov], dim=-1)
        p = self.sigmoid(self.out(self.dropout(concat))).squeeze(-1)
        return float(p[0, -1].item())

    @torch.no_grad()
    def update(
        self,
        user: int,
        concepts: List[int],
        correct: int,
        weights: Optional[List[float]] = None,
    ):
        u = int(user)
        y = 1 if int(correct) == 1 else 0
        cidx = self._first_cidx(concepts)
        if cidx is None:
            return

        h = self._hist.get(u)
        if h is None:
            h = {"qidx": [], "cidx": [], "y": []}
            self._hist[u] = h
        qidx = cidx  # proxy question index
        h["qidx"].append(qidx)
        h["cidx"].append(cidx)
        h["y"].append(y)
        if len(h["cidx"]) > self.seq_len:
            for k in ["qidx", "cidx", "y"]:
                h[k] = h[k][-self.seq_len:]

    # ===================== Offline training helpers =====================

    @staticmethod
    def _binary_auc(labels: List[int], scores: List[float]) -> float:
        if len(labels) == 0:
            return float("nan")
        y = np.array(labels, dtype=np.int64)
        s = np.array(scores, dtype=np.float64)
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
        pos_ranks_sum = ranks[y == 1].sum()
        auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)

    def _build_user_seqs(self, df):
        """
        Build per-user sequences:
          [(user_id, (q_seq, c_seq, y_seq))]
        """
        user_seqs: List[Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        for uid, g in df.groupby("UserId", sort=False):
            c_seq = g["cidx"].to_numpy(np.int64)
            if "QuestionId" in g.columns:
                q_seq = g["QuestionId"].astype(str).map(self.qid2qidx).to_numpy(np.int64)
            else:
                q_seq = c_seq.copy()
            y_seq = g["y"].to_numpy(np.int64)

            if len(c_seq) >= 2:
                if len(c_seq) > self.seq_len:
                    c_seq = c_seq[-self.seq_len:]
                    q_seq = q_seq[-self.seq_len:]
                    y_seq = y_seq[-self.seq_len:]
                user_seqs.append((int(uid), (q_seq, c_seq, y_seq)))
        return user_seqs

    def _split_train_val_by_user(
        self,
        user_seqs,
        val_frac: float,
        seed: int = 42,
    ):
        if val_frac <= 0 or len(user_seqs) < 2:
            return [s for _, s in user_seqs], []
        users = [u for u, _ in user_seqs]
        unique_users = np.array(sorted(set(users)))
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_users)
        n_users = len(unique_users)
        n_val = max(1, int(round(val_frac * n_users)))
        val_users = set(unique_users[:n_val])
        train_users = set(unique_users[n_val:])
        train_seqs = [seq for u, seq in user_seqs if u in train_users]
        val_seqs = [seq for u, seq in user_seqs if u in val_users]
        return train_seqs, val_seqs

    def _forward_batch(self, batch):
        """
        batch: list of (q_seq, c_seq, y_seq)
        Returns:
          - p: (B,L) probabilities
          - y_full: (B,L) labels 0/1
          - valid: (B,L) mask of valid positions (t>=1)
        """
        if len(batch) == 0:
            return None, None, None, 0

        L = max(len(s[0]) for s in batch)
        if L < 2:
            return None, None, None, 0

        B = len(batch)
        q = torch.zeros(B, L - 1, dtype=torch.long, device=self.device)
        c = torch.zeros(B, L - 1, dtype=torch.long, device=self.device)
        r = torch.zeros(B, L - 1, dtype=torch.long, device=self.device)
        qsh = torch.zeros(B, L - 1, dtype=torch.long, device=self.device)
        csh = torch.zeros(B, L - 1, dtype=torch.long, device=self.device)
        rsh = torch.zeros(B, L - 1, dtype=torch.long, device=self.device)
        valid = torch.zeros(B, L, dtype=torch.float32, device=self.device)

        for i, (q_seq, c_seq, y_seq) in enumerate(batch):
            T = len(q_seq)
            if T < 2:
                continue
            q_seq = np.asarray(q_seq, dtype=np.int64)
            c_seq = np.asarray(c_seq, dtype=np.int64)
            y_seq = np.asarray(y_seq, dtype=np.int64)

            if T > L:
                q_seq = q_seq[-L:]
                c_seq = c_seq[-L:]
                y_seq = y_seq[-L:]
                T = L

            q_i, c_i, r_i = q_seq[:-1], c_seq[:-1], y_seq[:-1]
            q_is, c_is, r_is = q_seq[1:], c_seq[1:], y_seq[1:]
            tlen = len(q_i)
            if tlen == 0:
                continue

            q[i, :tlen] = torch.from_numpy(q_i)
            c[i, :tlen] = torch.from_numpy(c_i)
            r[i, :tlen] = torch.from_numpy(r_i)
            qsh[i, :tlen] = torch.from_numpy(q_is)
            csh[i, :tlen] = torch.from_numpy(c_is)
            rsh[i, :tlen] = torch.from_numpy(r_is)
            valid[i, 1:(tlen + 1)] = 1.0

        q_data = torch.cat([c[:, 0:1], csh], dim=1)
        target = torch.cat([r[:, 0:1], rsh], dim=1)

        if q_data.numel() > 0:
            qmax = int(q_data.max().item())
            qmin = int(q_data.min().item())
            assert 0 <= qmin and qmax < self.num_c, \
                f"q_data indices out of range: [{qmin},{qmax}] vs num_c={self.num_c}"

        q_mean = self.mean_q_embed(q_data)
        q_cov = self.cov_q_embed(q_data)
        qa_mean = self.mean_qa_embed(target) + q_mean
        qa_cov = self.cov_qa_embed(target) + q_cov

        x_mean, x_cov = self.core(q_mean, q_cov, qa_mean, qa_cov, atten_type="w2")

        elu = nn.ELU()
        cov_act = elu(x_cov) + 1
        concat = torch.cat([x_mean, cov_act, q_mean, q_cov], dim=-1)
        p = self.sigmoid(self.out(self.dropout(concat))).squeeze(-1)

        total_valid = int(valid.sum().item())
        return p, target.float(), valid, total_valid

    def _evaluate_on_seqs(self, seqs, batch_size: int):
        """
        Evaluate BCE, ACC, AUC over a set of sequences.
        """
        if len(seqs) == 0:
            return float("nan"), float("nan"), float("nan")

        self.eval()
        total_loss_sum = 0.0
        total_count = 0
        all_probs: List[float] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for b_start in range(0, len(seqs), batch_size):
                batch = seqs[b_start:b_start + batch_size]
                p, y_full, valid, total_valid = self._forward_batch(batch)
                if p is None or total_valid == 0:
                    continue

                eps = 1e-7
                bce_mat = -(
                    y_full * torch.log(p.clamp_min(eps)) +
                    (1 - y_full) * torch.log((1 - p).clamp_min(eps))
                )
                loss_sum = (bce_mat * valid).sum()
                total_loss_sum += float(loss_sum.detach().cpu().item())
                total_count += int(valid.sum().item())

                mask = (valid > 0)
                probs = p[mask].detach().cpu().numpy().tolist()
                labels = y_full[mask].detach().cpu().numpy().astype(int).tolist()
                all_probs.extend(probs)
                all_labels.extend(labels)

        bce = total_loss_sum / max(1, total_count)
        if len(all_labels) == 0:
            return bce, float("nan"), float("nan")
        preds = [1 if s >= 0.5 else 0 for s in all_probs]
        correct = sum(int(a == b) for a, b in zip(preds, all_labels))
        acc = correct / max(1, len(all_labels))
        auc = self._binary_auc(all_labels, all_probs)
        return bce, acc, auc

    # ===================== Offline training =====================

    def fit(self, train_df, subjects_df=None, show_progress: bool = True, **kwargs):
        """
        Train UKT on the train split:
        - keep the first concept per interaction
        - build (qidx, cidx, y) sequences per user
        - per-student internal train/val split
        - early stopping on val AUC.
        """
        df = train_df.copy()
        cols = ["UserId", "QuestionId", "SubjectId", "IsCorrect"]
        if "DateAnswered" in df.columns:
            cols.append("DateAnswered")
        df = df[cols].copy()

        def _first_concept(lst):
            if isinstance(lst, (list, tuple)) and len(lst) > 0:
                for c in lst:
                    c_int = int(c)
                    if c_int in self.id2cidx:
                        return c_int
            return None

        df["cid"] = df["SubjectId"].apply(_first_concept)
        df = df.dropna(subset=["cid"]).copy()
        df["cid"] = df["cid"].astype(int)
        df["cidx"] = df["cid"].map(
            lambda c: self.id2cidx[int(c)] if int(c) in self.id2cidx else np.nan
        )
        df = df.dropna(subset=["cidx"]).copy()
        df["cidx"] = df["cidx"].astype(int)

        df["y"] = (df["IsCorrect"].astype(int) > 0).astype(int)

        if "QuestionId" in df.columns:
            uq = sorted(set(df["QuestionId"].astype(str)))
            self.qid2qidx = {q: i for i, q in enumerate(uq)}
            self.num_q = len(self.qid2qidx)

        if "DateAnswered" in df.columns:
            df = df.sort_values(["UserId", "DateAnswered"], kind="mergesort")
        else:
            df = df.sort_values(["UserId"], kind="mergesort")

        user_seqs = self._build_user_seqs(df)
        if len(user_seqs) == 0:
            return

        train_seqs, val_seqs = self._split_train_val_by_user(
            user_seqs, val_frac=self.val_frac
        )

        if len(train_seqs) == 0:
            return

        opt = optim.Adam(self.parameters(), lr=self.lr)

        try:
            from tqdm.auto import tqdm  # type: ignore

            def _p(it, desc):
                return tqdm(it, desc=desc) if show_progress else it

        except Exception:  # pragma: no cover

            def _p(it, desc):
                return it

        best_val_auc = -float("inf")
        best_state = None
        epochs_no_improve = 0

        for ep in range(self.epochs):
            self.train()
            total_loss = 0.0
            total_count = 0

            for b_start in _p(
                range(0, len(train_seqs), self.batch_size),
                desc=f"UKT train epoch {ep+1}/{self.epochs}",
            ):
                batch = train_seqs[b_start:b_start + self.batch_size]
                p, y_full, valid, total_valid = self._forward_batch(batch)
                if p is None or total_valid == 0:
                    continue

                eps = 1e-7
                bce_mat = -(
                    y_full * torch.log(p.clamp_min(eps)) +
                    (1 - y_full) * torch.log((1 - p).clamp_min(eps))
                )
                loss_sum = (bce_mat * valid).sum()
                count = valid.sum().clamp_min(1.0)
                loss = loss_sum / count

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                opt.step()

                total_loss += float(loss_sum.detach().cpu().item())
                total_count += int(count.detach().cpu().item())

            train_bce = total_loss / max(1, total_count)

            if len(val_seqs) > 0:
                val_bce, val_acc, val_auc = self._evaluate_on_seqs(
                    val_seqs, batch_size=self.batch_size
                )
                print(
                    f"UKT epoch {ep+1}/{self.epochs} — "
                    f"train_BCE={train_bce:.4f} | "
                    f"val_BCE={val_bce:.4f}, val_ACC={val_acc:.4f}, val_AUC={val_auc:.4f}"
                )

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {
                        k: v.cpu().clone() for k, v in self.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(
                            f"UKT early stopping at epoch {ep+1}, "
                            f"best val_AUC={best_val_auc:.4f}"
                        )
                        break
            else:
                print(f"UKT epoch {ep+1}/{self.epochs} — train_BCE={train_bce:.4f}")

        if best_state is not None:
            self.load_state_dict(best_state)
            self.to(self.device)
