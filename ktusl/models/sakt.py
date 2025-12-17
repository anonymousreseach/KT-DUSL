# ktusl/models/sakt.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Iterable

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import KTModel


def transformer_ffn(d_model: int, dropout: float = 0.1, expansion: int = 4) -> nn.Sequential:
    """FFN standard Transformer: d -> 4d -> d (GELU)"""
    return nn.Sequential(
        nn.Linear(d_model, expansion * d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(expansion * d_model, d_model),
    )


def ut_mask(tgt_len: int, src_len: Optional[int] = None) -> torch.Tensor:
    """
    Upper-triangular mask (causal). If src_len is None -> square (tgt_len = src_len).
    Returns a (tgt_len, src_len) mask with -inf above the diagonal.
    """
    if src_len is None:
        src_len = tgt_len
    mask = torch.triu(torch.ones((tgt_len, src_len), dtype=torch.bool), diagonal=1)
    return mask.float().masked_fill(mask, float('-inf'))


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Blocks(nn.Module):
    def __init__(self, emb_size: int, num_attn_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_size, num_attn_heads, dropout=dropout, batch_first=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(emb_size)

        self.FFN = transformer_ffn(emb_size, dropout)
        self.FFN_dropout = nn.Dropout(dropout)
        self.FFN_layer_norm = nn.LayerNorm(emb_size)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Expected q,k,v shapes: (B,T,E); permute to (T,B,E) for MHA.
        Apply a causal mask (size T x T).
        """
        qT, kT, vT = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        T = kT.shape[0]
        causal = ut_mask(T, T)  # (T,T)

        attn_emb, _ = self.attn(qT, kT, vT, attn_mask=causal)
        attn_emb = self.attn_dropout(attn_emb)
        attn_emb = attn_emb.permute(1, 0, 2)
        q = qT.permute(1, 0, 2)
        attn_out = self.attn_layer_norm(q + attn_emb)

        ffn_out = self.FFN(attn_out)
        ffn_out = self.FFN_dropout(ffn_out)
        out = self.FFN_layer_norm(attn_out + ffn_out)
        return out


class SAKT(KTModel, nn.Module):
    """
    SAKT (Self-Attentive KT) — compact version compatible with the trainer:
      - fit(train_df, subjects_df, show_progress=True): offline training
      - reset_state(), predict_concept_proba(user,c), update(user, concepts, correct): online phase

    Design choices:
      * Vocabulary = list of concept_ids (derived from metadata or traces).
      * During training, use the **first concept** in the list per interaction (simplicity/robustness).
      * Online, for multi-concept questions, the trainer combines via `combine_mode` (mean/noisy-or...).
    """

    def __init__(
        self,
        concept_ids: Iterable[int],
        seq_len: int = 200,
        emb_size: int = 128,
        num_attn_heads: int = 2,
        dropout: float = 0.2,
        num_en: int = 2,
        # Offline training
        epochs: int = 80,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[str] = None,
        seed: int = 42,
        # New hyperparameters
        val_frac: float = 0.1,
        patience: int = 8,
    ):
        nn.Module.__init__(self)
        KTModel.__init__(self)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- vocab & mapping ---
        self.concept_ids = list(sorted(set(int(x) for x in concept_ids)))
        self.id2idx = {cid: i for i, cid in enumerate(self.concept_ids)}
        self.num_c = len(self.concept_ids)

        # --- hyper ---
        self.seq_len = int(seq_len)
        self.emb_size = int(emb_size)
        self.num_attn_heads = int(num_attn_heads)
        self.dropout_p = float(dropout)
        self.num_en = int(num_en)

        # --- training hyper ---
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.val_frac = float(val_frac)
        self.patience = int(patience)

        # device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # --- embeddings ---
        self.interaction_emb = nn.Embedding(self.num_c * 2, self.emb_size)
        self.exercise_emb = nn.Embedding(self.num_c, self.emb_size)
        self.position_emb = nn.Embedding(self.seq_len, self.emb_size)

        # --- blocks ---
        self.blocks = get_clones(Blocks(self.emb_size, self.num_attn_heads, self.dropout_p), self.num_en)

        self.dropout = nn.Dropout(self.dropout_p)
        self.pred = nn.Linear(self.emb_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.to(self.device)

        # --- online state (per-user history) ---
        self._hist: Dict[int, Dict[str, List[int]]] = {}

    # ---------------- API KTModel (phase test online) ----------------

    def reset_state(self):
        self._hist.clear()

    def _concept_idx(self, concept: int) -> Optional[int]:
        return self.id2idx.get(int(concept))

    @torch.no_grad()
    def predict_concept_proba(self, user: int, concept: int) -> float:
        idx = self._concept_idx(concept)
        if idx is None:
            return 0.5

        hist = self._hist.get(int(user))
        if hist is None or len(hist.get("q", [])) == 0:
            return 0.5

        q_hist = hist["q"][-self.seq_len :]
        r_hist = hist["r"][-self.seq_len :]

        T = len(q_hist)
        if T == 0:
            return 0.5

        q = torch.tensor(q_hist, device=self.device, dtype=torch.long).unsqueeze(0)      # (1,T)
        r = torch.tensor(r_hist, device=self.device, dtype=torch.long).unsqueeze(0)      # (1,T)
        qry = torch.full_like(q, fill_value=idx)                                         # (1,T)

        x = q + self.num_c * r
        xemb = self.interaction_emb(x) + self.position_emb(torch.arange(T, device=self.device)).unsqueeze(0)
        qshftemb = self.exercise_emb(qry)

        self.eval()
        emb = xemb
        for i in range(self.num_en):
            emb = self.blocks[i](qshftemb, emb, emb)

        p = self.sigmoid(self.pred(self.dropout(emb))).squeeze(-1)   # (1,T)
        return float(p[0, -1].item())

    @torch.no_grad()
    def update(self, user: int, concepts: List[int], correct: int, weights: Optional[List[float]] = None):
        u = int(user)
        y = 1 if int(correct) == 1 else 0
        if len(concepts) == 0:
            return
        idxs = [self._concept_idx(c) for c in concepts]
        idxs = [i for i in idxs if i is not None]
        if len(idxs) == 0:
            return
        cidx = int(idxs[0])

        h = self._hist.get(u)
        if h is None:
            h = {"q": [], "r": []}
            self._hist[u] = h
        h["q"].append(cidx)
        h["r"].append(y)
        if len(h["q"]) > self.seq_len:
            h["q"] = h["q"][-self.seq_len :]
            h["r"] = h["r"][-self.seq_len :]

    # ------------------------ Helpers internes ------------------------

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
        Build sequences per user:
          [(user_id, (q_seq, r_seq))]
        """
        user_seqs: List[Tuple[int, Tuple[np.ndarray, np.ndarray]]] = []
        for uid, g in df.groupby("UserId", sort=False):
            q_seq = g["cid"].to_numpy(dtype=np.int64)
            r_seq = g["IsCorrect"].to_numpy(dtype=np.int64)
            if len(q_seq) >= 2:
                if len(q_seq) > self.seq_len:
                    q_seq = q_seq[-self.seq_len :]
                    r_seq = r_seq[-self.seq_len :]
                user_seqs.append((int(uid), (q_seq, r_seq)))
        return user_seqs

    def _split_train_val_by_user(
        self,
        user_seqs: List[Tuple[int, Tuple[np.ndarray, np.ndarray]]],
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

    def _build_batch_tensors(self, batch: List[Tuple[np.ndarray, np.ndarray]]):
        """
        batch: list of (q_seq, r_seq)
        Returns (q, r, qry, tgt) of shape (B,T).
        """
        if len(batch) == 0:
            return None, None, None, None, 0

        # Max length - 1 (we predict y_t for t >= 1)
        T = max(len(s[0]) for s in batch) - 1
        if T <= 0:
            return None, None, None, None, 0

        B = len(batch)
        q = torch.zeros(B, T, dtype=torch.long, device=self.device)
        r = torch.zeros(B, T, dtype=torch.long, device=self.device)
        qry = torch.zeros(B, T, dtype=torch.long, device=self.device)
        tgt = torch.zeros(B, T, dtype=torch.float32, device=self.device)
        valid = torch.zeros(B, T, dtype=torch.float32, device=self.device)

        for i, (q_seq, r_seq) in enumerate(batch):
            q_seq = np.asarray(q_seq, dtype=np.int64)
            r_seq = np.asarray(r_seq, dtype=np.int64)
            if len(q_seq) - 1 > T:
                q_seq = q_seq[-(T + 1):]
                r_seq = r_seq[-(T + 1):]

            q_i = q_seq[:-1]
            r_i = r_seq[:-1]
            qry_i = q_seq[1:]
            y_i = r_seq[1:]

            tlen = len(q_i)
            if tlen == 0:
                continue
            q[i, :tlen] = torch.from_numpy(q_i)
            r[i, :tlen] = torch.from_numpy(r_i)
            qry[i, :tlen] = torch.from_numpy(qry_i)
            tgt[i, :tlen] = torch.from_numpy(y_i.astype(np.float32))
            valid[i, :tlen] = 1.0

        total_valid = int(valid.sum().item())
        return q, r, qry, tgt, total_valid

    def _evaluate_on_seqs(
        self,
        seqs: List[Tuple[np.ndarray, np.ndarray]],
        bce_loss: nn.BCELoss,
        batch_size: int,
    ):
        if len(seqs) == 0:
            return float("nan"), float("nan"), float("nan")

        self.eval()
        total_loss_sum = 0.0
        total_targets = 0
        all_probs: List[float] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for b_start in range(0, len(seqs), batch_size):
                batch = seqs[b_start:b_start + batch_size]
                q, r, qry, tgt, total_valid = self._build_batch_tensors(batch)
                if total_valid == 0 or q is None:
                    continue

                T = q.shape[1]
                pos = torch.arange(T, device=self.device).unsqueeze(0)  # (1,T)
                x = q + self.num_c * r
                xemb = self.interaction_emb(x) + self.position_emb(pos)
                qshftemb = self.exercise_emb(qry)

                emb = xemb
                for i in range(self.num_en):
                    emb = self.blocks[i](qshftemb, emb, emb)

                p = self.sigmoid(self.pred(self.dropout(emb))).squeeze(-1)  # (B,T)

                # Valid-weighted BCE
                eps = 1e-7
                loss_mat = -(
                    tgt * torch.log(p.clamp_min(eps)) +
                    (1 - tgt) * torch.log((1 - p).clamp_min(eps))
                )
                loss_sum = (loss_mat).sum()

                total_loss_sum += float(loss_sum.detach().cpu().item())
                total_targets += total_valid

                # Collect probabilities / labels on valid positions
                mask = (tgt >= 0) & (p >= 0)  # all, though we could use 'valid'
                probs = p[mask].detach().cpu().numpy().tolist()
                labels = tgt[mask].detach().cpu().numpy().astype(int).tolist()
                all_probs.extend(probs)
                all_labels.extend(labels)

        bce = total_loss_sum / max(1, total_targets)
        if len(all_labels) == 0:
            return bce, float("nan"), float("nan")

        preds = [1 if x >= 0.5 else 0 for x in all_probs]
        correct = sum(int(p == y) for p, y in zip(preds, all_labels))
        acc = correct / max(1, len(all_labels))
        auc = self._binary_auc(all_labels, all_probs)
        return bce, acc, auc

    # ------------------------ OFFLINE TRAINING (per-student train/val split) ------------------------

    def fit(self, train_df, subjects_df, show_progress: bool = True, val_frac: Optional[float] = None, **kwargs):
        """
        Train on the 'train' split with a per-user train/val split.
        SAKT predicts y_t from (q_{<t}, r_{<t}) together with qry_t = q_t.
        "Single-skill" implementation: take the first concept per interaction.
        """
        df = train_df.copy()
        cols = ["UserId", "SubjectId", "IsCorrect"]
        if "DateAnswered" in df.columns:
            cols.append("DateAnswered")
        df = df[cols].copy()

        # SubjectId -> keep the first concept present in the vocab
        def _first_idx(lst):
            if isinstance(lst, (list, tuple)) and len(lst) > 0:
                for c in lst:
                    if c in self.id2idx:
                        return self.id2idx[c]
            return None

        df["cid"] = df["SubjectId"].apply(_first_idx)
        df = df.dropna(subset=["cid"]).copy()
        df["cid"] = df["cid"].astype(int)
        df["IsCorrect"] = (df["IsCorrect"].astype(int) > 0).astype(int)

        # Temporal sort
        if "DateAnswered" in df.columns:
            df = df.sort_values(["UserId", "DateAnswered"], kind="mergesort")
        else:
            df = df.sort_values(["UserId"], kind="mergesort")

        # Per-user sequences
        user_seqs = self._build_user_seqs(df)
        if len(user_seqs) == 0:
            return

        if val_frac is None:
            val_frac = self.val_frac

        train_seqs, val_seqs = self._split_train_val_by_user(user_seqs, val_frac=val_frac)

        if len(train_seqs) == 0:
            return

        opt = optim.Adam(self.parameters(), lr=self.lr)
        bce = nn.BCELoss(reduction="sum")

        # Optional tqdm
        try:
            from tqdm.auto import tqdm  # type: ignore
            def _p(it, desc): return tqdm(it, desc=desc) if show_progress else it
        except Exception:
            def _p(it, desc): return it

        best_val_auc = -float("inf")
        best_state = None
        epochs_no_improve = 0

        self.train()
        for ep in range(self.epochs):
            self.train()
            total_loss = 0.0
            total_count = 0

            for b_start in _p(range(0, len(train_seqs), self.batch_size),
                              desc=f"SAKT train epoch {ep+1}/{self.epochs}"):
                batch = train_seqs[b_start:b_start + self.batch_size]
                q, r, qry, tgt, total_valid = self._build_batch_tensors(batch)
                if total_valid == 0 or q is None:
                    continue

                T = q.shape[1]
                pos = torch.arange(T, device=self.device).unsqueeze(0)  # (1,T)
                x = q + self.num_c * r
                xemb = self.interaction_emb(x) + self.position_emb(pos)
                qshftemb = self.exercise_emb(qry)

                emb = xemb
                for i in range(self.num_en):
                    emb = self.blocks[i](qshftemb, emb, emb)

                p = self.sigmoid(self.pred(self.dropout(emb))).squeeze(-1)  # (B,T)

                eps = 1e-7
                loss_mat = -(
                    tgt * torch.log(p.clamp_min(eps)) +
                    (1 - tgt) * torch.log((1 - p).clamp_min(eps))
                )
                loss_sum = loss_mat.sum()
                count = float(total_valid)
                loss = loss_sum / max(1.0, count)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                opt.step()

                total_loss += float(loss_sum.detach().cpu().item())
                total_count += int(count)

            train_bce = total_loss / max(1, total_count)

            # Validation eval
            if len(val_seqs) > 0:
                val_bce, val_acc, val_auc = self._evaluate_on_seqs(val_seqs, bce_loss=bce, batch_size=self.batch_size)
                print(
                    f"SAKT epoch {ep+1}/{self.epochs} — "
                    f"train_BCE={train_bce:.4f} | "
                    f"val_BCE={val_bce:.4f}, val_ACC={val_acc:.4f}, val_AUC={val_auc:.4f}"
                )

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(
                            f"SAKT early stopping at epoch {ep+1}, "
                            f"best val_AUC={best_val_auc:.4f}"
                        )
                        break
            else:
                print(f"SAKT epoch {ep+1}/{self.epochs} — train_BCE={train_bce:.4f}")

        if best_state is not None:
            self.load_state_dict(best_state)
            self.to(self.device)
