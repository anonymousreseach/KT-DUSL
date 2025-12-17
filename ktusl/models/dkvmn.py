# ktusl/models/dkvmn.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import KTModel


class DKVMN(KTModel, nn.Module):
    """
    DKVMN (Dynamic Key-Value Memory Network for KT), adapted to the ktusl framework.

    Design:
      - Vocabulary = list of concept_ids (like SAKT).
      - During training, build (q_seq, r_seq) sequences per user:
          q_seq: concept indices (cid)
          r_seq: 0/1 (IsCorrect)
      - The model takes (q, r) of shape (B, T) and outputs p (B, T),
        probability of r_t = 1 at each time step t.
      - Online, to predict the success probability of concept c for
        user u:
          * retrieve the history (q_hist, r_hist)
          * append the target concept c (with a dummy r = 0)
          * run the sequence through DKVMN
          * use the last probability as the prediction.
    """

    def __init__(
        self,
        concept_ids: Iterable[int],
        seq_len: int = 200,
        dim_s: int = 128,
        size_m: int = 50,
        dropout: float = 0.2,
        # Offline training
        epochs: int = 80,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[str] = None,
        seed: int = 42,
        val_frac: float = 0.1,
        patience: int = 8,
    ):
        nn.Module.__init__(self)
        KTModel.__init__(self)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ----------------- vocab & mapping -----------------
        self.concept_ids = list(sorted(set(int(x) for x in concept_ids)))
        self.id2idx = {cid: i for i, cid in enumerate(self.concept_ids)}
        self.num_c = len(self.concept_ids)

        # ----------------- hyperparams -----------------
        self.seq_len = int(seq_len)
        self.dim_s = int(dim_s)
        self.size_m = int(size_m)
        self.dropout_p = float(dropout)

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.val_frac = float(val_frac)
        self.patience = int(patience)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # ----------------- DKVMN core -----------------
        # Embedding for keys (concepts)
        self.k_emb_layer = nn.Embedding(self.num_c, self.dim_s)

        # Key/value memory
        self.Mk = nn.Parameter(torch.empty(self.size_m, self.dim_s))
        self.Mv0 = nn.Parameter(torch.empty(self.size_m, self.dim_s))

        nn.init.kaiming_normal_(self.Mk)
        nn.init.kaiming_normal_(self.Mv0)

        # Embedding for interactions (concept + correct/incorrect)
        self.v_emb_layer = nn.Embedding(self.num_c * 2, self.dim_s)

        # Projection / prediction layers
        self.f_layer = nn.Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = nn.Dropout(self.dropout_p)
        self.p_layer = nn.Linear(self.dim_s, 1)

        # Memory update layers
        self.e_layer = nn.Linear(self.dim_s, self.dim_s)
        self.a_layer = nn.Linear(self.dim_s, self.dim_s)

        self.sigmoid = nn.Sigmoid()

        self.to(self.device)

        # ----------------- online state -----------------
        # per-user history: {user: {"q": [cid_idx], "r": [0/1]}}
        self._hist: Dict[int, Dict[str, List[int]]] = {}

    # ------------------------------------------------------------------
    # API KTModel (phase test online)
    # ------------------------------------------------------------------
    def reset_state(self):
        self._hist.clear()

    def _concept_idx(self, concept: int) -> Optional[int]:
        return self.id2idx.get(int(concept))

    @torch.no_grad()
    def predict_concept_proba(self, user: int, concept: int) -> float:
        """
        Online prediction.

        Take the user history (q_hist, r_hist), append the target concept
        with a dummy answer 0 to form a sequence (q, r). Run this sequence
        through DKVMN and read the last probability.
        """
        idx = self._concept_idx(concept)
        if idx is None:
            return 0.5

        hist = self._hist.get(int(user))
        q_hist = hist["q"] if hist is not None else []
        r_hist = hist["r"] if hist is not None else []

        # Append the target concept with a dummy 0 response
        q_seq = list(q_hist) + [idx]
        r_seq = list(r_hist) + [0]

        # Truncate to seq_len
        if len(q_seq) > self.seq_len:
            q_seq = q_seq[-self.seq_len :]
            r_seq = r_seq[-self.seq_len :]

        T = len(q_seq)
        if T == 0:
            return 0.5

        q = torch.tensor(q_seq, dtype=torch.long, device=self.device).unsqueeze(0)  # (1,T)
        r = torch.tensor(r_seq, dtype=torch.long, device=self.device).unsqueeze(0)  # (1,T)

        self.eval()
        p = self._forward_core(q, r)  # (1,T)
        return float(p[0, -1].item())

    @torch.no_grad()
    def update(self, user: int, concepts: List[int], correct: int, weights: Optional[List[float]] = None):
        """
        Update the online history:
        encode the first concept in the list (like SAKT/UKT) and the binary answer.
        """
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

    # ------------------------------------------------------------------
    # Noyau DKVMN (forward sur batch)
    # ------------------------------------------------------------------
    def _forward_core(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Implement DKVMN forward for a batch:
          q: (B, T) concept indices
          r: (B, T) 0/1
        Return:
          p: (B, T) success probabilities at each step.
        """
        B, T = q.shape

        x = q + self.num_c * r          # (B, T)
        k = self.k_emb_layer(q)         # (B, T, dim_s)
        v = self.v_emb_layer(x)         # (B, T, dim_s)

        # Initial value memory
        Mvt = self.Mv0.unsqueeze(0).repeat(B, 1, 1)  # (B, size_m, dim_s)
        Mv = [Mvt]

        # Attention weights over keys
        # w: (B, T, size_m)
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write process
        e = torch.sigmoid(self.e_layer(v))  # (B, T, dim_s)
        a = torch.tanh(self.a_layer(v))     # (B, T, dim_s)

        for et, at, wt in zip(
            e.permute(1, 0, 2),  # (T,B,dim_s) -> et: (B,dim_s)
            a.permute(1, 0, 2),
            w.permute(1, 0, 2),  # (T,B,size_m) -> wt: (B,size_m)
        ):
            # et: (B, dim_s), at: (B, dim_s), wt: (B, size_m)
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                  (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        # Mv: (B, T+1, size_m, dim_s)
        Mv = torch.stack(Mv, dim=1)

        # Read process
        # Read the memory BEFORE writing the current interaction:
        # Mv[:, :-1] corresponds to M_0,...,M_{T-1}
        # w: (B,T,size_m)
        read_content = (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2)  # (B, T, dim_s)

        f_in = torch.cat([read_content, k], dim=-1)  # (B, T, 2*dim_s)
        f = torch.tanh(self.f_layer(f_in))          # (B, T, dim_s)

        f = self.dropout_layer(f)
        p = self.sigmoid(self.p_layer(f)).squeeze(-1)  # (B, T)
        return p

    # for compatibility with a potential direct .forward batch call
    def forward(self, q, r, qtest: bool = False):
        p = self._forward_core(q, r)
        if not qtest:
            return p
        else:
            return p, None  # on pourrait retourner f si besoin

    # ------------------------------------------------------------------
    # Internal helpers (evaluation, batch, split, AUC)
    # ------------------------------------------------------------------
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
        Returns (q, r, tgt, total_valid) of shape (B,T).
        Here, DKVMN predicts r_t at each time step from the previous
        history, so no explicit shift is required (we could ignore t=0 if desired).
        """
        if len(batch) == 0:
            return None, None, None, 0

        T = max(len(s[0]) for s in batch)
        if T <= 0:
            return None, None, None, 0

        B = len(batch)
        q = torch.zeros(B, T, dtype=torch.long, device=self.device)
        r = torch.zeros(B, T, dtype=torch.long, device=self.device)
        tgt = torch.zeros(B, T, dtype=torch.float32, device=self.device)
        valid = torch.zeros(B, T, dtype=torch.float32, device=self.device)

        for i, (q_seq, r_seq) in enumerate(batch):
            q_seq = np.asarray(q_seq, dtype=np.int64)
            r_seq = np.asarray(r_seq, dtype=np.int64)

            if len(q_seq) > T:
                q_seq = q_seq[-T:]
                r_seq = r_seq[-T:]

            tlen = len(q_seq)
            if tlen == 0:
                continue

            q[i, :tlen] = torch.from_numpy(q_seq)
            r[i, :tlen] = torch.from_numpy(r_seq)
            tgt[i, :tlen] = torch.from_numpy(r_seq.astype(np.float32))
            valid[i, :tlen] = 1.0

        total_valid = int(valid.sum().item())
        return q, r, tgt, total_valid

    def _evaluate_on_seqs(
        self,
        seqs: List[Tuple[np.ndarray, np.ndarray]],
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
                q, r, tgt, total_valid = self._build_batch_tensors(batch)
                if total_valid == 0 or q is None:
                    continue

                p = self._forward_core(q, r)  # (B,T)

                eps = 1e-7
                loss_mat = -(
                    tgt * torch.log(p.clamp_min(eps)) +
                    (1 - tgt) * torch.log((1 - p).clamp_min(eps))
                )
                loss_sum = loss_mat.sum()

                total_loss_sum += float(loss_sum.detach().cpu().item())
                total_targets += int(total_valid)

                mask = tgt >= 0  # toutes les positions valides
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

    # ------------------------------------------------------------------
    # Offline training (per-user train/val split)
    # ------------------------------------------------------------------
    def fit(self, train_df, subjects_df, show_progress: bool = True, val_frac: Optional[float] = None, **kwargs):
        """
        Train DKVMN on the 'train' split with per-user train/val split.
        "Single-skill" implementation: take the first concept in the list per interaction,
        as done for SAKT.
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

        user_seqs = self._build_user_seqs(df)
        if len(user_seqs) == 0:
            return

        if val_frac is None:
            val_frac = self.val_frac

        train_seqs, val_seqs = self._split_train_val_by_user(user_seqs, val_frac=val_frac)

        if len(train_seqs) == 0:
            return

        opt = optim.Adam(self.parameters(), lr=self.lr)

        # Optional tqdm
        try:
            from tqdm.auto import tqdm  # type: ignore

            def _p(it, desc): return tqdm(it, desc=desc) if show_progress else it
        except Exception:
            def _p(it, desc): return it

        best_val_auc = -float("inf")
        best_state = None
        epochs_no_improve = 0

        for ep in range(self.epochs):
            self.train()
            total_loss = 0.0
            total_count = 0

            for b_start in _p(range(0, len(train_seqs), self.batch_size),
                              desc=f"DKVMN train epoch {ep+1}/{self.epochs}"):
                batch = train_seqs[b_start:b_start + self.batch_size]
                q, r, tgt, total_valid = self._build_batch_tensors(batch)
                if total_valid == 0 or q is None:
                    continue

                p = self._forward_core(q, r)  # (B,T)

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
                val_bce, val_acc, val_auc = self._evaluate_on_seqs(
                    val_seqs,
                    batch_size=self.batch_size,
                )
                print(
                    f"DKVMN epoch {ep+1}/{self.epochs} — "
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
                            f"DKVMN early stopping at epoch {ep+1}, "
                            f"best val_AUC={best_val_auc:.4f}"
                        )
                        break
            else:
                print(f"DKVMN epoch {ep+1}/{self.epochs} — train_BCE={train_bce:.4f}")

        if best_state is not None:
            self.load_state_dict(best_state)
            self.to(self.device)
