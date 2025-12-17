# ktusl/models/dkt.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import KTModel


class DKT(KTModel, nn.Module):
    """
    Deep Knowledge Tracing (LSTM) — compatible with the trainer:
      - fit(train, subjects) trains the weights (offline)
      - reset_state()/predict_concept_proba()/update() for the online phase
    """
    def __init__(
        self,
        concept_ids: Iterable[int],
        emb_size: int = 128,
        hidden_size: Optional[int] = None,
        dropout: float = 0.2,
        # Offline training
        epochs: int = 80,
        batch_size: int = 64,
        lr: float = 1e-3,
        max_seq_len: Optional[int] = 200,
        device: Optional[str] = None,
        seed: int = 42,
        # New training hyperparameters
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

        self.emb_size = int(emb_size)
        self.hidden_size = int(hidden_size or emb_size)
        self.dropout_p = float(dropout)

        # token <START> (index = 2C)
        self.start_index = 2 * self.num_c

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # --- network ---
        self.interaction_emb = nn.Embedding(self.start_index + 1, self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.num_c)
        self.sigmoid = nn.Sigmoid()

        self.to(self.device)

        # --- offline training hyperparameters ---
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.max_seq_len = max_seq_len

        # new hyperparameters
        self.val_frac = float(val_frac)
        self.patience = int(patience)

        # --- per-user online state ---
        # user -> (h, c, last_y); last_y ∈ [0,1]^C
        self._state: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Prior probability for cold-start (output with h=0)
        with torch.no_grad():
            h0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
            y0 = self.sigmoid(self.out(h0))  # (1,1,C)
            self._y_init = y0.squeeze(0).squeeze(0).detach()  # (C,)

    # ---------------- API KTModel (phase test online) ----------------

    def reset_state(self):
        self._state.clear()

    def predict_concept_proba(self, user: int, concept: int) -> float:
        idx = self.id2idx.get(int(concept))
        if idx is None:
            return float(self._y_init.mean().item())
        st = self._state.get(int(user))
        if st is None:
            return float(self._y_init[idx].item())
        _, _, last_y = st
        return float(last_y[idx].item())

    def update(self, user: int, concepts: List[int], correct: int, weights: Optional[List[float]] = None):
        """Advance the LSTM state by one step with the current interaction (concepts, y)."""
        u = int(user)
        y = 1 if int(correct) == 1 else 0

        idxs = [self.id2idx[c] for c in concepts if c in self.id2idx]
        if len(idxs) == 0:
            return

        inter_indices = [i + (y * self.num_c) for i in idxs]
        with torch.no_grad():
            emb = self.interaction_emb(
                torch.tensor(inter_indices, device=self.device, dtype=torch.long)
            )
            x_t = emb.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,emb_size)

        h_prev, c_prev, _ = self._state.get(
            u,
            (
                torch.zeros(1, 1, self.hidden_size, device=self.device),
                torch.zeros(1, 1, self.hidden_size, device=self.device),
                self._y_init.clone(),
            ),
        )

        self.eval()
        with torch.no_grad():
            h_t, (h_new, c_new) = self.lstm(x_t, (h_prev, c_prev))      # (1,1,H)
            h_t = self.dropout(h_t)                                     # no-op en eval()
            y_vec = self.sigmoid(self.out(h_t)).squeeze(0).squeeze(0)   # (C,)

        self._state[u] = (h_new.detach(), c_new.detach(), y_vec.detach())

    # ------------------------ Helpers internes ------------------------

    def _build_user_seqs(self, df) -> List[Tuple[int, List[Tuple[List[int], int]]]]:
        """
        Build sequences per user:
          [(user_id, seq)], where seq = [(cids_t, y_t)]
        """
        seqs_by_user: List[Tuple[int, List[Tuple[List[int], int]]]] = []

        for uid, g in df.groupby("UserId", sort=False):
            seq: List[Tuple[List[int], int]] = []
            for _, r in g.iterrows():
                cids = [self.id2idx[c] for c in r["SubjectId"] if c in self.id2idx]
                if len(cids) == 0:
                    continue
                y = int(r["IsCorrect"])
                seq.append((cids, y))
            if len(seq) >= 2:
                if self.max_seq_len is not None and len(seq) > self.max_seq_len:
                    seq = seq[-self.max_seq_len:]
                seqs_by_user.append((int(uid), seq))
        return seqs_by_user

    def _split_train_val_by_user(
        self,
        user_seqs: List[Tuple[int, List[Tuple[List[int], int]]]],
        val_frac: float,
        seed: int = 42,
    ):
        """
        Train/val split per user (not per interaction).
        Returns (train_seqs, val_seqs).
        """
        if val_frac <= 0 or len(user_seqs) < 2:
            # All sequences in train; no validation
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

    def _build_batch_tensors(self, batch: List[List[Tuple[List[int], int]]]):
        """
        From a batch of sequences, build:
          - X: (B,T,E)
          - targets: list of (i, t, c, y)
        """
        if len(batch) == 0:
            return None, []

        T = max(len(s) for s in batch)
        B = len(batch)
        X = torch.zeros(B, T, self.emb_size, device=self.device)
        targets: List[Tuple[int, int, int, float]] = []

        for i, seq in enumerate(batch):
            for t in range(T):
                if t == 0:
                    emb = self.interaction_emb(
                        torch.tensor([self.start_index], device=self.device, dtype=torch.long)
                    ).mean(dim=0)
                else:
                    if t - 1 < len(seq):
                        prev_cids, prev_y = seq[t - 1]
                        inter_idx = [c + prev_y * self.num_c for c in prev_cids]
                        emb = self.interaction_emb(
                            torch.tensor(inter_idx, device=self.device, dtype=torch.long)
                        ).mean(dim=0)
                    else:
                        emb = self.interaction_emb(
                            torch.tensor([self.start_index], device=self.device, dtype=torch.long)
                        ).mean(dim=0)
                X[i, t, :] = emb

                if t < len(seq):
                    cur_cids, cur_y = seq[t]
                    for c in cur_cids:
                        targets.append((i, t, c, float(cur_y)))

        return X, targets

    @staticmethod
    def _binary_auc(labels: List[int], scores: List[float]) -> float:
        """
        Binary AUC via the Mann–Whitney formula.
        labels ∈ {0,1}
        """
        if len(labels) == 0:
            return float("nan")
        y = np.array(labels, dtype=np.int64)
        s = np.array(scores, dtype=np.float64)
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        order = np.argsort(s)  # ascending
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
        pos_ranks_sum = ranks[y == 1].sum()
        auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)

    def _evaluate_on_seqs(
        self,
        seqs: List[List[Tuple[List[int], int]]],
        bce_loss: nn.BCELoss,
        batch_size: int,
    ):
        """
        Evaluate BCE, ACC, AUC on a set of sequences (offline).
        """
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
                X, targets = self._build_batch_tensors(batch)
                if X is None or len(targets) == 0:
                    continue

                H, _ = self.lstm(X)             # (B,T,H)
                H = self.dropout(H)             # no-op en eval()
                Y = self.sigmoid(self.out(H))   # (B,T,C)

                loss_sum = torch.zeros((), device=self.device)
                for i, t, c, y in targets:
                    p = Y[i, t, c]
                    loss_sum = loss_sum + bce_loss(p, torch.tensor(y, device=self.device))
                    all_probs.append(float(p.item()))
                    all_labels.append(int(y))

                total_loss_sum += float(loss_sum.detach().cpu().item())
                total_targets += len(targets)

        bce = total_loss_sum / max(1, total_targets)
        if len(all_labels) == 0:
            return bce, float("nan"), float("nan")

        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        correct = sum(int(p == y) for p, y in zip(preds, all_labels))
        acc = correct / max(1, len(all_labels))
        auc = self._binary_auc(all_labels, all_probs)

        return bce, acc, auc

    # ------------------------ OFFLINE TRAINING (per-student train/val split) ------------------------

    def fit(self, train_df, subjects_df=None, show_progress: bool = True, val_frac: Optional[float] = None, **kwargs):
        """
        Train the network with a per-user train/val split.
        - train_df: dataframe with columns ['UserId', 'SubjectId', 'IsCorrect', optionally 'DateAnswered']
        - subjects_df: unused here (API compatibility)
        """
        df = train_df.copy()
        cols = ["UserId", "SubjectId", "IsCorrect"]
        if "DateAnswered" in df.columns:
            cols.append("DateAnswered")
        df = df[cols].copy()

        # Ensure SubjectId is a list
        df["SubjectId"] = df["SubjectId"].apply(
            lambda x: x if isinstance(x, (list, tuple)) else []
        )

        # Temporal sorting (if available)
        if "DateAnswered" in df.columns:
            df = df.sort_values(["UserId", "DateAnswered"], kind="mergesort")
        else:
            df = df.sort_values(["UserId"], kind="mergesort").reset_index(drop=True)

        # Build per-user sequences
        user_seqs = self._build_user_seqs(df)
        if len(user_seqs) == 0:
            return

        # Per-user train / val split
        if val_frac is None:
            val_frac = self.val_frac

        train_seqs, val_seqs = self._split_train_val_by_user(user_seqs, val_frac=val_frac)

        if len(train_seqs) == 0:
            return

        self.train()
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

        for ep in range(self.epochs):
            self.train()
            total_loss = 0.0
            total_count = 0

            for b_start in _p(range(0, len(train_seqs), self.batch_size),
                              desc=f"DKT train epoch {ep+1}/{self.epochs}"):
                batch = train_seqs[b_start:b_start + self.batch_size]
                X, targets = self._build_batch_tensors(batch)
                if X is None or len(targets) == 0:
                    continue

                opt.zero_grad()
                H, _ = self.lstm(X)             # (B,T,H)
                H = self.dropout(H)
                Y = self.sigmoid(self.out(H))   # (B,T,C)

                loss_sum = torch.zeros((), device=self.device)
                for i, t, c, y in targets:
                    loss_sum = loss_sum + bce(Y[i, t, c], torch.tensor(y, device=self.device))
                count = float(len(targets))
                loss = loss_sum / max(1.0, count)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                opt.step()

                total_loss += float(loss_sum.detach().cpu().item())
                total_count += int(count)

            train_bce = total_loss / max(1, total_count)

            # Validation evaluation (if present)
            if len(val_seqs) > 0:
                val_bce, val_acc, val_auc = self._evaluate_on_seqs(
                    val_seqs, bce_loss=bce, batch_size=self.batch_size
                )
                print(
                    f"Epoch {ep+1}/{self.epochs} — "
                    f"train_BCE={train_bce:.4f} | "
                    f"val_BCE={val_bce:.4f}, val_ACC={val_acc:.4f}, val_AUC={val_auc:.4f}"
                )

                # Early stopping on val AUC
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(
                            f"Early stopping triggered at epoch {ep+1}, "
                            f"best val_AUC={best_val_auc:.4f}"
                        )
                        break
            else:
                # No validation: only log the train loss
                print(f"Epoch {ep+1}/{self.epochs} — train_BCE={train_bce:.4f}")

        # Restore the best checkpoint (if validation was tracked)
        if best_state is not None:
            self.load_state_dict(best_state)
            self.to(self.device)
