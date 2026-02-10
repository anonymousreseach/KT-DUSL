#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_next_correctness_eedi_all_subjects.py

EEDI online next-question correctness evaluation using ALL subjects attached
to each QuestionId (multi-tag).

This simplified version keeps a SINGLE fixed behavior:
- Unconditional SL propagation (former "P1") as the only supported mode.
- Removed all policy switches (P1/P2/P3/P4) and all uncertainty-threshold flags (u_th, min_evidence_up, etc.).

Inputs:
- --train_csv : interactions [UserId, QuestionId, IsCorrect]
- --subject_metadata_csv : hierarchy (SubjectId, ParentId)
- --question_metadata_csv : mapping QuestionId -> list of SubjectIds (multi-tag)

Outputs:
- Prints global metrics for: leaf_mean, leaf_sl, prop_mean, prop_sl
- Optionally writes run-level metrics CSV, per-user CSV, and per-interaction predictions CSV

Sharding:
- --shard_idx i --shard_count N keeps users u where (u % shard_count) == shard_idx

Per-interaction per-concept export (if --out_predictions_csv is provided):
- concept_ids (JSON list)
- u_leaf_by_concept (JSON dict ConceptId->u_leaf)
- u_prop_by_concept (JSON dict ConceptId->u_prop)
- evidence_by_concept (JSON dict ConceptId->(r+s) evidence at leaf before propagation)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from collections import defaultdict
import ast

import numpy as np
import pandas as pd


# -------------------------
# Metrics
# -------------------------

def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None

    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1, dtype=float)

    sorted_p = y_prob[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            avg_rank = float(np.mean(ranks[order[i:j + 1]]))
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    pos = (y_true == 1)
    n_pos = int(np.sum(pos))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    sum_ranks_pos = float(np.sum(ranks[pos]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def accuracy_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    y_pred = (y_prob >= float(thr)).astype(int)
    return float(np.mean(y_pred == y_true))


def f1_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    y_pred = (y_prob >= float(thr)).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def compute_metrics(
    y_true_list: List[int],
    y_prob_list: List[float],
    acc_thr: float,
) -> Tuple[int, Optional[float], float, float, float]:
    y_true = np.array(y_true_list, dtype=int)
    y_prob = np.array(y_prob_list, dtype=float)
    n = int(len(y_true))
    if n == 0:
        return 0, None, 0.0, 0.0, 0.0
    auc = roc_auc(y_true, y_prob)
    acc = accuracy_at_threshold(y_true, y_prob, thr=acc_thr)
    f1 = f1_at_threshold(y_true, y_prob, thr=acc_thr)
    ll = logloss(y_true, y_prob)
    return n, auc, acc, f1, ll


def append_df_to_csv(df_new: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    df_new.to_csv(path, index=False, mode="a", header=(not exists))


# -------------------------
# Aggregation
# -------------------------

def aggregate_probs(ps: List[float], mode: str) -> float:
    if not ps:
        return 0.5
    if mode == "mean":
        return float(np.mean(ps))
    if mode == "min":
        return float(np.min(ps))
    if mode == "max":
        return float(np.max(ps))
    raise ValueError(f"Unknown aggregation: {mode}")


def weighted_mean(values: List[float], weights: List[float], fallback: float = 0.5) -> float:
    if not values:
        return fallback
    wsum = float(np.sum(weights))
    if wsum <= 1e-12:
        return fallback
    return float(np.sum(np.array(values) * np.array(weights)) / wsum)


# -------------------------
# Leaf estimators (online)
# -------------------------

@dataclass
class LeafMeanState:
    n_attempts: DefaultDict[int, int]
    n_correct: DefaultDict[int, int]
    a_fallback: float

    def mastery(self, sid: int) -> float:
        n = self.n_attempts[sid]
        if n <= 0:
            return float(self.a_fallback)
        return float(self.n_correct[sid] / n)

    def update(self, sids: List[int], y: int) -> None:
        for sid in sids:
            self.n_attempts[sid] += 1
            if int(y) == 1:
                self.n_correct[sid] += 1


@dataclass
class LeafSLState:
    alpha: DefaultDict[int, float]
    beta: DefaultDict[int, float]
    a: float
    W: float

    @property
    def alpha0(self) -> float:
        return float(self.a * self.W)

    @property
    def beta0(self) -> float:
        return float((1.0 - self.a) * self.W)

    def ensure(self, sid: int) -> None:
        if sid not in self.alpha:
            self.alpha[sid] = float(self.alpha0)
            self.beta[sid] = float(self.beta0)

    def expected(self, sid: int) -> float:
        self.ensure(sid)
        s = self.alpha[sid] + self.beta[sid]
        if s <= 0:
            return float(self.a)
        return float(self.alpha[sid] / s)

    def opinion(self, sid: int) -> Tuple[float, float, float, float]:
        self.ensure(sid)
        r = max(0.0, self.alpha[sid] - self.alpha0)
        s = max(0.0, self.beta[sid] - self.beta0)
        denom = r + s + self.W
        if denom <= 1e-12:
            return 0.0, 0.0, 1.0, float(self.a)
        b = r / denom
        d = s / denom
        u = self.W / denom
        return float(b), float(d), float(u), float(self.a)

    def update(self, sids: List[int], y: int) -> None:
        for sid in sids:
            self.ensure(sid)
            if int(y) == 1:
                self.alpha[sid] += 1.0
            else:
                self.beta[sid] += 1.0


# -------------------------
# Propagation
# -------------------------

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def scalar_discount_to_a(x: float, t: float, a: float) -> float:
    return clamp01(a + float(t) * (float(x) - float(a)))


def propagate_mean_from_leaf(
    leaf_state: LeafMeanState,
    subj_ids: List[int],
    parent_of: Dict[int, Optional[int]],
    K: int,
    t_up: float,
    t_down: float,
    one_hop: bool = False,
    no_discount: bool = False,
) -> Dict[int, float]:
    obs: Dict[int, Optional[float]] = {}
    for sid in subj_ids:
        n = leaf_state.n_attempts[sid]
        obs[sid] = leaf_state.mastery(sid) if n > 0 else None

    base_state: Dict[int, float] = {}
    for sid in subj_ids:
        base_state[sid] = obs[sid] if obs[sid] is not None else float(leaf_state.a_fallback)

    def edge_msg(x: float, t: float) -> float:
        return float(x) if no_discount else scalar_discount_to_a(x, t=t, a=leaf_state.a_fallback)

    if one_hop:
        incoming: DefaultDict[int, List[float]] = defaultdict(list)
        for child, parent in parent_of.items():
            if parent is None:
                continue
            incoming[parent].append(edge_msg(base_state[child], t=t_up))
            incoming[child].append(edge_msg(base_state[parent], t=t_down))

        new_state: Dict[int, float] = {}
        for sid in subj_ids:
            inc = incoming.get(sid, [])
            inc_mean = float(np.mean(inc)) if inc else None
            if obs[sid] is not None:
                new_state[sid] = clamp01(obs[sid] if inc_mean is None else 0.5 * obs[sid] + 0.5 * inc_mean)
            else:
                new_state[sid] = clamp01(base_state[sid] if inc_mean is None else inc_mean)
        return new_state

    state = dict(base_state)
    for _ in range(int(max(0, K))):
        incoming: DefaultDict[int, List[float]] = defaultdict(list)
        for child, parent in parent_of.items():
            if parent is None:
                continue
            incoming[parent].append(edge_msg(state[child], t=t_up))
            incoming[child].append(edge_msg(state[parent], t=t_down))

        new_state: Dict[int, float] = {}
        for sid in subj_ids:
            inc = incoming.get(sid, [])
            inc_mean = float(np.mean(inc)) if inc else None
            if obs[sid] is not None:
                new_state[sid] = clamp01(obs[sid] if inc_mean is None else 0.5 * obs[sid] + 0.5 * inc_mean)
            else:
                new_state[sid] = clamp01(state[sid] if inc_mean is None else inc_mean)
        state = new_state

    return state


def normalize_opinion(b: float, d: float, u: float, a: float) -> Tuple[float, float, float, float]:
    b = clamp01(b)
    d = clamp01(d)
    u = clamp01(u)
    s = b + d + u
    if s <= 1e-12:
        return 0.0, 0.0, 1.0, float(a)
    b /= s
    d /= s
    u = clamp01(1.0 - b - d)
    return float(b), float(d), float(u), float(a)


def sl_discount(b: float, d: float, u: float, a: float, t: float) -> Tuple[float, float, float, float]:
    b, d, u, a = normalize_opinion(b, d, u, a)
    b2 = float(t) * b
    d2 = float(t) * d
    u2 = 1.0 - float(t) * (b + d)
    return normalize_opinion(b2, d2, u2, a)


def sl_consensus(
    w1: Tuple[float, float, float, float],
    w2: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    b1, d1, u1, a = normalize_opinion(*w1)
    b2, d2, u2, _a2 = normalize_opinion(*w2)
    Kc = u1 + u2 - (u1 * u2)
    if Kc <= 1e-12:
        return normalize_opinion(0.5 * (b1 + b2), 0.5 * (d1 + d2), 0.5 * (u1 + u2), a)
    b = (b1 * u2 + b2 * u1) / Kc
    d = (d1 * u2 + d2 * u1) / Kc
    u = (u1 * u2) / Kc
    return normalize_opinion(float(b), float(d), float(u), float(a))


def sl_consensus_many(ws: List[Tuple[float, float, float, float]], a: float) -> Tuple[float, float, float, float]:
    if not ws:
        return (0.0, 0.0, 1.0, float(a))
    cur = normalize_opinion(*ws[0])
    for w in ws[1:]:
        cur = sl_consensus(cur, w)
    return cur


def sl_expected(b: float, d: float, u: float, a: float) -> float:
    b, d, u, a = normalize_opinion(b, d, u, a)
    return float(b + a * u)


def propagate_sl_from_leaf(
    leaf_sl: LeafSLState,
    subj_ids: List[int],
    parent_of: Dict[int, Optional[int]],
    K: int,
    t_up: float,
    t_down: float,
    one_hop: bool = False,
    no_discount: bool = False,
) -> Dict[int, Tuple[float, float, float, float]]:
    # Observations at current step (leaf-level SL states)
    obs: Dict[int, Tuple[float, float, float, float]] = {}
    for sid in subj_ids:
        leaf_sl.ensure(sid)
        r = max(0.0, leaf_sl.alpha[sid] - leaf_sl.alpha0)
        s = max(0.0, leaf_sl.beta[sid] - leaf_sl.beta0)
        if (r + s) > 0.0:
            op = leaf_sl.opinion(sid)
        else:
            op = (0.0, 0.0, 1.0, float(leaf_sl.a))
        obs[sid] = normalize_opinion(*op)

    def edge_msg(op: Tuple[float, float, float, float], t: float) -> Tuple[float, float, float, float]:
        return normalize_opinion(*op) if no_discount else sl_discount(*op, t=t)

    if one_hop:
        incoming: DefaultDict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)
        for child, parent in parent_of.items():
            if parent is None:
                continue
            incoming[parent].append(edge_msg(obs[child], t=t_up))
            incoming[child].append(edge_msg(obs[parent], t=t_down))

        new_state: Dict[int, Tuple[float, float, float, float]] = {}
        for sid in subj_ids:
            prop = sl_consensus_many(incoming.get(sid, []), a=float(leaf_sl.a))
            new_state[sid] = sl_consensus(obs[sid], prop)
        return new_state

    state: Dict[int, Tuple[float, float, float, float]] = dict(obs)
    for _ in range(int(max(0, K))):
        incoming: DefaultDict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)

        for child, parent in parent_of.items():
            if parent is None:
                continue
            incoming[parent].append(edge_msg(state[child], t=t_up))
            incoming[child].append(edge_msg(state[parent], t=t_down))

        new_state: Dict[int, Tuple[float, float, float, float]] = {}
        for sid in subj_ids:
            prop = sl_consensus_many(incoming.get(sid, []), a=float(leaf_sl.a))
            new_state[sid] = sl_consensus(obs[sid], prop)

        state = new_state

    return state


# -------------------------
# EEDI: build hierarchy + q2s (MULTI-TAG)
# -------------------------

def build_hierarchy_from_subject_metadata(subject_metadata_csv: str) -> Tuple[List[int], Dict[int, Optional[int]]]:
    df = pd.read_csv(subject_metadata_csv)
    if "SubjectId" not in df.columns or "ParentId" not in df.columns:
        raise KeyError(f"Expected columns SubjectId, ParentId in {subject_metadata_csv}. Found: {list(df.columns)}")

    parent_of: Dict[int, Optional[int]] = {}
    all_ids = sorted(df["SubjectId"].dropna().astype(int).unique().tolist())

    for r in df.itertuples(index=False):
        sid = int(getattr(r, "SubjectId"))
        pid = getattr(r, "ParentId")
        parent_of[sid] = None if pd.isna(pid) else int(pid)

    return all_ids, parent_of


def _parse_subject_list(x: Any) -> List[int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (int, np.integer)):
        return [int(x)]
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []

    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                out = []
                for v in obj:
                    try:
                        out.append(int(v))
                    except Exception:
                        pass
                return out
            if isinstance(obj, (int, np.integer)):
                return [int(obj)]
        except Exception:
            pass

    for sep in ["|", ";", ",", " "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip() != ""]
            out = []
            for p in parts:
                try:
                    out.append(int(float(p)))
                except Exception:
                    continue
            return out

    try:
        return [int(float(s))]
    except Exception:
        return []


def build_q2s_all_subjects(question_metadata_csv: str) -> Dict[int, List[int]]:
    df = pd.read_csv(question_metadata_csv)
    if "QuestionId" not in df.columns:
        raise KeyError(f"Missing QuestionId in {question_metadata_csv}. Columns={list(df.columns)}")

    candidates = [
        "SubjectIds", "SubjectIdList", "SubjectId_list", "subject_ids", "subjects",
        "SubjectId", "SubjectID", "subject_id",
    ]
    subj_col = None
    for c in candidates:
        if c in df.columns:
            subj_col = c
            break
    if subj_col is None:
        raise KeyError(
            f"Could not find a subject column in {question_metadata_csv}. "
            f"Tried {candidates}. Columns={list(df.columns)}"
        )

    q2s: Dict[int, List[int]] = {}
    for r in df.itertuples(index=False):
        qid = int(getattr(r, "QuestionId"))
        raw = getattr(r, subj_col)
        sids = _parse_subject_list(raw)
        seen = set()
        sids_u = []
        for s in sids:
            if s not in seen:
                seen.add(s)
                sids_u.append(int(s))
        q2s[qid] = sids_u

    return q2s


def build_interactions_from_eedi_train(train_csv: str, max_users: int) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    needed = {"UserId", "QuestionId", "IsCorrect"}
    if not needed.issubset(set(df.columns)):
        raise KeyError(f"train file missing columns {needed}. Found: {list(df.columns)}")

    df = df[df["UserId"].notna() & df["QuestionId"].notna() & df["IsCorrect"].notna()].copy()
    df["UserId"] = df["UserId"].astype(int)
    df["QuestionId"] = df["QuestionId"].astype(int)
    df["IsCorrect"] = df["IsCorrect"].astype(int)

    user_ids = sorted(df["UserId"].unique().tolist())[: int(max_users)]
    df = df[df["UserId"].isin(user_ids)].copy()

    df["_row"] = np.arange(len(df), dtype=int)
    df = df.sort_values(["UserId", "_row"], kind="mergesort").drop(columns=["_row"])
    return df[["UserId", "QuestionId", "IsCorrect"]].reset_index(drop=True)


# -------------------------
# Online evaluation loop
# -------------------------

@dataclass
class MethodOutputs:
    y_true: List[int]
    y_prob: List[float]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--subject_metadata_csv", required=True)
    ap.add_argument("--question_metadata_csv", required=True,
                    help="Question metadata with ALL subjects (multi-tag).")

    ap.add_argument("--max_users", type=int, default=300)

    # Sharding
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--shard_count", type=int, default=1)

    ap.add_argument("--agg", default="mean", choices=["mean", "min", "max"])
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--W", type=float, default=2.0)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--t_up", type=float, default=0.8)
    ap.add_argument("--t_down", type=float, default=0.8)
    ap.add_argument("--one_hop", action="store_true")
    ap.add_argument("--sl_uncertainty_weighting", action="store_true")
    ap.add_argument("--no_discount", action="store_true")

    ap.add_argument("--acc_threshold", type=float, default=0.5)

    ap.add_argument("--out_csv", default="evaluation/eedi/metrics_online_next_correctness.csv")
    ap.add_argument("--out_user_csv", default=None)
    ap.add_argument("--out_predictions_csv", default=None)

    args = ap.parse_args()

    subj_ids, parent_of = build_hierarchy_from_subject_metadata(args.subject_metadata_csv)
    q2s = build_q2s_all_subjects(args.question_metadata_csv)

    df = build_interactions_from_eedi_train(args.train_csv, max_users=int(args.max_users))
    user_ids = sorted(df["UserId"].unique().tolist())[: int(args.max_users)]

    # Sharding
    if int(args.shard_count) > 1:
        if not (0 <= int(args.shard_idx) < int(args.shard_count)):
            raise ValueError("--shard_idx must be in [0, shard_count-1]")
        user_ids = [u for u in user_ids if (int(u) % int(args.shard_count)) == int(args.shard_idx)]
    df = df[df["UserId"].isin(user_ids)].copy()

    methods = {
        "leaf_mean": MethodOutputs([], []),
        "leaf_sl": MethodOutputs([], []),
        "prop_mean": MethodOutputs([], []),
        "prop_sl": MethodOutputs([], []),
    }

    per_user_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []

    for uid in user_ids:
        udf = df[df["UserId"] == uid].reset_index(drop=True)

        leaf_mean = LeafMeanState(defaultdict(int), defaultdict(int), a_fallback=float(args.a))
        leaf_sl = LeafSLState(defaultdict(float), defaultdict(float), a=float(args.a), W=float(args.W))

        user_methods = {k: MethodOutputs([], []) for k in methods.keys()}

        for step, row in enumerate(udf.itertuples(index=False)):
            qid = int(getattr(row, "QuestionId"))
            y = int(getattr(row, "IsCorrect"))

            sids = q2s.get(qid, [])
            sids = [s for s in sids if s in parent_of]

            # Leaf mean
            ps_mean = [leaf_mean.mastery(s) for s in sids] if sids else []
            p_leaf_mean = aggregate_probs(ps_mean, args.agg)

            # Leaf SL
            if not args.sl_uncertainty_weighting:
                ps_sl = [leaf_sl.expected(s) for s in sids] if sids else []
                p_leaf_sl = aggregate_probs(ps_sl, args.agg)
            else:
                vals, wts = [], []
                for s in sids:
                    b, d, u, a = leaf_sl.opinion(s)
                    vals.append(sl_expected(b, d, u, a))
                    wts.append(1.0 - u)
                p_leaf_sl = weighted_mean(vals, wts, fallback=0.5)

            # Propagate mean
            state_mean = propagate_mean_from_leaf(
                leaf_state=leaf_mean,
                subj_ids=subj_ids,
                parent_of=parent_of,
                K=int(args.K),
                t_up=float(args.t_up),
                t_down=float(args.t_down),
                one_hop=bool(args.one_hop),
                no_discount=bool(args.no_discount),
            )
            ps_pmean = [state_mean[s] for s in sids] if sids else []
            p_prop_mean = aggregate_probs(ps_pmean, args.agg)

            # Propagate SL (unconditional, fixed behavior)
            state_sl = propagate_sl_from_leaf(
                leaf_sl=leaf_sl,
                subj_ids=subj_ids,
                parent_of=parent_of,
                K=int(args.K),
                t_up=float(args.t_up),
                t_down=float(args.t_down),
                one_hop=bool(args.one_hop),
                no_discount=bool(args.no_discount),
            )

            if not args.sl_uncertainty_weighting:
                ps_psl = [sl_expected(*state_sl[s]) for s in sids] if sids else []
                p_prop_sl = aggregate_probs(ps_psl, args.agg)
            else:
                vals, wts = [], []
                for s in sids:
                    b, d, u, a = state_sl[s]
                    vals.append(sl_expected(b, d, u, a))
                    wts.append(1.0 - u)
                p_prop_sl = weighted_mean(vals, wts, fallback=0.5)

            # Collect overall method preds
            for name, p in [
                ("leaf_mean", p_leaf_mean),
                ("leaf_sl", p_leaf_sl),
                ("prop_mean", p_prop_mean),
                ("prop_sl", p_prop_sl),
            ]:
                methods[name].y_true.append(y)
                methods[name].y_prob.append(p)
                user_methods[name].y_true.append(y)
                user_methods[name].y_prob.append(p)

            # Per-concept export at this interaction
            if sids:
                u_leaf_by_concept = {int(s): float(leaf_sl.opinion(s)[2]) for s in sids}
                u_prop_by_concept = {int(s): float(state_sl[s][2]) for s in sids}

                ev_by_concept: Dict[int, float] = {}
                for s in sids:
                    leaf_sl.ensure(s)
                    r = max(0.0, leaf_sl.alpha[s] - leaf_sl.alpha0)
                    t = max(0.0, leaf_sl.beta[s] - leaf_sl.beta0)
                    ev_by_concept[int(s)] = float(r + t)

                concept_ids = [int(s) for s in sids]
            else:
                u_leaf_by_concept = {}
                u_prop_by_concept = {}
                ev_by_concept = {}
                concept_ids = []

            if args.out_predictions_csv is not None:
                pred_rows.append({
                    "UserId": int(uid),
                    "step": int(step),
                    "QuestionId": int(qid),
                    "IsCorrect": int(y),

                    "p_leaf_mean": float(p_leaf_mean),
                    "p_leaf_sl": float(p_leaf_sl),
                    "p_prop_mean": float(p_prop_mean),
                    "p_prop_sl": float(p_prop_sl),

                    # Per-concept details as JSON strings
                    "concept_ids": json.dumps(concept_ids, separators=(",", ":")),
                    "u_leaf_by_concept": json.dumps(u_leaf_by_concept, separators=(",", ":")),
                    "u_prop_by_concept": json.dumps(u_prop_by_concept, separators=(",", ":")),
                    "evidence_by_concept": json.dumps(ev_by_concept, separators=(",", ":")),

                    "K": int(args.K),
                    "t_up": float(args.t_up),
                    "t_down": float(args.t_down),
                    "no_discount": bool(args.no_discount),
                    "one_hop": bool(args.one_hop),
                    "sl_uncertainty_weighting": bool(args.sl_uncertainty_weighting),
                    "agg": str(args.agg),
                    "a": float(args.a),
                    "W": float(args.W),
                    "acc_threshold": float(args.acc_threshold),
                    "shard_count": int(args.shard_count),
                    "shard_idx": int(args.shard_idx),
                })

            # Online update AFTER logging
            leaf_mean.update(sids, y)
            leaf_sl.update(sids, y)

        if args.out_user_csv is not None:
            for mname, out in user_methods.items():
                n, auc, acc, f1, ll = compute_metrics(out.y_true, out.y_prob, acc_thr=float(args.acc_threshold))
                per_user_rows.append({
                    "UserId": int(uid),
                    "method": str(mname),
                    "n_predictions": int(n),
                    "auc": None if auc is None else float(auc),
                    "accuracy": float(acc),
                    "f1": float(f1),
                    "logloss": float(ll),
                    "K": int(args.K),
                    "t_up": float(args.t_up),
                    "t_down": float(args.t_down),
                    "no_discount": bool(args.no_discount),
                    "one_hop": bool(args.one_hop),
                    "sl_uncertainty_weighting": bool(args.sl_uncertainty_weighting),
                    "agg": str(args.agg),
                    "a": float(args.a),
                    "W": float(args.W),
                    "acc_threshold": float(args.acc_threshold),
                    "shard_count": int(args.shard_count),
                    "shard_idx": int(args.shard_idx),
                })

    run_meta = {
        "dataset": "eedi_multi",
        "train_csv": str(args.train_csv),
        "subject_metadata_csv": str(args.subject_metadata_csv),
        "question_metadata_csv": str(args.question_metadata_csv),
        "max_users": int(args.max_users),
        "n_users_evaluated": int(len(user_ids)),
        "agg": str(args.agg),
        "a": float(args.a),
        "W": float(args.W),
        "K": int(args.K),
        "t_up": float(args.t_up),
        "t_down": float(args.t_down),
        "one_hop": bool(args.one_hop),
        "sl_uncertainty_weighting": bool(args.sl_uncertainty_weighting),
        "no_discount": bool(args.no_discount),
        "acc_threshold": float(args.acc_threshold),
        "shard_count": int(args.shard_count),
        "shard_idx": int(args.shard_idx),
    }

    rows_out: List[Dict[str, Any]] = []
    for name, out in methods.items():
        n, auc, acc, f1, ll = compute_metrics(out.y_true, out.y_prob, acc_thr=float(args.acc_threshold))
        auc_str = "NA" if auc is None else f"{auc:.6f}"
        print(
            f"{name:10s} n={n:7d}  AUC={auc_str}  ACC@{args.acc_threshold:.2f}={acc:.6f}  "
            f"F1@{args.acc_threshold:.2f}={f1:.6f}  LogLoss={ll:.6f}"
        )

        row = dict(run_meta)
        row.update({
            "method": name,
            "n_predictions": int(n),
            "auc": None if auc is None else float(auc),
            "accuracy": float(acc),
            "f1": float(f1),
            "logloss": float(ll),
        })
        rows_out.append(row)

    out_path = Path(args.out_csv)
    append_df_to_csv(pd.DataFrame(rows_out), out_path)
    print("Appended:", str(out_path))

    if args.out_user_csv is not None:
        out_user = Path(args.out_user_csv)
        out_user.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_user_rows).to_csv(out_user, index=False)
        print("Wrote:", str(out_user))

    if args.out_predictions_csv is not None:
        out_pred = Path(args.out_predictions_csv)
        out_pred.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pred_rows).to_csv(out_pred, index=False)
        print("Wrote:", str(out_pred))


if __name__ == "__main__":
    main()
