#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_next_correctness_junyi.py

Online next-question correctness evaluation for Junyi.

This version is simplified:
- Single fixed behavior (former "P1"): unconditional propagation for SL (no policy switch).
- Removed all uncertainty-threshold / conditional-propagation flags (u_th, min_evidence_up, etc.).
- Still supports optional sharding:
    --shard_idx i
    --shard_count N
  Sharding is applied AFTER building the interactions DF, on integer UserId:
      keep u iff (u % shard_count) == shard_idx
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd


# -------------------------
# Small utilities (robust columns)
# -------------------------

def _norm_col(c: str) -> str:
    return c.strip().replace("\t", " ").replace("\u00a0", " ")


def _canon(c: str) -> str:
    return _norm_col(c).lower().replace(" ", "").replace("_", "")


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lookup = {_canon(c): c for c in df.columns}
    for cand in candidates:
        rc = lookup.get(_canon(cand))
        if rc is not None:
            return rc
    if required:
        raise KeyError(f"Cannot find column among {candidates}. Available: {list(df.columns)}")
    return None


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Metrics
# -------------------------

def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None

    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1, dtype=float)

    # average ranks for ties
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


# -------------------------
# Domain parsing: tree -> q2s + edges
# -------------------------

def parse_domain(domain_json: Dict[str, Any]) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, Optional[int]]]:
    roots = ((domain_json.get("pkg", {}) or {}).get("roots", []) or [])
    if not roots:
        roots = domain_json.get("roots", []) or domain_json.get("subjects", []) or []

    q2s: Dict[int, List[int]] = {}
    children_of: DefaultDict[int, List[int]] = defaultdict(list)
    parent_of: Dict[int, Optional[int]] = {}

    def visit(node: Dict[str, Any], parent: Optional[int]) -> None:
        sid = int(node["subject_id"])
        parent_of[sid] = parent

        for q in (node.get("questions", []) or []):
            qid = q.get("question_id", q.get("id"))
            if qid is None:
                continue
            qid = int(qid)
            q2s.setdefault(qid, []).append(sid)

        for ch in (node.get("children", []) or []):
            csid = int(ch["subject_id"])
            children_of[sid].append(csid)
            visit(ch, sid)

    for r in roots:
        if isinstance(r, dict) and "subject_id" in r:
            visit(r, None)

    for qid, sids in q2s.items():
        q2s[qid] = sorted(list(dict.fromkeys(sids)))

    if not q2s:
        raise ValueError(
            "Could not extract question->subject mapping from domain_json. "
            "Ensure leaf nodes contain a 'questions' list."
        )

    return q2s, dict(children_of), parent_of


def all_subject_ids(children_of: Dict[int, List[int]], parent_of: Dict[int, Optional[int]]) -> List[int]:
    ids = set(children_of.keys()) | set(parent_of.keys())
    for vs in children_of.values():
        ids.update(vs)
    return sorted(ids)


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

        new_state = {}
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
    conditional: bool = False,  # kept for API compatibility; always False in this script
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
# Online evaluation loop
# -------------------------

@dataclass
class MethodOutputs:
    y_true: List[int]
    y_prob: List[float]


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
# JUNYI raw log -> interactions
# -------------------------

def build_interactions_from_junyi_log(
    log_df: pd.DataFrame,
    qmap_df: pd.DataFrame,
    max_users: int,
    time_sort: bool,
    out_user_map_csv: Optional[str] = None,
) -> pd.DataFrame:
    uuid_col = pick_col(log_df, ["uuid"])
    ucid_col = pick_col(log_df, ["uc id", "ucid", "uc_id", "uc"])
    correct_col = pick_col(log_df, ["is_correct", "iscorrect", "correct"])
    ts_col = pick_col(log_df, ["timestamp_TW", "timestamp_T", "timestamp"], required=False)

    q_ucid_col = pick_col(qmap_df, ["ucid", "uc id", "uc_id"])
    q_qid_col = pick_col(qmap_df, ["QuestionId", "question_id", "qid"])
    qmap = dict(zip(qmap_df[q_ucid_col].astype(str), qmap_df[q_qid_col].astype(int)))

    ucids = log_df[ucid_col].astype(str)
    qids = ucids.map(qmap)
    keep = qids.notna()

    df = log_df.loc[keep].copy()
    df["QuestionId"] = qids.loc[keep].astype(int)

    y_raw = df[correct_col]
    if y_raw.dtype == bool:
        df["IsCorrect"] = y_raw.astype(int)
    else:
        yr = y_raw.astype(str).str.strip().str.lower()
        df["IsCorrect"] = yr.map({"true": 1, "false": 0, "1": 1, "0": 0})
        if df["IsCorrect"].isna().any():
            df["IsCorrect"] = pd.to_numeric(y_raw, errors="coerce")
        df = df[df["IsCorrect"].notna()].copy()
        df["IsCorrect"] = df["IsCorrect"].astype(int)

    user_codes, user_uniques = pd.factorize(df[uuid_col].astype(str), sort=True)
    df["UserId"] = user_codes.astype(int)

    if out_user_map_csv:
        outp = Path(out_user_map_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"uuid": user_uniques.astype(str), "UserId": list(range(len(user_uniques)))}) \
            .to_csv(outp, index=False)

    user_ids = sorted(df["UserId"].unique().tolist())[: int(max_users)]
    df = df[df["UserId"].isin(user_ids)].copy()

    if time_sort and ts_col is not None and ts_col in df.columns:
        try:
            df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
            if df["_ts"].notna().mean() >= 0.5:
                df = df.sort_values(["UserId", "_ts"], kind="mergesort")
            else:
                df = df.sort_values(["UserId"], kind="mergesort")
        finally:
            df.drop(columns=["_ts"], inplace=True, errors="ignore")
    else:
        df = df.sort_values(["UserId"], kind="mergesort")

    return df[["UserId", "QuestionId", "IsCorrect"]].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--domain_json", required=True)
    ap.add_argument("--log_problem_csv", required=True)
    ap.add_argument("--question_id_map_csv", required=True)
    ap.add_argument("--out_user_map_csv", default=None)

    ap.add_argument("--max_users", type=int, default=300)
    ap.add_argument("--time_sort", action="store_true")

    # Sharding (optional)
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

    ap.add_argument("--acc_threshold", type=float, default=0.5,
                    help="Decision threshold used for ACC and F1 (default 0.5).")

    ap.add_argument("--out_csv", default="evaluation/junyi/metrics_online_next_correctness.csv")
    ap.add_argument("--out_user_csv", default=None)
    ap.add_argument("--out_predictions_csv", default=None)

    args = ap.parse_args()

    domain = load_json(args.domain_json)
    q2s, children_of, parent_of = parse_domain(domain)
    subj_ids = all_subject_ids(children_of, parent_of)

    log_df = pd.read_csv(args.log_problem_csv)
    log_df.columns = [_norm_col(c) for c in log_df.columns]

    qmap_df = pd.read_csv(args.question_id_map_csv)
    qmap_df.columns = [_norm_col(c) for c in qmap_df.columns]

    df = build_interactions_from_junyi_log(
        log_df=log_df,
        qmap_df=qmap_df,
        max_users=int(args.max_users),
        time_sort=bool(args.time_sort),
        out_user_map_csv=args.out_user_map_csv,
    )

    user_ids = sorted(df["UserId"].unique().tolist())
    user_ids = user_ids[: int(args.max_users)]

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
        udf = df[df["UserId"] == uid].copy().reset_index(drop=True)

        leaf_mean = LeafMeanState(defaultdict(int), defaultdict(int), a_fallback=float(args.a))
        leaf_sl = LeafSLState(defaultdict(float), defaultdict(float), a=float(args.a), W=float(args.W))

        user_methods = {k: MethodOutputs([], []) for k in methods.keys()}

        for step, row in enumerate(udf.itertuples(index=False)):
            qid = int(getattr(row, "QuestionId"))
            y = int(getattr(row, "IsCorrect"))
            sids = q2s.get(qid, [])

            # Leaf-mean
            ps_mean = [leaf_mean.mastery(s) for s in sids] if sids else []
            p_leaf_mean = aggregate_probs(ps_mean, args.agg)

            # Leaf-SL
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

            # Mean propagation
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

            # SL propagation (unconditional, fixed behavior)
            state_sl = propagate_sl_from_leaf(
                leaf_sl=leaf_sl,
                subj_ids=subj_ids,
                parent_of=parent_of,
                K=int(args.K),
                t_up=float(args.t_up),
                t_down=float(args.t_down),
                one_hop=bool(args.one_hop),
                no_discount=bool(args.no_discount),
                conditional=False,
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

            # Online update with the observed outcome
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
        "dataset": "junyi",
        "domain_json": str(args.domain_json),
        "log_problem_csv": str(args.log_problem_csv),
        "question_id_map_csv": str(args.question_id_map_csv),
        "max_users": int(args.max_users),
        "n_users_evaluated": int(len(user_ids)),
        "time_sort": bool(args.time_sort),
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
            f"{name:10s} n={n:7d}  AUC={auc_str}  "
            f"ACC@{args.acc_threshold:.2f}={acc:.6f}  "
            f"F1@{args.acc_threshold:.2f}={f1:.6f}  "
            f"LogLoss={ll:.6f}"
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
