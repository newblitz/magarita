"""Macro F0.5, computed exactly as the competition scores it: per-S1-entity
F_beta (beta=0.5), averaged across all S1 entities, singletons included (an
empty-empty prediction scores 1.0; any false match on a true singleton scores 0.0).
"""

from __future__ import annotations

from typing import Iterable, Mapping


def f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def per_entity_f_beta(predicted: Iterable[str], truth: Iterable[str], beta: float = 0.5) -> float:
    predicted = set(predicted)
    truth = set(truth)
    if not truth and not predicted:
        return 1.0
    if not predicted:
        return 0.0  # recall = 0 with truth non-empty
    if not truth:
        return 0.0  # any prediction on a true singleton scores 0
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return f_beta_score(precision, recall, beta)


def macro_f_beta(
    predicted_by_s1: Mapping[str, Iterable[str]],
    truth_by_s1: Mapping[str, Iterable[str]],
    all_s1_ids: Iterable[str],
    beta: float = 0.5,
) -> float:
    """Macro-averaged F_beta across ``all_s1_ids`` (every S1 must be scored, even
    if it's absent from one of the mappings -- absent means "predicted nothing" /
    "truth is nothing")."""
    scores = []
    for s1 in all_s1_ids:
        pred = predicted_by_s1.get(s1, ())
        truth = truth_by_s1.get(s1, ())
        scores.append(per_entity_f_beta(pred, truth, beta))
    return sum(scores) / len(scores) if scores else 0.0


def macro_f_beta_by_group(
    predicted_by_s1: Mapping[str, Iterable[str]],
    truth_by_s1: Mapping[str, Iterable[str]],
    group_by_s1: Mapping[str, str],
    beta: float = 0.5,
) -> dict[str, float]:
    """Macro F_beta broken out by an arbitrary group (e.g. country) -- used to
    catch silent generalization failures (e.g. on France) before they surface
    as a leaderboard surprise."""
    buckets: dict[str, list[str]] = {}
    for s1, group in group_by_s1.items():
        buckets.setdefault(group, []).append(s1)
    return {
        group: macro_f_beta(predicted_by_s1, truth_by_s1, ids, beta)
        for group, ids in buckets.items()
    }
