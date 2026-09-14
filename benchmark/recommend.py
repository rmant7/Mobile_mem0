"""
Turns one full hybrid-weight sweep into a single recommended semantic
weight — pure and synthetic-data-testable, so the actual decision rule can
be verified without llama_cpp, and so the printed RECOMMENDED_SEMANTIC_WEIGHT
line is never "whatever looked good to a human skimming a table" but a
fixed, inspectable rule applied the same way every run.

Implements this task's own Part C robustness checks:
  - a candidate weight must beat lexical-only MRR and Recall@1 (a weight
    that doesn't is not a candidate at all — the point of adding semantic
    is winning, not merely avoiding a large loss);
  - a candidate weight must not hurt identifier/exact recall_at_1 by more
    than REGRESSION_TOLERANCE vs lexical-only (these are the categories
    where the lexical algorithm is already close to perfect; semantic
    weight should not be allowed to buy other categories' wins with them);
  - low_overlap's movement is reported for every weight but never used to
    reject one — it is semantic's own best category by design;
  - moving to a larger candidate weight requires its MRR gain over the
    smaller one to clear bootstrap noise, and the final pick's own gain
    over the lexical baseline must also clear its bootstrap noise band —
    otherwise this function reports the result as ambiguous.

Per this task's own explicit instruction, an ambiguous result is never
resolved by arbitrarily picking a mid-sweep value (e.g. 0.30/0.35): the
fallback is always FALLBACK_WEIGHT (the existing production baseline),
with the reason spelled out so a human can see more data is needed.
"""
from __future__ import annotations

# "Does not hurt" tolerance for identifier/exact recall_at_1 vs lexical-only
# — a plain equality check would reject a weight over one query's worth of
# floating-point/tie-break noise (identifier has 7 queries, exact has 18).
REGRESSION_TOLERANCE = 0.01

# An MRR gain must exceed this many bootstrap standard deviations to count
# as real rather than resampling noise — both when comparing candidate
# weights against each other and when comparing the final pick against the
# lexical-only baseline.
NOISE_MULTIPLIER = 1.0

# This task's own explicit fallback: the current, already-shipped
# production value, to keep when the sweep is ambiguous.
FALLBACK_WEIGHT = 0.20

# Categories a regression check is applied to — see module docstring.
GUARDED_CATEGORIES = ("identifier", "exact")


def recommend_weight(lexical_overall: dict, lexical_by_category: dict, sweep: dict) -> dict:
    """
    lexical_overall: AggregateMetrics.as_dict() for the lexical-only baseline.
    lexical_by_category: {category: AggregateMetrics.as_dict()} for the same.
    sweep: {weight: {"overall": as_dict(), "by_category": {cat: as_dict()},
                      "bootstrap": bootstrap_mrr()'s own dict}}, one entry
        per swept semantic weight.

    Returns {"weight": float, "ambiguous": bool, "reason": str,
             "candidates": [weights that passed the non-regression gate]}.
    """
    weights = sorted(sweep.keys())

    candidates = []
    for w in weights:
        overall = sweep[w]["overall"]
        by_cat = sweep[w]["by_category"]

        if overall["mrr"] < lexical_overall["mrr"]:
            continue
        if overall["recall_at_1"] < lexical_overall["recall_at_1"]:
            continue

        regressed = False
        for cat in GUARDED_CATEGORIES:
            lex_cat_metrics = lexical_by_category.get(cat)
            cur_cat_metrics = by_cat.get(cat)
            if lex_cat_metrics is None or cur_cat_metrics is None:
                continue
            if cur_cat_metrics["recall_at_1"] < lex_cat_metrics["recall_at_1"] - REGRESSION_TOLERANCE:
                regressed = True
                break
        if regressed:
            continue

        candidates.append(w)

    if not candidates:
        return {
            "weight": FALLBACK_WEIGHT,
            "ambiguous": True,
            "reason": (
                "No swept weight both beat lexical-only MRR/Recall@1 and avoided "
                f"hurting identifier/exact recall_at_1 by more than {REGRESSION_TOLERANCE}; "
                f"keeping the existing {FALLBACK_WEIGHT} baseline and flagging that more data is needed."
            ),
            "candidates": [],
        }

    # Walk candidates from smallest to largest, only moving up when the gain
    # clears bootstrap noise on MRR — otherwise the smaller weight already
    # explains the data and there is no reason to prefer the larger one's risk
    # (more semantic weight than the data supports is itself a risk).
    best = candidates[0]
    for w in candidates[1:]:
        prev_mrr = sweep[best]["overall"]["mrr"]
        cur_mrr = sweep[w]["overall"]["mrr"]
        noise = NOISE_MULTIPLIER * max(sweep[w]["bootstrap"]["mrr_std"], sweep[best]["bootstrap"]["mrr_std"])
        if cur_mrr - prev_mrr > noise:
            best = w

    best_mrr = sweep[best]["overall"]["mrr"]
    best_std = sweep[best]["bootstrap"]["mrr_std"]
    gain_over_lexical = best_mrr - lexical_overall["mrr"]
    ambiguous = gain_over_lexical <= NOISE_MULTIPLIER * best_std

    if ambiguous:
        return {
            "weight": FALLBACK_WEIGHT,
            "ambiguous": True,
            "reason": (
                f"Best surviving weight {best} beats lexical MRR by only {gain_over_lexical:.4f}, "
                f"within its own bootstrap noise band (std={best_std:.4f}); keeping the existing "
                f"{FALLBACK_WEIGHT} baseline and flagging that more data is needed."
            ),
            "candidates": candidates,
        }

    return {
        "weight": best,
        "ambiguous": False,
        "reason": (
            f"Weight {best} beats lexical-only MRR/Recall@1 by {gain_over_lexical:.4f} "
            f"(clears its own bootstrap noise band, std={best_std:.4f}), does not hurt "
            f"identifier/exact recall_at_1 by more than {REGRESSION_TOLERANCE}, and any larger "
            "candidate weight's extra gain did not clear bootstrap noise."
        ),
        "candidates": candidates,
    }
