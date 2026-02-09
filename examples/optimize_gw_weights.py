"""
Optimize Gent-Wevelgem specific weights against 2024 + 2025 actual results.

GW is a sprinters' classic — the general model under-ranks sprinters.
This script finds weights that maximize prediction accuracy for GW specifically.
"""

import sys
import os
import itertools
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from procyclingstats.classics_predictor import ClassicsPredictor, RACE_WEIGHT_OVERRIDES
from tests.test_backtest_2025 import (
    BIRTHDATES,
    SPECIALTY_PTS,
    RIDER_TEAMS,
    PRE_2025_RESULTS,
    RESULTS_2025,
    _r,
    score_predictions,
    build_startlist,
    build_rider_data,
)


# GW actual results
GW_2024 = {
    "name": "Gent-Wevelgem 2024",
    "base_url": "race/gent-wevelgem",
    "date": date(2024, 3, 24),
    "distance": 250.0,
    "actual_top10": [
        "rider/mads-pedersen",         # 1
        "rider/mathieu-van-der-poel",  # 2
        None, None, None, None, None,
        "rider/alexander-kristoff",    # 8
        None,
        "rider/davide-ballerini",      # 10
    ],
}

GW_2025 = {
    "name": "Gent-Wevelgem 2025",
    "base_url": "race/gent-wevelgem",
    "date": date(2025, 3, 30),
    "distance": 250.0,
    "actual_top10": [
        "rider/mads-pedersen",
        "rider/tim-merlier",
        "rider/jonathan-milan",
        "rider/alexander-kristoff",
        "rider/hugo-hofstetter",
        "rider/davide-ballerini",
        "rider/biniam-girmay",
        None,  # Jenno Berckmoes
        None,  # Jordi Meeus
        "rider/laurenz-rex",
    ],
}


def build_pre_omloop_2024_data():
    """Pre-Omloop 2024 data (only 2023 + prep)."""
    cutoff = date(2024, 2, 28)
    rider_data = {}
    for rider_url in BIRTHDATES:
        results = []
        for r in PRE_2025_RESULTS.get(rider_url, []):
            parts = r["date"].split("-")
            rd = date(int(parts[0]), int(parts[1]), int(parts[2]))
            if rd < cutoff:
                results.append(r)
        results.append(_r("2024-1-28", 10, "race/prep/2024", "1.Pro"))
        results.append(_r("2024-2-10", 8, "race/prep/2024", "1.Pro"))
        results.append(_r("2024-2-20", 12, "race/prep/2024", "1.Pro"))
        pts = SPECIALTY_PTS.get(rider_url, {})
        rider_data[rider_url] = {
            "profile": {
                "birthdate": BIRTHDATES[rider_url],
                "points_per_speciality": {
                    "one_day_races": pts.get("one_day", 500),
                    "gc": pts.get("gc", 0),
                    "time_trial": pts.get("tt", 0),
                    "sprint": pts.get("sprint", 0),
                    "climber": pts.get("climber", 0),
                },
            },
            "results": results,
            "team": RIDER_TEAMS.get(rider_url, ""),
        }
    return rider_data


def test_gw_weights(weights):
    """Test given weights on both GW 2024 and 2025, return composite score."""
    # Temporarily override GW weights
    old_override = RACE_WEIGHT_OVERRIDES.get("race/gent-wevelgem")
    RACE_WEIGHT_OVERRIDES["race/gent-wevelgem"] = weights

    predictor = ClassicsPredictor()
    startlist = build_startlist()

    # GW 2025 — use accumulating data (data before GW date)
    rider_data_2025 = build_rider_data(GW_2025["date"])
    preds_2025 = predictor.predict_from_data(
        GW_2025["base_url"], 2025, GW_2025["distance"],
        GW_2025["date"], startlist, rider_data_2025,
    )
    scores_2025 = score_predictions(preds_2025, GW_2025["actual_top10"])

    # GW 2024 — use pre-Omloop 2024 data
    rider_data_2024 = build_pre_omloop_2024_data()
    preds_2024 = predictor.predict_from_data(
        GW_2024["base_url"], 2024, GW_2024["distance"],
        GW_2024["date"], startlist, rider_data_2024,
    )
    scores_2024 = score_predictions(preds_2024, GW_2024["actual_top10"])

    # Restore
    if old_override:
        RACE_WEIGHT_OVERRIDES["race/gent-wevelgem"] = old_override
    else:
        del RACE_WEIGHT_OVERRIDES["race/gent-wevelgem"]

    # Composite across both years
    composite = 0.0
    for s in [scores_2024, scores_2025]:
        composite += (
            0.30 * s["top10_hit_rate"]
            + 0.30 * s["top5_hit_rate"]
            + 0.20 * s["winner_in_top5"]
            + 0.10 * s["winner_in_top3"]
            + 0.10 * max(0, 1.0 - s["avg_rank_error"] / 20)
        )
    return composite / 2, scores_2024, scores_2025


def optimize():
    """Grid search for best GW weights."""
    best_score = -1
    best_weights = None
    best_s24 = None
    best_s25 = None
    tested = 0

    # Sprint capability is the key variable for GW
    for sp in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for cp in [0.10, 0.15, 0.20, 0.25]:
            for tm in [0.05, 0.10, 0.15]:
                for cc in [0.05, 0.10, 0.15]:
                    for py in [0.03, 0.06, 0.10]:
                        major = sp + cp + tm + cc + py
                        rest = 1.0 - major
                        if rest < 0.10 or rest > 0.50:
                            continue

                        weights = {
                            "sprint_capability": sp,
                            "classic_pedigree": cp,
                            "terrain_match": tm,
                            "cobble_capability": cc,
                            "previous_year": py,
                            "recent_form": rest * 0.18,
                            "momentum": rest * 0.16,
                            "team_strength": rest * 0.16,
                            "preparation": rest * 0.14,
                            "specialty_score": rest * 0.12,
                            "injury_penalty": rest * 0.10,
                            "uphill_sprint": rest * 0.08,
                            "age_distance_fit": rest * 0.06,
                        }

                        score, s24, s25 = test_gw_weights(weights)
                        tested += 1

                        if score > best_score:
                            best_score = score
                            best_weights = weights.copy()
                            best_s24 = s24
                            best_s25 = s25

    print(f"\nTested {tested} weight combinations\n")
    print(f"Best composite score: {best_score:.3f}")
    print(f"\nBest GW weights:")
    for k, v in sorted(best_weights.items(), key=lambda x: -x[1]):
        print(f"  {k:<25} {v:.3f}")
    print(f"\n2024 GW: top10={best_s24['top10_hit_rate']:.0%} "
          f"top5={best_s24['top5_hit_rate']:.0%} "
          f"winner_top3={'YES' if best_s24['winner_in_top3'] else 'no'} "
          f"winner_top5={'YES' if best_s24['winner_in_top5'] else 'no'}")
    print(f"2025 GW: top10={best_s25['top10_hit_rate']:.0%} "
          f"top5={best_s25['top5_hit_rate']:.0%} "
          f"winner_top3={'YES' if best_s25['winner_in_top3'] else 'no'} "
          f"winner_top5={'YES' if best_s25['winner_in_top5'] else 'no'}")

    # Also show what default weights give
    print(f"\n--- For comparison, DEFAULT weights on GW ---")
    RACE_WEIGHT_OVERRIDES.pop("race/gent-wevelgem", None)
    default_score, ds24, ds25 = test_gw_weights(
        {k: v for k, v in zip(
            ["sprint_capability", "classic_pedigree", "terrain_match",
             "cobble_capability", "previous_year", "recent_form", "momentum",
             "team_strength", "preparation", "specialty_score", "injury_penalty",
             "uphill_sprint", "age_distance_fit"],
            [0.08, 0.15, 0.15, 0.12, 0.05, 0.046, 0.046,
             0.059, 0.053, 0.046, 0.046, 0.12, 0.033]
        )}
    )
    print(f"Default composite: {default_score:.3f}")
    print(f"2024 GW: top10={ds24['top10_hit_rate']:.0%} "
          f"top5={ds24['top5_hit_rate']:.0%}")
    print(f"2025 GW: top10={ds25['top10_hit_rate']:.0%} "
          f"top5={ds25['top5_hit_rate']:.0%}")

    return best_weights


if __name__ == "__main__":
    optimize()
