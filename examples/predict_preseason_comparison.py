"""
Compare pre-Omloop predictions for 2024 and 2025 spring classics.

Tests the model's preseason predictive power:
- 2024: uses only 2023 (and earlier) results + prep placeholders
- 2025: uses only 2023-2024 results + prep placeholders

No in-season data is used — this answers the question:
"How well can the model predict at the start of the season?"
"""

import sys
import os
from datetime import date
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from procyclingstats.classics_predictor import (
    ClassicsPredictor,
    CLASSICS_METADATA,
)
from tests.test_backtest_2025 import (
    BIRTHDATES,
    SPECIALTY_PTS,
    RIDER_TEAMS,
    PRE_2025_RESULTS,
    RACES_2025,
    RESULTS_2025,
    _r,
    score_predictions,
)


# =====================================================================
# 2024 ACTUAL RESULTS (reconstructed from PRE_2025_RESULTS data)
# =====================================================================

RACES_2024 = [
    {
        "name": "Strade Bianche",
        "base_url": "race/strade-bianche",
        "date": date(2024, 3, 2),
        "distance": 185.0,
        "actual_top10": [
            "rider/tadej-pogacar",        # 1
            "rider/mathieu-van-der-poel",  # 2
            None, None, None, None, None,
            "rider/magnus-cort",           # 8
            None,
            "rider/tim-wellens",           # 10
        ],
    },
    {
        "name": "Milano-Sanremo",
        "base_url": "race/milano-sanremo",
        "date": date(2024, 3, 16),
        "distance": 291.0,
        "actual_top10": [
            "rider/jasper-philipsen",      # 1
            "rider/michael-matthews",      # 2
            "rider/tadej-pogacar",         # 3
            "rider/mads-pedersen",         # 3 (tied)
            None, None, None,
            "rider/mathieu-van-der-poel",  # 8
            None,
            "rider/filippo-ganna",         # 10
        ],
    },
    {
        "name": "E3 Saxo Classic",
        "base_url": "race/e3-harelbeke",
        "date": date(2024, 3, 22),
        "distance": 209.0,
        "actual_top10": [
            "rider/mathieu-van-der-poel",  # 1
            "rider/jasper-stuyven",        # 2
            "rider/wout-van-aert",         # 3
            "rider/tim-wellens",           # 4
            "rider/matteo-jorgenson",      # 5
            None, None,
            "rider/filippo-ganna",         # 8
            None,
            "rider/stefan-kueng",          # 10
        ],
    },
    {
        "name": "Gent-Wevelgem",
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
    },
    {
        "name": "Ronde van Vlaanderen",
        "base_url": "race/ronde-van-vlaanderen",
        "date": date(2024, 3, 31),
        "distance": 260.0,
        "actual_top10": [
            "rider/mathieu-van-der-poel",  # 1
            "rider/tadej-pogacar",         # 2
            None, None,
            "rider/wout-van-aert",         # 5
            None, None,
            "rider/mads-pedersen",         # 8
            "rider/jasper-stuyven",        # 8
            "rider/tiesj-benoot",          # 10
        ],
    },
    {
        "name": "Paris-Roubaix",
        "base_url": "race/paris-roubaix",
        "date": date(2024, 4, 7),
        "distance": 260.0,
        "actual_top10": [
            "rider/mathieu-van-der-poel",  # 1
            "rider/jasper-philipsen",      # 2
            "rider/mads-pedersen",         # 3
            None,
            "rider/florian-vermeersch",    # 5
            None, None,
            "rider/wout-van-aert",         # 8
            None, None,
        ],
    },
    {
        "name": "Amstel Gold Race",
        "base_url": "race/amstel-gold-race",
        "date": date(2024, 4, 14),
        "distance": 256.0,
        "actual_top10": [
            "rider/tom-pidcock",           # 1
            None,
            "rider/tiesj-benoot",          # 3
            None,
            "rider/wout-van-aert",         # 5
            "rider/michael-matthews",      # 6
            None,
            "rider/ben-healy",             # 8
            "rider/mattias-skjelmose",     # 8
            "rider/remco-evenepoel",       # 10
        ],
    },
    {
        "name": "La Flèche Wallonne",
        "base_url": "race/la-fleche-wallone",
        "date": date(2024, 4, 17),
        "distance": 195.0,
        "actual_top10": [
            None,                           # 1 (winner not in DB)
            "rider/kevin-vauquelin",       # 2
            None,
            "rider/tadej-pogacar",         # 4
            None,
            "rider/ben-healy",             # 6
            None,
            "rider/remco-evenepoel",       # 8
            None,
            "rider/mattias-skjelmose",     # 10
        ],
    },
    {
        "name": "Liège-Bastogne-Liège",
        "base_url": "race/liege-bastogne-liege",
        "date": date(2024, 4, 21),
        "distance": 260.0,
        "actual_top10": [
            "rider/tadej-pogacar",         # 1
            None,
            "rider/mathieu-van-der-poel",  # 3
            None,
            "rider/remco-evenepoel",       # 5
            None, None,
            "rider/ben-healy",             # 8
            "rider/daniel-martinez",       # 8
            "rider/neilson-powless",       # 10
        ],
    },
]


# =====================================================================
# Build pre-Omloop rider data for each year
# =====================================================================

def build_pre_omloop_2024_data():
    """
    Build rider data using only results BEFORE Omloop 2024 (Feb 28, 2024).

    This means: 2023 and earlier results + early 2024 prep races (Jan/Feb).
    No 2024 race results from March onwards.
    """
    cutoff = date(2024, 2, 28)
    rider_data = {}

    for rider_url in BIRTHDATES:
        results = []

        # Filter PRE_2025_RESULTS to only before cutoff
        for r in PRE_2025_RESULTS.get(rider_url, []):
            parts = r["date"].split("-")
            rd = date(int(parts[0]), int(parts[1]), int(parts[2]))
            if rd < cutoff:
                results.append(r)

        # Add placeholder prep for early 2024
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


def build_pre_omloop_2025_data():
    """
    Build rider data using only results BEFORE Omloop 2025 (Feb 28, 2025).

    This means: 2023-2024 results (all of PRE_2025_RESULTS) + prep placeholders.
    No 2025 race results at all.
    """
    rider_data = {}

    for rider_url in BIRTHDATES:
        # All pre-2025 results (2023-2024)
        results = list(PRE_2025_RESULTS.get(rider_url, []))

        # Add placeholder prep for early 2025
        results.append(_r("2025-1-28", 10, "race/prep/2025", "1.Pro"))
        results.append(_r("2025-2-10", 8, "race/prep/2025", "1.Pro"))
        results.append(_r("2025-2-20", 12, "race/prep/2025", "1.Pro"))

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


def build_startlist():
    return [
        {"rider_url": url, "rider_name": url.split("/")[-1].replace("-", " ").title()}
        for url in BIRTHDATES
    ]


# =====================================================================
# Run backtest for a given year
# =====================================================================

def run_preseason_backtest(
    year: int,
    races: List[Dict],
    rider_data: Dict,
    verbose: bool = True,
) -> Dict[str, float]:
    """Run predictions using pre-Omloop data and score against actuals."""
    predictor = ClassicsPredictor()
    startlist = build_startlist()

    all_scores = []

    for race in races:
        predictions = predictor.predict_from_data(
            race_base_url=race["base_url"],
            year=year,
            distance=race["distance"],
            race_date=race["date"],
            startlist=startlist,
            rider_data=rider_data,
        )

        scores = score_predictions(predictions, race["actual_top10"])
        scores["race"] = race["name"]
        all_scores.append(scores)

        if verbose:
            actual_known = [u for u in race["actual_top10"] if u]
            print(f"\n  {race['name']} ({race['date']})")
            print(f"  {'Pred':>4}  {'Rider':<35} {'Actual':>6}")
            print(f"  {'-'*4}  {'-'*35} {'-'*6}")
            for p in predictions[:15]:
                actual_pos = ""
                for i, au in enumerate(race["actual_top10"], 1):
                    if au == p["rider_url"]:
                        actual_pos = str(i)
                        break
                marker = " *" if actual_pos else ""
                print(f"  {p['rank']:>4}  {p['rider_name']:<35} "
                      f"{actual_pos:>6}{marker}")

            print(f"  -- Top-10 hit: {scores['top10_hit_rate']:.0%} | "
                  f"Top-5 hit: {scores['top5_hit_rate']:.0%} | "
                  f"Winner top-3: {'YES' if scores['winner_in_top3'] else 'no'} | "
                  f"Winner top-5: {'YES' if scores['winner_in_top5'] else 'no'}")

    if not all_scores:
        return {}

    n = len(all_scores)
    agg = {
        "avg_top10_hit": sum(s["top10_hit_rate"] for s in all_scores) / n,
        "avg_top5_hit": sum(s["top5_hit_rate"] for s in all_scores) / n,
        "winner_top3_pct": sum(s["winner_in_top3"] for s in all_scores) / n,
        "winner_top5_pct": sum(s["winner_in_top5"] for s in all_scores) / n,
        "avg_rank_error": sum(s["avg_rank_error"] for s in all_scores) / n,
        "n_races": n,
    }
    return agg


# =====================================================================
# Main: compare 2024 and 2025 pre-Omloop predictions
# =====================================================================

def main():
    # -- 2024 PRE-OMLOOP BACKTEST --
    print("=" * 75)
    print("  2024 SPRING CLASSICS — PRE-OMLOOP DATA ONLY")
    print("  (Using only 2023 results + early 2024 prep)")
    print("=" * 75)

    rider_data_2024 = build_pre_omloop_2024_data()

    # Show how much data each key rider has
    print("\n  Pre-Omloop 2024 data available per rider:")
    key_riders = [
        "rider/mathieu-van-der-poel", "rider/tadej-pogacar",
        "rider/mads-pedersen", "rider/wout-van-aert",
        "rider/remco-evenepoel", "rider/tom-pidcock",
        "rider/jasper-philipsen", "rider/filippo-ganna",
    ]
    for r in key_riders:
        n_results = len(rider_data_2024[r]["results"])
        n_classic = sum(1 for res in rider_data_2024[r]["results"]
                        if any(k in res.get("stage_url", "")
                               for k in ["ronde-van", "paris-roubaix", "milano-san",
                                          "liege-bastogne", "strade-bianche", "e3-",
                                          "gent-wevelgem", "amstel", "fleche"]))
        name = r.split("/")[-1].replace("-", " ").title()
        print(f"    {name:<35} {n_results:>2} results ({n_classic} classics)")

    agg_2024 = run_preseason_backtest(2024, RACES_2024, rider_data_2024, verbose=True)

    # -- 2025 PRE-OMLOOP BACKTEST --
    print(f"\n\n{'=' * 75}")
    print("  2025 SPRING CLASSICS — PRE-OMLOOP DATA ONLY")
    print("  (Using only 2023-2024 results + early 2025 prep)")
    print("=" * 75)

    rider_data_2025 = build_pre_omloop_2025_data()

    # Filter RACES_2025 to skip sprinters-only races
    races_2025_scorable = [r for r in RACES_2025 if not r.get("is_sprinters_race")]

    print("\n  Pre-Omloop 2025 data available per rider:")
    for r in key_riders:
        n_results = len(rider_data_2025[r]["results"])
        n_classic = sum(1 for res in rider_data_2025[r]["results"]
                        if any(k in res.get("stage_url", "")
                               for k in ["ronde-van", "paris-roubaix", "milano-san",
                                          "liege-bastogne", "strade-bianche", "e3-",
                                          "gent-wevelgem", "amstel", "fleche"]))
        name = r.split("/")[-1].replace("-", " ").title()
        print(f"    {name:<35} {n_results:>2} results ({n_classic} classics)")

    agg_2025 = run_preseason_backtest(2025, races_2025_scorable, rider_data_2025, verbose=True)

    # -- COMPARISON --
    print(f"\n\n{'=' * 75}")
    print("  COMPARISON: PRE-OMLOOP PREDICTION ACCURACY")
    print(f"{'=' * 75}")
    print(f"\n  {'Metric':<30} {'2024':>10} {'2025':>10} {'Delta':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    metrics = [
        ("Avg top-10 hit rate", "avg_top10_hit"),
        ("Avg top-5 hit rate", "avg_top5_hit"),
        ("Winner in top-3", "winner_top3_pct"),
        ("Winner in top-5", "winner_top5_pct"),
        ("Avg rank error", "avg_rank_error"),
    ]

    for label, key in metrics:
        v24 = agg_2024.get(key, 0)
        v25 = agg_2025.get(key, 0)
        delta = v25 - v24
        if key == "avg_rank_error":
            fmt = ".1f"
            sign = "+" if delta > 0 else ""
            print(f"  {label:<30} {v24:>9{fmt}} {v25:>9{fmt}} {sign}{delta:>9{fmt}}")
        else:
            print(f"  {label:<30} {v24:>9.0%} {v25:>9.0%} {delta:>+9.0%}")

    print(f"\n  Races scored: 2024={agg_2024.get('n_races', 0)}, "
          f"2025={agg_2025.get('n_races', 0)}")
    print(f"\n  Note: 2024 pre-Omloop has LESS historical data (only 2023)")
    print(f"  while 2025 pre-Omloop has 2023+2024 (two full seasons).")
    print(f"  The model's accuracy with less data shows its robustness.")


if __name__ == "__main__":
    main()
