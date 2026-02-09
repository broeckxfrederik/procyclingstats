"""
Predict the full 2025 spring classics season using only pre-Omloop data.

This tests the model's predictive power at the start of the season, before
any 2025 race results have accumulated. Only pre-2025 history (2023-2024)
and placeholder prep results are used.

All 19 races from Omloop Het Nieuwsblad through Liège-Bastogne-Liège.
"""

import sys
import os
from datetime import date

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from procyclingstats.classics_predictor import ClassicsPredictor, CLASSICS_METADATA

# Import rider database from the backtest
from tests.test_backtest_2025 import (
    BIRTHDATES,
    SPECIALTY_PTS,
    RIDER_TEAMS,
    PRE_2025_RESULTS,
    _r,
)


# =====================================================================
# Full 2025 Spring Classics Calendar (19 races)
# =====================================================================

SEASON_2025 = [
    {
        "name": "Omloop Het Nieuwsblad",
        "base_url": "race/omloop-het-nieuwsblad",
        "date": date(2025, 2, 28),
        "distance": 200.0,
    },
    {
        "name": "Kuurne-Brussel-Kuurne",
        "base_url": "race/kuurne-brussel-kuurne",
        "date": date(2025, 3, 1),
        "distance": 197.0,
    },
    {
        "name": "GP Samyn",
        "base_url": "race/gp-samyn",
        "date": date(2025, 3, 3),
        "distance": 200.0,
    },
    {
        "name": "Strade Bianche",
        "base_url": "race/strade-bianche",
        "date": date(2025, 3, 7),
        "distance": 213.0,
    },
    {
        "name": "Nokere Koerse",
        "base_url": "race/nokere-koerse",
        "date": date(2025, 3, 18),
        "distance": 195.0,
    },
    {
        "name": "Bredene-Koksijde Classic",
        "base_url": "race/bredene-koksijde-classic",
        "date": date(2025, 3, 20),
        "distance": 200.0,
    },
    {
        "name": "Milano-Sanremo",
        "base_url": "race/milano-sanremo",
        "date": date(2025, 3, 21),
        "distance": 289.0,
    },
    {
        "name": "Brugge-De Panne",
        "base_url": "race/brugge-de-panne",
        "date": date(2025, 3, 25),
        "distance": 205.0,
    },
    {
        "name": "E3 Saxo Classic",
        "base_url": "race/e3-harelbeke",
        "date": date(2025, 3, 27),
        "distance": 209.0,
    },
    {
        "name": "Gent-Wevelgem",
        "base_url": "race/gent-wevelgem",
        "date": date(2025, 3, 29),
        "distance": 250.0,
    },
    {
        "name": "Dwars door Vlaanderen",
        "base_url": "race/dwars-door-vlaanderen",
        "date": date(2025, 4, 1),
        "distance": 185.0,
    },
    {
        "name": "Ronde van Vlaanderen",
        "base_url": "race/ronde-van-vlaanderen",
        "date": date(2025, 4, 5),
        "distance": 270.0,
    },
    {
        "name": "Scheldeprijs",
        "base_url": "race/scheldeprijs",
        "date": date(2025, 4, 8),
        "distance": 200.0,
    },
    {
        "name": "Paris-Roubaix",
        "base_url": "race/paris-roubaix",
        "date": date(2025, 4, 12),
        "distance": 259.0,
    },
    {
        "name": "Ronde van Limburg",
        "base_url": "race/ronde-van-limburg",
        "date": date(2025, 4, 15),
        "distance": 200.0,
    },
    {
        "name": "Brabantse Pijl",
        "base_url": "race/brabantse-pijl",
        "date": date(2025, 4, 17),
        "distance": 200.0,
    },
    {
        "name": "Amstel Gold Race",
        "base_url": "race/amstel-gold-race",
        "date": date(2025, 4, 19),
        "distance": 256.0,
    },
    {
        "name": "La Flèche Wallonne",
        "base_url": "race/la-fleche-wallone",
        "date": date(2025, 4, 22),
        "distance": 205.0,
    },
    {
        "name": "Liège-Bastogne-Liège",
        "base_url": "race/liege-bastogne-liege",
        "date": date(2025, 4, 26),
        "distance": 260.0,
    },
]


# =====================================================================
# Build rider data using ONLY pre-Omloop data (no 2025 race results)
# =====================================================================

def build_preseason_rider_data():
    """
    Build rider data using only pre-2025 history + prep placeholders.

    No 2025 race results are included — this simulates predicting
    the entire season at the start, before Omloop Het Nieuwsblad.
    """
    rider_data = {}
    for rider_url in BIRTHDATES:
        # Pre-2025 results only (2023-2024)
        results = list(PRE_2025_RESULTS.get(rider_url, []))

        # Add placeholder prep results (Jan/Feb training races)
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
    """Build startlist with all riders in the database."""
    return [
        {"rider_url": url, "rider_name": url.split("/")[-1].replace("-", " ").title()}
        for url in BIRTHDATES
    ]


# =====================================================================
# Main: run predictions for all 19 races
# =====================================================================

def main():
    predictor = ClassicsPredictor()
    rider_data = build_preseason_rider_data()
    startlist = build_startlist()

    print("=" * 75)
    print("  2025 SPRING CLASSICS PREDICTIONS (Pre-Omloop data only)")
    print("  Using only 2023-2024 results + prep placeholders")
    print("  13-feature model with optimized weights")
    print("=" * 75)

    for race in SEASON_2025:
        meta = CLASSICS_METADATA.get(race["base_url"], {})
        race_type = meta.get("type", "?")
        if hasattr(race_type, "value"):
            race_type = race_type.value.replace("_", " ").title()

        predictions = predictor.predict_from_data(
            race_base_url=race["base_url"],
            year=2025,
            distance=race["distance"],
            race_date=race["date"],
            startlist=startlist,
            rider_data=rider_data,
        )

        print(f"\n{'=' * 75}")
        print(f"  {race['name'].upper()} — {race['date'].strftime('%d %b')} "
              f"({race['distance']:.0f} km) [{race_type}]")
        print(f"{'=' * 75}")
        print(f"  {'Rank':>4}  {'Rider':<35} {'Score':>6}")
        print(f"  {'-'*4}  {'-'*35} {'-'*6}")

        for p in predictions[:15]:
            print(f"  {p['rank']:>4}  {p['rider_name']:<35} {p['score']:>6.1f}")

    # Summary: who appears most in predicted top-3 across all races
    print(f"\n\n{'=' * 75}")
    print("  SEASON SUMMARY: Most top-3 predicted finishes")
    print(f"{'=' * 75}")

    top3_counts = {}
    top5_counts = {}
    top10_counts = {}
    for race in SEASON_2025:
        predictions = predictor.predict_from_data(
            race_base_url=race["base_url"],
            year=2025,
            distance=race["distance"],
            race_date=race["date"],
            startlist=startlist,
            rider_data=rider_data,
        )
        for p in predictions[:3]:
            name = p["rider_name"]
            top3_counts[name] = top3_counts.get(name, 0) + 1
        for p in predictions[:5]:
            name = p["rider_name"]
            top5_counts[name] = top5_counts.get(name, 0) + 1
        for p in predictions[:10]:
            name = p["rider_name"]
            top10_counts[name] = top10_counts.get(name, 0) + 1

    print(f"\n  {'Rider':<35} {'Top-3':>5} {'Top-5':>5} {'Top-10':>6}")
    print(f"  {'-'*35} {'-'*5} {'-'*5} {'-'*6}")
    for name, count in sorted(top3_counts.items(), key=lambda x: -x[1]):
        t5 = top5_counts.get(name, 0)
        t10 = top10_counts.get(name, 0)
        print(f"  {name:<35} {count:>5} {t5:>5} {t10:>6}")

    # Also show riders who appear in top-10 but not top-3
    for name in sorted(top10_counts.keys()):
        if name not in top3_counts:
            t5 = top5_counts.get(name, 0)
            t10 = top10_counts.get(name, 0)
            print(f"  {name:<35} {'0':>5} {t5:>5} {t10:>6}")


if __name__ == "__main__":
    main()
