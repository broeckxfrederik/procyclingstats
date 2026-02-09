"""
Example: Predict rider rankings for Paris-Roubaix 2024.

This example demonstrates two approaches:

1. **Offline prediction** with pre-collected data (fast, no HTTP requests).
   Use this when you've already scraped rider data or want to experiment
   with different weights without re-fetching.

2. **Live prediction** that fetches data from procyclingstats.com.
   This is slower (one HTTP request per rider) but fully automated.

Run:
    python -m examples.predict_classic
"""

from datetime import date

from procyclingstats import ClassicsPredictor


def offline_prediction_example():
    """
    Predict Paris-Roubaix 2024 with synthetic rider data.

    This shows how the model handles the key factors:
    - Age vs distance (young riders fade in 260km monuments)
    - Recent form (preseason results)
    - Classic pedigree (historical Roubaix finishes)
    - Injury detection (late season start, gaps)
    """
    print("=" * 60)
    print("OFFLINE PREDICTION: Paris-Roubaix 2024 (synthetic data)")
    print("=" * 60)

    predictor = ClassicsPredictor()

    # Synthetic rider profiles modeled after classic archetypes
    rider_data = {
        "rider/cobble-king": {
            "profile": {
                "birthdate": "1995-3-15",
                "points_per_speciality": {
                    "one_day_races": 3800,
                    "gc": 200, "time_trial": 500,
                    "sprint": 800, "climber": 100,
                },
            },
            "results": [
                # Strong preseason cobbles campaign
                {"date": "2024-3-2", "rank": 3, "stage_url": "race/strade-bianche/2024", "class": "1.UWT"},
                {"date": "2024-3-23", "rank": 1, "stage_url": "race/e3-harelbeke/2024", "class": "1.UWT"},
                {"date": "2024-3-27", "rank": 2, "stage_url": "race/gent-wevelgem/2024", "class": "1.UWT"},
                {"date": "2024-3-31", "rank": 1, "stage_url": "race/dwars-door-vlaanderen/2024", "class": "1.UWT"},
                # Previous Roubaix results
                {"date": "2023-4-9", "rank": 1, "stage_url": "race/paris-roubaix/2023", "class": "1.UWT"},
                {"date": "2022-4-17", "rank": 3, "stage_url": "race/paris-roubaix/2022", "class": "1.UWT"},
                # Season base
                {"date": "2024-2-1", "rank": 5, "stage_url": "race/a/2024", "class": "1.Pro"},
                {"date": "2024-2-10", "rank": 8, "stage_url": "race/b/2024", "class": "1.Pro"},
                {"date": "2024-2-18", "rank": 3, "stage_url": "race/c/2024", "class": "1.Pro"},
                {"date": "2024-1-28", "rank": 12, "stage_url": "race/d/2024", "class": "1.Pro"},
            ],
        },
        "rider/young-sprinter": {
            "profile": {
                "birthdate": "2001-7-20",
                "points_per_speciality": {
                    "one_day_races": 1200,
                    "gc": 50, "time_trial": 200,
                    "sprint": 1500, "climber": 30,
                },
            },
            "results": [
                # Some good results but limited classics experience
                {"date": "2024-3-15", "rank": 5, "stage_url": "race/a/2024", "class": "1.UWT"},
                {"date": "2024-3-25", "rank": 8, "stage_url": "race/b/2024", "class": "1.UWT"},
                {"date": "2024-2-10", "rank": 1, "stage_url": "race/c/2024", "class": "1.Pro"},
                {"date": "2024-2-20", "rank": 3, "stage_url": "race/d/2024", "class": "1.Pro"},
                {"date": "2024-1-28", "rank": 15, "stage_url": "race/e/2024", "class": "1.Pro"},
            ],
        },
        "rider/comeback-veteran": {
            "profile": {
                "birthdate": "1990-11-5",
                "points_per_speciality": {
                    "one_day_races": 2800,
                    "gc": 300, "time_trial": 600,
                    "sprint": 400, "climber": 200,
                },
            },
            "results": [
                # Late start - injury comeback
                {"date": "2024-3-20", "rank": 18, "stage_url": "race/a/2024", "class": "1.UWT"},
                {"date": "2024-3-30", "rank": 12, "stage_url": "race/b/2024", "class": "1.UWT"},
                # Strong Roubaix history though
                {"date": "2023-4-9", "rank": 5, "stage_url": "race/paris-roubaix/2023", "class": "1.UWT"},
                {"date": "2022-4-17", "rank": 2, "stage_url": "race/paris-roubaix/2022", "class": "1.UWT"},
                {"date": "2021-4-11", "rank": 1, "stage_url": "race/paris-roubaix/2021", "class": "1.UWT"},
            ],
        },
        "rider/steady-domestique": {
            "profile": {
                "birthdate": "1997-5-10",
                "points_per_speciality": {
                    "one_day_races": 900,
                    "gc": 100, "time_trial": 300,
                    "sprint": 200, "climber": 100,
                },
            },
            "results": [
                # Consistent but unremarkable
                {"date": "2024-3-2", "rank": 30, "stage_url": "race/a/2024", "class": "1.UWT"},
                {"date": "2024-3-15", "rank": 25, "stage_url": "race/b/2024", "class": "1.UWT"},
                {"date": "2024-3-23", "rank": 28, "stage_url": "race/c/2024", "class": "1.UWT"},
                {"date": "2024-2-5", "rank": 20, "stage_url": "race/d/2024", "class": "1.Pro"},
                {"date": "2024-2-15", "rank": 22, "stage_url": "race/e/2024", "class": "1.Pro"},
                {"date": "2024-1-28", "rank": 18, "stage_url": "race/f/2024", "class": "1.Pro"},
            ],
        },
    }

    startlist = [
        {"rider_url": url, "rider_name": name}
        for url, name in [
            ("rider/cobble-king", "VAN DER COBBLE Kevin"),
            ("rider/young-sprinter", "SPEEDY Junior"),
            ("rider/comeback-veteran", "OLDMAN Classics"),
            ("rider/steady-domestique", "STEADY Worker"),
        ]
    ]

    predictions = predictor.predict_from_data(
        race_base_url="race/paris-roubaix",
        year=2024,
        distance=260.0,
        race_date=date(2024, 4, 7),
        startlist=startlist,
        rider_data=rider_data,
    )

    # Print rankings
    print(f"\n{'Rank':<6}{'Rider':<30}{'Score':<8}")
    print("-" * 44)
    for p in predictions:
        print(f"{p['rank']:<6}{p['rider_name']:<30}{p['score']:<8}")

    # Print detailed explanations
    print()
    for p in predictions:
        print(predictor.explain(p))
        print()


def live_prediction_example():
    """
    Predict a classic by scraping live data from procyclingstats.com.

    WARNING: This makes many HTTP requests (one per rider on the startlist).
    It will take several minutes for a full startlist of ~200 riders.

    Uncomment and run if you want to test with real data.
    """
    print("=" * 60)
    print("LIVE PREDICTION: Paris-Roubaix (fetched from PCS)")
    print("=" * 60)
    print("This fetches live data and may take several minutes...")
    print()

    predictor = ClassicsPredictor()
    predictions = predictor.predict("race/paris-roubaix/2024")

    print(f"\n{'Rank':<6}{'Rider':<35}{'Score':<8}")
    print("-" * 49)
    for p in predictions[:20]:
        print(f"{p['rank']:<6}{p['rider_name']:<35}{p['score']:<8}")


def compare_classic_distances():
    """
    Show how age-distance scoring differs across classics.

    Demonstrates the core insight: young riders score better in shorter
    races while veterans thrive in longer monuments.
    """
    print("=" * 60)
    print("AGE-DISTANCE FIT: How age impacts scoring across classics")
    print("=" * 60)

    predictor = ClassicsPredictor()
    ages = [21, 23, 25, 27, 29, 31, 33, 35, 37]
    classics = [
        ("Strade Bianche", 185),
        ("La Flèche Wallonne", 195),
        ("Ronde van Vlaanderen", 260),
        ("Paris-Roubaix", 260),
        ("Milano-Sanremo", 300),
    ]

    print(f"\n{'Age':<6}", end="")
    for name, _ in classics:
        print(f"{name:<22}", end="")
    print()
    print("-" * (6 + 22 * len(classics)))

    for age in ages:
        print(f"{age:<6}", end="")
        for _, distance in classics:
            score = predictor._score_age_distance(age, distance)
            bar = "#" * int(score * 15)
            print(f"{score:.2f} {bar:<16}", end="")
        print()


if __name__ == "__main__":
    offline_prediction_example()
    print("\n")
    compare_classic_distances()

    # Uncomment to run live prediction (slow, many HTTP requests):
    # live_prediction_example()
