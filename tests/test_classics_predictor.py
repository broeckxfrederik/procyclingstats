"""
Tests for the ClassicsPredictor module.

All tests use synthetic data (no live scraping) to validate the feature
scoring logic and overall prediction pipeline.
"""

import math
from datetime import date

import pytest

from procyclingstats.classics_predictor import (
    CLASSICS_METADATA,
    DEFAULT_WEIGHTS,
    TEAM_TIERS,
    ClassicType,
    ClassicsPredictor,
    TerrainType,
    _age_from_birthdate,
    _extract_year,
    _parse_date,
    _race_base_url,
    _similar_terrain_classics,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic rider data builders
# ---------------------------------------------------------------------------

def _make_results(entries):
    """Build a list of result dicts from compact tuples.

    Each entry: (date_str, rank, stage_url, race_class)
    """
    results = []
    for date_str, rank, stage_url, race_class in entries:
        results.append({
            "date": date_str,
            "rank": rank,
            "stage_url": stage_url,
            "class": race_class,
            "stage_name": stage_url,
            "distance": 200,
            "pcs_points": 0,
            "uci_points": 0,
        })
    return results


def _make_rider_data(birthdate, one_day_pts=1000, results=None):
    """Build a rider data dict suitable for predict_from_data."""
    return {
        "profile": {
            "birthdate": birthdate,
            "points_per_speciality": {
                "one_day_races": one_day_pts,
                "gc": 0,
                "time_trial": 0,
                "sprint": 0,
                "climber": 0,
            },
        },
        "results": results or [],
    }


# ---------------------------------------------------------------------------
# Unit tests: utility functions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    def test_parse_date(self):
        assert _parse_date("2024-4-7") == date(2024, 4, 7)
        assert _parse_date("2023-12-25") == date(2023, 12, 25)

    def test_race_base_url(self):
        assert _race_base_url("race/paris-roubaix/2024") == "race/paris-roubaix"
        assert _race_base_url("race/milano-sanremo/2023/result") == "race/milano-sanremo"
        assert _race_base_url("race/il-lombardia") == "race/il-lombardia"

    def test_extract_year(self):
        assert _extract_year("race/paris-roubaix/2024") == 2024
        assert _extract_year("race/milano-sanremo/2023/result") == 2023
        assert _extract_year("race/paris-roubaix") is None

    def test_age_from_birthdate(self):
        # Born 1998-09-21, check on 2024-04-07 -> 25
        assert _age_from_birthdate("1998-9-21", date(2024, 4, 7)) == 25
        # After birthday -> 26
        assert _age_from_birthdate("1998-9-21", date(2024, 10, 1)) == 26

    def test_similar_terrain_classics(self):
        cobbles = _similar_terrain_classics(TerrainType.COBBLES)
        assert "race/paris-roubaix" in cobbles
        hilly = _similar_terrain_classics(TerrainType.HILLY)
        assert "race/liege-bastogne-liege" in hilly
        assert "race/il-lombardia" in hilly


# ---------------------------------------------------------------------------
# Unit tests: individual feature scores
# ---------------------------------------------------------------------------

class TestAgeDistanceScore:
    """Test age-distance suitability scoring."""

    def setup_method(self):
        self.predictor = ClassicsPredictor()

    def test_optimal_age_for_monument(self):
        """A 30-year-old should score high for a 260km monument."""
        score = self.predictor._score_age_distance(30, 260.0)
        assert score > 0.85

    def test_young_rider_penalized_in_monument(self):
        """A 22-year-old should be penalized for a 260km race."""
        young = self.predictor._score_age_distance(22, 260.0)
        optimal = self.predictor._score_age_distance(30, 260.0)
        assert young < optimal
        assert young < 0.5

    def test_young_rider_ok_in_short_classic(self):
        """A 24-year-old should score better in a 185km classic."""
        short = self.predictor._score_age_distance(24, 185.0)
        long = self.predictor._score_age_distance(24, 300.0)
        assert short > long

    def test_veteran_moderate_penalty(self):
        """A 37-year-old gets a penalty but less severe than a 21-year-old
        in a monument."""
        veteran = self.predictor._score_age_distance(37, 260.0)
        very_young = self.predictor._score_age_distance(21, 260.0)
        assert veteran > very_young

    def test_unknown_age_returns_neutral(self):
        score = self.predictor._score_age_distance(None, 260.0)
        assert score == 0.5

    def test_san_remo_distance_favors_experience(self):
        """At 300km (Milan-San Remo), age 31 should score higher than 24."""
        experienced = self.predictor._score_age_distance(31, 300.0)
        young = self.predictor._score_age_distance(24, 300.0)
        assert experienced > young
        assert experienced > 0.8


class TestRecentFormScore:
    """Test recent form scoring."""

    def setup_method(self):
        self.predictor = ClassicsPredictor()
        self.race_date = date(2024, 4, 7)

    def test_no_results_returns_zero(self):
        score = self.predictor._score_recent_form([], self.race_date)
        assert score == 0.0

    def test_recent_win_scores_high(self):
        results = _make_results([
            ("2024-3-30", 1, "race/gent-wevelgem/2024", "1.UWT"),
        ])
        score = self.predictor._score_recent_form(results, self.race_date)
        assert score > 0.9

    def test_recency_weighting(self):
        """Recent good results + old bad results should beat old good + recent bad.

        With a single result the weighted average equals the result score
        regardless of recency, so we need multiple results to observe the
        effect of recency weighting.
        """
        # Rider A: won recently, mediocre result long ago
        rider_a = _make_results([
            ("2024-3-31", 1, "race/a/2024", "1.UWT"),
            ("2024-1-15", 15, "race/b/2024", "1.UWT"),
        ])
        # Rider B: mediocre recently, won long ago
        rider_b = _make_results([
            ("2024-3-31", 15, "race/a/2024", "1.UWT"),
            ("2024-1-15", 1, "race/b/2024", "1.UWT"),
        ])
        score_a = self.predictor._score_recent_form(rider_a, self.race_date)
        score_b = self.predictor._score_recent_form(rider_b, self.race_date)
        assert score_a > score_b

    def test_class_weighting(self):
        """A WorldTour good result + low-class bad result should outscore
        a low-class good result + WorldTour bad result, because the WT
        result carries more weight in the average."""
        # Good WT result + bad low-class result
        rider_a = _make_results([
            ("2024-3-20", 3, "race/a/2024", "1.UWT"),
            ("2024-3-10", 18, "race/b/2024", "2.2"),
        ])
        # Bad WT result + good low-class result
        rider_b = _make_results([
            ("2024-3-20", 18, "race/a/2024", "1.UWT"),
            ("2024-3-10", 3, "race/b/2024", "2.2"),
        ])
        score_a = self.predictor._score_recent_form(rider_a, self.race_date)
        score_b = self.predictor._score_recent_form(rider_b, self.race_date)
        assert score_a > score_b

    def test_dnf_scores_zero(self):
        results = _make_results([
            ("2024-3-25", "DNF", "race/a/2024", "1.UWT"),
        ])
        score = self.predictor._score_recent_form(results, self.race_date)
        assert score == 0.0

    def test_mixed_results(self):
        """Mix of good and mediocre results should give moderate score."""
        results = _make_results([
            ("2024-3-30", 2, "race/a/2024", "1.UWT"),
            ("2024-3-15", 15, "race/b/2024", "1.UWT"),
            ("2024-2-10", 8, "race/c/2024", "1.Pro"),
        ])
        score = self.predictor._score_recent_form(results, self.race_date)
        assert 0.3 < score < 0.9


class TestClassicPedigreeScore:
    """Test classic pedigree (historical classics performance) scoring."""

    def setup_method(self):
        self.predictor = ClassicsPredictor()

    def test_no_history_returns_zero(self):
        score = self.predictor._score_classic_pedigree(
            [], "race/paris-roubaix", 2024
        )
        assert score == 0.0

    def test_previous_win_scores_high(self):
        results = _make_results([
            ("2023-4-9", 1, "race/paris-roubaix/2023", "1.UWT"),
        ])
        score = self.predictor._score_classic_pedigree(
            results, "race/paris-roubaix", 2024
        )
        assert score > 0.7

    def test_recency_matters(self):
        """Last year's result matters more than 4 years ago."""
        recent = _make_results([
            ("2023-4-9", 3, "race/paris-roubaix/2023", "1.UWT"),
        ])
        old = _make_results([
            ("2020-4-9", 3, "race/paris-roubaix/2020", "1.UWT"),
        ])
        recent_score = self.predictor._score_classic_pedigree(
            recent, "race/paris-roubaix", 2024
        )
        old_score = self.predictor._score_classic_pedigree(
            old, "race/paris-roubaix", 2024
        )
        assert recent_score > old_score

    def test_multiple_editions(self):
        """Consistent top-10 results across editions should score high."""
        results = _make_results([
            ("2023-4-9", 5, "race/paris-roubaix/2023", "1.UWT"),
            ("2022-4-17", 8, "race/paris-roubaix/2022", "1.UWT"),
            ("2021-4-3", 3, "race/paris-roubaix/2021", "1.UWT"),
        ])
        score = self.predictor._score_classic_pedigree(
            results, "race/paris-roubaix", 2024
        )
        assert score > 0.7

    def test_ignores_different_classic(self):
        """Results in a different classic shouldn't count."""
        results = _make_results([
            ("2023-3-18", 1, "race/milano-sanremo/2023", "1.UWT"),
        ])
        score = self.predictor._score_classic_pedigree(
            results, "race/paris-roubaix", 2024
        )
        assert score == 0.0


class TestSpecialtyScore:
    """Test PCS specialty score."""

    def setup_method(self):
        self.predictor = ClassicsPredictor()

    def test_high_one_day_points(self):
        profile = {
            "points_per_speciality": {"one_day_races": 4000}
        }
        score = self.predictor._score_specialty(profile)
        assert score > 0.95

    def test_zero_points(self):
        profile = {
            "points_per_speciality": {"one_day_races": 0}
        }
        score = self.predictor._score_specialty(profile)
        assert score == 0.0

    def test_moderate_points(self):
        profile = {
            "points_per_speciality": {"one_day_races": 1000}
        }
        score = self.predictor._score_specialty(profile)
        assert 0.5 < score < 1.0

    def test_missing_speciality(self):
        score = self.predictor._score_specialty({})
        assert score == 0.0


class TestPreviousYearScore:
    """Test previous year result scoring."""

    def setup_method(self):
        self.predictor = ClassicsPredictor()

    def test_previous_year_win(self):
        results = _make_results([
            ("2023-4-9", 1, "race/paris-roubaix/2023", "1.UWT"),
        ])
        score = self.predictor._score_previous_year(
            results, "race/paris-roubaix", TerrainType.COBBLES, 2024
        )
        assert score > 0.5

    def test_no_previous_year(self):
        score = self.predictor._score_previous_year(
            [], "race/paris-roubaix", TerrainType.COBBLES, 2024
        )
        assert score == 0.0

    def test_similar_terrain_contributes(self):
        """Good results in a similar-terrain classic should help."""
        # No result in Paris-Roubaix, but good result in another cobble race
        # (Only Paris-Roubaix has pure COBBLES terrain in our metadata,
        #  so test with HILLY terrain which has multiple classics)
        results = _make_results([
            ("2023-4-23", 2, "race/liege-bastogne-liege/2023", "1.UWT"),
        ])
        # Predict for Il Lombardia (also hilly)
        score = self.predictor._score_previous_year(
            results, "race/il-lombardia", TerrainType.HILLY, 2024
        )
        assert score > 0.0


class TestPreparationScore:
    """Test season preparation scoring."""

    def setup_method(self):
        self.predictor = ClassicsPredictor()
        self.race_date = date(2024, 4, 7)  # ~97 days into season

    def test_optimal_preparation(self):
        """~25 race days by early April should be optimal."""
        results = _make_results([
            (f"2024-{1 + i // 10}-{1 + i % 28}", 10, "race/a/2024", "1.UWT")
            for i in range(25)
        ])
        score = self.predictor._score_preparation(results, self.race_date)
        assert score > 0.8

    def test_underprepared(self):
        """Only 3 race days should score low."""
        results = _make_results([
            ("2024-3-1", 10, "race/a/2024", "1.UWT"),
            ("2024-3-5", 10, "race/a/2024", "1.UWT"),
            ("2024-3-10", 10, "race/a/2024", "1.UWT"),
        ])
        score = self.predictor._score_preparation(results, self.race_date)
        assert score < 0.5

    def test_no_races(self):
        score = self.predictor._score_preparation([], self.race_date)
        assert score < 0.2


class TestInjuryScore:
    """Test injury detection scoring."""

    def setup_method(self):
        self.predictor = ClassicsPredictor()
        self.race_date = date(2024, 4, 7)

    def test_no_races_severe_penalty(self):
        score = self.predictor._score_injury([], self.race_date)
        assert score == 0.3

    def test_normal_season_no_penalty(self):
        """Regular racing pattern should give score near 1.0."""
        results = _make_results([
            ("2024-1-28", 15, "race/a/2024", "1.Pro"),
            ("2024-2-10", 10, "race/b/2024", "1.Pro"),
            ("2024-2-25", 8, "race/c/2024", "1.UWT"),
            ("2024-3-9", 5, "race/d/2024", "1.UWT"),
            ("2024-3-24", 3, "race/e/2024", "1.UWT"),
        ])
        score = self.predictor._score_injury(results, self.race_date)
        assert score > 0.85

    def test_late_start_penalty(self):
        """Starting the season in March should be penalized for an April
        classic."""
        results = _make_results([
            ("2024-3-15", 10, "race/a/2024", "1.UWT"),
            ("2024-3-25", 8, "race/b/2024", "1.UWT"),
        ])
        score = self.predictor._score_injury(results, self.race_date)
        # Late start should reduce score
        assert score < 0.9

    def test_large_gap_penalty(self):
        """A 40-day gap mid-season suggests injury."""
        results = _make_results([
            ("2024-1-28", 10, "race/a/2024", "1.Pro"),
            ("2024-2-5", 10, "race/b/2024", "1.Pro"),
            # 40+ day gap
            ("2024-3-20", 10, "race/c/2024", "1.UWT"),
        ])
        score = self.predictor._score_injury(results, self.race_date)
        assert score < 0.95

    def test_recent_dnf_penalty(self):
        """A DNF in the last 30 days is a concern."""
        results = _make_results([
            ("2024-1-28", 10, "race/a/2024", "1.Pro"),
            ("2024-2-15", 5, "race/b/2024", "1.UWT"),
            ("2024-3-10", 8, "race/c/2024", "1.UWT"),
            ("2024-3-25", "DNF", "race/d/2024", "1.UWT"),
        ])
        score = self.predictor._score_injury(results, self.race_date)
        assert score < 0.9


# ---------------------------------------------------------------------------
# Integration tests: full prediction pipeline
# ---------------------------------------------------------------------------

class TestPredictionPipeline:
    """Test the full prediction pipeline with synthetic data."""

    def setup_method(self):
        self.predictor = ClassicsPredictor()
        self.race_date = date(2024, 4, 7)

    def _build_startlist(self, riders):
        return [
            {"rider_url": url, "rider_name": name}
            for url, name in riders
        ]

    def test_experienced_classics_rider_beats_young_rider(self):
        """
        Van Aert-like profile (age 29, good form, classics history)
        should rank above a talented 22-year-old neo-pro in a monument.
        """
        experienced_results = _make_results([
            # Strong preseason
            ("2024-3-2", 1, "race/strade-bianche/2024", "1.UWT"),
            ("2024-3-9", 2, "race/milano-sanremo/2024", "1.UWT"),
            ("2024-3-23", 3, "race/e3-harelbeke/2024", "1.UWT"),
            ("2024-3-27", 1, "race/gent-wevelgem/2024", "1.UWT"),
            # Previous year pedigree
            ("2023-4-2", 2, "race/ronde-van-vlaanderen/2023", "1.UWT"),
            ("2023-4-9", 1, "race/paris-roubaix/2023", "1.UWT"),
            # More season prep
            ("2024-2-1", 5, "race/a/2024", "1.Pro"),
            ("2024-2-10", 3, "race/b/2024", "1.Pro"),
            ("2024-2-18", 4, "race/c/2024", "1.Pro"),
            ("2024-1-28", 8, "race/d/2024", "1.Pro"),
        ])

        young_results = _make_results([
            # Decent preseason but limited
            ("2024-3-15", 8, "race/a/2024", "1.UWT"),
            ("2024-3-25", 12, "race/b/2024", "1.UWT"),
            ("2024-2-5", 15, "race/c/2024", "1.Pro"),
            ("2024-2-20", 10, "race/d/2024", "1.Pro"),
        ])

        rider_data = {
            "rider/experienced-pro": _make_rider_data(
                "1995-4-15", one_day_pts=3500, results=experienced_results
            ),
            "rider/young-talent": _make_rider_data(
                "2002-6-1", one_day_pts=800, results=young_results
            ),
        }

        startlist = self._build_startlist([
            ("rider/experienced-pro", "EXPERIENCED Pro"),
            ("rider/young-talent", "YOUNG Talent"),
        ])

        predictions = self.predictor.predict_from_data(
            race_base_url="race/paris-roubaix",
            year=2024,
            distance=260.0,
            race_date=self.race_date,
            startlist=startlist,
            rider_data=rider_data,
        )

        assert len(predictions) == 2
        assert predictions[0]["rider_url"] == "rider/experienced-pro"
        assert predictions[0]["rank"] == 1
        assert predictions[0]["score"] > predictions[1]["score"]

    def test_young_rider_competitive_in_short_classic(self):
        """
        A talented 24-year-old should have a smaller gap (or even beat)
        an older rider in a shorter 185km classic like Strade Bianche.
        """
        young_results = _make_results([
            ("2024-2-20", 1, "race/a/2024", "1.UWT"),
            ("2024-3-1", 2, "race/b/2024", "1.UWT"),
            ("2024-2-5", 5, "race/c/2024", "1.Pro"),
            ("2024-1-28", 8, "race/d/2024", "1.Pro"),
            # Previous strade bianche
            ("2023-3-4", 5, "race/strade-bianche/2023", "1.UWT"),
        ])

        older_results = _make_results([
            ("2024-2-20", 8, "race/a/2024", "1.UWT"),
            ("2024-3-1", 12, "race/b/2024", "1.UWT"),
            ("2024-2-5", 15, "race/c/2024", "1.Pro"),
            ("2024-1-28", 10, "race/d/2024", "1.Pro"),
            # Previous strade bianche
            ("2023-3-4", 10, "race/strade-bianche/2023", "1.UWT"),
        ])

        rider_data = {
            "rider/young-talent": _make_rider_data(
                "2000-5-1", one_day_pts=1500, results=young_results
            ),
            "rider/older-rider": _make_rider_data(
                "1991-3-15", one_day_pts=2000, results=older_results
            ),
        }

        startlist = self._build_startlist([
            ("rider/young-talent", "YOUNG Talent"),
            ("rider/older-rider", "OLDER Rider"),
        ])

        predictions = self.predictor.predict_from_data(
            race_base_url="race/strade-bianche",
            year=2024,
            distance=185.0,
            race_date=date(2024, 3, 2),
            startlist=startlist,
            rider_data=rider_data,
        )

        # Young rider should rank first given better form and decent age fit
        assert predictions[0]["rider_url"] == "rider/young-talent"

    def test_injured_rider_penalized(self):
        """A rider coming back from injury should rank lower."""
        healthy_results = _make_results([
            ("2024-1-28", 10, "race/a/2024", "1.Pro"),
            ("2024-2-10", 5, "race/b/2024", "1.UWT"),
            ("2024-2-25", 3, "race/c/2024", "1.UWT"),
            ("2024-3-10", 2, "race/d/2024", "1.UWT"),
            ("2024-3-24", 4, "race/e/2024", "1.UWT"),
        ])

        injured_results = _make_results([
            # Late start (March only) suggests returning from injury
            ("2024-3-15", 20, "race/a/2024", "1.UWT"),
            ("2024-3-25", 15, "race/b/2024", "1.UWT"),
        ])

        rider_data = {
            "rider/healthy": _make_rider_data(
                "1996-6-1", one_day_pts=2500, results=healthy_results
            ),
            "rider/injured": _make_rider_data(
                "1996-6-1", one_day_pts=2500, results=injured_results
            ),
        }

        startlist = self._build_startlist([
            ("rider/healthy", "HEALTHY Rider"),
            ("rider/injured", "INJURED Rider"),
        ])

        predictions = self.predictor.predict_from_data(
            race_base_url="race/ronde-van-vlaanderen",
            year=2024,
            distance=260.0,
            race_date=self.race_date,
            startlist=startlist,
            rider_data=rider_data,
        )

        assert predictions[0]["rider_url"] == "rider/healthy"

    def test_prediction_output_structure(self):
        """Verify the structure of prediction output."""
        rider_data = {
            "rider/test": _make_rider_data(
                "1996-1-1",
                one_day_pts=2000,
                results=_make_results([
                    ("2024-3-15", 5, "race/a/2024", "1.UWT"),
                ]),
            ),
        }

        startlist = [{"rider_url": "rider/test", "rider_name": "Test Rider"}]

        predictions = self.predictor.predict_from_data(
            race_base_url="race/paris-roubaix",
            year=2024,
            distance=260.0,
            race_date=self.race_date,
            startlist=startlist,
            rider_data=rider_data,
        )

        assert len(predictions) == 1
        p = predictions[0]
        assert "rider_name" in p
        assert "rider_url" in p
        assert "rank" in p
        assert "score" in p
        assert "features" in p
        assert isinstance(p["score"], float)
        assert isinstance(p["features"], dict)

        expected_features = {
            "recent_form", "classic_pedigree", "specialty_score",
            "age_distance_fit", "previous_year", "preparation",
            "injury_penalty", "terrain_match", "sprint_capability",
            "momentum", "team_strength",
        }
        assert set(p["features"].keys()) == expected_features

    def test_custom_weights(self):
        """Custom weights should change prediction ordering.

        Rider A: excellent current form with many race days, no pedigree.
        Rider B: terrible form but dominant Roubaix history.

        With form-heavy weights rider A wins; with pedigree-heavy weights
        rider B wins.
        """
        rider_a_results = _make_results(
            [
                ("2024-3-30", 1, "race/gent-wevelgem/2024", "1.UWT"),
                ("2024-3-23", 1, "race/e3-harelbeke/2024", "1.UWT"),
                ("2024-3-15", 2, "race/strade-bianche/2024", "1.UWT"),
                ("2024-3-9", 3, "race/a/2024", "1.UWT"),
            ]
            # Add 20 filler race days for good preparation score
            + [
                (f"2024-{1 + i // 15}-{1 + i % 28}", 10,
                 f"race/prep{i}/2024", "1.Pro")
                for i in range(20)
            ]
        )
        rider_b_results = _make_results([
            # Very poor current form (all outside top 40)
            ("2024-3-30", 50, "race/a/2024", "1.UWT"),
            ("2024-3-20", 45, "race/b/2024", "1.UWT"),
            ("2024-2-10", 40, "race/c/2024", "1.Pro"),
            ("2024-2-20", 42, "race/d/2024", "1.Pro"),
            ("2024-1-28", 38, "race/e/2024", "1.Pro"),
            # But dominant Roubaix history
            ("2023-4-9", 1, "race/paris-roubaix/2023", "1.UWT"),
            ("2022-4-17", 1, "race/paris-roubaix/2022", "1.UWT"),
            ("2021-4-11", 2, "race/paris-roubaix/2021", "1.UWT"),
        ])

        rider_data = {
            "rider/form-rider": _make_rider_data(
                "1996-1-1", one_day_pts=3000, results=rider_a_results
            ),
            "rider/pedigree-rider": _make_rider_data(
                "1996-1-1", one_day_pts=2500, results=rider_b_results
            ),
        }

        startlist = self._build_startlist([
            ("rider/form-rider", "FORM Rider"),
            ("rider/pedigree-rider", "PEDIGREE Rider"),
        ])

        # With form-heavy weights, form rider should win
        form_predictor = ClassicsPredictor(weights={
            "recent_form": 0.50,
            "classic_pedigree": 0.05,
            "specialty_score": 0.10,
            "age_distance_fit": 0.10,
            "previous_year": 0.05,
            "preparation": 0.15,
            "injury_penalty": 0.05,
        })
        form_pred = form_predictor.predict_from_data(
            "race/paris-roubaix", 2024, 260.0, self.race_date,
            startlist, rider_data,
        )
        assert form_pred[0]["rider_url"] == "rider/form-rider"

        # With pedigree-heavy weights, pedigree rider should win
        pedigree_predictor = ClassicsPredictor(weights={
            "recent_form": 0.05,
            "classic_pedigree": 0.60,
            "specialty_score": 0.05,
            "age_distance_fit": 0.05,
            "previous_year": 0.15,
            "preparation": 0.05,
            "injury_penalty": 0.05,
        })
        pedigree_pred = pedigree_predictor.predict_from_data(
            "race/paris-roubaix", 2024, 260.0, self.race_date,
            startlist, rider_data,
        )
        assert pedigree_pred[0]["rider_url"] == "rider/pedigree-rider"


class TestExplain:
    """Test the explain method."""

    def test_explain_output(self):
        predictor = ClassicsPredictor()
        prediction = {
            "rider_name": "Test Rider",
            "rider_url": "rider/test",
            "rank": 1,
            "score": 78.5,
            "features": {
                "recent_form": 0.85,
                "classic_pedigree": 0.70,
                "specialty_score": 0.90,
                "age_distance_fit": 0.82,
                "previous_year": 0.60,
                "preparation": 0.95,
                "injury_penalty": 1.0,
                "terrain_match": 0.75,
                "sprint_capability": 0.60,
                "momentum": 0.70,
                "team_strength": 0.80,
            },
        }
        explanation = predictor.explain(prediction)
        assert "Test Rider" in explanation
        assert "78.5" in explanation
        assert "Recent form" in explanation
        assert "Age-distance" in explanation
        assert "Terrain-rider match" in explanation
        assert "Team strength" in explanation


class TestTerrainMatchScore:
    """Test terrain-rider specialty matching."""

    def setup_method(self):
        self.predictor = ClassicsPredictor()

    def test_climber_scores_high_for_hilly_race(self):
        """A strong climber should score well for LBL (climbing_difficulty=0.85)."""
        profile = {
            "points_per_speciality": {
                "one_day_races": 2000, "climber": 3000,
                "gc": 2000, "time_trial": 500, "sprint": 100,
            }
        }
        meta = CLASSICS_METADATA["race/liege-bastogne-liege"]
        score = self.predictor._score_terrain_match(
            profile, TerrainType.HILLY, meta
        )
        assert score > 0.7

    def test_sprinter_scores_low_for_hilly_race(self):
        """A pure sprinter should score lower than a climber for a hilly race."""
        sprinter_profile = {
            "points_per_speciality": {
                "one_day_races": 500, "climber": 50,
                "gc": 50, "time_trial": 100, "sprint": 3000,
            }
        }
        climber_profile = {
            "points_per_speciality": {
                "one_day_races": 2000, "climber": 3000,
                "gc": 2000, "time_trial": 500, "sprint": 100,
            }
        }
        meta = CLASSICS_METADATA["race/liege-bastogne-liege"]
        sprinter_score = self.predictor._score_terrain_match(
            sprinter_profile, TerrainType.HILLY, meta
        )
        climber_score = self.predictor._score_terrain_match(
            climber_profile, TerrainType.HILLY, meta
        )
        assert sprinter_score < climber_score

    def test_tt_specialist_scores_high_for_cobbles(self):
        """A TT/power rider should score well for Roubaix (cobbles)."""
        profile = {
            "points_per_speciality": {
                "one_day_races": 1500, "climber": 50,
                "gc": 100, "time_trial": 3000, "sprint": 200,
            }
        }
        meta = CLASSICS_METADATA["race/paris-roubaix"]
        score = self.predictor._score_terrain_match(
            profile, TerrainType.COBBLES, meta
        )
        assert score > 0.7

    def test_empty_profile_returns_zero(self):
        score = self.predictor._score_terrain_match({}, TerrainType.HILLY, {})
        assert score == 0.0


class TestSprintCapabilityScore:
    """Test sprint capability scoring."""

    def setup_method(self):
        self.predictor = ClassicsPredictor()

    def test_sprinter_high_in_sprint_race(self):
        """A sprinter should score high in Gent-Wevelgem (sprint_finish_prob=0.6)."""
        profile = {
            "points_per_speciality": {
                "one_day_races": 1000, "sprint": 3000,
            }
        }
        meta = CLASSICS_METADATA["race/gent-wevelgem"]
        score = self.predictor._score_sprint_capability(profile, meta)
        assert score > 0.7

    def test_sprinter_lower_in_fleche(self):
        """A sprinter should score lower in Flèche Wallonne (sprint_finish_prob=0.0)."""
        profile = {
            "points_per_speciality": {
                "one_day_races": 500, "sprint": 3000,
            }
        }
        meta_gw = CLASSICS_METADATA["race/gent-wevelgem"]
        meta_fl = CLASSICS_METADATA["race/la-fleche-wallone"]
        score_gw = self.predictor._score_sprint_capability(profile, meta_gw)
        score_fl = self.predictor._score_sprint_capability(profile, meta_fl)
        assert score_gw > score_fl

    def test_puncheur_scores_everywhere(self):
        """A puncheur with high one_day pts should score decently even in non-sprint races."""
        profile = {
            "points_per_speciality": {
                "one_day_races": 4000, "sprint": 200,
            }
        }
        meta = CLASSICS_METADATA["race/la-fleche-wallone"]
        score = self.predictor._score_sprint_capability(profile, meta)
        assert score > 0.9  # Basically all punch, sprint_prob=0


class TestMomentumScore:
    """Test form momentum scoring."""

    def setup_method(self):
        self.predictor = ClassicsPredictor()
        self.race_date = date(2024, 4, 7)

    def test_no_recent_results_low(self):
        score = self.predictor._score_momentum([], self.race_date)
        assert score == 0.3

    def test_improving_form_beats_declining(self):
        """A rider improving should outscore a declining rider."""
        improving = _make_results([
            ("2024-3-30", 2, "race/a/2024", "1.UWT"),
            ("2024-3-20", 5, "race/b/2024", "1.UWT"),
            ("2024-3-10", 10, "race/c/2024", "1.UWT"),  # recent = better
            ("2024-2-20", 20, "race/d/2024", "1.UWT"),  # earlier = worse
            ("2024-2-10", 18, "race/e/2024", "1.UWT"),
        ])
        declining = _make_results([
            ("2024-3-30", 15, "race/a/2024", "1.UWT"),
            ("2024-3-20", 18, "race/b/2024", "1.UWT"),
            ("2024-3-10", 20, "race/c/2024", "1.UWT"),  # recent = worse
            ("2024-2-20", 2, "race/d/2024", "1.UWT"),  # earlier = better
            ("2024-2-10", 3, "race/e/2024", "1.UWT"),
        ])
        score_up = self.predictor._score_momentum(improving, self.race_date)
        score_down = self.predictor._score_momentum(declining, self.race_date)
        assert score_up > score_down

    def test_recent_wins_boost(self):
        """Recent wins should give a momentum boost."""
        wins = _make_results([
            ("2024-3-30", 1, "race/a/2024", "1.UWT"),
            ("2024-3-25", 1, "race/b/2024", "1.UWT"),
        ])
        score = self.predictor._score_momentum(wins, self.race_date)
        assert score > 0.7


class TestTeamStrengthScore:
    """Test team strength scoring."""

    def setup_method(self):
        self.predictor = ClassicsPredictor()

    def test_tier1_team_scores_highest(self):
        score = self.predictor._score_team_strength(
            {"team": "uae-team-emirates"}
        )
        assert score == 1.0

    def test_tier2_team_scores_medium(self):
        score = self.predictor._score_team_strength(
            {"team": "ineos-grenadiers"}
        )
        assert score == 0.7

    def test_unknown_team_scores_low(self):
        score = self.predictor._score_team_strength(
            {"team": "some-small-team"}
        )
        assert score == 0.4

    def test_no_team_neutral(self):
        score = self.predictor._score_team_strength({})
        assert score == 0.5


class TestClassicsMetadata:
    """Test that classics metadata is complete and consistent."""

    def test_all_monuments_present(self):
        monuments = [
            url for url, meta in CLASSICS_METADATA.items()
            if meta["type"] == ClassicType.MONUMENT
        ]
        assert len(monuments) == 5
        monument_names = {CLASSICS_METADATA[u]["name"] for u in monuments}
        assert "Milano-Sanremo" in monument_names
        assert "Paris-Roubaix" in monument_names

    def test_distances_reasonable(self):
        for url, meta in CLASSICS_METADATA.items():
            assert 150 <= meta["typical_distance"] <= 320, (
                f"{url} has unreasonable distance {meta['typical_distance']}"
            )

    def test_months_valid(self):
        for url, meta in CLASSICS_METADATA.items():
            assert 1 <= meta["month"] <= 12, (
                f"{url} has invalid month {meta['month']}"
            )

    def test_sprint_finish_prob_valid(self):
        for url, meta in CLASSICS_METADATA.items():
            assert 0.0 <= meta.get("sprint_finish_prob", 0.2) <= 1.0, (
                f"{url} has invalid sprint_finish_prob"
            )

    def test_climbing_difficulty_valid(self):
        for url, meta in CLASSICS_METADATA.items():
            assert 0.0 <= meta.get("climbing_difficulty", 0.4) <= 1.0, (
                f"{url} has invalid climbing_difficulty"
            )
