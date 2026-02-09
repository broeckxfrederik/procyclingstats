"""
Classics prediction engine for professional cycling.

Predicts rider performance in one-day classics based on multiple weighted
factors including age-distance suitability, recent form, injury indicators,
specialty scores, preparation level, and historical pedigree.

Key insight: young riders tend to struggle in longer monuments (250+ km)
like Milan-San Remo, Paris-Roubaix, and Tour of Flanders, while seasoned
pros (28-33) thrive at those distances. Shorter classics are more open
to younger talent.

Usage:

>>> from procyclingstats import ClassicsPredictor
>>> predictor = ClassicsPredictor()
>>> predictions = predictor.predict("race/paris-roubaix/2024")
>>> for p in predictions[:10]:
...     print(f"{p['rank']:>2}. {p['rider_name']:<30} {p['score']:.1f}")

You can also feed pre-collected data to avoid live scraping:

>>> predictor = ClassicsPredictor()
>>> predictions = predictor.predict(
...     "race/paris-roubaix/2024",
...     rider_data={"rider/mathieu-van-der-poel": { ... }},
... )
"""

import math
from datetime import date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .race_scraper import Race
from .race_startlist_scraper import RaceStartlist
from .rider_results_scraper import RiderResults
from .rider_scraper import Rider
from .stage_scraper import Stage


class ClassicType(Enum):
    """Classification of classic races by distance and prestige."""
    MONUMENT = "monument"
    MAJOR_CLASSIC = "major_classic"
    SEMI_CLASSIC = "semi_classic"


class TerrainType(Enum):
    """Terrain profile of classic races."""
    FLAT_PUNCH = "flat_punch"
    COBBLES = "cobbles"
    COBBLES_HILLS = "cobbles_hills"
    HILLY = "hilly"
    MOUNTAINOUS = "mountainous"


# Default feature weights for the prediction model.
# Calibrated against the 2025 spring classics season
# (Strade Bianche through Liège-Bastogne-Liège).
# Weights sum to 1.0. Adjust to shift emphasis.
DEFAULT_WEIGHTS = {
    "classic_pedigree": 0.15,
    "terrain_match": 0.15,
    "uphill_sprint": 0.12,
    "cobble_capability": 0.12,
    "sprint_capability": 0.08,
    "team_strength": 0.059,
    "preparation": 0.053,
    "previous_year": 0.05,
    "specialty_score": 0.046,
    "injury_penalty": 0.046,
    "recent_form": 0.046,
    "momentum": 0.046,
    "age_distance_fit": 0.033,
}

# Known classics with metadata for the prediction model.
# Keys are the PCS race base URL (without year).
CLASSICS_METADATA = {
    "race/milano-sanremo": {
        "type": ClassicType.MONUMENT,
        "terrain": TerrainType.FLAT_PUNCH,
        "typical_distance": 300,
        "month": 3,
        "name": "Milano-Sanremo",
        "sprint_finish_prob": 0.65,
        "climbing_difficulty": 0.2,
        "uphill_finish_prob": 0.15,  # Poggio is selective but often a sprint
        "cobble_difficulty": 0.0,
    },
    "race/ronde-van-vlaanderen": {
        "type": ClassicType.MONUMENT,
        "terrain": TerrainType.COBBLES_HILLS,
        "typical_distance": 260,
        "month": 4,
        "name": "Ronde van Vlaanderen",
        "sprint_finish_prob": 0.1,
        "climbing_difficulty": 0.5,
        "uphill_finish_prob": 0.3,   # Oude Kwaremont/Paterberg selective
        "cobble_difficulty": 0.8,    # Major cobble race
    },
    "race/paris-roubaix": {
        "type": ClassicType.MONUMENT,
        "terrain": TerrainType.COBBLES,
        "typical_distance": 260,
        "month": 4,
        "name": "Paris-Roubaix",
        "sprint_finish_prob": 0.15,
        "climbing_difficulty": 0.05,
        "uphill_finish_prob": 0.0,   # Flat velodrome finish
        "cobble_difficulty": 1.0,    # The cobble race
    },
    "race/liege-bastogne-liege": {
        "type": ClassicType.MONUMENT,
        "terrain": TerrainType.HILLY,
        "typical_distance": 260,
        "month": 4,
        "name": "Liège-Bastogne-Liège",
        "sprint_finish_prob": 0.05,
        "climbing_difficulty": 0.85,
        "uphill_finish_prob": 0.5,   # Often decided on final climbs
        "cobble_difficulty": 0.0,
    },
    "race/il-lombardia": {
        "type": ClassicType.MONUMENT,
        "terrain": TerrainType.HILLY,
        "typical_distance": 240,
        "month": 10,
        "name": "Il Lombardia",
        "sprint_finish_prob": 0.05,
        "climbing_difficulty": 0.8,
        "uphill_finish_prob": 0.5,   # San Fermo della Battaglia
        "cobble_difficulty": 0.0,
    },
    "race/strade-bianche": {
        "type": ClassicType.MAJOR_CLASSIC,
        "terrain": TerrainType.HILLY,
        "typical_distance": 185,
        "month": 3,
        "name": "Strade Bianche",
        "sprint_finish_prob": 0.1,
        "climbing_difficulty": 0.6,
        "uphill_finish_prob": 0.7,   # Piazza del Campo uphill sprint
        "cobble_difficulty": 0.1,    # Gravel roads, not pavé
    },
    "race/e3-harelbeke": {
        "type": ClassicType.MAJOR_CLASSIC,
        "terrain": TerrainType.COBBLES_HILLS,
        "typical_distance": 205,
        "month": 3,
        "name": "E3 Saxo Classic",
        "sprint_finish_prob": 0.15,
        "climbing_difficulty": 0.35,
        "uphill_finish_prob": 0.2,   # Paterberg/Taaienberg selective
        "cobble_difficulty": 0.6,    # Significant cobble sections
    },
    "race/gent-wevelgem": {
        "type": ClassicType.MAJOR_CLASSIC,
        "terrain": TerrainType.COBBLES_HILLS,
        "typical_distance": 250,
        "month": 3,
        "name": "Gent-Wevelgem",
        "sprint_finish_prob": 0.6,
        "climbing_difficulty": 0.2,
        "uphill_finish_prob": 0.1,   # Kemmelberg selective but flat finish
        "cobble_difficulty": 0.4,    # Some cobble sectors
    },
    "race/amstel-gold-race": {
        "type": ClassicType.MAJOR_CLASSIC,
        "terrain": TerrainType.HILLY,
        "typical_distance": 260,
        "month": 4,
        "name": "Amstel Gold Race",
        "sprint_finish_prob": 0.2,
        "climbing_difficulty": 0.55,
        "uphill_finish_prob": 0.6,   # Cauberg finish, often small group sprint
        "cobble_difficulty": 0.0,
    },
    "race/la-fleche-wallone": {
        "type": ClassicType.MAJOR_CLASSIC,
        "terrain": TerrainType.HILLY,
        "typical_distance": 195,
        "month": 4,
        "name": "La Flèche Wallonne",
        "sprint_finish_prob": 0.0,
        "climbing_difficulty": 0.75,
        "uphill_finish_prob": 0.95,  # Mur de Huy — the uphill sprint
        "cobble_difficulty": 0.0,
    },
    "race/san-sebastian": {
        "type": ClassicType.MAJOR_CLASSIC,
        "terrain": TerrainType.HILLY,
        "typical_distance": 225,
        "month": 7,
        "name": "Clásica San Sebastián",
        "sprint_finish_prob": 0.1,
        "climbing_difficulty": 0.7,
        "uphill_finish_prob": 0.4,
        "cobble_difficulty": 0.0,
    },
    "race/dwars-door-vlaanderen": {
        "type": ClassicType.SEMI_CLASSIC,
        "terrain": TerrainType.COBBLES_HILLS,
        "typical_distance": 185,
        "month": 3,
        "name": "Dwars door Vlaanderen",
        "sprint_finish_prob": 0.2,
        "climbing_difficulty": 0.3,
        "uphill_finish_prob": 0.2,
        "cobble_difficulty": 0.5,    # Significant cobble sections
    },
    "race/brabantse-pijl": {
        "type": ClassicType.SEMI_CLASSIC,
        "terrain": TerrainType.HILLY,
        "typical_distance": 200,
        "month": 4,
        "name": "Brabantse Pijl",
        "sprint_finish_prob": 0.15,
        "climbing_difficulty": 0.5,
        "uphill_finish_prob": 0.4,
        "cobble_difficulty": 0.0,
    },
    "race/kuurne-brussel-kuurne": {
        "type": ClassicType.SEMI_CLASSIC,
        "terrain": TerrainType.FLAT_PUNCH,
        "typical_distance": 197,
        "month": 3,
        "name": "Kuurne-Brussel-Kuurne",
        "sprint_finish_prob": 0.8,
        "climbing_difficulty": 0.1,
        "uphill_finish_prob": 0.0,
        "cobble_difficulty": 0.1,
    },
    "race/omloop-het-nieuwsblad": {
        "type": ClassicType.MAJOR_CLASSIC,
        "terrain": TerrainType.COBBLES_HILLS,
        "typical_distance": 200,
        "month": 3,
        "name": "Omloop Het Nieuwsblad",
        "sprint_finish_prob": 0.15,
        "climbing_difficulty": 0.35,
        "uphill_finish_prob": 0.15,
        "cobble_difficulty": 0.5,
    },
    "race/gp-samyn": {
        "type": ClassicType.SEMI_CLASSIC,
        "terrain": TerrainType.COBBLES,
        "typical_distance": 200,
        "month": 3,
        "name": "GP Samyn",
        "sprint_finish_prob": 0.1,
        "climbing_difficulty": 0.1,
        "uphill_finish_prob": 0.0,
        "cobble_difficulty": 0.5,
    },
    "race/nokere-koerse": {
        "type": ClassicType.SEMI_CLASSIC,
        "terrain": TerrainType.FLAT_PUNCH,
        "typical_distance": 195,
        "month": 3,
        "name": "Nokere Koerse",
        "sprint_finish_prob": 0.7,
        "climbing_difficulty": 0.1,
        "uphill_finish_prob": 0.0,
        "cobble_difficulty": 0.3,
    },
    "race/bredene-koksijde-classic": {
        "type": ClassicType.SEMI_CLASSIC,
        "terrain": TerrainType.FLAT_PUNCH,
        "typical_distance": 200,
        "month": 3,
        "name": "Bredene-Koksijde Classic",
        "sprint_finish_prob": 0.6,
        "climbing_difficulty": 0.05,
        "uphill_finish_prob": 0.0,
        "cobble_difficulty": 0.0,
    },
    "race/brugge-de-panne": {
        "type": ClassicType.MAJOR_CLASSIC,
        "terrain": TerrainType.FLAT_PUNCH,
        "typical_distance": 205,
        "month": 3,
        "name": "Brugge-De Panne",
        "sprint_finish_prob": 0.75,
        "climbing_difficulty": 0.05,
        "uphill_finish_prob": 0.0,
        "cobble_difficulty": 0.1,
    },
    "race/scheldeprijs": {
        "type": ClassicType.SEMI_CLASSIC,
        "terrain": TerrainType.FLAT_PUNCH,
        "typical_distance": 200,
        "month": 4,
        "name": "Scheldeprijs",
        "sprint_finish_prob": 0.95,
        "climbing_difficulty": 0.0,
        "uphill_finish_prob": 0.0,
        "cobble_difficulty": 0.0,
    },
    "race/ronde-van-limburg": {
        "type": ClassicType.SEMI_CLASSIC,
        "terrain": TerrainType.HILLY,
        "typical_distance": 200,
        "month": 4,
        "name": "Ronde van Limburg",
        "sprint_finish_prob": 0.4,
        "climbing_difficulty": 0.35,
        "uphill_finish_prob": 0.2,
        "cobble_difficulty": 0.0,
    },
}

# Known team tiers for team strength scoring.
# Tier 1 = top classics teams, tier 2 = strong teams, tier 3 = others.
TEAM_TIERS = {
    # Tier 1 — dominant classics squads
    "uae-team-emirates": 1,
    "team-visma-lease-a-bike": 1,
    "alpecin-deceuninck": 1,
    "lidl-trek": 1,
    "soudal-quick-step": 1,
    # Tier 2 — strong contenders
    "ineos-grenadiers": 2,
    "groupama-fdj": 2,
    "bahrain-victorious": 2,
    "team-jayco-alula": 2,
    "ef-education-easypost": 2,
    "intermarche-wanty": 2,
    "lotto-dstny": 2,
    "movistar-team": 2,
    "decathlon-ag2r-la-mondiale-team": 2,
    "cofidis": 2,
    "team-dsm-firmenich-postnl": 2,
    "uno-x-mobility": 2,
}


def _parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD or YYYY-M-D format."""
    parts = date_str.split("-")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def _race_base_url(race_url: str) -> str:
    """
    Extract the race base URL (without year) from a full race URL.

    >>> _race_base_url("race/paris-roubaix/2024")
    'race/paris-roubaix'
    >>> _race_base_url("race/milano-sanremo/2023/result")
    'race/milano-sanremo'
    """
    parts = race_url.replace("https://www.procyclingstats.com/", "").split("/")
    # race/<name> is the base, everything after is year/stage/etc.
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return race_url


def _extract_year(race_url: str) -> Optional[int]:
    """Extract year from a race URL like 'race/paris-roubaix/2024'."""
    parts = race_url.replace("https://www.procyclingstats.com/", "").split("/")
    for part in parts:
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def _age_from_birthdate(birthdate_str: str, reference_date: date) -> int:
    """Calculate age at a given date from a birthdate string."""
    bd = _parse_date(birthdate_str)
    age = reference_date.year - bd.year
    if (reference_date.month, reference_date.day) < (bd.month, bd.day):
        age -= 1
    return age


def _similar_terrain_classics(terrain: TerrainType) -> List[str]:
    """Get race base URLs of classics with similar terrain."""
    return [
        url for url, meta in CLASSICS_METADATA.items()
        if meta["terrain"] == terrain
    ]


class ClassicsPredictor:
    """
    Predicts rider performance in professional cycling classics.

    The model scores each rider on a startlist across multiple features,
    each capturing a different aspect of classics readiness:

    - **age_distance_fit**: Young riders (<25) fade in long monuments (250+ km)
      while experienced riders (28-33) peak at those distances. Shorter
      classics suit a wider age range.
    - **recent_form**: Weighted average of results in the 90 days before the
      race. More recent results and higher-class races count more.
    - **classic_pedigree**: Track record in this specific classic over the
      last 5 years, weighted by recency.
    - **specialty_score**: PCS one-day-race specialty points relative to the
      best rider on the startlist.
    - **previous_year**: Result in this same classic last year and in
      similar-terrain classics.
    - **preparation**: Number of race days this season. Underprepared riders
      score lower; overly fatigued riders get a small penalty.
    - **injury_penalty**: Detects late-season starts, large gaps in the
      calendar, and recent DNFs as signs of injury or illness.

    Each feature is normalized to [0, 1] and combined via configurable
    weights (default weights emphasize recent form and pedigree).

    :param weights: Dict of feature name -> weight. Defaults to
        ``DEFAULT_WEIGHTS``. Weights are normalized to sum to 1.0.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        raw = weights if weights is not None else DEFAULT_WEIGHTS.copy()
        total = sum(raw.values())
        self.weights = {k: v / total for k, v in raw.items()}

    def predict(
        self,
        race_url: str,
        year: Optional[int] = None,
        rider_data: Optional[Dict[str, Dict[str, Any]]] = None,
        startlist: Optional[List[Dict[str, Any]]] = None,
        race_distance: Optional[float] = None,
        race_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict rider rankings for a classic race.

        :param race_url: PCS race URL, e.g. ``"race/paris-roubaix/2024"``.
        :param year: Race year. Extracted from URL if not given.
        :param rider_data: Optional pre-collected rider data keyed by rider
            URL. Each value should have keys: ``"profile"`` (dict with
            ``birthdate``, ``points_per_speciality``), ``"results"`` (list of
            result dicts from RiderResults).
            When None, data is fetched live via the scrapers (slow).
        :param startlist: Optional pre-fetched startlist (list of dicts with
            at least ``rider_url`` and ``rider_name``). Fetched live if None.
        :param race_distance: Override race distance in km. If None, looked
            up from ``CLASSICS_METADATA`` or fetched from the race page.
        :param race_date: Override race date. If None, estimated from
            ``CLASSICS_METADATA`` or fetched from the race page.
        :return: List of prediction dicts sorted by score (best first).
            Each dict has: ``rider_name``, ``rider_url``, ``rank``,
            ``score``, ``features`` (dict of individual feature scores).
        """
        if year is None:
            year = _extract_year(race_url)
        if year is None:
            year = date.today().year

        base_url = _race_base_url(race_url)
        race_meta = CLASSICS_METADATA.get(base_url, {})

        # Resolve race distance
        if race_distance is None:
            race_distance = race_meta.get("typical_distance")
        if race_distance is None:
            race_distance = self._fetch_race_distance(race_url)

        # Resolve race date
        if race_date is None:
            month = race_meta.get("month")
            if month:
                race_date = date(year, month, 15)
            else:
                race_date = self._fetch_race_date(race_url)

        # Get terrain info
        terrain = race_meta.get("terrain", TerrainType.HILLY)

        # Fetch startlist if not provided
        if startlist is None:
            startlist = self._fetch_startlist(race_url, year)

        # Compute features for each rider
        predictions = []
        for rider_entry in startlist:
            rider_url = rider_entry.get("rider_url", "")
            rider_name = rider_entry.get("rider_name", "Unknown")

            # Get rider data (pre-collected or fetch live)
            if rider_data and rider_url in rider_data:
                rdata = rider_data[rider_url]
            else:
                rdata = self._fetch_rider_data(rider_url, year)

            if rdata is None:
                continue

            features = self._compute_features(
                rdata, base_url, terrain, race_distance, race_date, year
            )

            score = sum(
                self.weights.get(fname, 0) * fval
                for fname, fval in features.items()
            )

            predictions.append({
                "rider_name": rider_name,
                "rider_url": rider_url,
                "score": round(score * 100, 1),
                "features": {k: round(v, 3) for k, v in features.items()},
            })

        # Sort by score descending and assign ranks
        predictions.sort(key=lambda p: p["score"], reverse=True)
        for i, p in enumerate(predictions, 1):
            p["rank"] = i

        return predictions

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        rider_data: Dict[str, Any],
        race_base_url: str,
        terrain: TerrainType,
        distance: float,
        race_date: date,
        year: int,
    ) -> Dict[str, float]:
        """Compute all feature scores for a single rider."""
        profile = rider_data.get("profile", {})
        results = rider_data.get("results", [])

        # Age at race date
        birthdate = profile.get("birthdate")
        age = None
        if birthdate:
            try:
                age = _age_from_birthdate(birthdate, race_date)
            except (ValueError, IndexError):
                age = None

        race_meta = CLASSICS_METADATA.get(race_base_url, {})

        features = {}

        features["age_distance_fit"] = self._score_age_distance(age, distance)
        features["recent_form"] = self._score_recent_form(results, race_date)
        features["classic_pedigree"] = self._score_classic_pedigree(
            results, race_base_url, year
        )
        features["specialty_score"] = self._score_specialty(profile)
        features["previous_year"] = self._score_previous_year(
            results, race_base_url, terrain, year
        )
        features["preparation"] = self._score_preparation(results, race_date)
        features["injury_penalty"] = self._score_injury(results, race_date)
        features["terrain_match"] = self._score_terrain_match(
            profile, terrain, race_meta
        )
        features["sprint_capability"] = self._score_sprint_capability(
            profile, race_meta
        )
        features["uphill_sprint"] = self._score_uphill_sprint(
            profile, race_meta
        )
        features["cobble_capability"] = self._score_cobble_capability(
            profile, race_meta
        )
        features["momentum"] = self._score_momentum(results, race_date)
        features["team_strength"] = self._score_team_strength(rider_data)

        return features

    @staticmethod
    def _score_age_distance(age: Optional[int], distance: float) -> float:
        """
        Score rider's age suitability for the race distance.

        Young riders (<25) are penalized more heavily in longer races (250+ km)
        because they tend to fade in the finale of monuments. The optimal age
        shifts upward as distance increases:

        - 180 km classic: optimal ~26, broad tolerance
        - 260 km monument: optimal ~29, narrower tolerance
        - 300 km Milan-San Remo: optimal ~31, narrow tolerance

        Uses a Gaussian curve centered on the distance-dependent optimum.
        """
        if age is None:
            return 0.5  # neutral if unknown

        # Optimal age increases with distance
        optimal_age = 26.0 + (distance - 180.0) * 0.042
        # Tolerance (sigma) narrows for longer races
        sigma = max(3.5, 5.5 - (distance - 180.0) * 0.012)

        score = math.exp(-0.5 * ((age - optimal_age) / sigma) ** 2)

        # Extra penalty for very young riders in long races
        if age < 24 and distance >= 250:
            youth_penalty = 0.7 + (age - 20) * 0.075  # 20y=0.7, 23y=0.925
            youth_penalty = max(0.5, min(1.0, youth_penalty))
            score *= youth_penalty

        return score

    @staticmethod
    def _score_recent_form(
        results: List[Dict[str, Any]],
        race_date: date,
        window_days: int = 90,
    ) -> float:
        """
        Score recent race form in the window before the target race.

        Results are weighted by:
        - Recency: most recent races count more
        - Race class: WorldTour results weighted higher
        - Finishing position: top placements score highest

        Returns 0.0 if no results in the window.
        """
        relevant = []
        for r in results:
            try:
                rdate = _parse_date(r["date"])
            except (KeyError, ValueError, IndexError):
                continue
            days_ago = (race_date - rdate).days
            if 0 < days_ago <= window_days:
                relevant.append((r, days_ago))

        if not relevant:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for result, days_ago in relevant:
            # Recency weight: linearly from 1.0 (today) to 0.5 (window edge)
            recency_w = 1.0 - (days_ago / window_days) * 0.5

            # Race class weight
            race_class = str(result.get("class", ""))
            if "UWT" in race_class or "WC" in race_class:
                class_w = 1.3
            elif "Pro" in race_class or "1." in race_class:
                class_w = 1.0
            else:
                class_w = 0.7

            # Result score from rank
            rank = result.get("rank")
            if isinstance(rank, int) and rank > 0:
                # 1st=1.0, 5th=0.8, 10th=0.55, 20th=0.05, >20th tapers to 0
                result_score = max(0.0, 1.0 - (rank - 1) * 0.05)
            elif isinstance(rank, str) and rank.isdigit():
                r = int(rank)
                result_score = max(0.0, 1.0 - (r - 1) * 0.05)
            else:
                result_score = 0.0  # DNF, DNS, DSQ

            weight = recency_w * class_w
            total_score += result_score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _score_classic_pedigree(
        results: List[Dict[str, Any]],
        race_base_url: str,
        current_year: int,
        years_back: int = 5,
    ) -> float:
        """
        Score historical performance in this specific classic.

        Looks at the last ``years_back`` editions. Recent editions are
        weighted more heavily. A top-5 finish last year is a strong signal;
        a top-20 three years ago still contributes.
        """
        classic_results = []
        for r in results:
            stage_url = r.get("stage_url", "")
            if race_base_url not in stage_url:
                continue
            try:
                result_year = _parse_date(r["date"]).year
            except (KeyError, ValueError, IndexError):
                continue
            if current_year - years_back <= result_year < current_year:
                classic_results.append((r, result_year))

        if not classic_results:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for result, result_year in classic_results:
            years_ago = current_year - result_year
            # Recency: last year=1.0, 2 years=0.7, 3=0.5, 4=0.36, 5=0.28
            recency_w = 1.0 / (1.0 + years_ago * 0.4)

            rank = result.get("rank")
            if isinstance(rank, int) and rank > 0:
                # Gentler decay: top 50 all contribute something
                result_score = max(0.0, 1.0 - (rank - 1) * 0.018)
            elif isinstance(rank, str) and rank.isdigit():
                r = int(rank)
                result_score = max(0.0, 1.0 - (r - 1) * 0.018)
            else:
                result_score = 0.0

            total_score += result_score * recency_w
            total_weight += recency_w

        return total_score / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _score_specialty(profile: Dict[str, Any]) -> float:
        """
        Score based on PCS one-day-race specialty points.

        Normalizes against a reference maximum (top classics riders typically
        have 4000+ one-day-race points). Uses a logarithmic scale so the
        difference between 0 and 1000 matters more than 3000 vs 4000.
        """
        specialties = profile.get("points_per_speciality", {})
        one_day_pts = specialties.get("one_day_races", 0)

        if one_day_pts <= 0:
            return 0.0

        # Log-scaled normalization. A rider with ~4000 pts scores ~1.0
        # One with ~1000 pts scores ~0.83, ~200 pts scores ~0.64
        reference = 4000.0
        score = math.log1p(one_day_pts) / math.log1p(reference)
        return min(1.0, score)

    @staticmethod
    def _score_previous_year(
        results: List[Dict[str, Any]],
        race_base_url: str,
        terrain: TerrainType,
        current_year: int,
    ) -> float:
        """
        Score based on last year's result in this classic and in
        similar-terrain classics.

        Direct result in the same race last year is weighted 60%.
        Results in other classics with similar terrain are weighted 40%.
        """
        prev_year = current_year - 1
        same_race_score = 0.0
        similar_scores = []

        similar_urls = _similar_terrain_classics(terrain)

        for r in results:
            try:
                result_year = _parse_date(r["date"]).year
            except (KeyError, ValueError, IndexError):
                continue

            if result_year != prev_year:
                continue

            stage_url = r.get("stage_url", "")
            rank = r.get("rank")
            if isinstance(rank, str) and rank.isdigit():
                rank = int(rank)
            if not isinstance(rank, int) or rank <= 0:
                continue

            rank_score = max(0.0, 1.0 - (rank - 1) * 0.025)

            if race_base_url in stage_url:
                same_race_score = max(same_race_score, rank_score)
            else:
                for similar_url in similar_urls:
                    if similar_url in stage_url and similar_url != race_base_url:
                        similar_scores.append(rank_score)
                        break

        similar_avg = (
            sum(similar_scores) / len(similar_scores)
            if similar_scores
            else 0.0
        )

        return same_race_score * 0.6 + similar_avg * 0.4

    @staticmethod
    def _score_preparation(
        results: List[Dict[str, Any]], race_date: date
    ) -> float:
        """
        Score based on season preparation level (race days before classic).

        Optimal range is 20-35 race days for spring classics.
        Fewer days = underprepared. More days = potential fatigue.
        For autumn classics (Il Lombardia), the optimal is higher (50-70).
        """
        season_start = date(race_date.year, 1, 1)
        race_days = 0
        for r in results:
            try:
                rdate = _parse_date(r["date"])
            except (KeyError, ValueError, IndexError):
                continue
            if season_start <= rdate < race_date:
                race_days += 1

        # Adjust optimal range based on when in the season the race is
        days_into_season = (race_date - season_start).days
        # For a race 90 days in (April): optimal 20-35
        # For a race 270 days in (October): optimal 50-70
        optimal_low = max(10, int(days_into_season * 0.20))
        optimal_high = max(20, int(days_into_season * 0.35))

        if optimal_low <= race_days <= optimal_high:
            return 1.0
        elif race_days < optimal_low:
            return max(0.1, race_days / optimal_low)
        else:
            # Gradual fatigue penalty
            excess = race_days - optimal_high
            return max(0.3, 1.0 - excess * 0.02)

    @staticmethod
    def _score_injury(
        results: List[Dict[str, Any]], race_date: date
    ) -> float:
        """
        Detect injury/illness indicators and penalize accordingly.

        Checks for:
        - Late season start (no races before mid-February for spring classics)
        - Large gaps (>28 days) in the racing calendar
        - Recent DNFs or DNS in the last 30 days

        Returns 1.0 (no injury signs) down to ~0.3 (severe indicators).
        """
        season_start = date(race_date.year, 1, 1)
        season_results = []
        for r in results:
            try:
                rdate = _parse_date(r["date"])
            except (KeyError, ValueError, IndexError):
                continue
            if season_start <= rdate < race_date:
                season_results.append((rdate, r))

        if not season_results:
            return 0.3  # No races this season = major red flag

        season_results.sort(key=lambda x: x[0])
        penalty = 1.0

        # Late season start
        first_race_date = season_results[0][0]
        expected_start = date(race_date.year, 1, 25)
        if first_race_date > expected_start:
            delay = (first_race_date - expected_start).days
            # Each week of delay costs ~5%
            penalty *= max(0.5, 1.0 - delay * 0.007)

        # Calendar gaps > 28 days
        for i in range(1, len(season_results)):
            gap = (season_results[i][0] - season_results[i - 1][0]).days
            if gap > 28:
                # Longer gaps are more concerning
                gap_penalty = max(0.85, 1.0 - (gap - 28) * 0.003)
                penalty *= gap_penalty

        # Recent DNFs (last 30 days)
        recent_cutoff = race_date - timedelta(days=30)
        dnf_count = 0
        for rdate, r in season_results:
            if rdate >= recent_cutoff:
                rank = r.get("rank")
                if rank is None or (isinstance(rank, str) and
                                    rank in ("DNF", "DNS", "DSQ", "OTL")):
                    dnf_count += 1

        if dnf_count > 0:
            penalty *= max(0.6, 1.0 - dnf_count * 0.12)

        return max(0.3, penalty)

    @staticmethod
    def _score_terrain_match(
        profile: Dict[str, Any],
        terrain: TerrainType,
        race_meta: Dict[str, Any],
    ) -> float:
        """
        Score how well a rider's specialty profile matches the race terrain.

        Different terrains reward different specialty combinations:
        - COBBLES: power-based (one_day + time_trial)
        - HILLY / MOUNTAINOUS: climbing ability (one_day + climber)
        - FLAT_PUNCH: sprint + one_day
        - COBBLES_HILLS: all-round (one_day + time_trial + climber)

        Uses the climbing_difficulty from race metadata to blend between
        climber-friendly and power-friendly scoring.
        """
        specialties = profile.get("points_per_speciality", {})
        one_day = specialties.get("one_day_races", 0)
        sprint = specialties.get("sprint", 0)
        climber = specialties.get("climber", 0)
        tt = specialties.get("time_trial", 0)
        gc = specialties.get("gc", 0)

        climbing_diff = race_meta.get("climbing_difficulty", 0.4)

        # Terrain-specific scoring
        if terrain == TerrainType.COBBLES:
            # Power + endurance: TT (raw power) + one_day
            raw = one_day * 0.5 + tt * 0.35 + gc * 0.15
            ref = 4000.0
        elif terrain == TerrainType.HILLY or terrain == TerrainType.MOUNTAINOUS:
            # Climbing ability scales with difficulty
            raw = (
                one_day * (0.4 - climbing_diff * 0.15)
                + climber * climbing_diff * 0.5
                + gc * climbing_diff * 0.3
                + tt * (1 - climbing_diff) * 0.1
            )
            ref = 3500.0
        elif terrain == TerrainType.FLAT_PUNCH:
            # Sprint + endurance
            raw = one_day * 0.35 + sprint * 0.4 + tt * 0.15 + gc * 0.1
            ref = 3500.0
        elif terrain == TerrainType.COBBLES_HILLS:
            # All-round: cobbles power + hills
            raw = one_day * 0.4 + tt * 0.2 + climber * 0.2 + gc * 0.1 + sprint * 0.1
            ref = 3500.0
        else:
            raw = one_day
            ref = 4000.0

        if raw <= 0:
            return 0.0

        score = math.log1p(raw) / math.log1p(ref)
        return min(1.0, score)

    @staticmethod
    def _score_sprint_capability(
        profile: Dict[str, Any],
        race_meta: Dict[str, Any],
    ) -> float:
        """
        Score sprint capability relative to race sprint finish probability.

        Races like Gent-Wevelgem and Milano-Sanremo often end in sprints,
        so sprint specialists get a boost. For races that rarely end in
        sprints (Flèche Wallonne, LBL), this feature contributes little.
        """
        sprint_prob = race_meta.get("sprint_finish_prob", 0.2)
        specialties = profile.get("points_per_speciality", {})
        sprint_pts = specialties.get("sprint", 0)
        one_day_pts = specialties.get("one_day_races", 0)

        # Sprint ability (log-normalized)
        ref_sprint = 2000.0
        if sprint_pts > 0:
            sprint_score = math.log1p(sprint_pts) / math.log1p(ref_sprint)
            sprint_score = min(1.0, sprint_score)
        else:
            sprint_score = 0.0

        # Non-sprint ability (punching power / solo ability)
        ref_one_day = 4000.0
        if one_day_pts > 0:
            punch_score = math.log1p(one_day_pts) / math.log1p(ref_one_day)
            punch_score = min(1.0, punch_score)
        else:
            punch_score = 0.0

        # Blend based on sprint finish probability
        return sprint_score * sprint_prob + punch_score * (1.0 - sprint_prob)

    @staticmethod
    def _score_uphill_sprint(
        profile: Dict[str, Any],
        race_meta: Dict[str, Any],
    ) -> float:
        """
        Score uphill sprint (puncheur) ability for races with uphill finishes.

        Races like La Flèche Wallonne (Mur de Huy), Strade Bianche, and
        Amstel Gold Race reward riders who can sprint uphill — a blend of
        climbing explosiveness and one-day punch. This is fundamentally
        different from flat sprint ability: Pogačar, Pidcock, and Vauquelin
        thrive here, not Philipsen or Merlier.

        The score scales with ``uphill_finish_prob`` from race metadata.
        For flat races (Roubaix, Kuurne), this feature stays neutral.
        """
        uphill_prob = race_meta.get("uphill_finish_prob", 0.2)
        specialties = profile.get("points_per_speciality", {})
        climber_pts = specialties.get("climber", 0)
        one_day_pts = specialties.get("one_day_races", 0)
        gc_pts = specialties.get("gc", 0)

        # Puncheur score: explosive climbing + one-day ability
        # Climber pts are weighted highest because uphill sprints favor
        # riders with raw climbing power who can accelerate on gradients.
        # GC contributes because GC riders tend to have the sustained
        # power to survive uphill finishes.
        raw = climber_pts * 0.45 + one_day_pts * 0.35 + gc_pts * 0.20
        ref = 4000.0

        if raw <= 0:
            puncheur_score = 0.0
        else:
            puncheur_score = math.log1p(raw) / math.log1p(ref)
            puncheur_score = min(1.0, puncheur_score)

        # Generic one-day ability as fallback for non-uphill races
        if one_day_pts > 0:
            generic_score = math.log1p(one_day_pts) / math.log1p(ref)
            generic_score = min(1.0, generic_score)
        else:
            generic_score = 0.0

        # Blend: in uphill finishes, puncheur ability dominates;
        # in flat finishes, falls back to generic one-day ability
        return puncheur_score * uphill_prob + generic_score * (1.0 - uphill_prob)

    @staticmethod
    def _score_cobble_capability(
        profile: Dict[str, Any],
        race_meta: Dict[str, Any],
    ) -> float:
        """
        Score cobblestone race capability.

        Cobble classics (Paris-Roubaix, Ronde, E3, Dwars) reward a very
        specific rider profile: high sustained power (TT points), one-day
        endurance, and the build/handling to survive rough surfaces. This
        is its own specialty — pure TT riders or pure climbers don't
        necessarily handle cobbles well.

        The cobble score blends TT and one-day points, applied proportionally
        to ``cobble_difficulty`` from race metadata. For non-cobble races
        (LBL, Flèche), this feature stays neutral.
        """
        cobble_diff = race_meta.get("cobble_difficulty", 0.0)
        specialties = profile.get("points_per_speciality", {})
        one_day_pts = specialties.get("one_day_races", 0)
        tt_pts = specialties.get("time_trial", 0)
        sprint_pts = specialties.get("sprint", 0)

        # Cobble ability: power (TT) + endurance (one_day) + some sprint
        # TT riders have the raw wattage to push through cobbles.
        # One-day riders have the tactical and endurance ability.
        # Sprint contributes slightly for the group sprint scenarios.
        raw = one_day_pts * 0.45 + tt_pts * 0.40 + sprint_pts * 0.15
        ref = 4000.0

        if raw <= 0:
            cobble_score = 0.0
        else:
            cobble_score = math.log1p(raw) / math.log1p(ref)
            cobble_score = min(1.0, cobble_score)

        # Generic one-day ability as fallback for non-cobble races
        if one_day_pts > 0:
            generic_score = math.log1p(one_day_pts) / math.log1p(ref)
            generic_score = min(1.0, generic_score)
        else:
            generic_score = 0.0

        # Blend: in cobble races, cobble ability dominates;
        # in non-cobble races, falls back to generic one-day ability
        return cobble_score * cobble_diff + generic_score * (1.0 - cobble_diff)

    @staticmethod
    def _score_momentum(
        results: List[Dict[str, Any]],
        race_date: date,
    ) -> float:
        """
        Score recent form trajectory (improving vs declining).

        Compares form in the last 30 days vs 30-60 days before the race.
        An upward trajectory (recent results better than older ones) scores
        higher. Also rewards recent podiums and wins as confidence boosters.
        """
        recent = []  # last 30 days
        earlier = []  # 30-60 days ago

        for r in results:
            try:
                rdate = _parse_date(r["date"])
            except (KeyError, ValueError, IndexError):
                continue
            days_ago = (race_date - rdate).days
            if days_ago <= 0:
                continue

            rank = r.get("rank")
            if isinstance(rank, str) and rank.isdigit():
                rank = int(rank)
            if not isinstance(rank, int) or rank <= 0:
                continue

            result_score = max(0.0, 1.0 - (rank - 1) * 0.05)

            if days_ago <= 30:
                recent.append(result_score)
            elif days_ago <= 60:
                earlier.append(result_score)

        if not recent:
            return 0.3  # No recent activity = low momentum

        recent_avg = sum(recent) / len(recent)

        # Trajectory bonus: compare recent vs earlier
        if earlier:
            earlier_avg = sum(earlier) / len(earlier)
            trajectory = recent_avg - earlier_avg
            # Trajectory from -1 to +1, map to 0.3 - 0.7 contribution
            trajectory_score = 0.5 + trajectory * 0.5
            trajectory_score = max(0.0, min(1.0, trajectory_score))
        else:
            trajectory_score = 0.5  # No comparison data

        # Recent podium/win bonus
        podium_count = sum(1 for s in recent if s >= 0.85)  # top-3
        podium_bonus = min(0.3, podium_count * 0.1)

        # Combine: base form (50%) + trajectory (30%) + podium bonus (20%)
        score = recent_avg * 0.5 + trajectory_score * 0.3 + podium_bonus / 0.3 * 0.2
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_team_strength(rider_data: Dict[str, Any]) -> float:
        """
        Score the strength of a rider's team.

        Top classics teams (UAE, Visma, Alpecin, Lidl-Trek, Soudal-QS)
        provide tactical advantages: positioning, leadout trains, team
        support in crosswinds and on cobbles.

        Team info is read from ``rider_data["team"]`` (a team slug).
        Falls back to 0.5 if unknown.
        """
        team = rider_data.get("team", "")
        if not team:
            return 0.5  # Unknown team = neutral

        tier = TEAM_TIERS.get(team, 3)
        if tier == 1:
            return 1.0
        elif tier == 2:
            return 0.7
        else:
            return 0.4

    # ------------------------------------------------------------------
    # Data fetching (live scraping, used when rider_data is not provided)
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_startlist(
        race_url: str, year: int
    ) -> List[Dict[str, Any]]:
        """Fetch startlist for a race from procyclingstats."""
        base = _race_base_url(race_url)
        url = f"{base}/{year}/startlist"
        try:
            sl = RaceStartlist(url)
            return sl.startlist("rider_name", "rider_url")
        except (ValueError, AttributeError):
            return []

    @staticmethod
    def _fetch_rider_data(
        rider_url: str, year: int
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch profile and results for a single rider.

        Returns dict with keys ``"profile"`` and ``"results"``, or None
        on failure.
        """
        try:
            rider = Rider(rider_url)
            profile = {
                "birthdate": rider.birthdate(),
                "points_per_speciality": rider.points_per_speciality(),
            }
        except (ValueError, AttributeError):
            profile = {}

        # Fetch current season results
        all_results = []
        for target_year in (year, year - 1):
            try:
                results_url = f"{rider_url}/results"
                rr = RiderResults(results_url)
                season_results = rr.results()
                all_results.extend(season_results)
            except (ValueError, AttributeError):
                pass

        if not profile and not all_results:
            return None

        return {"profile": profile, "results": all_results}

    @staticmethod
    def _fetch_race_distance(race_url: str) -> float:
        """Fetch race distance. Falls back to 220 km."""
        try:
            stage = Stage(race_url)
            return stage.distance()
        except (ValueError, AttributeError):
            return 220.0

    @staticmethod
    def _fetch_race_date(race_url: str) -> date:
        """Fetch race date. Falls back to today."""
        try:
            race = Race(race_url)
            date_str = race.startdate()
            return _parse_date(date_str)
        except (ValueError, AttributeError, IndexError):
            return date.today()

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def predict_from_data(
        self,
        race_base_url: str,
        year: int,
        distance: float,
        race_date: date,
        startlist: List[Dict[str, Any]],
        rider_data: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Predict without any live scraping. All data must be pre-collected.

        This is the recommended entry point for batch analysis or when
        you've already collected data (avoids repeated HTTP requests).

        :param race_base_url: e.g. ``"race/paris-roubaix"``
        :param year: Race year.
        :param distance: Race distance in km.
        :param race_date: Date of the race.
        :param startlist: List of dicts with ``rider_url`` and ``rider_name``.
        :param rider_data: Dict mapping rider URL to data dict.
        :return: Ranked predictions.
        """
        return self.predict(
            race_url=f"{race_base_url}/{year}",
            year=year,
            rider_data=rider_data,
            startlist=startlist,
            race_distance=distance,
            race_date=race_date,
        )

    def explain(self, prediction: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation of a rider's prediction.

        :param prediction: Single prediction dict from ``predict()``.
        :return: Explanation string.
        """
        features = prediction.get("features", {})
        name = prediction.get("rider_name", "Unknown")
        score = prediction.get("score", 0)
        rank = prediction.get("rank", "?")

        lines = [f"#{rank} {name} (score: {score})"]
        lines.append("-" * 40)

        labels = {
            "recent_form": "Recent form (90 days)",
            "classic_pedigree": "Classic pedigree (5 years)",
            "specialty_score": "One-day race specialty",
            "age_distance_fit": "Age-distance suitability",
            "previous_year": "Previous year results",
            "preparation": "Season preparation",
            "injury_penalty": "Injury/fitness indicator",
            "terrain_match": "Terrain-rider match",
            "sprint_capability": "Sprint capability (flat)",
            "uphill_sprint": "Uphill sprint (puncheur)",
            "cobble_capability": "Cobble capability",
            "momentum": "Form momentum",
            "team_strength": "Team strength",
        }

        for feat_name, label in labels.items():
            val = features.get(feat_name, 0)
            weight = self.weights.get(feat_name, 0)
            bar_len = int(val * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            contribution = val * weight * 100
            lines.append(
                f"  {label:<30} [{bar}] {val:.2f} "
                f"(w={weight:.2f}, +{contribution:.1f})"
            )

        return "\n".join(lines)
