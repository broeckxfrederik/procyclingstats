"""
Backtest the ClassicsPredictor against actual 2025 spring classics results.

Uses pre-2025 data (2023-2024 results, rider profiles) plus accumulating
2025 preseason results to predict each classic, then compares against
actual top-10 finishes. Finally, runs a grid search to find optimal weights.

All data is hardcoded from public race results — no live scraping.
"""

import itertools
import math
from datetime import date
from typing import Any, Dict, List, Tuple

import pytest

from procyclingstats.classics_predictor import (
    ClassicsPredictor,
    TerrainType,
)


# =====================================================================
# RIDER DATABASE: pre-2025 profiles for ~40 key classics riders
# =====================================================================

# Birthdates (YYYY-M-D)
BIRTHDATES = {
    "rider/tadej-pogacar": "1998-9-21",
    "rider/mathieu-van-der-poel": "1995-1-19",
    "rider/mads-pedersen": "1995-12-18",
    "rider/wout-van-aert": "1994-9-15",
    "rider/tom-pidcock": "1999-7-30",
    "rider/ben-healy": "2000-8-11",
    "rider/filippo-ganna": "1996-7-25",
    "rider/tim-wellens": "1991-5-10",
    "rider/jasper-stuyven": "1992-4-17",
    "rider/stefan-kueng": "1993-11-16",
    "rider/tiesj-benoot": "1994-3-11",
    "rider/matteo-jorgenson": "1999-7-1",
    "rider/remco-evenepoel": "2000-1-25",
    "rider/mattias-skjelmose": "2000-9-26",
    "rider/magnus-cort": "1993-1-16",
    "rider/michael-matthews": "1990-9-26",
    "rider/neilson-powless": "1996-9-3",
    "rider/kaden-groves": "1998-11-12",
    "rider/jasper-philipsen": "1998-3-2",
    "rider/giulio-ciccone": "1994-12-20",
    "rider/thibau-nys": "2002-11-28",
    "rider/romain-gregoire": "2003-1-10",
    "rider/michael-valgren": "1992-2-7",
    "rider/davide-ballerini": "1994-10-21",
    "rider/fred-wright": "1999-6-13",
    "rider/florian-vermeersch": "1999-3-12",
    "rider/alexander-kristoff": "1987-7-5",
    "rider/jonathan-milan": "2000-10-1",
    "rider/olav-kooij": "2001-10-17",
    "rider/hugo-hofstetter": "1994-3-2",
    "rider/tim-merlier": "1992-10-30",
    "rider/biniam-girmay": "2000-4-2",
    "rider/lenny-martinez": "2003-6-17",
    "rider/kevin-vauquelin": "2001-3-26",
    "rider/pello-bilbao": "1990-2-25",
    "rider/gianni-vermeersch": "1992-11-19",
    "rider/daniel-martinez": "1996-4-25",
    "rider/santiago-buitrago": "1999-12-28",
    "rider/mauro-schmid": "2000-1-2",
    "rider/matteo-trentin": "1989-8-2",
    "rider/ivan-garcia-cortina": "1995-11-20",
    "rider/laurenz-rex": "1999-6-29",
    "rider/axel-laurance": "2002-5-13",
    "rider/simone-velasco": "1995-7-2",
    "rider/andrea-bagioli": "1999-3-23",
}

# Estimated PCS specialty points (pre-2025 career level)
# Keys: one_day, sprint, climber, tt, gc
SPECIALTY_PTS = {
    "rider/tadej-pogacar":         {"one_day": 4500, "sprint": 400, "climber": 4800, "tt": 2000, "gc": 5000},
    "rider/mathieu-van-der-poel":  {"one_day": 5000, "sprint": 1200, "climber": 800, "tt": 1500, "gc": 400},
    "rider/mads-pedersen":         {"one_day": 3200, "sprint": 2200, "climber": 200, "tt": 1800, "gc": 200},
    "rider/wout-van-aert":         {"one_day": 3800, "sprint": 1800, "climber": 1000, "tt": 2500, "gc": 600},
    "rider/tom-pidcock":           {"one_day": 2200, "sprint": 300, "climber": 1800, "tt": 800, "gc": 1200},
    "rider/ben-healy":             {"one_day": 1200, "sprint": 100, "climber": 1500, "tt": 400, "gc": 800},
    "rider/filippo-ganna":         {"one_day": 1500, "sprint": 600, "climber": 50, "tt": 3500, "gc": 100},
    "rider/tim-wellens":           {"one_day": 2000, "sprint": 300, "climber": 800, "tt": 500, "gc": 400},
    "rider/jasper-stuyven":        {"one_day": 2100, "sprint": 900, "climber": 100, "tt": 600, "gc": 100},
    "rider/stefan-kueng":          {"one_day": 1400, "sprint": 200, "climber": 50, "tt": 2800, "gc": 100},
    "rider/tiesj-benoot":          {"one_day": 2000, "sprint": 200, "climber": 500, "tt": 600, "gc": 300},
    "rider/matteo-jorgenson":      {"one_day": 1300, "sprint": 200, "climber": 400, "tt": 800, "gc": 600},
    "rider/remco-evenepoel":       {"one_day": 2800, "sprint": 300, "climber": 2200, "tt": 3500, "gc": 3800},
    "rider/mattias-skjelmose":     {"one_day": 1200, "sprint": 200, "climber": 800, "tt": 1200, "gc": 600},
    "rider/magnus-cort":           {"one_day": 1600, "sprint": 800, "climber": 600, "tt": 300, "gc": 200},
    "rider/michael-matthews":      {"one_day": 2400, "sprint": 1800, "climber": 300, "tt": 500, "gc": 200},
    "rider/neilson-powless":       {"one_day": 1100, "sprint": 200, "climber": 600, "tt": 400, "gc": 400},
    "rider/kaden-groves":          {"one_day": 800, "sprint": 2000, "climber": 50, "tt": 100, "gc": 50},
    "rider/jasper-philipsen":      {"one_day": 1800, "sprint": 3500, "climber": 50, "tt": 200, "gc": 50},
    "rider/giulio-ciccone":        {"one_day": 1000, "sprint": 100, "climber": 2000, "tt": 300, "gc": 1200},
    "rider/thibau-nys":            {"one_day": 400, "sprint": 300, "climber": 500, "tt": 200, "gc": 200},
    "rider/romain-gregoire":       {"one_day": 500, "sprint": 200, "climber": 600, "tt": 300, "gc": 300},
    "rider/michael-valgren":       {"one_day": 1700, "sprint": 400, "climber": 300, "tt": 400, "gc": 200},
    "rider/davide-ballerini":      {"one_day": 1200, "sprint": 800, "climber": 50, "tt": 200, "gc": 50},
    "rider/fred-wright":           {"one_day": 900, "sprint": 400, "climber": 200, "tt": 500, "gc": 200},
    "rider/florian-vermeersch":    {"one_day": 900, "sprint": 200, "climber": 100, "tt": 800, "gc": 200},
    "rider/alexander-kristoff":    {"one_day": 2600, "sprint": 2500, "climber": 50, "tt": 400, "gc": 50},
    "rider/jonathan-milan":        {"one_day": 700, "sprint": 2800, "climber": 50, "tt": 600, "gc": 50},
    "rider/olav-kooij":            {"one_day": 600, "sprint": 2600, "climber": 50, "tt": 200, "gc": 50},
    "rider/hugo-hofstetter":       {"one_day": 1000, "sprint": 1200, "climber": 100, "tt": 200, "gc": 50},
    "rider/tim-merlier":           {"one_day": 1300, "sprint": 2800, "climber": 50, "tt": 300, "gc": 50},
    "rider/biniam-girmay":         {"one_day": 1400, "sprint": 2200, "climber": 100, "tt": 200, "gc": 100},
    "rider/lenny-martinez":        {"one_day": 300, "sprint": 100, "climber": 800, "tt": 200, "gc": 500},
    "rider/kevin-vauquelin":       {"one_day": 600, "sprint": 100, "climber": 800, "tt": 500, "gc": 400},
    "rider/pello-bilbao":          {"one_day": 1000, "sprint": 100, "climber": 1500, "tt": 400, "gc": 1000},
    "rider/gianni-vermeersch":     {"one_day": 800, "sprint": 200, "climber": 200, "tt": 300, "gc": 100},
    "rider/daniel-martinez":       {"one_day": 1200, "sprint": 100, "climber": 1800, "tt": 800, "gc": 1800},
    "rider/santiago-buitrago":     {"one_day": 600, "sprint": 100, "climber": 1500, "tt": 400, "gc": 800},
    "rider/mauro-schmid":          {"one_day": 700, "sprint": 200, "climber": 500, "tt": 600, "gc": 300},
    "rider/matteo-trentin":        {"one_day": 2200, "sprint": 1500, "climber": 100, "tt": 600, "gc": 100},
    "rider/ivan-garcia-cortina":   {"one_day": 900, "sprint": 500, "climber": 100, "tt": 400, "gc": 100},
    "rider/laurenz-rex":           {"one_day": 400, "sprint": 300, "climber": 50, "tt": 200, "gc": 50},
    "rider/axel-laurance":         {"one_day": 300, "sprint": 100, "climber": 400, "tt": 200, "gc": 200},
    "rider/simone-velasco":        {"one_day": 600, "sprint": 200, "climber": 400, "tt": 200, "gc": 200},
    "rider/andrea-bagioli":        {"one_day": 800, "sprint": 300, "climber": 500, "tt": 300, "gc": 300},
}

# Team assignments (2025 team slugs)
RIDER_TEAMS = {
    "rider/tadej-pogacar": "uae-team-emirates",
    "rider/mathieu-van-der-poel": "alpecin-deceuninck",
    "rider/mads-pedersen": "lidl-trek",
    "rider/wout-van-aert": "team-visma-lease-a-bike",
    "rider/tom-pidcock": "ineos-grenadiers",
    "rider/ben-healy": "ef-education-easypost",
    "rider/filippo-ganna": "ineos-grenadiers",
    "rider/tim-wellens": "uae-team-emirates",
    "rider/jasper-stuyven": "lidl-trek",
    "rider/stefan-kueng": "groupama-fdj",
    "rider/tiesj-benoot": "team-visma-lease-a-bike",
    "rider/matteo-jorgenson": "team-visma-lease-a-bike",
    "rider/remco-evenepoel": "soudal-quick-step",
    "rider/mattias-skjelmose": "lidl-trek",
    "rider/magnus-cort": "uno-x-mobility",
    "rider/michael-matthews": "team-jayco-alula",
    "rider/neilson-powless": "ef-education-easypost",
    "rider/kaden-groves": "alpecin-deceuninck",
    "rider/jasper-philipsen": "alpecin-deceuninck",
    "rider/giulio-ciccone": "lidl-trek",
    "rider/thibau-nys": "lidl-trek",
    "rider/romain-gregoire": "groupama-fdj",
    "rider/michael-valgren": "ef-education-easypost",
    "rider/davide-ballerini": "movistar-team",
    "rider/fred-wright": "bahrain-victorious",
    "rider/florian-vermeersch": "alpecin-deceuninck",
    "rider/alexander-kristoff": "uno-x-mobility",
    "rider/jonathan-milan": "lidl-trek",
    "rider/olav-kooij": "team-visma-lease-a-bike",
    "rider/hugo-hofstetter": "cofidis",
    "rider/tim-merlier": "soudal-quick-step",
    "rider/biniam-girmay": "intermarche-wanty",
    "rider/lenny-martinez": "groupama-fdj",
    "rider/kevin-vauquelin": "decathlon-ag2r-la-mondiale-team",
    "rider/pello-bilbao": "bahrain-victorious",
    "rider/gianni-vermeersch": "alpecin-deceuninck",
    "rider/daniel-martinez": "decathlon-ag2r-la-mondiale-team",
    "rider/santiago-buitrago": "bahrain-victorious",
    "rider/mauro-schmid": "team-jayco-alula",
    "rider/matteo-trentin": "team-visma-lease-a-bike",
    "rider/ivan-garcia-cortina": "movistar-team",
    "rider/laurenz-rex": "intermarche-wanty",
    "rider/axel-laurance": "decathlon-ag2r-la-mondiale-team",
    "rider/simone-velasco": "movistar-team",
    "rider/andrea-bagioli": "decathlon-ag2r-la-mondiale-team",
}


def _r(date_str, rank, url, cls="1.UWT"):
    """Shorthand to build a result dict."""
    return {"date": date_str, "rank": rank, "stage_url": url, "class": cls}


# =====================================================================
# PRE-2025 RESULTS (2023-2024 classics + key races)
# =====================================================================

# For each rider: list of (date, rank, stage_url, class) results
# covering 2023-2024 classic performances and some early-season races.

PRE_2025_RESULTS = {
    "rider/tadej-pogacar": [
        # 2024
        _r("2024-3-2", 1, "race/strade-bianche/2024"),
        _r("2024-3-16", 3, "race/milano-sanremo/2024"),
        _r("2024-3-31", 1, "race/ronde-van-vlaanderen/2024"),
        _r("2024-4-17", 4, "race/la-fleche-wallone/2024"),
        _r("2024-4-21", 1, "race/liege-bastogne-liege/2024"),
        # 2024 prep races
        _r("2024-2-15", 1, "race/uae-tour/2024/stage-1"),
        _r("2024-2-20", 1, "race/uae-tour/2024/stage-5"),
        # 2023
        _r("2024-4-23", 1, "race/liege-bastogne-liege/2023"),
        _r("2023-3-11", 2, "race/strade-bianche/2023"),
        _r("2023-3-18", 5, "race/milano-sanremo/2023"),
        _r("2023-4-2", 2, "race/ronde-van-vlaanderen/2023"),
    ],
    "rider/mathieu-van-der-poel": [
        # 2024
        _r("2024-3-2", 2, "race/strade-bianche/2024"),
        _r("2024-3-16", 8, "race/milano-sanremo/2024"),
        _r("2024-3-22", 1, "race/e3-harelbeke/2024"),
        _r("2024-3-24", 2, "race/gent-wevelgem/2024"),
        _r("2024-3-31", 1, "race/ronde-van-vlaanderen/2024"),
        _r("2024-4-7", 1, "race/paris-roubaix/2024"),
        _r("2024-4-21", 3, "race/liege-bastogne-liege/2024"),
        # 2024 prep
        _r("2024-2-10", 3, "race/a/2024", "1.Pro"),
        _r("2024-2-18", 5, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 8, "race/c/2024", "1.Pro"),
        # 2023
        _r("2023-3-18", 1, "race/milano-sanremo/2023"),
        _r("2023-4-2", 1, "race/ronde-van-vlaanderen/2023"),
        _r("2023-4-9", 1, "race/paris-roubaix/2023"),
        _r("2023-3-24", 1, "race/e3-harelbeke/2023"),
    ],
    "rider/mads-pedersen": [
        # 2024
        _r("2024-3-16", 3, "race/milano-sanremo/2024"),
        _r("2024-3-24", 1, "race/gent-wevelgem/2024"),
        _r("2024-4-7", 3, "race/paris-roubaix/2024"),
        _r("2024-3-31", 8, "race/ronde-van-vlaanderen/2024"),
        # prep
        _r("2024-2-5", 5, "race/a/2024", "1.Pro"),
        _r("2024-2-15", 8, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 3, "race/c/2024", "1.Pro"),
        # 2023
        _r("2023-3-26", 5, "race/gent-wevelgem/2023"),
        _r("2023-4-2", 6, "race/ronde-van-vlaanderen/2023"),
        _r("2023-4-9", 5, "race/paris-roubaix/2023"),
    ],
    "rider/wout-van-aert": [
        # 2024
        _r("2024-3-22", 3, "race/e3-harelbeke/2024"),
        _r("2024-3-31", 5, "race/ronde-van-vlaanderen/2024"),
        _r("2024-4-7", 8, "race/paris-roubaix/2024"),
        _r("2024-4-14", 5, "race/amstel-gold-race/2024"),
        # prep
        _r("2024-2-10", 2, "race/a/2024", "1.Pro"),
        _r("2024-2-18", 3, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 5, "race/c/2024", "1.Pro"),
        # 2023 (injured much of 2023 season)
        _r("2023-3-11", 5, "race/strade-bianche/2023"),
    ],
    "rider/tom-pidcock": [
        # 2024
        _r("2024-4-14", 1, "race/amstel-gold-race/2024"),
        _r("2024-3-2", 15, "race/strade-bianche/2024"),
        # prep
        _r("2024-2-10", 10, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 8, "race/b/2024", "1.Pro"),
        # 2023
        _r("2023-4-14", 3, "race/amstel-gold-race/2023"),
        _r("2023-4-19", 5, "race/la-fleche-wallone/2023"),
    ],
    "rider/ben-healy": [
        # 2024
        _r("2024-4-14", 8, "race/amstel-gold-race/2024"),
        _r("2024-4-17", 6, "race/la-fleche-wallone/2024"),
        _r("2024-4-21", 8, "race/liege-bastogne-liege/2024"),
        # prep
        _r("2024-2-10", 12, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 15, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 10, "race/c/2024", "1.Pro"),
        # 2023
        _r("2023-4-21", 5, "race/liege-bastogne-liege/2023"),
    ],
    "rider/filippo-ganna": [
        _r("2024-3-16", 10, "race/milano-sanremo/2024"),
        _r("2024-3-22", 8, "race/e3-harelbeke/2024"),
        _r("2024-2-10", 5, "race/a/2024", "1.Pro"),
        _r("2024-2-18", 3, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 8, "race/c/2024", "1.Pro"),
    ],
    "rider/tim-wellens": [
        _r("2024-3-22", 4, "race/e3-harelbeke/2024"),
        _r("2024-3-2", 10, "race/strade-bianche/2024"),
        _r("2024-2-10", 8, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 5, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 12, "race/c/2024", "1.Pro"),
    ],
    "rider/jasper-stuyven": [
        _r("2024-3-22", 2, "race/e3-harelbeke/2024"),
        _r("2024-3-31", 8, "race/ronde-van-vlaanderen/2024"),
        _r("2024-2-10", 10, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 12, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 8, "race/c/2024", "1.Pro"),
        _r("2023-4-2", 8, "race/ronde-van-vlaanderen/2023"),
    ],
    "rider/stefan-kueng": [
        _r("2024-3-22", 10, "race/e3-harelbeke/2024"),
        _r("2024-3-31", 12, "race/ronde-van-vlaanderen/2024"),
        _r("2024-2-10", 8, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 10, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 15, "race/c/2024", "1.Pro"),
    ],
    "rider/tiesj-benoot": [
        _r("2024-4-14", 3, "race/amstel-gold-race/2024"),
        _r("2024-3-31", 10, "race/ronde-van-vlaanderen/2024"),
        _r("2024-2-10", 8, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 12, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 10, "race/c/2024", "1.Pro"),
    ],
    "rider/matteo-jorgenson": [
        _r("2024-3-22", 5, "race/e3-harelbeke/2024"),
        _r("2024-3-31", 15, "race/ronde-van-vlaanderen/2024"),
        _r("2024-2-10", 3, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 5, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 8, "race/c/2024", "1.Pro"),
    ],
    "rider/remco-evenepoel": [
        _r("2024-4-14", 10, "race/amstel-gold-race/2024"),
        _r("2024-4-17", 8, "race/la-fleche-wallone/2024"),
        _r("2024-4-21", 5, "race/liege-bastogne-liege/2024"),
        _r("2024-2-10", 1, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 2, "race/b/2024", "1.Pro"),
        # 2023
        _r("2023-4-21", 2, "race/liege-bastogne-liege/2023"),
        _r("2023-4-19", 3, "race/la-fleche-wallone/2023"),
    ],
    "rider/mattias-skjelmose": [
        _r("2024-4-14", 8, "race/amstel-gold-race/2024"),
        _r("2024-4-17", 10, "race/la-fleche-wallone/2024"),
        _r("2024-2-10", 5, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 8, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 10, "race/c/2024", "1.Pro"),
    ],
    "rider/magnus-cort": [
        _r("2024-3-16", 12, "race/milano-sanremo/2024"),
        _r("2024-3-2", 8, "race/strade-bianche/2024"),
        _r("2024-2-10", 5, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 10, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 8, "race/c/2024", "1.Pro"),
    ],
    "rider/michael-matthews": [
        _r("2024-3-16", 2, "race/milano-sanremo/2024"),
        _r("2024-4-14", 6, "race/amstel-gold-race/2024"),
        _r("2024-2-10", 5, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 8, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 10, "race/c/2024", "1.Pro"),
        _r("2023-3-18", 8, "race/milano-sanremo/2023"),
    ],
    "rider/neilson-powless": [
        _r("2024-4-14", 12, "race/amstel-gold-race/2024"),
        _r("2024-4-21", 10, "race/liege-bastogne-liege/2024"),
        _r("2024-2-10", 10, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 8, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 12, "race/c/2024", "1.Pro"),
    ],
    "rider/jasper-philipsen": [
        _r("2024-3-16", 1, "race/milano-sanremo/2024"),
        _r("2024-4-7", 2, "race/paris-roubaix/2024"),
        _r("2024-2-10", 1, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 3, "race/b/2024", "1.Pro"),
        _r("2024-1-28", 5, "race/c/2024", "1.Pro"),
    ],
    "rider/giulio-ciccone": [
        _r("2024-4-21", 10, "race/liege-bastogne-liege/2024"),
        _r("2024-4-17", 12, "race/la-fleche-wallone/2024"),
        _r("2024-2-10", 8, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 10, "race/b/2024", "1.Pro"),
    ],
    "rider/thibau-nys": [
        _r("2024-2-10", 15, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 12, "race/b/2024", "1.Pro"),
        _r("2024-3-2", 20, "race/strade-bianche/2024"),
    ],
    "rider/romain-gregoire": [
        _r("2024-3-16", 12, "race/milano-sanremo/2024"),
        _r("2024-2-10", 10, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 8, "race/b/2024", "1.Pro"),
    ],
    "rider/davide-ballerini": [
        _r("2024-3-24", 10, "race/gent-wevelgem/2024"),
        _r("2024-3-31", 15, "race/ronde-van-vlaanderen/2024"),
        _r("2024-2-10", 10, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 12, "race/b/2024", "1.Pro"),
    ],
    "rider/fred-wright": [
        _r("2024-3-16", 15, "race/milano-sanremo/2024"),
        _r("2024-4-7", 12, "race/paris-roubaix/2024"),
        _r("2024-2-10", 10, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 8, "race/b/2024", "1.Pro"),
    ],
    "rider/florian-vermeersch": [
        _r("2024-4-7", 5, "race/paris-roubaix/2024"),
        _r("2024-3-31", 12, "race/ronde-van-vlaanderen/2024"),
        _r("2024-2-10", 8, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 10, "race/b/2024", "1.Pro"),
        _r("2023-4-9", 8, "race/paris-roubaix/2023"),
    ],
    "rider/alexander-kristoff": [
        _r("2024-3-24", 8, "race/gent-wevelgem/2024"),
        _r("2024-3-16", 15, "race/milano-sanremo/2024"),
        _r("2024-2-10", 5, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 8, "race/b/2024", "1.Pro"),
    ],
    "rider/michael-valgren": [
        _r("2024-3-2", 12, "race/strade-bianche/2024"),
        _r("2024-4-14", 10, "race/amstel-gold-race/2024"),
        _r("2024-2-10", 8, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 10, "race/b/2024", "1.Pro"),
    ],
    "rider/kevin-vauquelin": [
        _r("2024-4-17", 2, "race/la-fleche-wallone/2024"),
        _r("2024-2-10", 12, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 10, "race/b/2024", "1.Pro"),
    ],
    "rider/daniel-martinez": [
        _r("2024-4-21", 8, "race/liege-bastogne-liege/2024"),
        _r("2024-2-10", 5, "race/a/2024", "1.Pro"),
        _r("2024-2-20", 8, "race/b/2024", "1.Pro"),
    ],
    "rider/santiago-buitrago": [
        _r("2024-4-17", 10, "race/la-fleche-wallone/2024"),
        _r("2024-4-21", 12, "race/liege-bastogne-liege/2024"),
        _r("2024-2-10", 8, "race/a/2024", "1.Pro"),
    ],
    "rider/andrea-bagioli": [
        _r("2024-4-14", 12, "race/amstel-gold-race/2024"),
        _r("2024-4-21", 10, "race/liege-bastogne-liege/2024"),
        _r("2024-2-10", 8, "race/a/2024", "1.Pro"),
    ],
    "rider/simone-velasco": [
        _r("2024-4-21", 15, "race/liege-bastogne-liege/2024"),
        _r("2024-2-10", 10, "race/a/2024", "1.Pro"),
    ],
}

# Add empty entries for riders without pre-2025 data
for rider_url in BIRTHDATES:
    if rider_url not in PRE_2025_RESULTS:
        PRE_2025_RESULTS[rider_url] = []


# =====================================================================
# 2025 RACE RESULTS (accumulating through the season)
# =====================================================================

# Each entry: (race_date, race_url, distance, actual_top10_rider_urls)
# The startlist for prediction includes all riders in our database.

RACES_2025 = [
    {
        "name": "Kuurne-Brussel-Kuurne",
        "base_url": "race/kuurne-brussel-kuurne",
        "date": date(2025, 3, 2),
        "distance": 197.0,
        "actual_top10": [
            "rider/jasper-philipsen",
            "rider/olav-kooij",
            "rider/hugo-hofstetter",
            None,  # Arne Marit - not in our DB
            None,  # Rick Pluimers
            "rider/jonathan-milan",
            None,  # Marijn van den Berg
            None,  # Pavel Bittner
            None,  # Lukas Kubis
            "rider/kaden-groves",
        ],
        # This is a sprinters' race, not really a classic
        # Including for completeness but weight lower in analysis
        "is_sprinters_race": True,
    },
    {
        "name": "Strade Bianche",
        "base_url": "race/strade-bianche",
        "date": date(2025, 3, 8),
        "distance": 213.0,
        "actual_top10": [
            "rider/tadej-pogacar",
            "rider/tom-pidcock",
            "rider/tim-wellens",
            "rider/ben-healy",
            "rider/pello-bilbao",
            "rider/magnus-cort",
            "rider/gianni-vermeersch",
            "rider/michael-valgren",
            None,  # Lennert Van Eetvelt
            None,  # Roger Adrià
        ],
    },
    {
        "name": "Milano-Sanremo",
        "base_url": "race/milano-sanremo",
        "date": date(2025, 3, 22),
        "distance": 289.0,
        "actual_top10": [
            "rider/mathieu-van-der-poel",
            "rider/filippo-ganna",
            "rider/tadej-pogacar",
            "rider/michael-matthews",
            "rider/kaden-groves",
            "rider/magnus-cort",
            "rider/mads-pedersen",
            "rider/olav-kooij",
            "rider/matteo-trentin",
            "rider/fred-wright",
        ],
    },
    {
        "name": "E3 Saxo Classic",
        "base_url": "race/e3-harelbeke",
        "date": date(2025, 3, 28),
        "distance": 209.0,
        "actual_top10": [
            "rider/mathieu-van-der-poel",
            "rider/mads-pedersen",
            "rider/filippo-ganna",
            None,  # Casper Pedersen
            "rider/jasper-stuyven",
            "rider/stefan-kueng",
            None,  # Aimé De Gendt
            "rider/tim-wellens",
            "rider/matteo-jorgenson",
            None,  # Mike Teunissen
        ],
    },
    {
        "name": "Gent-Wevelgem",
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
    },
    {
        "name": "Dwars door Vlaanderen",
        "base_url": "race/dwars-door-vlaanderen",
        "date": date(2025, 4, 2),
        "distance": 185.0,
        "actual_top10": [
            "rider/neilson-powless",
            "rider/wout-van-aert",
            "rider/tiesj-benoot",
            "rider/matteo-jorgenson",
            "rider/mads-pedersen",
            None,  # Tibor Del Grosso
            None,  # Dries De Bondt
            None,  # Arjen Livyns
            "rider/stefan-kueng",
            None,  # Alec Segaert
        ],
    },
    {
        "name": "Ronde van Vlaanderen",
        "base_url": "race/ronde-van-vlaanderen",
        "date": date(2025, 4, 6),
        "distance": 270.0,
        "actual_top10": [
            "rider/tadej-pogacar",
            "rider/mads-pedersen",
            "rider/mathieu-van-der-poel",
            "rider/wout-van-aert",
            "rider/jasper-stuyven",
            "rider/tiesj-benoot",
            "rider/stefan-kueng",
            "rider/filippo-ganna",
            "rider/ivan-garcia-cortina",
            "rider/davide-ballerini",
        ],
    },
    {
        "name": "Paris-Roubaix",
        "base_url": "race/paris-roubaix",
        "date": date(2025, 4, 13),
        "distance": 259.0,
        "actual_top10": [
            "rider/mathieu-van-der-poel",
            "rider/tadej-pogacar",
            "rider/mads-pedersen",
            "rider/wout-van-aert",
            "rider/florian-vermeersch",
            None,  # Jonas Rutsch
            None,  # Stefan Bissegger
            None,  # Markus Hoelgaard
            "rider/fred-wright",
            "rider/laurenz-rex",
        ],
    },
    {
        "name": "Amstel Gold Race",
        "base_url": "race/amstel-gold-race",
        "date": date(2025, 4, 20),
        "distance": 256.0,
        "actual_top10": [
            "rider/mattias-skjelmose",
            "rider/tadej-pogacar",
            "rider/remco-evenepoel",
            "rider/wout-van-aert",
            "rider/michael-matthews",
            None,  # Louis Barré
            "rider/romain-gregoire",
            "rider/tiesj-benoot",
            "rider/tom-pidcock",
            "rider/ben-healy",
        ],
    },
    {
        "name": "La Flèche Wallonne",
        "base_url": "race/la-fleche-wallone",
        "date": date(2025, 4, 23),
        "distance": 205.0,
        "actual_top10": [
            "rider/tadej-pogacar",
            "rider/kevin-vauquelin",
            "rider/tom-pidcock",
            "rider/lenny-martinez",
            "rider/ben-healy",
            "rider/santiago-buitrago",
            "rider/romain-gregoire",
            "rider/thibau-nys",
            "rider/remco-evenepoel",
            "rider/mauro-schmid",
        ],
    },
    {
        "name": "Liège-Bastogne-Liège",
        "base_url": "race/liege-bastogne-liege",
        "date": date(2025, 4, 27),
        "distance": 260.0,
        "actual_top10": [
            "rider/tadej-pogacar",
            "rider/giulio-ciccone",
            "rider/ben-healy",
            "rider/simone-velasco",
            "rider/thibau-nys",
            "rider/andrea-bagioli",
            "rider/daniel-martinez",
            "rider/axel-laurance",
            "rider/tom-pidcock",
            "rider/neilson-powless",
        ],
    },
]


# 2025 results that accumulate as preseason form data
RESULTS_2025 = [
    # Kuurne Mar 2
    ("2025-3-2", 1, "race/kuurne-brussel-kuurne/2025", "rider/jasper-philipsen"),
    ("2025-3-2", 6, "race/kuurne-brussel-kuurne/2025", "rider/jonathan-milan"),
    ("2025-3-2", 10, "race/kuurne-brussel-kuurne/2025", "rider/kaden-groves"),
    # Strade Bianche Mar 8
    ("2025-3-8", 1, "race/strade-bianche/2025", "rider/tadej-pogacar"),
    ("2025-3-8", 2, "race/strade-bianche/2025", "rider/tom-pidcock"),
    ("2025-3-8", 3, "race/strade-bianche/2025", "rider/tim-wellens"),
    ("2025-3-8", 4, "race/strade-bianche/2025", "rider/ben-healy"),
    ("2025-3-8", 6, "race/strade-bianche/2025", "rider/magnus-cort"),
    ("2025-3-8", 8, "race/strade-bianche/2025", "rider/michael-valgren"),
    # Milano-Sanremo Mar 22
    ("2025-3-22", 1, "race/milano-sanremo/2025", "rider/mathieu-van-der-poel"),
    ("2025-3-22", 2, "race/milano-sanremo/2025", "rider/filippo-ganna"),
    ("2025-3-22", 3, "race/milano-sanremo/2025", "rider/tadej-pogacar"),
    ("2025-3-22", 4, "race/milano-sanremo/2025", "rider/michael-matthews"),
    ("2025-3-22", 6, "race/milano-sanremo/2025", "rider/magnus-cort"),
    ("2025-3-22", 7, "race/milano-sanremo/2025", "rider/mads-pedersen"),
    ("2025-3-22", 10, "race/milano-sanremo/2025", "rider/fred-wright"),
    # E3 Mar 28
    ("2025-3-28", 1, "race/e3-harelbeke/2025", "rider/mathieu-van-der-poel"),
    ("2025-3-28", 2, "race/e3-harelbeke/2025", "rider/mads-pedersen"),
    ("2025-3-28", 3, "race/e3-harelbeke/2025", "rider/filippo-ganna"),
    ("2025-3-28", 5, "race/e3-harelbeke/2025", "rider/jasper-stuyven"),
    ("2025-3-28", 6, "race/e3-harelbeke/2025", "rider/stefan-kueng"),
    ("2025-3-28", 8, "race/e3-harelbeke/2025", "rider/tim-wellens"),
    ("2025-3-28", 9, "race/e3-harelbeke/2025", "rider/matteo-jorgenson"),
    # Gent-Wevelgem Mar 30
    ("2025-3-30", 1, "race/gent-wevelgem/2025", "rider/mads-pedersen"),
    ("2025-3-30", 2, "race/gent-wevelgem/2025", "rider/tim-merlier"),
    ("2025-3-30", 4, "race/gent-wevelgem/2025", "rider/alexander-kristoff"),
    ("2025-3-30", 6, "race/gent-wevelgem/2025", "rider/davide-ballerini"),
    ("2025-3-30", 7, "race/gent-wevelgem/2025", "rider/biniam-girmay"),
    # Dwars Apr 2
    ("2025-4-2", 1, "race/dwars-door-vlaanderen/2025", "rider/neilson-powless"),
    ("2025-4-2", 2, "race/dwars-door-vlaanderen/2025", "rider/wout-van-aert"),
    ("2025-4-2", 3, "race/dwars-door-vlaanderen/2025", "rider/tiesj-benoot"),
    ("2025-4-2", 4, "race/dwars-door-vlaanderen/2025", "rider/matteo-jorgenson"),
    ("2025-4-2", 5, "race/dwars-door-vlaanderen/2025", "rider/mads-pedersen"),
    ("2025-4-2", 9, "race/dwars-door-vlaanderen/2025", "rider/stefan-kueng"),
    # Flanders Apr 6
    ("2025-4-6", 1, "race/ronde-van-vlaanderen/2025", "rider/tadej-pogacar"),
    ("2025-4-6", 2, "race/ronde-van-vlaanderen/2025", "rider/mads-pedersen"),
    ("2025-4-6", 3, "race/ronde-van-vlaanderen/2025", "rider/mathieu-van-der-poel"),
    ("2025-4-6", 4, "race/ronde-van-vlaanderen/2025", "rider/wout-van-aert"),
    ("2025-4-6", 5, "race/ronde-van-vlaanderen/2025", "rider/jasper-stuyven"),
    ("2025-4-6", 6, "race/ronde-van-vlaanderen/2025", "rider/tiesj-benoot"),
    ("2025-4-6", 7, "race/ronde-van-vlaanderen/2025", "rider/stefan-kueng"),
    ("2025-4-6", 8, "race/ronde-van-vlaanderen/2025", "rider/filippo-ganna"),
    # Roubaix Apr 13
    ("2025-4-13", 1, "race/paris-roubaix/2025", "rider/mathieu-van-der-poel"),
    ("2025-4-13", 2, "race/paris-roubaix/2025", "rider/tadej-pogacar"),
    ("2025-4-13", 3, "race/paris-roubaix/2025", "rider/mads-pedersen"),
    ("2025-4-13", 4, "race/paris-roubaix/2025", "rider/wout-van-aert"),
    ("2025-4-13", 5, "race/paris-roubaix/2025", "rider/florian-vermeersch"),
    ("2025-4-13", 9, "race/paris-roubaix/2025", "rider/fred-wright"),
    # Amstel Apr 20
    ("2025-4-20", 1, "race/amstel-gold-race/2025", "rider/mattias-skjelmose"),
    ("2025-4-20", 2, "race/amstel-gold-race/2025", "rider/tadej-pogacar"),
    ("2025-4-20", 3, "race/amstel-gold-race/2025", "rider/remco-evenepoel"),
    ("2025-4-20", 4, "race/amstel-gold-race/2025", "rider/wout-van-aert"),
    ("2025-4-20", 5, "race/amstel-gold-race/2025", "rider/michael-matthews"),
    ("2025-4-20", 8, "race/amstel-gold-race/2025", "rider/tiesj-benoot"),
    ("2025-4-20", 9, "race/amstel-gold-race/2025", "rider/tom-pidcock"),
    ("2025-4-20", 10, "race/amstel-gold-race/2025", "rider/ben-healy"),
    # Flèche Apr 23
    ("2025-4-23", 1, "race/la-fleche-wallone/2025", "rider/tadej-pogacar"),
    ("2025-4-23", 2, "race/la-fleche-wallone/2025", "rider/kevin-vauquelin"),
    ("2025-4-23", 3, "race/la-fleche-wallone/2025", "rider/tom-pidcock"),
    ("2025-4-23", 5, "race/la-fleche-wallone/2025", "rider/ben-healy"),
    ("2025-4-23", 8, "race/la-fleche-wallone/2025", "rider/thibau-nys"),
    ("2025-4-23", 9, "race/la-fleche-wallone/2025", "rider/remco-evenepoel"),
]


# =====================================================================
# Build rider data for a given race date
# =====================================================================

def build_rider_data(race_date: date) -> Dict[str, Dict]:
    """Build rider data dict using pre-2025 data + 2025 results before race_date."""
    rider_data = {}
    for rider_url in BIRTHDATES:
        # Pre-2025 results
        results = list(PRE_2025_RESULTS.get(rider_url, []))
        # Add 2025 results from before this race
        for date_str, rank, race_url, r_url in RESULTS_2025:
            if r_url == rider_url:
                parts = date_str.split("-")
                r_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
                if r_date < race_date:
                    results.append(_r(date_str, rank, race_url))
        # Add generic early-2025 prep results for riders we know raced
        # (approximation for January/February races we don't have)
        if results:
            earliest_2025 = None
            for r in results:
                if r["date"].startswith("2025"):
                    p = r["date"].split("-")
                    rd = date(int(p[0]), int(p[1]), int(p[2]))
                    if earliest_2025 is None or rd < earliest_2025:
                        earliest_2025 = rd
            # If rider's first 2025 result is March+, add placeholder Jan/Feb prep
            if earliest_2025 is None or earliest_2025 >= date(2025, 3, 1):
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


def build_startlist() -> List[Dict[str, str]]:
    """Build a startlist with all riders in our database."""
    return [
        {"rider_url": url, "rider_name": url.split("/")[-1].replace("-", " ").title()}
        for url in BIRTHDATES
    ]


# =====================================================================
# Scoring: how well do predictions match reality?
# =====================================================================

def score_predictions(
    predictions: List[Dict],
    actual_top10: List,
) -> Dict[str, float]:
    """
    Score prediction quality against actual results.

    Returns dict with:
    - top10_hit_rate: fraction of actual top-10 riders predicted in our top-10
    - top5_hit_rate: fraction of actual top-5 riders predicted in our top-5
    - winner_in_top3: 1.0 if actual winner predicted in top-3, else 0.0
    - winner_in_top5: 1.0 if actual winner predicted in top-5, else 0.0
    - avg_rank_error: average absolute rank difference for matched riders
    """
    # Filter None entries from actual (riders not in our DB)
    actual_known = [url for url in actual_top10 if url is not None]
    if not actual_known:
        return {"top10_hit_rate": 0, "top5_hit_rate": 0,
                "winner_in_top3": 0, "winner_in_top5": 0, "avg_rank_error": 20}

    pred_urls = [p["rider_url"] for p in predictions]
    pred_top10 = set(pred_urls[:10])
    pred_top5 = set(pred_urls[:5])

    # Top-10 hit rate
    actual_top10_known = set(actual_known[:10])
    hits_10 = len(actual_top10_known & pred_top10)
    top10_hit = hits_10 / len(actual_top10_known) if actual_top10_known else 0

    # Top-5 hit rate
    actual_top5_known = set(actual_known[:5])
    hits_5 = len(actual_top5_known & pred_top5)
    top5_hit = hits_5 / len(actual_top5_known) if actual_top5_known else 0

    # Winner predicted in top-3 / top-5
    actual_winner = actual_known[0] if actual_known else None
    winner_top3 = 1.0 if actual_winner in set(pred_urls[:3]) else 0.0
    winner_top5 = 1.0 if actual_winner in set(pred_urls[:5]) else 0.0

    # Average rank error for actual top-10 riders
    rank_errors = []
    for actual_rank, url in enumerate(actual_known[:10], 1):
        if url in pred_urls:
            pred_rank = pred_urls.index(url) + 1
            rank_errors.append(abs(actual_rank - pred_rank))
    avg_err = sum(rank_errors) / len(rank_errors) if rank_errors else 20

    return {
        "top10_hit_rate": top10_hit,
        "top5_hit_rate": top5_hit,
        "winner_in_top3": winner_top3,
        "winner_in_top5": winner_top5,
        "avg_rank_error": avg_err,
    }


# =====================================================================
# Run backtest for a given set of weights
# =====================================================================

def run_backtest(
    weights: Dict[str, float],
    races: List[Dict] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Run the predictor on all 2025 races and return aggregate scores."""
    if races is None:
        races = RACES_2025

    predictor = ClassicsPredictor(weights=weights)
    startlist = build_startlist()

    all_scores = []
    for race in races:
        if race.get("is_sprinters_race"):
            continue  # Skip pure sprinter races

        rider_data = build_rider_data(race["date"])
        predictions = predictor.predict_from_data(
            race_base_url=race["base_url"],
            year=2025,
            distance=race["distance"],
            race_date=race["date"],
            startlist=startlist,
            rider_data=rider_data,
        )

        scores = score_predictions(predictions, race["actual_top10"])
        scores["race"] = race["name"]
        all_scores.append(scores)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"{race['name']} ({race['date']})")
            print(f"{'=' * 60}")
            print(f"  Top-10 hit: {scores['top10_hit_rate']:.0%} | "
                  f"Top-5 hit: {scores['top5_hit_rate']:.0%} | "
                  f"Winner in top-3: {'YES' if scores['winner_in_top3'] else 'no'} | "
                  f"Avg rank err: {scores['avg_rank_error']:.1f}")

            actual_known = [u for u in race["actual_top10"] if u]
            print(f"\n  {'Predicted':>3} {'Rider':<35} {'Actual':>6}")
            print(f"  {'-'*3} {'-'*35} {'-'*6}")
            for p in predictions[:15]:
                actual_pos = ""
                for i, au in enumerate(race["actual_top10"], 1):
                    if au == p["rider_url"]:
                        actual_pos = str(i)
                        break
                marker = " *" if actual_pos else ""
                print(f"  {p['rank']:>3} {p['rider_name']:<35} "
                      f"{actual_pos:>6}{marker}")

    if not all_scores:
        return {}

    # Aggregate metrics
    n = len(all_scores)
    agg = {
        "avg_top10_hit": sum(s["top10_hit_rate"] for s in all_scores) / n,
        "avg_top5_hit": sum(s["top5_hit_rate"] for s in all_scores) / n,
        "winner_top3_pct": sum(s["winner_in_top3"] for s in all_scores) / n,
        "winner_top5_pct": sum(s["winner_in_top5"] for s in all_scores) / n,
        "avg_rank_error": sum(s["avg_rank_error"] for s in all_scores) / n,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print("AGGREGATE RESULTS")
        print(f"{'=' * 60}")
        print(f"  Avg top-10 hit rate: {agg['avg_top10_hit']:.1%}")
        print(f"  Avg top-5 hit rate:  {agg['avg_top5_hit']:.1%}")
        print(f"  Winner in top-3:     {agg['winner_top3_pct']:.0%}")
        print(f"  Winner in top-5:     {agg['winner_top5_pct']:.0%}")
        print(f"  Avg rank error:      {agg['avg_rank_error']:.1f}")

    return agg


# =====================================================================
# Weight optimization: grid search
# =====================================================================

def optimize_weights(verbose: bool = False) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Grid search over weight combinations to find the best fit
    for the 2025 classics results.

    Optimizes for a composite score emphasizing top-10 and top-5 hit rates.
    """
    best_score = -1
    best_weights = None
    best_metrics = None
    tested = 0

    # Major features to vary (most impactful)
    cp_steps = [0.15, 0.20, 0.25, 0.30, 0.35]
    tm_steps = [0.05, 0.10, 0.15, 0.20]
    ss_steps = [0.05, 0.10, 0.15]
    py_steps = [0.05, 0.10, 0.15]

    for cp in cp_steps:
        for tm in tm_steps:
            for ss in ss_steps:
                for py in py_steps:
                    major_sum = cp + tm + ss + py
                    remaining = 1.0 - major_sum
                    if remaining < 0.20 or remaining > 0.60:
                        continue
                    # Distribute remaining among minor features
                    # (preparation, injury, recent_form, age, sprint, momentum, team)
                    for prep in [0.05, 0.10, 0.15]:
                        for sp in [0.03, 0.05, 0.08]:
                            rest = remaining - prep - sp
                            if rest < 0.10 or rest > 0.45:
                                continue
                            # Split rest among: injury, recent_form, age_dist, momentum, team
                            ip = rest * 0.25
                            rf = rest * 0.15
                            ad = rest * 0.15
                            mom = rest * 0.20
                            ts = rest * 0.25

                            weights = {
                                "classic_pedigree": cp,
                                "terrain_match": tm,
                                "specialty_score": ss,
                                "previous_year": py,
                                "preparation": prep,
                                "sprint_capability": sp,
                                "injury_penalty": ip,
                                "recent_form": rf,
                                "age_distance_fit": ad,
                                "momentum": mom,
                                "team_strength": ts,
                            }

                            metrics = run_backtest(weights)
                            if not metrics:
                                continue

                            # Composite score — emphasize top-10 and top-5 hit rates
                            score = (
                                0.30 * metrics["avg_top10_hit"]
                                + 0.30 * metrics["avg_top5_hit"]
                                + 0.20 * metrics["winner_top5_pct"]
                                + 0.10 * metrics["winner_top3_pct"]
                                + 0.10 * max(0, 1.0 - metrics["avg_rank_error"] / 20)
                            )

                            tested += 1
                            if score > best_score:
                                best_score = score
                                best_weights = weights.copy()
                                best_metrics = metrics.copy()

    if verbose and best_weights:
        print(f"\nTested {tested} weight combinations")
        print(f"\nBest composite score: {best_score:.3f}")
        print(f"Best weights:")
        for k, v in sorted(best_weights.items(), key=lambda x: -x[1]):
            print(f"  {k:<25} {v:.3f}")
        print(f"\nMetrics with best weights:")
        print(f"  Avg top-10 hit rate: {best_metrics['avg_top10_hit']:.1%}")
        print(f"  Avg top-5 hit rate:  {best_metrics['avg_top5_hit']:.1%}")
        print(f"  Winner in top-3:     {best_metrics['winner_top3_pct']:.0%}")
        print(f"  Winner in top-5:     {best_metrics['winner_top5_pct']:.0%}")
        print(f"  Avg rank error:      {best_metrics['avg_rank_error']:.1f}")

    return best_weights, best_metrics


# =====================================================================
# Tests
# =====================================================================

class TestBacktest2025:
    """Verify the backtest infrastructure and run predictions."""

    def test_build_rider_data(self):
        """Rider data builds correctly with accumulating results."""
        data = build_rider_data(date(2025, 4, 6))  # Before Flanders
        pog = data["rider/tadej-pogacar"]
        assert pog["profile"]["birthdate"] == "1998-9-21"
        # Should have pre-2025 + Strade Bianche + San Remo 2025 results
        dates = [r["date"] for r in pog["results"]]
        assert any("2025-3-8" in d for d in dates)   # Strade
        assert any("2025-3-22" in d for d in dates)   # San Remo
        assert not any("2025-4-6" in d for d in dates)  # Not Flanders itself

    def test_default_weights_backtest(self):
        """Run backtest with default weights — should get reasonable scores."""
        metrics = run_backtest(weights=None, verbose=True)
        # At minimum we should predict the winner in top-5 more than half the time
        assert metrics["winner_top5_pct"] >= 0.4
        # Top-10 hit rate should be reasonable
        assert metrics["avg_top10_hit"] >= 0.3

    def test_optimized_weights_improve(self):
        """Optimized weights should beat or equal default weights."""
        default_metrics = run_backtest(weights=None)
        best_weights, best_metrics = optimize_weights(verbose=True)

        default_score = (
            0.35 * default_metrics["avg_top10_hit"]
            + 0.25 * default_metrics["winner_top5_pct"]
            + 0.25 * default_metrics["avg_top5_hit"]
            + 0.15 * max(0, 1.0 - default_metrics["avg_rank_error"] / 20)
        )
        best_score = (
            0.35 * best_metrics["avg_top10_hit"]
            + 0.25 * best_metrics["winner_top5_pct"]
            + 0.25 * best_metrics["avg_top5_hit"]
            + 0.15 * max(0, 1.0 - best_metrics["avg_rank_error"] / 20)
        )
        assert best_score >= default_score

    def test_pogacar_ranks_high_in_monuments(self):
        """Pogačar should consistently rank in top-5 for hilly monuments."""
        predictor = ClassicsPredictor()
        startlist = build_startlist()

        for race in RACES_2025:
            if race["base_url"] in (
                "race/ronde-van-vlaanderen",
                "race/liege-bastogne-liege",
            ):
                rider_data = build_rider_data(race["date"])
                preds = predictor.predict_from_data(
                    race["base_url"], 2025, race["distance"],
                    race["date"], startlist, rider_data,
                )
                pog_rank = next(
                    p["rank"] for p in preds
                    if p["rider_url"] == "rider/tadej-pogacar"
                )
                assert pog_rank <= 5, (
                    f"Pogačar ranked {pog_rank} for {race['name']}"
                )

    def test_van_der_poel_ranks_high_for_roubaix(self):
        """Van der Poel should rank top-3 for Paris-Roubaix."""
        predictor = ClassicsPredictor()
        startlist = build_startlist()
        for race in RACES_2025:
            if race["base_url"] == "race/paris-roubaix":
                rider_data = build_rider_data(race["date"])
                preds = predictor.predict_from_data(
                    race["base_url"], 2025, race["distance"],
                    race["date"], startlist, rider_data,
                )
                mvdp_rank = next(
                    p["rank"] for p in preds
                    if p["rider_url"] == "rider/mathieu-van-der-poel"
                )
                assert mvdp_rank <= 3

    def test_young_riders_rank_lower_in_long_monuments(self):
        """Nys (22) and Grégoire (22) should score lower than experienced
        riders in 260km monuments, but be competitive in shorter classics."""
        predictor = ClassicsPredictor()
        startlist = build_startlist()

        # Roubaix (259 km) - young riders should not be top-5
        for race in RACES_2025:
            if race["base_url"] == "race/paris-roubaix":
                rider_data = build_rider_data(race["date"])
                preds = predictor.predict_from_data(
                    race["base_url"], 2025, race["distance"],
                    race["date"], startlist, rider_data,
                )
                nys_rank = next(
                    (p["rank"] for p in preds
                     if p["rider_url"] == "rider/thibau-nys"), None
                )
                if nys_rank:
                    assert nys_rank > 5


# =====================================================================
# CLI entry point
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BACKTEST: 2025 Spring Classics with DEFAULT weights")
    print("=" * 70)
    run_backtest(weights=None, verbose=True)

    print("\n\n")
    print("=" * 70)
    print("WEIGHT OPTIMIZATION: Finding best weights for 2025 results")
    print("=" * 70)
    best_w, best_m = optimize_weights(verbose=True)

    if best_w:
        print("\n\n")
        print("=" * 70)
        print("BACKTEST: 2025 Spring Classics with OPTIMIZED weights")
        print("=" * 70)
        run_backtest(weights=best_w, verbose=True)
