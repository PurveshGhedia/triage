"""Prompts for the Gemini-vision perception tools.

Bump PROMPT_VERSION whenever any prompt changes: it is part of the cache key, so old
cached answers are never served for a new prompt.
"""
PROMPT_VERSION = "v1"

DAMAGE_LABELS = ["no-damage", "minor-damage", "major-damage", "destroyed"]
ACCESS_LABELS = ["access-clear", "access-degraded", "access-blocked"]

_SCALE = """Use the xBD / Joint Damage Scale for buildings:
- no-damage: no sign of water, fire or structural damage.
- minor-damage: partial roof/wall damage, surrounding water or debris, but the structure is intact.
- major-damage: partial collapse, wall or roof failure, or the building is surrounded or filled by water.
- destroyed: completely collapsed, scorched to the foundation, or fully submerged / no longer recognisable."""

_IMAGES = """You are given two satellite images of the SAME ~400 m square area, in this order:
1. BEFORE the disaster. 2. AFTER the disaster.
(If only one image is provided, it is the AFTER image.)"""

DAMAGE_PROMPT = f"""{_IMAGES}

Assess how badly the buildings in this area were damaged by the disaster, comparing before and after.
{_SCALE}

Judge the tile as a whole: give the damage level that best describes the typical-to-worst affected
buildings, not a single outlier. Compare before vs after; ignore differences caused only by lighting,
season or camera angle.

Reply as JSON with:
- label: one of {DAMAGE_LABELS}
- confidence: your confidence from 0.0 to 1.0 (be honest: use below 0.6 if the imagery is unclear)
- evidence: one or two short sentences naming what you actually see (e.g. flood water, collapsed roofs)."""

SECOND_PASS_PROMPT = f"""{_IMAGES}

This is a SECOND OPINION. A first review of this area was uncertain or disagreed with neighbouring areas,
so be systematic: scan the area quadrant by quadrant, compare individual buildings before and after,
and decide how many are affected and how severely.
{_SCALE}

Give the level that best describes the area as a whole after that building-by-building review.

Reply as JSON with:
- label: one of {DAMAGE_LABELS}
- confidence: 0.0 to 1.0 (honest; below 0.6 if unclear)
- evidence: one or two short sentences on what the building-level comparison showed."""

ACCESS_PROMPT = f"""{_IMAGES}

Judge ROAD ACCESS for emergency vehicles in the AFTER image, compared with BEFORE.
- access-clear: roads and bridges are visibly passable.
- access-degraded: some roads are flooded, blocked by debris, or partly unusable, but a route probably exists.
- access-blocked: the main roads/bridges look impassable (deep flooding, collapsed structures, large debris).
If there are no visible roads, answer access-clear with low confidence.

Reply as JSON with:
- label: one of {ACCESS_LABELS}
- confidence: 0.0 to 1.0 (honest)
- evidence: one short sentence naming what you see on the roads."""
