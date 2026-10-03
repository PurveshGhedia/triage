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


# --- variant "fractions": estimate the share of buildings at each level -----------------------------
# xBD's ground truth is the damage level of EVERY building, so a tile's truth is a mixture. Asking for
# the mixture (and averaging it) matches that, instead of asking for one "typical-to-worst" level.

_FRACTIONS_REPLY = """Reply as JSON with:
- no_damage, minor_damage, major_damage, destroyed: the share of buildings at each level, each between
  0 and 1, adding up to 1.0
- confidence: how sure you are of those shares, 0.0 to 1.0. Use 0.9 or more ONLY when buildings are clearly
  visible in both images and the changes are unambiguous; 0.6-0.8 if the imagery is blurry, buildings are
  small or hidden, or the two images do not line up; below 0.6 if you cannot really tell. If you can see
  no buildings at all, set no_damage to 1.0 and confidence below 0.4.
- evidence: one or two short sentences naming what you actually see."""

DAMAGE_FRACTIONS_PROMPT = f"""{_IMAGES}

Estimate what SHARE of the buildings in this area fall into each damage level, comparing before and after.
{_SCALE}

Count only buildings that exist in the BEFORE image. Most areas contain many undamaged buildings: do not let a
few damaged ones, or water standing in empty fields, parks or roads away from buildings, dominate your estimate.
Ignore differences caused only by lighting, season or camera angle.

{_FRACTIONS_REPLY}"""

SECOND_PASS_FRACTIONS_PROMPT = f"""{_IMAGES}

This is a SECOND OPINION: a first review was uncertain or disagreed with neighbouring areas. Be systematic:
scan the area quadrant by quadrant, compare individual buildings before and after, and tally how many are
affected and how severely, then convert the tally into shares.
{_SCALE}

Count only buildings that exist in the BEFORE image, and do not let water in empty land dominate.

{_FRACTIONS_REPLY}"""
