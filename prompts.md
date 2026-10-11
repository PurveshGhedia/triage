# Prompts sent by the agents

Images are attached to the vision prompts: the BEFORE image first, then the AFTER image, downscaled to at most 768 px. Replies are forced into a JSON schema (an enumerated label or four damage shares, a confidence, and an evidence sentence). Prompt version: v1.

## Agent 1, vision tools (Gemini 2.5 Flash on Vertex AI)

### Damage assessment: variant 'classes' (tool name: classifier)

You are given two satellite images of the SAME ~400 m square area, in this order:
1. BEFORE the disaster. 2. AFTER the disaster.
(If only one image is provided, it is the AFTER image.)

Assess how badly the buildings in this area were damaged by the disaster, comparing before and after.
Use the xBD / Joint Damage Scale for buildings:
- no-damage: no sign of water, fire or structural damage.
- minor-damage: partial roof/wall damage, surrounding water or debris, but the structure is intact.
- major-damage: partial collapse, wall or roof failure, or the building is surrounded or filled by water.
- destroyed: completely collapsed, scorched to the foundation, or fully submerged / no longer recognisable.

Judge the tile as a whole: give the damage level that best describes the typical-to-worst affected
buildings, not a single outlier. Compare before vs after; ignore differences caused only by lighting,
season or camera angle.

Reply as JSON with:
- label: one of ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
- confidence: your confidence from 0.0 to 1.0 (be honest: use below 0.6 if the imagery is unclear)
- evidence: one or two short sentences naming what you actually see (e.g. flood water, collapsed roofs).

### Damage assessment: variant 'fractions' (tool name: classifier)

You are given two satellite images of the SAME ~400 m square area, in this order:
1. BEFORE the disaster. 2. AFTER the disaster.
(If only one image is provided, it is the AFTER image.)

Estimate what SHARE of the buildings in this area fall into each damage level, comparing before and after.
Use the xBD / Joint Damage Scale for buildings:
- no-damage: no sign of water, fire or structural damage.
- minor-damage: partial roof/wall damage, surrounding water or debris, but the structure is intact.
- major-damage: partial collapse, wall or roof failure, or the building is surrounded or filled by water.
- destroyed: completely collapsed, scorched to the foundation, or fully submerged / no longer recognisable.

Count only buildings that exist in the BEFORE image. Most areas contain many undamaged buildings: do not let a
few damaged ones, or water standing in empty fields, parks or roads away from buildings, dominate your estimate.
Ignore differences caused only by lighting, season or camera angle.

Reply as JSON with:
- no_damage, minor_damage, major_damage, destroyed: the share of buildings at each level, each between
  0 and 1, adding up to 1.0
- confidence: how sure you are of those shares, 0.0 to 1.0. Use 0.9 or more ONLY when buildings are clearly
  visible in both images and the changes are unambiguous; 0.6-0.8 if the imagery is blurry, buildings are
  small or hidden, or the two images do not line up; below 0.6 if you cannot really tell. If you can see
  no buildings at all, set no_damage to 1.0 and confidence below 0.4.
- evidence: one or two short sentences naming what you actually see.

### Second opinion: variant 'classes' (tool name: sam)

You are given two satellite images of the SAME ~400 m square area, in this order:
1. BEFORE the disaster. 2. AFTER the disaster.
(If only one image is provided, it is the AFTER image.)

This is a SECOND OPINION. A first review of this area was uncertain or disagreed with neighbouring areas,
so be systematic: scan the area quadrant by quadrant, compare individual buildings before and after,
and decide how many are affected and how severely.
Use the xBD / Joint Damage Scale for buildings:
- no-damage: no sign of water, fire or structural damage.
- minor-damage: partial roof/wall damage, surrounding water or debris, but the structure is intact.
- major-damage: partial collapse, wall or roof failure, or the building is surrounded or filled by water.
- destroyed: completely collapsed, scorched to the foundation, or fully submerged / no longer recognisable.

Give the level that best describes the area as a whole after that building-by-building review.

Reply as JSON with:
- label: one of ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
- confidence: 0.0 to 1.0 (honest; below 0.6 if unclear)
- evidence: one or two short sentences on what the building-level comparison showed.

### Second opinion: variant 'fractions' (tool name: sam)

You are given two satellite images of the SAME ~400 m square area, in this order:
1. BEFORE the disaster. 2. AFTER the disaster.
(If only one image is provided, it is the AFTER image.)

This is a SECOND OPINION: a first review was uncertain or disagreed with neighbouring areas. Be systematic:
scan the area quadrant by quadrant, compare individual buildings before and after, and tally how many are
affected and how severely, then convert the tally into shares.
Use the xBD / Joint Damage Scale for buildings:
- no-damage: no sign of water, fire or structural damage.
- minor-damage: partial roof/wall damage, surrounding water or debris, but the structure is intact.
- major-damage: partial collapse, wall or roof failure, or the building is surrounded or filled by water.
- destroyed: completely collapsed, scorched to the foundation, or fully submerged / no longer recognisable.

Count only buildings that exist in the BEFORE image, and do not let water in empty land dominate.

Reply as JSON with:
- no_damage, minor_damage, major_damage, destroyed: the share of buildings at each level, each between
  0 and 1, adding up to 1.0
- confidence: how sure you are of those shares, 0.0 to 1.0. Use 0.9 or more ONLY when buildings are clearly
  visible in both images and the changes are unambiguous; 0.6-0.8 if the imagery is blurry, buildings are
  small or hidden, or the two images do not line up; below 0.6 if you cannot really tell. If you can see
  no buildings at all, set no_damage to 1.0 and confidence below 0.4.
- evidence: one or two short sentences naming what you actually see.

### Road access (tool name: detector)

You are given two satellite images of the SAME ~400 m square area, in this order:
1. BEFORE the disaster. 2. AFTER the disaster.
(If only one image is provided, it is the AFTER image.)

Judge ROAD ACCESS for emergency vehicles in the AFTER image, compared with BEFORE.
- access-clear: roads and bridges are visibly passable.
- access-degraded: some roads are flooded, blocked by debris, or partly unusable, but a route probably exists.
- access-blocked: the main roads/bridges look impassable (deep flooding, collapsed structures, large debris).
If there are no visible roads, answer access-clear with low confidence.

Reply as JSON with:
- label: one of ['access-clear', 'access-degraded', 'access-blocked']
- confidence: 0.0 to 1.0 (honest)
- evidence: one short sentence naming what you see on the roads.

## Agent 1, tool selector (optional: Gemini picks which tool to try next)

### System prompt

You are the tool-selection reasoner for a disaster-damage perception agent. Given the tools already tried on this satellite tile and their results, choose which untried tool to try next to best assess damage. Respond with ONLY a JSON object: {"next_tool": "<tool_name>", "reasoning": "<one short sentence>"}. No markdown, no extra text.

### Example user prompt (after a low-confidence classifier reading; detector and SAM still untried)

Tool-call history so far:
- classifier: confidence=0.55, label=minor-damage

Untried tools available:
- detector: Object detector for roads, bridges, and buildings — best when access/infrastructure state matters, not just overall damage severity.
- sam: Zero-shot segmentation fallback — best for ambiguous or heavily occluded tiles where the classifier and detector both struggle.

Which untried tool should be called next, and why?

## Agent 2, planner

No LLM and no prompt. It clusters tiles into zones with a distance graph, scores each zone from the tools' damage and road-access readings, detects neighbouring tiles that disagree, and asks Agent 1 to re-inspect them. Everything it does is deterministic and unit-tested.
