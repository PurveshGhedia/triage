const test = require("node:test");
const assert = require("node:assert/strict");
const S = require("../../dashboard/static/state.js");

let seq = 0;
const ev = (type, extra, ts) => Object.assign({ seq: ++seq, ts: ts || "2026-10-02T10:00:00Z", run_id: "r", type }, extra);
const tile = (id, over) => Object.assign({
  tile_id: id, status: "assessed", damage_severity: "major-damage", confidence: 0.9,
  latitude: 10, longitude: 20, failure_reason: null, tool_errors: 0, tool_calls: [],
}, over || {});
const zone = (id, priority, tiles, extra) => Object.assign({
  zone_id: id, version: 1, tile_ids: tiles, damage_score: priority / 2, accessibility_penalty: priority / 2,
  priority_score: priority, status: "ranked", escalated_tile_ids: [], ungeolocated_tile_ids: [],
  conflicting_tile_pairs: [],
}, extra || {});

function sampleRun() {
  seq = 0;
  return [
    ev("transport_selected", { transport: "a2a", agent1_url: "http://a", agent2_url: "http://b" }),
    ev("run_started", { batch_id: "b1", disaster: "storm", tile_count: 3, adjacency_radius_m: 700, wave_size: 2,
      tiles: [{ tile_id: "storm_00000001_post_disaster", latitude: 10, longitude: 20 },
              { tile_id: "storm_00000002_post_disaster", latitude: 10.001, longitude: 20 },
              { tile_id: "storm_00000003_post_disaster", latitude: 10.002, longitude: 20 }] }),
    ev("wave_arrived", { wave: 1, waves: 2, tile_ids: ["storm_00000001_post_disaster", "storm_00000002_post_disaster"] }),
    ev("tile_finished", { tile: tile("storm_00000001_post_disaster") }),
    ev("tile_finished", { tile: tile("storm_00000002_post_disaster", { status: "escalated", damage_severity: null, confidence: null, failure_reason: "all tools failed" }) }),
    ev("zones_ranked", { version: 1, wave: 1, round: 0, reinspection_requests: [],
      zones: [zone("zone-storm_00000001_post_disaster", 4.5, ["storm_00000001_post_disaster", "storm_00000002_post_disaster"])] }),
    ev("wave_arrived", { wave: 2, waves: 2, tile_ids: ["storm_00000003_post_disaster"] }),
    ev("tile_finished", { tile: tile("storm_00000003_post_disaster", { damage_severity: "destroyed" }) }),
    ev("zones_ranked", { version: 2, wave: 2, round: 0, reinspection_requests: [{ tile_id: "storm_00000002_post_disaster", reason: "x" }],
      zones: [zone("zone-storm_00000001_post_disaster", 5.5, ["storm_00000001_post_disaster", "storm_00000002_post_disaster", "storm_00000003_post_disaster"])] }),
    ev("reinspection_requested", { round: 1, requests: [{ tile_id: "storm_00000002_post_disaster", reason: "x" }] }),
    ev("tile_reinspected", { round: 1, before: tile("storm_00000002_post_disaster", { status: "escalated", damage_severity: null }),
      after: tile("storm_00000002_post_disaster", { damage_severity: "minor-damage" }) }),
    ev("zones_ranked", { version: 3, wave: null, round: 1, reinspection_requests: [],
      zones: [zone("zone-storm_00000001_post_disaster", 4.0, ["storm_00000001_post_disaster", "storm_00000002_post_disaster", "storm_00000003_post_disaster"])] }),
    ev("run_finished", { zone_count: 1, escalated_tile_ids: [], versions: [1, 2, 3] }),
  ];
}

test("full reduction builds tiles, versions, status and metadata", () => {
  const s = S.reduce(sampleRun());
  assert.equal(s.status, "finished");
  assert.equal(s.run.disaster, "storm");
  assert.equal(s.run.streaming, true);
  assert.deepEqual(s.transport, { name: "a2a", agent1: "http://a", agent2: "http://b" });
  assert.equal(s.versions.length, 3);
  assert.equal(S.currentVersion(s).version, 3);
  assert.equal(Object.keys(s.tiles).length, 3);
});

test("queued tiles appear at run start and fill in as they arrive", () => {
  const events = sampleRun();
  const atStart = S.reduce(events, events[1].seq);
  assert.ok(Object.values(atStart.tiles).every((t) => t.status === "queued"));
  assert.equal(S.stats(atStart).arrived, 0);
  const afterWave1 = S.reduce(events, events[5].seq);
  assert.equal(S.stats(afterWave1).arrived, 2);
  assert.equal(afterWave1.wave.current, 1);
});

test("reducing up to a sequence number is a faithful rewind", () => {
  const events = sampleRun();
  const v1 = S.reduce(events, events[5].seq);
  assert.equal(v1.versions.length, 1);
  assert.equal(S.currentVersion(v1).zones[0].priority_score, 4.5);
  const later = S.reduce(events);
  assert.equal(S.currentVersion(later).zones[0].priority_score, 4.0);
});

test("reinspection records before/after on the tile and updates its reading", () => {
  const s = S.reduce(sampleRun());
  const t = s.tiles["storm_00000002_post_disaster"];
  assert.equal(t.status, "assessed");
  assert.equal(t.damage_severity, "minor-damage");
  assert.equal(t.reinspections.length, 1);
  assert.equal(t.reinspections[0].before.status, "escalated");
});

test("stats count assessed, escalated, reinspected, versions and fallbacks", () => {
  const events = sampleRun();
  events.splice(3, 0, Object.assign({ seq: 99, ts: "t", run_id: "r", type: "transport_fallback", agent: "planner", error: "boom" }));
  const s = S.reduce(events);
  const st = S.stats(s);
  assert.equal(st.versions, 3);
  assert.equal(st.reinspected, 1);
  assert.equal(st.fallbacks, 1);
  assert.equal(st.total, 3);
});

test("failed run sets status and error", () => {
  const s = S.reduce([ev("run_started", { batch_id: "b", tile_count: 0, adjacency_radius_m: 500, tiles: [] }), ev("run_failed", { error: "kaboom" })]);
  assert.equal(s.status, "failed");
  assert.equal(s.error, "kaboom");
});

test("unknown event types are ignored, not fatal", () => {
  const s = S.reduce([ev("something_new", {}), ev("run_started", { batch_id: "b", tile_count: 0, adjacency_radius_m: 500, tiles: [] })]);
  assert.equal(s.status, "running");
});

test("labels and short ids", () => {
  assert.equal(S.shortId("hurricane-harvey_00000155_post_disaster"), "155");
  assert.equal(S.zoneLabel({ zone_id: "zone-hurricane-harvey_00000155_post_disaster" }), "Zone 155");
  assert.equal(S.zoneLabel({ zone_id: "zone-ungeolocated-hurricane-harvey_00000009_post_disaster" }), "No location 9");
});

test("colours: damage ladder, escalated purple, queued grey, priority green->red", () => {
  assert.equal(S.damageColor(0), "#22c55e");
  assert.equal(S.damageColor(3), "#ef4444");
  assert.equal(S.damageColor(null), "#6b7280");
  assert.equal(S.tileColor({ status: "escalated" }), "#a855f7");
  assert.equal(S.tileColor({ status: "queued" }), "#475569");
  assert.equal(S.tileColor({ status: "assessed", damage_severity: "destroyed" }), "#ef4444");
  assert.equal(S.priorityColor(0), "hsl(120, 78%, 50%)");
  assert.equal(S.priorityColor(6), "hsl(0, 78%, 50%)");
  assert.equal(S.priorityColor(99), "hsl(0, 78%, 50%)");
});

test("rank movement against the previous version", () => {
  const prev = [{ zone_id: "a" }, { zone_id: "b" }, { zone_id: "c" }];
  const now = [{ zone_id: "c" }, { zone_id: "a" }, { zone_id: "d" }];
  const d = S.rankDiff(prev, now);
  assert.deepEqual(d.map((x) => x.delta), [2, -1, null]);   // c moved up 2, a down 1, d is new
  assert.deepEqual(S.rankDiff(null, now).map((x) => x.delta), [null, null, null]);
});

test("convex hull of a square with an interior point", () => {
  const hull = S.convexHull([[0, 0], [0, 2], [2, 2], [2, 0], [1, 1]]);
  assert.equal(hull.length, 4);
  assert.ok(!hull.some((p) => p[0] === 1 && p[1] === 1));
  assert.equal(S.convexHull([[1, 1]]).length, 1);
  assert.equal(S.convexHull([[1, 1], [1, 1]]).length, 1);          // duplicates collapse
  assert.equal(S.convexHull([[0, 0], [1, 1], [2, 2]]).length, 2);  // collinear -> segment
});

test("tile corners form a ~square footprint of the requested size", () => {
  const c = S.tileCorners(0, 0, 100);
  const latSpan = (c[0][0] - c[2][0]) * 111320;
  const lonSpan = (c[1][1] - c[0][1]) * 111320;
  assert.ok(Math.abs(latSpan - 200) < 0.5 && Math.abs(lonSpan - 200) < 0.5);
  const hi = S.tileCorners(60, 0, 100);                             // longitudes stretch away from equator
  assert.ok((hi[1][1] - hi[0][1]) > (c[1][1] - c[0][1]) * 1.9);
});

test("replay delay follows real gaps, scaled and clamped", () => {
  assert.equal(S.replayDelayMs("2026-10-02T10:00:00Z", "2026-10-02T10:00:01Z", 1), 1000);
  assert.equal(S.replayDelayMs("2026-10-02T10:00:00Z", "2026-10-02T10:00:01Z", 10), 100);
  assert.equal(S.replayDelayMs("2026-10-02T10:00:00Z", "2026-10-02T10:00:00Z", 1), 25);    // floor
  assert.equal(S.replayDelayMs("2026-10-02T10:00:00Z", "2026-10-02T10:05:00Z", 1), 1200);  // ceiling
});

test("tileGeometry prefers the true footprint and falls back to a square", () => {
  const fp = { bounds: [[10, 20], [10.01, 20.02]], corners: [[10.01, 20], [10.01, 20.02], [10, 20.02], [10, 20]] };
  const exact = S.tileGeometry({ latitude: 99, longitude: 99 }, fp, 230);
  assert.equal(exact.exact, true);
  assert.deepEqual(exact.bounds, fp.bounds);
  const approx = S.tileGeometry({ latitude: 10, longitude: 20 }, undefined, 230);
  assert.equal(approx.exact, false);
  assert.ok(approx.bounds[0][0] < 10 && approx.bounds[1][0] > 10);     // south < centre < north
  assert.ok(approx.bounds[0][1] < 20 && approx.bounds[1][1] > 20);     // west < centre < east
  assert.equal(approx.corners.length, 4);
  assert.equal(S.tileGeometry({ latitude: 10, longitude: 20 }, { bounds: null }, 230).exact, false);  // malformed -> fallback
});

test("run records which tools were used, defaulting to mock", () => {
  const mk = (extra) => S.reduce([ev("run_started", Object.assign({ batch_id: "b", tile_count: 0, adjacency_radius_m: 500, tiles: [] }, extra))]);
  assert.equal(mk({}).run.tools, "mock");
  assert.equal(mk({ tools: "vision" }).run.tools, "vision");
});
