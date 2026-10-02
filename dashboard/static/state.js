/* TRIAGE dashboard state: turns the run's event log into what the UI shows.
 * Pure functions only (no DOM), so it runs in the browser AND under `node --test`.
 * Replay works by re-reducing the events up to any sequence number. */
(function (root) {
  "use strict";

  const DAMAGE_SCORES = {
    "no-damage": 0, "minor-damage": 1, "partial-damage": 1.5,
    "major-damage": 2, "destroyed": 3, "severe-damage": 3,
  };
  const MAX_PRIORITY = 6;

  function initialState() {
    return {
      run: null, transport: null, status: "idle", error: null,
      tiles: {}, versions: [], wave: null, reinspection: null,
      fallbacks: [], feed: [], lastSeq: 0,
    };
  }

  function shortId(tileId) {
    const m = /_(\d+)_post_disaster$/.exec(tileId || "");
    return m ? m[1].replace(/^0+/, "") || "0" : tileId;
  }

  function zoneLabel(zone) {
    if (!zone) return "";
    if (zone.zone_id.startsWith("zone-ungeolocated-")) return "No location " + shortId(zone.zone_id.slice(18));
    return "Zone " + shortId(zone.zone_id.slice(5));
  }

  function damageScore(label) {
    return label in DAMAGE_SCORES ? DAMAGE_SCORES[label] : null;
  }

  function damageColor(score) {
    if (score === null || score === undefined) return "#6b7280";
    const stops = ["#22c55e", "#eab308", "#f97316", "#ef4444"];
    return stops[Math.max(0, Math.min(3, Math.round(score)))];
  }

  function priorityColor(priority, max) {
    const t = Math.max(0, Math.min(1, priority / (max || MAX_PRIORITY)));
    return "hsl(" + Math.round(120 - 120 * t) + ", 78%, 50%)";
  }

  function tileColor(tile) {
    if (!tile || tile.status === "queued") return "#475569";
    if (tile.status === "escalated") return "#a855f7";
    return damageColor(damageScore(tile.damage_severity));
  }

  function feed(state, e, kind, text) {
    state.feed.push({ seq: e.seq, ts: e.ts, kind: kind, text: text });
    if (state.feed.length > 400) state.feed.shift();
  }

  function sev(t) { return t && t.damage_severity ? t.damage_severity : "no reading"; }

  function applyEvent(state, e) {
    state.lastSeq = e.seq;
    switch (e.type) {
      case "transport_selected":
        state.transport = { name: e.transport, agent1: e.agent1_url || null, agent2: e.agent2_url || null };
        feed(state, e, "system", e.transport === "a2a" ? "Agents connected over A2A" : "Agents running in-process");
        break;
      case "run_started":
        state.status = "running";
        state.run = {
          run_id: e.run_id, batch_id: e.batch_id, disaster: e.disaster || null,
          tile_count: e.tile_count, radius: e.adjacency_radius_m, streaming: !!e.wave_size,
          started: e.ts,
        };
        (e.tiles || []).forEach(function (t) {
          state.tiles[t.tile_id] = {
            tile_id: t.tile_id, latitude: t.latitude, longitude: t.longitude,
            status: "queued", damage_severity: null, confidence: null,
            tool_calls: [], reinspections: [], failure_reason: null,
          };
        });
        feed(state, e, "system", "Run started: " + e.tile_count + " tiles, adjacency " + Math.round(e.adjacency_radius_m) + " m");
        break;
      case "wave_arrived":
        state.wave = { current: e.wave, total: e.waves };
        feed(state, e, "feed", "Wave " + e.wave + "/" + e.waves + ": " + e.tile_ids.length + " tile(s) received");
        break;
      case "tile_finished": {
        const t = e.tile, prev = state.tiles[t.tile_id] || { reinspections: [] };
        state.tiles[t.tile_id] = Object.assign({}, prev, t, { reinspections: prev.reinspections || [] });
        const verdict = t.status === "assessed" ? sev(t) + " (" + t.confidence + ")" : "escalated" + (t.failure_reason ? " - " + t.failure_reason : "");
        feed(state, e, t.status === "assessed" ? "agent1" : "warn", "Agent 1 · " + shortId(t.tile_id) + ": " + verdict);
        break;
      }
      case "zones_ranked":
        state.versions.push({
          version: e.version, seq: e.seq, wave: e.wave, round: e.round,
          zones: e.zones, requests: e.reinspection_requests || [],
        });
        feed(state, e, "agent2", "Agent 2 · v" + e.version + ": " + e.zones.length + " zone(s) ranked");
        break;
      case "reinspection_requested":
        state.reinspection = { round: e.round, tile_ids: e.requests.map(function (r) { return r.tile_id; }) };
        feed(state, e, "loop", "Re-inspection round " + e.round + ": " + e.requests.length + " tile(s) sent back to Agent 1");
        break;
      case "tile_reinspected": {
        const a = e.after, t = state.tiles[a.tile_id];
        if (t) {
          t.reinspections = (t.reinspections || []).concat([{ round: e.round, before: e.before, after: a }]);
          Object.assign(t, a, { reinspections: t.reinspections });
        }
        const changed = sev(e.before) !== sev(a);
        feed(state, e, "loop", "Second opinion · " + shortId(a.tile_id) + ": " + sev(e.before) + (changed ? " → " + sev(a) : " (confirmed)"));
        break;
      }
      case "reinspection_no_change":
        feed(state, e, "loop", "Second opinions changed nothing; stopping");
        break;
      case "reinspection_exhausted":
        feed(state, e, "warn", "Re-inspection cap reached; " + e.unresolved.length + " tile(s) unresolved");
        break;
      case "tile_missing":
        feed(state, e, "warn", "Tile " + shortId(e.tile_id) + " never reported; added as escalated");
        break;
      case "transport_fallback":
        state.fallbacks.push({ agent: e.agent, error: e.error });
        feed(state, e, "warn", "A2A call to " + e.agent + " failed; using in-process agent");
        break;
      case "run_finished":
        state.status = "finished";
        state.reinspection = null;
        feed(state, e, "system", "Run finished: " + e.zone_count + " zone(s), " + e.escalated_tile_ids.length + " escalated");
        break;
      case "run_failed":
        state.status = "failed";
        state.error = e.error;
        feed(state, e, "warn", "Run failed: " + e.error);
        break;
      default:
        break;
    }
    return state;
  }

  function reduce(events, upToSeq) {
    const state = initialState();
    for (let i = 0; i < events.length; i++) {
      if (upToSeq !== undefined && upToSeq !== null && events[i].seq > upToSeq) break;
      applyEvent(state, events[i]);
    }
    return state;
  }

  function currentVersion(state) {
    return state.versions.length ? state.versions[state.versions.length - 1] : null;
  }

  /* Rank movement of each zone versus the previous version (by zone id). */
  function rankDiff(prevZones, zones) {
    const prev = {};
    (prevZones || []).forEach(function (z, i) { prev[z.zone_id] = i + 1; });
    return zones.map(function (z, i) {
      const was = prev[z.zone_id];
      return { zone_id: z.zone_id, rank: i + 1, prevRank: was === undefined ? null : was,
               delta: was === undefined ? null : was - (i + 1) };
    });
  }

  function stats(state) {
    const tiles = Object.values(state.tiles);
    const arrived = tiles.filter(function (t) { return t.status !== "queued"; });
    return {
      total: tiles.length,
      arrived: arrived.length,
      assessed: arrived.filter(function (t) { return t.status === "assessed"; }).length,
      escalated: arrived.filter(function (t) { return t.status === "escalated"; }).length,
      reinspected: tiles.filter(function (t) { return (t.reinspections || []).length > 0; }).length,
      versions: state.versions.length,
      fallbacks: state.fallbacks.length,
      toolErrors: arrived.reduce(function (n, t) { return n + (t.tool_errors || 0); }, 0),
    };
  }

  /* Four corners of a square tile footprint centred on (lat, lon), half-side in metres. */
  function tileCorners(lat, lon, halfM) {
    const dLat = halfM / 111320;
    const dLon = halfM / (111320 * Math.cos(lat * Math.PI / 180));
    return [[lat + dLat, lon - dLon], [lat + dLat, lon + dLon], [lat - dLat, lon + dLon], [lat - dLat, lon - dLon]];
  }

  /* Convex hull (Andrew's monotone chain) of [lat, lon] points, returned as [lat, lon]. */
  function convexHull(points) {
    const pts = points.map(function (p) { return [p[1], p[0]]; })
      .sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    const uniq = pts.filter(function (p, i) { return i === 0 || p[0] !== pts[i - 1][0] || p[1] !== pts[i - 1][1]; });
    if (uniq.length < 3) return uniq.map(function (p) { return [p[1], p[0]]; });
    const cross = function (o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); };
    const lower = [];
    uniq.forEach(function (p) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    });
    const upper = [];
    for (let i = uniq.length - 1; i >= 0; i--) {
      const p = uniq[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    }
    lower.pop(); upper.pop();
    return lower.concat(upper).map(function (p) { return [p[1], p[0]]; });
  }

  /* Replay pacing: real gap between two events, sped up, clamped so it never stalls or flickers. */
  function replayDelayMs(prevTs, ts, speed) {
    const gap = Math.max(0, new Date(ts) - new Date(prevTs));
    return Math.max(25, Math.min(1200, gap / (speed || 1)));
  }

  const api = {
    initialState, applyEvent, reduce, currentVersion, rankDiff, stats, zoneLabel, shortId,
    damageScore, damageColor, priorityColor, tileColor, tileCorners, convexHull, replayDelayMs,
    MAX_PRIORITY: MAX_PRIORITY,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TriageState = api;
})(typeof self !== "undefined" ? self : this);
