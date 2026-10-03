/* TRIAGE dashboard UI. All state comes from the run's event log (see state.js);
 * this file only fetches events and draws them. Text from events is always set via
 * textContent (never innerHTML), so model output cannot inject markup. */
(function () {
  "use strict";
  const S = window.TriageState;
  const $ = (id) => document.getElementById(id);

  function h(tag, props) {
    const el = document.createElement(tag);
    Object.entries(props || {}).forEach(([k, v]) => {
      if (v === null || v === undefined || v === false) return;
      if (k === "class") el.className = v;
      else if (k === "style") Object.assign(el.style, v);
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    });
    for (let i = 2; i < arguments.length; i++) {
      [].concat(arguments[i]).forEach((c) => {
        if (c === null || c === undefined || c === false) return;
        el.append(c.nodeType ? c : document.createTextNode(String(c)));
      });
    }
    return el;
  }

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let detail = "";
      try { detail = (await r.json()).detail; } catch (e) { /* not json */ }
      throw new Error(detail || "HTTP " + r.status);
    }
    return r.json();
  }

  // ---- app state -----------------------------------------------------------------
  let events = [], shown = 0, following = true, runId = null, mode = "idle";
  let pollTimer = null, replayTimer = null, allowRun = true;
  let map = null, tileLayer = null, zoneLayer = null, fitted = false;
  let tileRects = {}, arrivedBefore = new Set(), selectedZone = null, lastDetailSig = "";
  let tileImages = {}, imageryMode = "post", imageryChosen = false, footprintsFor = null, footprints = {}, footprintsReady = false;
  const footprintCache = {};
  const truthCache = {};
  const TILE_HALF_M = 230;

  // ---- map -----------------------------------------------------------------------
  function initMap() {
    if (!window.L) {
      const b = $("banner");
      b.hidden = false;
      b.textContent = "Map library unavailable (offline?) — rankings and details still work";
      return;
    }
    map = L.map("map", { zoomControl: true }).setView([20, 0], 2);
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 19, attribution: "Imagery © Esri" }).addTo(map);
    map.createPane("imagery");
    map.getPane("imagery").style.zIndex = 350;      // above the basemap, below tile outlines and zones
    map.getPane("imagery").classList.add("leaflet-imagery-pane");
    tileLayer = L.layerGroup().addTo(map);
    zoneLayer = L.layerGroup().addTo(map);
  }

  function geometry(t) {
    return S.tileGeometry(t, footprints[t.tile_id], TILE_HALF_M);
  }

  function imageUrl(t, kind) {
    const disaster = lastState.run && lastState.run.disaster;
    return "/api/image/" + encodeURIComponent(disaster) + "/" + kind + "/" + encodeURIComponent(t.tile_id) + "?w=640";
  }

  function haveFootprints() { return Object.keys(footprints).length > 0; }

  /* Draw the real xBD imagery for tiles that have arrived, at their true ground position. */
  function syncImagery(t, geo, queued) {
    const want = imageryMode !== "basemap" && geo.exact && !queued && lastState.run && lastState.run.disaster;
    let ov = tileImages[t.tile_id];
    if (!want) {
      if (ov) { map.removeLayer(ov); delete tileImages[t.tile_id]; }
      return;
    }
    const url = imageUrl(t, imageryMode);
    if (!ov) {
      ov = L.imageOverlay(url, geo.bounds, { pane: "imagery", opacity: 0, interactive: false });
      ov.addTo(map);
      tileImages[t.tile_id] = ov;
      ov._triageMode = imageryMode;
      setTimeout(() => { if (ov.setOpacity) ov.setOpacity(1); }, 30);   // fade the tile in as it "arrives"
    } else if (ov._triageMode !== imageryMode) {
      ov.setUrl(url);
      ov._triageMode = imageryMode;
    }
  }

  async function ensureFootprints(disaster) {
    if (!disaster || footprintsFor === disaster) return;
    footprintsFor = disaster;
    footprintsReady = false;
    if (!(disaster in footprintCache)) {
      try { footprintCache[disaster] = await api("/api/footprints/" + encodeURIComponent(disaster)); }
      catch (e) { footprintCache[disaster] = {}; }
    }
    footprints = footprintCache[disaster];
    if (!haveFootprints()) imageryMode = "basemap";
    else if (imageryMode === "basemap" && !imageryChosen) imageryMode = "post";
    footprintsReady = true;     // only now is the map framed, so it fits once, to the true footprints
    fitted = false;
    render();
  }

  function currentZones(state) {
    const v = S.currentVersion(state);
    return v ? v.zones : [];
  }

  function zoneOfTile(state, tileId) {
    return currentZones(state).find((z) => z.tile_ids.indexOf(tileId) >= 0);
  }

  function renderMap(state) {
    if (!map) return;
    const located = Object.values(state.tiles).filter((t) => t.latitude !== null && t.latitude !== undefined);
    const inReinspection = new Set(state.reinspection ? state.reinspection.tile_ids : []);
    const nowArrived = new Set();

    located.forEach((t) => {
      const geo = geometry(t);
      const bounds = geo.bounds;
      const queued = t.status === "queued";
      const showingImagery = imageryMode !== "basemap" && geo.exact;
      const style = {
        color: S.tileColor(t), weight: showingImagery ? 3 : 1.5, fillColor: S.tileColor(t),
        fillOpacity: queued ? 0.06 : (showingImagery ? 0.14 : 0.5),
        dashArray: queued || t.status === "escalated" ? "4 3" : null,
      };
      let rect = tileRects[t.tile_id];
      if (!rect) {
        rect = L.rectangle(bounds, style).addTo(tileLayer);
        rect.on("click", () => { const z = zoneOfTile(lastState, t.tile_id); if (z) selectZone(z.zone_id, false); });
        tileRects[t.tile_id] = rect;
      } else {
        rect.setStyle(style);
        if (rect.setBounds) rect.setBounds(bounds);
      }
      syncImagery(t, geo, queued);
      const tip = S.shortId(t.tile_id) + " · " + (queued ? "waiting" : (t.damage_severity || t.status));
      rect.bindTooltip(tip, { sticky: true });
      if (!queued) nowArrived.add(t.tile_id);
      const el = rect.getElement && rect.getElement();
      if (el) {
        el.classList.toggle("reinspect", inReinspection.has(t.tile_id));
        if (!queued && !arrivedBefore.has(t.tile_id)) {
          el.classList.add("arrive");
          setTimeout(() => el.classList.remove("arrive"), 1200);
        }
      }
    });
    arrivedBefore = nowArrived;

    if (!fitted && located.length && (footprintsReady || !(state.run && state.run.disaster))) {
      const pts = [];
      located.forEach((t) => geometry(t).corners.forEach((c) => pts.push(c)));
      map.fitBounds(L.latLngBounds(pts), { padding: [50, 50] });
      fitted = true;
    }

    zoneLayer.clearLayers();
    currentZones(state).forEach((z, i) => {
      const pts = [];
      z.tile_ids.forEach((id) => {
        const t = state.tiles[id];
        if (t && t.latitude !== null && t.latitude !== undefined)
          geometry(t).corners.forEach((c) => pts.push(c));
      });
      if (!pts.length) return;
      const color = S.priorityColor(z.priority_score);
      const selected = z.zone_id === selectedZone;
      const poly = L.polygon(S.convexHull(pts), {
        color: selected ? "#ffffff" : color, weight: selected ? 5 : 3, fillColor: color, fillOpacity: 0.1,
        dashArray: z.status === "needs_manual_review" ? "8 6" : null,
      }).addTo(zoneLayer);
      poly.on("click", () => selectZone(z.zone_id, false));
      const centre = poly.getBounds().getCenter();
      L.marker(centre, {
        interactive: false,
        icon: L.divIcon({ className: "", iconSize: [0, 0],
          html: '<div class="rankbadge" style="border-color:' + color + '">#' + (i + 1) + " · " + z.priority_score.toFixed(1) + "</div>" }),
      }).addTo(zoneLayer);
      (z.conflicting_tile_pairs || []).forEach((pair) => {
        const a = state.tiles[pair[0]], b = state.tiles[pair[1]];
        if (a && b && a.latitude !== null && b.latitude !== null)
          L.polyline([[a.latitude, a.longitude], [b.latitude, b.longitude]],
            { color: "#ef4444", weight: 3, dashArray: "2 6" }).addTo(zoneLayer);
      });
    });
  }

  // ---- side panel ----------------------------------------------------------------
  function totalVersions() { return events.filter((e) => e.type === "zones_ranked").length; }

  function versionHint(v) {
    if (!v) return "";
    if (v.round > 0) return "after re-inspection round " + v.round;
    if (v.wave) return "after wave " + v.wave + " of the live feed";
    return "initial ranking";
  }

  function renderSide(state) {
    const total = totalVersions();
    const slider = $("versionSlider");
    slider.max = Math.max(1, total);
    slider.value = Math.max(1, state.versions.length);
    slider.disabled = total < 2;
    const v = S.currentVersion(state);
    $("versionTitle").textContent = v ? "Ranking · v" + v.version + (total > 1 ? " of " + total : "") : "Ranking";
    $("versionHint").textContent = versionHint(v);
    $("latestBtn").hidden = following || shown >= events.length;

    const st = S.stats(state);
    const stat = (n, label, cls) => h("div", { class: "stat " + (cls || "") }, h("b", {}, n), h("span", {}, label));
    const box = $("stats");
    box.replaceChildren(
      stat(st.arrived + "/" + st.total, "tiles in"),
      stat(st.assessed, "assessed"),
      stat(st.escalated, "escalated", st.escalated ? "warn" : ""),
      stat(st.reinspected, "re-checked"),
    );

    const list = $("ranking");
    const zones = currentZones(state);
    if (!zones.length) {
      list.replaceChildren(h("div", { class: "empty" },
        state.status === "running" ? "Waiting for the first ranking…" : "Pick a scenario and press ▶ Run, or replay a past run."));
      return;
    }
    const prev = state.versions.length > 1 ? state.versions[state.versions.length - 2].zones : null;
    const diff = S.rankDiff(prev, zones);
    list.replaceChildren(...zones.map((z, i) => {
      const d = diff[i];
      let delta = null;
      if (prev && d.prevRank === null) delta = h("span", { class: "delta new" }, "NEW");
      else if (prev && d.delta > 0) delta = h("span", { class: "delta up" }, "▲" + d.delta);
      else if (prev && d.delta < 0) delta = h("span", { class: "delta down" }, "▼" + -d.delta);
      const dw = Math.min(100, (z.damage_score / S.MAX_PRIORITY) * 100);
      const aw = Math.min(100 - dw, (z.accessibility_penalty / S.MAX_PRIORITY) * 100);
      return h("div", { class: "zone" + (z.zone_id === selectedZone ? " selected" : ""), onclick: () => selectZone(z.zone_id, true) },
        h("div", { class: "row" },
          h("div", { class: "rank" }, "#" + (i + 1)),
          h("div", { class: "name" }, S.zoneLabel(z)),
          h("div", { class: "score", style: { color: S.priorityColor(z.priority_score) } }, z.priority_score.toFixed(2)),
          delta),
        h("div", { class: "bar", title: "damage " + z.damage_score + " + access " + z.accessibility_penalty },
          h("div", { class: "d", style: { width: dw + "%" } }), h("div", { class: "a", style: { width: aw + "%" } })),
        h("div", { class: "meta" },
          h("span", {}, "damage " + z.damage_score.toFixed(1)), h("span", {}, "access " + z.accessibility_penalty.toFixed(1)),
          h("span", {}, z.tile_ids.length + (z.tile_ids.length === 1 ? " tile" : " tiles")),
          z.status === "needs_manual_review" ? h("span", { class: "tag review" }, "needs review") : null,
          (z.conflicting_tile_pairs || []).length ? h("span", { class: "tag conflict" }, z.conflicting_tile_pairs.length + " conflict") : null,
          (z.ungeolocated_tile_ids || []).length ? h("span", { class: "tag review" }, "no location") : null));
    }));
  }

  function renderHud(state) {
    const hud = $("hud");
    const chips = [];
    if (state.run) {
      chips.push(h("div", { class: "chip" }, h("b", {}, state.run.disaster || state.run.batch_id), state.run.streaming ? "simulated live feed" : "batch"));
      const real = state.run.tools === "vision";
      chips.push(h("div", { class: "chip", style: real ? { borderColor: "#38bdf8", color: "#7dd3fc" } : { borderColor: "#f59e0b", color: "#fbbf24" },
        title: real ? "Gemini looked at the real before/after imagery" : "Labels come from random stand-in tools that ignore the image" },
        real ? "Gemini vision" : "⚠ mock perception (random labels)"));
      if (state.wave) chips.push(h("div", { class: "chip" }, "wave ", h("b", {}, state.wave.current + "/" + state.wave.total)));
      if (state.reinspection) chips.push(h("div", { class: "chip", style: { borderColor: "#fcd34d", color: "#fcd34d" } },
        "⟳ second opinion · round " + state.reinspection.round));
    }
    if (haveFootprints()) {
      const mk = (mode, label) => h("button", { class: imageryMode === mode ? "on" : "", onclick: () => {
        imageryMode = mode; imageryChosen = true; render(); } }, label);
      chips.push(h("div", { class: "seg", title: "What to show under the tile outlines" },
        mk("post", "After"), mk("pre", "Before"), mk("basemap", "Basemap")));
    }
    hud.replaceChildren(...chips);
    const legend = $("legend");
    if (!legend.firstChild) {
      const row = (c, t, dashed) => h("div", {}, h("i", { style: { background: dashed ? "transparent" : c, border: dashed ? "1.5px dashed " + c : "none" } }), t);
      legend.replaceChildren(h("div", { style: { fontWeight: 700, marginBottom: "3px" } }, "Tile damage"),
        row("#22c55e", "none"), row("#eab308", "minor"), row("#f97316", "major"), row("#ef4444", "destroyed"),
        row("#a855f7", "escalated — human review", true), row("#475569", "not yet received", true),
        h("div", { style: { marginTop: "5px", color: "#8da0bd" } }, "Zone outline = priority (green→red)"));
    }
  }

  function renderStatus(state) {
    const parts = [];
    if (state.transport) parts.push(h("span", { class: "pill " + (state.transport.name === "a2a" ? "a2a" : "") }, state.transport.name === "a2a" ? "A2A" : "IN-PROCESS"));
    if (state.fallbacks.length) parts.push(h("span", { class: "pill failed", title: state.fallbacks[0].error }, "FALLBACK ×" + state.fallbacks.length));
    const label = mode === "replay" ? "REPLAY" : (state.status || "idle").toUpperCase();
    parts.push(h("span", { class: "pill " + (mode === "replay" ? "running" : state.status) }, label));
    $("status").replaceChildren(...parts);
  }

  function renderFeed(state) {
    const box = $("feed");
    const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    box.replaceChildren(...state.feed.slice(-120).map((f) => {
      const t = new Date(f.ts);
      const stamp = isNaN(t) ? "" : t.toTimeString().slice(0, 8) + "  ";
      return h("div", { class: "f-" + f.kind }, stamp + f.text);
    }));
    if (nearBottom) box.scrollTop = box.scrollHeight;
  }

  // ---- zone detail ---------------------------------------------------------------
  function selectZone(zoneId, fly) {
    selectedZone = zoneId;
    lastDetailSig = "";
    render();
    if (fly && map && zoneLayer) {
      zoneLayer.eachLayer((l) => {
        if (l.getBounds && l.options && l.options.color === "#ffffff") map.fitBounds(l.getBounds(), { padding: [90, 90], maxZoom: 17 });
      });
    }
  }

  const TRUTH_LABELS = ["no damage", "minor", "major", "destroyed"];

  function truthLine(disaster, tileId) {
    const key = disaster + "/" + tileId;
    if (!(key in truthCache)) {
      truthCache[key] = null;
      api("/api/truth/" + encodeURIComponent(disaster) + "/" + encodeURIComponent(tileId))
        .then((t) => { truthCache[key] = t; lastDetailSig = ""; render(); })
        .catch(() => { truthCache[key] = { error: true }; lastDetailSig = ""; render(); });
    }
    const t = truthCache[key];
    if (!t) return h("div", { class: "truth" }, "xBD ground truth: loading…");
    if (t.error || !t.buildings) return h("div", { class: "truth" }, "xBD ground truth: no labelled buildings");
    return h("div", { class: "truth" }, "xBD ground truth: " + t.buildings + " buildings · worst " +
      TRUTH_LABELS[t.worst] + " · mean " + t.mean + " (0–3)");
  }

  function tileCard(state, id) {
    const t = state.tiles[id];
    if (!t) return null;
    const disaster = state.run && state.run.disaster;
    const sev = t.status === "escalated" ? "escalated" : (t.damage_severity || "—");
    const badge = h("span", { class: "sev", style: { background: S.tileColor(t) } }, sev);
    const parts = [h("h4", {}, h("span", {}, "Tile " + S.shortId(id)), badge)];
    if (disaster) {
      const post = "/api/image/" + encodeURIComponent(disaster) + "/post/" + encodeURIComponent(id) + "?w=520";
      const pre = "/api/image/" + encodeURIComponent(disaster) + "/pre/" + encodeURIComponent(id) + "?w=520";
      const cmp = h("div", { class: "cmp" },
        h("img", { class: "pre", src: pre, loading: "lazy", alt: "before" }),
        h("img", { class: "post", src: post, loading: "lazy", alt: "after" }),
        h("span", { class: "tag before" }, "before"), h("span", { class: "tag after" }, "after"),
        h("input", { type: "range", min: "0", max: "100", value: "50", "aria-label": "before/after",
          oninput: (e) => cmp.style.setProperty("--pos", e.target.value + "%") }));
      parts.push(cmp);
    }
    if (t.confidence !== null && t.confidence !== undefined) parts.push(h("div", {}, "Confidence " + t.confidence));
    if (t.failure_reason) parts.push(h("div", { class: "tag conflict" }, t.failure_reason));
    const steps = (t.tool_calls || []).map((c) => h("div", { class: "step" },
      h("b", {}, c.tool), " → " + c.label + " (" + c.confidence + ")",
      c.evidence ? h("div", { class: "seen" }, "sees: " + c.evidence) : null,
      c.reasoning ? h("div", { class: "why" }, "“" + c.reasoning + "”") : null));
    (t.reinspections || []).forEach((r) => {
      const was = r.before && r.before.damage_severity ? r.before.damage_severity : "no reading";
      steps.push(h("div", { class: "step", style: { color: "#fcd34d" } },
        "⟳ round " + r.round + " second opinion: " + was + " → " + (r.after.damage_severity || "no reading")));
    });
    if (steps.length) parts.push(h("div", { class: "timeline" }, steps));
    if ($("truthToggle").checked && disaster) parts.push(truthLine(disaster, id));
    return h("div", { class: "tcard" }, parts);
  }

  function renderDetail(state) {
    const panel = $("detail");
    const zone = selectedZone ? currentZones(state).find((z) => z.zone_id === selectedZone) : null;
    if (!zone) { panel.hidden = true; lastDetailSig = ""; return; }
    const sig = [selectedZone, S.currentVersion(state).version, $("truthToggle").checked,
      zone.tile_ids.map((id) => { const t = state.tiles[id]; return id + ":" + (t ? t.status + (t.damage_severity || "") + (t.reinspections || []).length : ""); }).join(",")].join("|");
    panel.hidden = false;
    if (sig === lastDetailSig) return;
    lastDetailSig = sig;
    $("detailTitle").textContent = S.zoneLabel(zone) + " · priority " + zone.priority_score.toFixed(2) +
      " (damage " + zone.damage_score.toFixed(1) + " + access " + zone.accessibility_penalty.toFixed(1) + ")";
    const ids = zone.tile_ids.slice().sort((a, b) => (S.damageScore((state.tiles[b] || {}).damage_severity) || -1) - (S.damageScore((state.tiles[a] || {}).damage_severity) || -1));
    $("detailBody").replaceChildren(...ids.map((id) => tileCard(state, id)).filter(Boolean));
  }

  // ---- render loop ---------------------------------------------------------------
  let lastState = S.initialState();

  function render() {
    lastState = S.reduce(events.slice(0, shown));
    if (lastState.run && lastState.run.disaster) ensureFootprints(lastState.run.disaster);
    renderStatus(lastState);
    renderHud(lastState);
    renderMap(lastState);
    renderSide(lastState);
    renderFeed(lastState);
    renderDetail(lastState);
  }

  // ---- loading, live polling, replay ------------------------------------------------
  function stopAll() {
    clearTimeout(pollTimer); clearTimeout(replayTimer);
    pollTimer = replayTimer = null;
  }

  function resetView() {
    events = []; shown = 0; following = true; fitted = false; selectedZone = null; lastDetailSig = "";
    if (tileLayer) { tileLayer.clearLayers(); zoneLayer.clearLayers(); }
    if (map) Object.values(tileImages).forEach((ov) => map.removeLayer(ov));
    tileImages = {}; tileRects = {}; arrivedBefore = new Set();
  }

  async function poll() {
    try {
      const after = events.length ? events[events.length - 1].seq : 0;
      const data = await api("/api/runs/" + runId + "/events?after=" + after);
      events = events.concat(data.events);
      if (following) shown = events.length;
      render();
      if (data.finished) { mode = "idle"; loadRuns(); return; }
    } catch (e) {
      $("versionHint").textContent = "connection problem: " + e.message;
    }
    pollTimer = setTimeout(poll, 700);
  }

  function beginLive(id) {
    stopAll(); resetView(); runId = id; mode = "live";
    $("runSelect").value = id;
    poll();
  }

  async function showRun(id) {
    stopAll(); resetView(); runId = id; mode = "idle";
    const data = await api("/api/runs/" + id + "/events");
    events = data.events; shown = events.length;
    const finished = events.some((e) => e.type === "run_finished" || e.type === "run_failed");
    render();
    if (!finished) { mode = "live"; poll(); }
  }

  async function replay() {
    if (!runId) return;
    stopAll();
    const data = await api("/api/runs/" + runId + "/events");
    resetView();
    events = data.events; shown = 0; following = false; mode = "replay";
    const speed = Number($("speed").value) || 1;
    const step = () => {
      if (shown >= events.length) { mode = "idle"; following = true; render(); return; }
      shown += 1;
      render();
      const delay = shown < events.length ? S.replayDelayMs(events[shown - 1].ts, events[shown].ts, speed) : 0;
      replayTimer = setTimeout(step, delay);
    };
    step();
  }

  async function startRun() {
    const btn = $("runBtn");
    btn.disabled = true;
    try {
      const body = {
        disaster: $("disaster").value, transport: $("transport").value,
        wave_size: $("streaming").checked ? 3 : null, wave_interval_s: 2.0,
        tools: $("vision").checked ? "vision" : "mock", use_llm_selector: $("llmselect").checked,
      };
      const res = await api("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      await loadRuns();
      beginLive(res.run_id);
    } catch (e) {
      const b = $("banner");
      b.hidden = false; b.textContent = "Could not start run: " + e.message;
      setTimeout(() => { b.hidden = true; }, 6000);
    } finally {
      btn.disabled = !allowRun;
    }
  }

  async function loadDisasters() {
    const list = await api("/api/disasters");
    $("disaster").replaceChildren(...list.map((d) => h("option", { value: d.disaster }, d.disaster + (d.tile_count ? " (" + d.tile_count + " tiles)" : ""))));
  }

  async function loadRuns() {
    const runs = await api("/api/runs");
    const sel = $("runSelect");
    const keep = runId;
    sel.replaceChildren(...runs.map((r) => {
      const t = new Date(r.started_at);
      const when = isNaN(t) ? "" : t.toTimeString().slice(0, 5);
      return h("option", { value: r.run_id }, when + " · " + (r.disaster || r.batch_id || r.run_id) + " · " + r.status);
    }));
    if (!runs.length) sel.replaceChildren(h("option", { value: "" }, "no runs yet"));
    if (keep) sel.value = keep;
    return runs;
  }

  async function init() {
    initMap();
    $("runBtn").addEventListener("click", startRun);
    $("replayBtn").addEventListener("click", replay);
    $("runSelect").addEventListener("change", (e) => { if (e.target.value) showRun(e.target.value); });
    $("detailClose").addEventListener("click", () => { selectedZone = null; render(); });
    $("truthToggle").addEventListener("change", () => { lastDetailSig = ""; render(); });
    $("latestBtn").addEventListener("click", () => { following = true; shown = events.length; render(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") { selectedZone = null; render(); } });
    $("versionSlider").addEventListener("input", (e) => {
      const n = Number(e.target.value);
      const ranked = events.map((ev, i) => ({ ev, i })).filter((x) => x.ev.type === "zones_ranked");
      if (!ranked[n - 1]) return;
      clearTimeout(replayTimer); replayTimer = null;
      if (mode === "replay") mode = "idle";
      following = false; shown = ranked[n - 1].i + 1;
      render();
    });
    render();
    try {
      const health = await api("/api/health");
      allowRun = health.allow_run;
      $("runBtn").disabled = !allowRun;
      if (!allowRun) $("runBtn").title = "Starting runs is disabled on this server";
      await loadDisasters();
      const runs = await loadRuns();
      if (runs.length) showRun(runs[0].run_id);
    } catch (e) {
      $("versionHint").textContent = "Could not reach the server: " + e.message;
    }
  }

  init();
})();
