"""Gemini-vision tools: image pairing, parsing, caching, failure behaviour, and how they
plug into Agent 1. The model itself is replaced by a fake `generate` function."""
from __future__ import annotations

import json
import struct
import zlib

import pytest

from agents.orchestrator.pipeline import tile_summary
from agents.perception.agent import PerceptionAgent
from agents.perception.schemas import TileStatus
from tools.vision.client import (
    GeminiAccessTool,
    GeminiDamageTool,
    load_image,
    make_vertex_generate_fn,
    parse_reading,
    pre_path_for,
)
from tools.vision.prompts import ACCESS_LABELS, DAMAGE_LABELS, DAMAGE_PROMPT, SECOND_PASS_PROMPT


def make_png(shade: int = 128) -> bytes:
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes([0, shade, shade, shade]))) + chunk(b"IEND", b""))


@pytest.fixture
def demo(tmp_path):
    d = tmp_path / "demo" / "storm"
    (d / "images").mkdir(parents=True)
    (d / "pre").mkdir()
    post = d / "images" / "storm_00000001_post_disaster.png"
    pre = d / "pre" / "storm_00000001_pre_disaster.png"
    post.write_bytes(make_png(200))
    pre.write_bytes(make_png(50))
    return post, pre


class FakeModel:
    """Stand-in for Gemini: records every call, returns scripted JSON."""

    def __init__(self, reply=None):
        self.calls = []
        self.reply = reply or {"label": "major-damage", "confidence": 0.82, "evidence": "flooded streets"}

    def __call__(self, images, prompt, schema):
        self.calls.append({"images": len(images), "prompt": prompt, "schema": schema})
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


def damage_tool(model, tmp_path, **kw):
    return GeminiDamageTool(generate_fn=model, cache_dir=tmp_path / "cache", **kw)


# --- finding the before image ------------------------------------------------------------


def test_pre_path_demo_layout(demo):
    post, pre = demo
    assert pre_path_for(post) == pre


def test_pre_path_native_xbd_layout(tmp_path):
    post = tmp_path / "x_00000001_post_disaster.png"
    pre = tmp_path / "x_00000001_pre_disaster.png"
    post.write_bytes(make_png()); pre.write_bytes(make_png())
    assert pre_path_for(post) == pre


def test_pre_path_missing_or_not_a_post_tile(tmp_path):
    lonely = tmp_path / "images" / "x_00000001_post_disaster.png"
    lonely.parent.mkdir(); lonely.write_bytes(make_png())
    assert pre_path_for(lonely) is None
    assert pre_path_for(tmp_path / "random_name.png") is None


# --- images ------------------------------------------------------------------------------


def test_large_images_are_shrunk_before_sending(tmp_path):
    try:
        from PIL import Image as PIL
    except ImportError:
        pytest.skip("Pillow not installed")
    big = tmp_path / "big.png"
    PIL.new("RGB", (2000, 1000), (10, 120, 30)).save(big)
    data, mime = load_image(big, max_side=768)
    from io import BytesIO
    assert mime == "image/jpeg" and max(PIL.open(BytesIO(data)).size) == 768


def test_without_pillow_the_raw_bytes_are_sent(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                        lambda n, *a, **k: (_ for _ in ()).throw(ImportError()) if n == "PIL" else real_import(n, *a, **k))
    p = tmp_path / "t.png"; p.write_bytes(make_png())
    data, mime = load_image(p)
    assert data == make_png() and mime == "image/png"


# --- parsing -----------------------------------------------------------------------------


def test_parse_valid_and_fenced_json():
    assert parse_reading('{"label": "destroyed", "confidence": 0.9, "evidence": "collapsed"}', DAMAGE_LABELS) == ("destroyed", 0.9, "collapsed")
    fenced = '```json\n{"label": "no-damage", "confidence": 0.7, "evidence": ""}\n```'
    assert parse_reading(fenced, DAMAGE_LABELS)[0] == "no-damage"


@pytest.mark.parametrize("bad", ["", "   ", "not json", "[1, 2]", '{"label": "catastrophic", "confidence": 0.9}',
                                 '{"confidence": 0.9}', '{"label": "access-clear", "confidence": 1}'])
def test_unusable_answers_raise_value_error(bad):
    with pytest.raises(ValueError):
        parse_reading(bad, DAMAGE_LABELS)


def test_confidence_is_clamped_and_defaulted():
    assert parse_reading('{"label": "destroyed", "confidence": 7}', DAMAGE_LABELS)[1] == 1.0
    assert parse_reading('{"label": "destroyed", "confidence": -2}', DAMAGE_LABELS)[1] == 0.0
    assert parse_reading('{"label": "destroyed"}', DAMAGE_LABELS)[1] == 0.5
    assert parse_reading('{"label": "destroyed", "confidence": "high"}', DAMAGE_LABELS)[1] == 0.5


def test_evidence_is_trimmed():
    assert len(parse_reading(json.dumps({"label": "destroyed", "confidence": 1, "evidence": "x" * 5000}), DAMAGE_LABELS)[2]) == 400


# --- the tools ---------------------------------------------------------------------------


def test_damage_tool_sends_before_and_after_and_returns_a_normal_tool_result(demo, tmp_path):
    post, _ = demo
    model = FakeModel()
    result = damage_tool(model, tmp_path).run(str(post))

    assert model.calls[0]["images"] == 2 and model.calls[0]["prompt"] == DAMAGE_PROMPT
    assert (result.tool_name, result.label, result.confidence) == ("classifier", "major-damage", 0.82)
    assert result.raw_output["evidence"] == "flooded streets" and result.raw_output["has_before"] is True


def test_single_image_is_sent_when_there_is_no_before_image(tmp_path):
    lonely = tmp_path / "images" / "x_00000001_post_disaster.png"
    lonely.parent.mkdir(); lonely.write_bytes(make_png())
    model = FakeModel()
    result = damage_tool(model, tmp_path).run(str(lonely))
    assert model.calls[0]["images"] == 1 and result.raw_output["has_before"] is False


def test_identical_inputs_are_served_from_cache_without_calling_the_model(demo, tmp_path):
    post, _ = demo
    model = FakeModel()
    tool = damage_tool(model, tmp_path)
    first, second = tool.run(str(post)), tool.run(str(post))

    assert len(model.calls) == 1
    assert (second.label, second.confidence) == (first.label, first.confidence)
    assert first.raw_output["cached"] is False and second.raw_output["cached"] is True
    other_tool_instance = damage_tool(FakeModel(), tmp_path)        # a fresh process sharing the same cache dir
    assert other_tool_instance.run(str(post)).raw_output["cached"] is True


def test_cache_is_keyed_on_image_content_tool_and_prompt(demo, tmp_path):
    post, pre = demo
    model = FakeModel()
    tool = damage_tool(model, tmp_path)
    tool.run(str(post))
    post.write_bytes(make_png(77))                                   # the image changed
    tool.run(str(post))
    assert len(model.calls) == 2

    second = damage_tool(model, tmp_path, name="sam", second_pass=True)   # different tool + prompt
    second.run(str(post))
    assert len(model.calls) == 3 and model.calls[-1]["prompt"] == SECOND_PASS_PROMPT


def test_editing_the_prompt_prompt_version_or_model_invalidates_cached_answers(demo, tmp_path, monkeypatch):
    """Old answers must never be served for a new prompt/model, even with the same tool name and images."""
    import tools.vision.client as vc

    post, _ = demo
    model = FakeModel()
    tool = damage_tool(model, tmp_path)
    tool.run(str(post))
    assert len(model.calls) == 1

    tool.prompt = tool.prompt + " (edited)"                  # same name, same images, new prompt
    tool.run(str(post))
    assert len(model.calls) == 2

    monkeypatch.setattr(vc, "PROMPT_VERSION", "v-next")      # the maintainer bumped the version
    tool.run(str(post))
    assert len(model.calls) == 3

    tool.model = "gemini-other"                              # different model
    tool.run(str(post))
    assert len(model.calls) == 4


def test_bad_model_answers_raise_and_are_never_cached(demo, tmp_path):
    post, _ = demo
    model = FakeModel(reply='{"label": "apocalyptic", "confidence": 1}')
    tool = damage_tool(model, tmp_path)
    with pytest.raises(ValueError):
        tool.run(str(post))
    assert not list((tmp_path / "cache").glob("*.json")) if (tmp_path / "cache").exists() else True
    model.reply = {"label": "minor-damage", "confidence": 0.6, "evidence": "ok"}
    assert tool.run(str(post)).label == "minor-damage"               # next try is a real call


def test_a_corrupt_cache_entry_is_discarded_and_refetched(demo, tmp_path):
    post, _ = demo
    model = FakeModel()
    tool = damage_tool(model, tmp_path)
    tool.run(str(post))
    for f in (tmp_path / "cache").glob("*.json"):
        f.write_text("{ not json")
    result = tool.run(str(post))
    assert len(model.calls) == 2 and result.label == "major-damage"


def test_model_exceptions_propagate_so_agent_fault_handling_applies(demo, tmp_path):
    post, _ = demo

    def boom(images, prompt, schema):
        raise ConnectionError("429 quota")

    with pytest.raises(ConnectionError):
        damage_tool(boom, tmp_path).run(str(post))


def test_access_tool_is_not_a_damage_tool_and_uses_access_labels(demo, tmp_path):
    post, _ = demo
    model = FakeModel({"label": "access-blocked", "confidence": 0.7, "evidence": "flooded main road"})
    tool = GeminiAccessTool(generate_fn=model, cache_dir=tmp_path / "cache")
    result = tool.run(str(post))
    assert tool.provides_damage is False and GeminiDamageTool.provides_damage is True
    assert (tool.name, result.label) == ("detector", "access-blocked") and tool.labels == ACCESS_LABELS
    model.reply = {"label": "destroyed", "confidence": 1, "evidence": ""}      # a damage label is invalid here
    post.write_bytes(make_png(9))
    with pytest.raises(ValueError):
        tool.run(str(post))


# --- inside Agent 1 -------------------------------------------------------------------------


def agent_with(model, tmp_path):
    tools = [GeminiDamageTool(generate_fn=model, cache_dir=tmp_path / "c"),
             GeminiAccessTool(generate_fn=model, cache_dir=tmp_path / "c")]
    return PerceptionAgent(tools=tools, run_auxiliary_tools=True, sleep=lambda s: None)


def test_evidence_flows_into_the_tile_record_and_dashboard_event(demo, tmp_path):
    post, _ = demo

    class Router(FakeModel):
        def __call__(self, images, prompt, schema):
            self.calls.append(1)
            if "ROAD ACCESS" in prompt:
                return json.dumps({"label": "access-degraded", "confidence": 0.7, "evidence": "debris on road"})
            return json.dumps({"label": "destroyed", "confidence": 0.9, "evidence": "roofs collapsed"})

    tile = agent_with(Router(), tmp_path).process_tile("storm_00000001_post_disaster", str(post))
    assert tile.status == TileStatus.ASSESSED and tile.damage_severity == "destroyed"
    assert [r.tool_name for r in tile.tool_call_history] == ["classifier", "detector"]
    assert tile.tool_call_history[0].evidence == "roofs collapsed"
    summary = tile_summary(tile)
    assert summary["tool_calls"][1]["evidence"] == "debris on road"


def test_flaky_model_is_retried_by_the_agent(demo, tmp_path):
    post, _ = demo
    state = {"n": 0}

    def flaky(images, prompt, schema):
        state["n"] += 1
        if state["n"] < 3:
            raise ConnectionError("503")
        return json.dumps({"label": "minor-damage", "confidence": 0.9, "evidence": "ok"})

    agent = PerceptionAgent(tools=[GeminiDamageTool(generate_fn=flaky, cache_dir=tmp_path / "c")], sleep=lambda s: None)
    tile = agent.process_tile("t", str(post))
    assert tile.status == TileStatus.ASSESSED and len(tile.tool_errors) == 2


# --- the real Vertex call (only its wiring; no network) -------------------------------------


def test_vertex_generate_fn_builds_the_request_correctly(monkeypatch):
    genai = pytest.importorskip("google.genai")
    seen = {}

    class FakeResponse:
        text = '{"label": "no-damage", "confidence": 1, "evidence": ""}'

    class FakeModels:
        def generate_content(self, model, contents, config):
            seen.update(model=model, contents=contents, config=config)
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kw):
            seen["client_kwargs"] = kw
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    gen = make_vertex_generate_fn(project="proj", location="us-central1", model="gemini-2.5-flash")
    out = gen([(b"AAA", "image/jpeg"), (b"BBB", "image/jpeg")], "the prompt", __import__("tools.vision.client", fromlist=["DamageReading"]).DamageReading)

    assert out == FakeResponse.text
    assert seen["client_kwargs"] == {"vertexai": True, "project": "proj", "location": "us-central1"}
    assert seen["model"] == "gemini-2.5-flash"
    assert len(seen["contents"]) == 3 and seen["contents"][-1] == "the prompt"        # 2 images + prompt
    assert seen["config"].response_mime_type == "application/json"
    assert seen["config"].temperature == 0.0
    assert seen["config"].thinking_config.thinking_budget == 0
