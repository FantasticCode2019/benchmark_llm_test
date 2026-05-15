"""Smoke tests for the Ollama descriptor helpers in
``llm_bench.clients.ollama_client``. Pure functions only — each test pins
one branch of the jq projection used by ``ollama_describe_model``.
"""
from __future__ import annotations

from llm_bench.clients.ollama_client import (
    _build_ollama_descriptor,
    _find_ollama_entry,
    _ollama_first_name,
    _ollama_max_context,
    _ollama_processor_split,
    ollama_discover_model_id,
    ollama_supports_thinking,
)


class TestProcessorSplit:
    def test_not_loaded_when_running_is_none(self) -> None:
        assert _ollama_processor_split(None) == "not loaded"

    def test_full_gpu_when_size_equals_size_vram(self) -> None:
        assert (
            _ollama_processor_split({"size": 1024, "size_vram": 1024})
            == "100% GPU"
        )

    def test_full_cpu_when_size_vram_is_zero(self) -> None:
        assert (
            _ollama_processor_split({"size": 1024, "size_vram": 0})
            == "100% CPU"
        )

    def test_split_string_when_mixed(self) -> None:
        # 60% GPU / 40% CPU split.
        assert (
            _ollama_processor_split({"size": 1000, "size_vram": 600})
            == "60% GPU / 40% CPU"
        )


class TestMaxContext:
    def test_picks_first_arch_prefixed_key(self) -> None:
        show = {
            "model_info": {
                "general.parameter_count": 26_000_000_000,
                "qwen3.context_length": 131_072,
                "qwen3.attention.head_count": 32,
            },
        }
        assert _ollama_max_context(show) == 131_072

    def test_returns_none_when_missing(self) -> None:
        assert _ollama_max_context({}) is None
        assert _ollama_max_context({"model_info": {"unrelated": 1}}) is None


class TestFirstName:
    def test_prefers_first_non_empty_source(self) -> None:
        ps = [{"name": "alpha:7b"}]
        tags = [{"name": "beta:13b"}]
        # /api/ps wins when populated.
        assert _ollama_first_name(ps, tags) == "alpha:7b"
        # Falls back to /api/tags when /api/ps empty.
        assert _ollama_first_name([], tags) == "beta:13b"

    def test_returns_none_when_all_empty(self) -> None:
        assert _ollama_first_name([], []) is None


class TestFindOllamaEntry:
    def test_matches_on_name_or_model(self) -> None:
        models = [
            {"name": "gemma3:12b", "size": 100},
            {"model": "qwen3:14b", "size": 200},
        ]
        assert _find_ollama_entry(models, "gemma3:12b") == models[0]
        assert _find_ollama_entry(models, "qwen3:14b") == models[1]

    def test_returns_none_when_missing(self) -> None:
        assert _find_ollama_entry([], "x") is None
        assert _find_ollama_entry(None, "x") is None


class TestBuildDescriptor:
    def test_fields_match_jq_projection(self) -> None:
        # 16 GiB on disk, 20 GiB resident, 20 GiB VRAM (= full GPU),
        # so kvcache_gb = 20 - 16 = 4, ram_gb = 20 - 20 = 0.
        gib = 1024 ** 3
        running = {
            "name": "demo:13b",
            "size": 20 * gib,
            "size_vram": 20 * gib,
            "context_length": 131072,
        }
        ondisk = {"name": "demo:13b", "size": 16 * gib}
        show = {
            "details": {
                "family": "demo",
                "parameter_size": "13B",
                "quantization_level": "Q4_K_M",
            },
            "model_info": {"demo.context_length": 262144},
        }
        out = _build_ollama_descriptor("demo:13b", show, running, ondisk)
        assert out["model"] == "demo:13b"
        assert out["family"] == "demo"
        assert out["parameter_size"] == "13B"
        assert out["quantization"] == "Q4_K_M"
        assert out["max_context"] == 262144
        assert out["runtime_context"] == 131072
        assert out["disk_gb"] == 16.0
        assert out["total_gb"] == 20.0
        assert out["vram_gb"] == 20.0
        assert out["ram_gb"] == 0.0
        assert out["kvcache_gb"] == 4.0
        assert out["processor"] == "100% GPU"
        assert out["loaded"] is True

    def test_descriptor_when_not_loaded(self) -> None:
        out = _build_ollama_descriptor("demo:13b", {}, None, None)
        assert out["loaded"] is False
        assert out["processor"] == "not loaded"
        assert out["total_gb"] == 0.0


class TestOllamaSupportsThinking:
    """Pin the contract that the probe queries the CALLER-supplied
    model — never auto-picks from /api/ps / /api/tags.

    The earlier "guess the model from the daemon's first hit" behaviour
    silently flipped True / False between runs whenever the daemon
    hosted more than one model: /api/ps was empty before any prompt
    landed (so we fell back to /api/tags) and the tag ordering is
    implementation-defined. These tests would have caught that.
    """

    def test_uses_caller_supplied_model_not_daemon_first_hit(
            self, monkeypatch) -> None:
        # If the helper accidentally auto-discovers, this asserts the
        # discovered name would have been "other:7b" (NOT thinking).
        # Passing the caller's "target:1b" must short-circuit that
        # lookup and probe the configured name directly.
        calls: list[str] = []

        def fake_show(_base, model, *, timeout):
            calls.append(model)
            if model == "target:1b":
                return {"capabilities": ["completion", "thinking"]}
            return {"capabilities": ["completion"]}

        monkeypatch.setattr(
            "llm_bench.clients.ollama_client._ollama_show", fake_show)
        # Sentinel: any call to ps/tags helpers would prove the
        # auto-discovery regression came back.
        def fail_listing(*args, **kw):
            raise AssertionError(
                "supports_thinking must NOT call /api/ps or /api/tags")
        monkeypatch.setattr(
            "llm_bench.clients.ollama_client._ollama_get_models_list",
            fail_listing)

        assert ollama_supports_thinking(
            "http://ollama.local", "target:1b") is True
        assert calls == ["target:1b"]

    def test_returns_false_when_model_unknown_to_daemon(
            self, monkeypatch) -> None:
        # /api/show returns {} for an unknown model name (the
        # underlying helper swallows the 4xx and yields an empty
        # dict). The probe must collapse to False rather than raise.
        monkeypatch.setattr(
            "llm_bench.clients.ollama_client._ollama_show",
            lambda *a, **kw: {})
        assert ollama_supports_thinking(
            "http://ollama.local", "no-such:1b") is False

    def test_returns_false_when_capabilities_missing(
            self, monkeypatch) -> None:
        monkeypatch.setattr(
            "llm_bench.clients.ollama_client._ollama_show",
            lambda *a, **kw: {"some_other_field": 1})
        assert ollama_supports_thinking(
            "http://ollama.local", "x:1b") is False

    def test_returns_false_when_model_empty(self) -> None:
        # Defensive: an empty model name short-circuits before any
        # HTTP call -- a caller passing "" is a config bug, not a
        # reason to crash.
        assert ollama_supports_thinking("http://ollama.local", "") is False


class TestOllamaDiscoverModelId:
    """``/v1/models`` discovery — recovers the daemon's canonical id
    even when the operator-supplied ``model_name`` has trailing
    annotations like ``(Unsloth GGUF)`` that the API won't accept.
    """

    def test_returns_first_data_entry_id(self, monkeypatch) -> None:
        body = (
            '{"object":"list","data":[{"created":1778761066,'
            '"id":"gemma4:26b-a4b-it-ud-q4_K_XL","object":"model",'
            '"owned_by":"library"}]}'
        )
        monkeypatch.setattr(
            "llm_bench.clients.ollama_client.http_get_status",
            lambda *a, **kw: (200, body))
        assert (ollama_discover_model_id("http://ollama.local")
                == "gemma4:26b-a4b-it-ud-q4_K_XL")

    def test_skips_non_dict_entries_until_first_string_id(
            self, monkeypatch) -> None:
        body = (
            '{"data":['
            'null,'
            '{"id":""},'         # empty id, skip
            '{"id":42},'          # non-string id, skip
            '{"id":"chosen:1b"}'
            ']}'
        )
        monkeypatch.setattr(
            "llm_bench.clients.ollama_client.http_get_status",
            lambda *a, **kw: (200, body))
        assert (ollama_discover_model_id("http://ollama.local")
                == "chosen:1b")

    def test_returns_none_on_non_2xx(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "llm_bench.clients.ollama_client.http_get_status",
            lambda *a, **kw: (503, "service unavailable"))
        assert ollama_discover_model_id("http://ollama.local") is None

    def test_returns_none_on_bad_json(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "llm_bench.clients.ollama_client.http_get_status",
            lambda *a, **kw: (200, "<<not json>>"))
        assert ollama_discover_model_id("http://ollama.local") is None

    def test_returns_none_when_data_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "llm_bench.clients.ollama_client.http_get_status",
            lambda *a, **kw: (200, '{"object":"list"}'))
        assert ollama_discover_model_id("http://ollama.local") is None

    def test_returns_none_when_data_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "llm_bench.clients.ollama_client.http_get_status",
            lambda *a, **kw: (200, '{"data":[]}'))
        assert ollama_discover_model_id("http://ollama.local") is None
