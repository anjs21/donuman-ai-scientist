"""OpenAI/Ollama planner backends against a local mock server (no key, no network)."""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import pytest
from warden.planner import CRITIC_SCHEMA, Ollama, OpenAI, api_schema, make_planner, schema_for

KEY = "sk-test-not-a-real-key"


class Mock:
    """Serves queued (status, body) replies and records requests."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                status, reply = mock.replies.pop(0) if mock.replies else (200, chat_reply({"action": "finish", "reason": "r"}))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                if status == 429:
                    self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(json.dumps(reply).encode())

            def log_message(self, *a):
                pass
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"


def chat_reply(decision, finish="stop", refusal=None):
    return {"choices": [{"finish_reason": finish,
                         "message": {"content": json.dumps(decision), "refusal": refusal}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3}}


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)


def test_openai_sends_schema_and_bearer_key_and_counts_usage(key):
    with Mock([(200, chat_reply({"action": "finish", "reason": "done"}))]) as m:
        o = OpenAI({"model": "m", "base_url": m.url})
        assert o.plan({"x": 1})["action"] == "finish"
    r = m.requests[0]
    assert r["path"] == "/v1/chat/completions" and r["auth"] == "Bearer " + KEY
    fmt = r["body"]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["schema"] == api_schema(schema_for())
    assert fmt["json_schema"]["strict"] is False
    assert r["body"]["messages"][0]["role"] == "system"
    assert o.describe()["usage"] == {"calls": 1, "prompt_tokens": 11, "completion_tokens": 3}
    assert KEY not in json.dumps(o.describe())


def test_openai_reply_is_validated_locally(key):
    with Mock([(200, chat_reply({"action": "dance", "reason": "x"}))]) as m:
        with pytest.raises(Exception, match="dance"):
            OpenAI({"model": "m", "base_url": m.url}).plan({})


def test_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("MY_KEY", raising=False)
    with Mock([]) as m:
        with pytest.raises(ValueError, match="MY_KEY"):
            OpenAI({"model": "m", "base_url": m.url, "api_key_env": "MY_KEY"}).plan({})
    assert m.requests == []


def test_rate_limit_is_retried_then_succeeds(key):
    with Mock([(429, {"error": {"message": "slow down"}}),
               (200, chat_reply({"approve": True, "reason": "ok"}))]) as m:
        r = OpenAI({"model": "m", "base_url": m.url, "max_retries": 2}).critique({}, {"evidence": [], "diagnoses": []})
    assert r["approve"] and len(m.requests) == 2


def test_api_errors_surface_message_without_key(key):
    with Mock([(401, {"error": {"message": "Incorrect API key provided"}})]) as m:
        with pytest.raises(RuntimeError) as err:
            OpenAI({"model": "m", "base_url": m.url}).plan({})
    assert "401" in str(err.value) and "Incorrect API key" in str(err.value)
    assert KEY not in str(err.value)


@pytest.mark.parametrize("reply,match", [(chat_reply({}, refusal="no"), "refused"),
                                         (chat_reply({}, finish="length"), "truncated")])
def test_refusal_and_truncation_are_errors(key, reply, match):
    with Mock([(200, reply)]) as m:
        with pytest.raises(RuntimeError, match=match):
            OpenAI({"model": "m", "base_url": m.url}).plan({})


def test_temperature_can_be_omitted_for_models_that_reject_it(key):
    with Mock([(200, chat_reply({"action": "finish", "reason": "r"}))]) as m:
        OpenAI({"model": "m", "base_url": m.url, "temperature": None, "max_completion_tokens": 900}).plan({})
    body = m.requests[0]["body"]
    assert "temperature" not in body and body["max_completion_tokens"] == 900


def test_make_planner_legacy_config_is_ollama_for_both_roles():
    p = make_planner({"host": "http://localhost:11434", "model": "qwen2.5:7b"})
    assert isinstance(p.planner, Ollama) and p.planner is p.critic


def test_make_planner_routes_roles_to_separate_backends(key):
    with Mock([(200, chat_reply({"approve": False, "reason": "unsupported"}))]) as m:
        p = make_planner({"timeout_s": 5,
                          "planner": {"provider": "ollama", "model": "local"},
                          "critic": {"provider": "openai", "model": "remote", "base_url": m.url}})
        assert isinstance(p.planner, Ollama) and isinstance(p.critic, OpenAI)
        assert p.critic.timeout == 5
        review = p.critique({"action": "propose"}, {"evidence": [], "diagnoses": []})
    assert review == {"approve": False, "reason": "unsupported"}
    assert m.requests[0]["body"]["response_format"]["json_schema"]["schema"] == CRITIC_SCHEMA
    assert p.describe()["critic"]["model"] == "remote"


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="provider"):
        make_planner({"provider": "other", "model": "x"})


def test_llm_check_cli_against_mock(key, tmp_path, capsys):
    from warden.__main__ import main
    with Mock([(200, chat_reply({"ok": True, "echo": "WARDEN"})),
               (200, chat_reply({"action": "analogs", "reason": "check"})),
               (200, chat_reply({"approve": False, "reason": "check"}))]) as m:
        cfg = tmp_path / "c.json"
        cfg.write_text(json.dumps({"llm": {"provider": "openai", "model": "m", "base_url": m.url}}))
        assert main(["llm-check", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "planner: openai m" in out and "critic: same backend" in out and KEY not in out
    assert out.count("campaign schema accepted") == 2
    assert m.requests[1]["body"]["response_format"]["json_schema"]["schema"]["type"] == "object"


@pytest.mark.skipif(not (os.environ.get("WARDEN_LIVE_LLM") and os.environ.get("OPENAI_API_KEY")),
                    reason="set WARDEN_LIVE_LLM=1 and OPENAI_API_KEY for one real request")
def test_live_openai_structured_reply():
    from warden.__main__ import configuration
    cfg = configuration(os.environ.get("WARDEN_LLM_CONFIG", "config/v3-openai.json"))["llm"]
    p = make_planner(cfg)
    reply = p.critic.chat("Reply with JSON only.", {"task": "Return ok=true and echo the word WARDEN."},
                          {"type": "object", "properties": {"ok": {"type": "boolean"}, "echo": {"type": "string"}},
                           "required": ["ok", "echo"], "additionalProperties": False})
    assert reply["ok"] is True


def test_ollama_caps_reply_length_and_reports_runaway():
    """Regression: a 7B planner generated for 600 s under the JSON grammar."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            Handler.seen = body
            reply = {"message": {"content": "{\"action\": \"propose\", " + " " * 5000}, "done_reason": "length"}
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps(reply).encode())
        def log_message(self, *a): pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    t = Thread(target=server.serve_forever, daemon=True); t.start()
    try:
        o = Ollama({"host": f"http://127.0.0.1:{server.server_port}", "model": "m", "num_predict": 512})
        with pytest.raises(RuntimeError, match="num_predict=512"):
            o.plan({})
        assert Handler.seen["options"]["num_predict"] == 512
    finally:
        server.shutdown(); server.server_close(); t.join()


def test_openai_gets_root_object_schema_but_branch_rules_still_apply(key):
    """Regression: a root anyOf schema was rejected by OpenAI (400) on every step."""
    with Mock([(200, chat_reply({"action": "propose", "reason": "no candidates"}))]) as m:
        o = OpenAI({"model": "m", "base_url": m.url})
        with pytest.raises(Exception, match="candidates"):
            o.plan({"available_actions": ["propose", "analogs"]})
    sent = m.requests[0]["body"]["response_format"]["json_schema"]["schema"]
    assert sent["type"] == "object" and "anyOf" not in sent
    assert sent["properties"]["action"]["enum"] == ["propose", "analogs"]
    assert sent["required"] == ["action", "reason"]


def test_campaign_stops_on_backend_config_error_and_resumes(key, tmp_path):
    from warden.campaign import run
    from warden.store import Store
    store = Store(tmp_path / "c.sqlite")
    config = {"campaign": {"max_steps": 5, "max_candidates": 2}}
    with Mock([(400, {"error": {"message": "Invalid schema"}})]) as m:
        state = run(store, make_planner({"provider": "openai", "model": "m", "base_url": m.url}),
                    lambda ids, f: None, config)
    assert state["status"] == "stopped_backend_error" and state["step"] == 0 and len(m.requests) == 1
    with Mock([(200, chat_reply({"action": "analogs", "reason": "r"}))] * 5) as m:
        state = run(store, make_planner({"provider": "openai", "model": "m", "base_url": m.url}),
                    lambda ids, f: None, config)
    assert state["status"] != "stopped_backend_error" and state["step"] == 5


def test_missing_ollama_model_is_a_config_error():
    from warden.planner import BackendConfigError

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(404); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps({"error": "model 'qwen2.5:7b' not found"}).encode())
        def log_message(self, *a): pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    t = Thread(target=server.serve_forever, daemon=True); t.start()
    try:
        with pytest.raises(BackendConfigError, match="not found"):
            Ollama({"host": f"http://127.0.0.1:{server.server_port}", "model": "qwen2.5:7b"}).plan({})
    finally:
        server.shutdown(); server.server_close(); t.join()
