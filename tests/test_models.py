"""fetch-model: fixed registry, license gate, atomic download, provenance."""
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import pytest
from warden import models

PAYLOAD = b"x" * (2 << 20)


@pytest.fixture
def server(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = PAYLOAD if self.path == "/ok.model" else PAYLOAD[:10]
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a): pass
    s = HTTPServer(("127.0.0.1", 0), Handler)
    t = Thread(target=s.serve_forever, daemon=True); t.start()
    url = f"http://127.0.0.1:{s.server_port}"
    monkeypatch.setitem(models.REGISTRY, "test-asl", {"url": url + "/ok.model", "license": "ASL", "theory": "t", "min_mace": "0"})
    monkeypatch.setitem(models.REGISTRY, "test-tiny", {"url": url + "/tiny.model", "license": "MIT", "theory": "t", "min_mace": "0"})
    yield
    s.shutdown(); s.server_close(); t.join()


def test_registry_is_official_github_only():
    assert all(e["url"].startswith("https://github.com/ACEsuit/") for e in models.REGISTRY.values())


def test_unknown_name_and_unaccepted_license_refused(tmp_path, server):
    with pytest.raises(ValueError, match="Unknown"):
        models.fetch("https://evil.example/x.model", tmp_path / "m")
    with pytest.raises(ValueError, match="accept-license"):
        models.fetch("test-asl", tmp_path / "m")
    assert not (tmp_path / "m").exists()


def test_download_writes_file_and_provenance_and_refuses_overwrite(tmp_path, server):
    out = tmp_path / "models" / "mace.model"
    r = models.fetch("test-asl", out, accept_license=True, log=lambda *a: None)
    assert out.read_bytes() == PAYLOAD and r["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    prov = models.provenance(out)
    assert prov["matches_file"] and prov["license_accepted"] is True and prov["name"] == "test-asl"
    with pytest.raises(ValueError, match="--force"):
        models.fetch("test-asl", out, accept_license=True)
    out.write_bytes(b"changed")
    assert models.provenance(out)["matches_file"] is False


def test_truncated_or_tiny_download_leaves_nothing(tmp_path, server):
    with pytest.raises(IOError):
        models.fetch("test-tiny", tmp_path / "m", log=lambda *a: None)
    assert list(tmp_path.iterdir()) == []
