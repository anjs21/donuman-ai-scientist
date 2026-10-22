"""Literature can be retrieved and imported without live screening prerequisites."""
import json
import subprocess
import sys
from pathlib import Path

from warden.store import Store


ROOT = Path(__file__).resolve().parents[1]


def test_bond_call_records_importable_evidence_and_protects_outputs(tmp_path):
    server = tmp_path / 'server.py'
    server.write_text('''from mcp.server.fastmcp import FastMCP
mcp = FastMCP("literature-test")
@mcp.tool()
def search(query: str, source: str) -> str:
    return "MOCK passage: " + query + " from " + source
if __name__ == "__main__":
    mcp.run(transport="stdio")
''')
    config = tmp_path / 'bond.json'
    config.write_text(json.dumps({'transport': 'stdio', 'command': sys.executable,
                                 'args': [str(server)], 'allowed_tools': ['search']}))
    arguments = tmp_path / 'arguments.json'
    arguments.write_text(json.dumps({'query': 'amine versus silanol', 'source': 'chemrxiv'}))
    output = tmp_path / 'evidence.json'

    def call(tool='search'):
        return subprocess.run([sys.executable, '-m', 'warden', 'bond-call',
                               '--config', str(config), '--tool', tool,
                               '--arguments', str(arguments), '--out', str(output)],
                              cwd=ROOT, capture_output=True, text=True, timeout=30)

    rejected = call('unapproved')
    assert rejected.returncode == 2 and not output.exists()
    arguments.write_text('[]')
    rejected = call()
    assert rejected.returncode == 2 and 'JSON object' in rejected.stderr
    arguments.write_text(json.dumps({'query': 42, 'source': 'chemrxiv'}))
    assert call().returncode == 2 and not output.exists()
    arguments.write_text(json.dumps({'query': 'amine versus silanol', 'source': 'chemrxiv'}))
    result = call()
    assert result.returncode == 0, result.stderr
    original = output.read_bytes()
    row, = json.loads(original)
    assert row['kind'] == 'bond_result' and 'MOCK passage' in row['text']
    assert row['source']['arguments']['source'] == 'chemrxiv'
    assert row['source']['tool'] == 'search' and row['source']['retrieved_at']
    assert call().returncode == 2
    assert output.read_bytes() == original

    db = tmp_path / 'campaign.sqlite'
    result = subprocess.run([sys.executable, '-m', 'warden', 'add-evidence',
                             '--db', str(db), '--file', str(output)],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    store = Store(db)
    try:
        evidence, = store.all('evidence')
        assert evidence['text'] == row['text']
        assert evidence['source'] == row['source']
    finally:
        store.close()
