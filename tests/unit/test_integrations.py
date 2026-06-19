"""Unit tests for the integrations subsystem (MCP registry + connectors).

These exercise:
  * the registry (register/get/dedupe/config-loading/disabled/unsupported-kind),
  * a fake in-memory connector (no real MCP server),
  * the labeled local filesystem connector (read-only, traversal-guarded),
  * graceful degradation when the optional ``mcp`` SDK is absent,
  * ingestion of connector content into a real MemoryStore as observations.

Embeddings come from the deterministic fake provider in conftest, so no Ollama
or network is required.
"""

from __future__ import annotations

import pytest

from openbird.integrations.mcp import (
    ConnectorConfig,
    FilesystemMCPConnector,
    MCPConnector,
    MCPRegistry,
    MCPResource,
    MCPWriteDisabledError,
    mcp_available,
)
from openbird.memory.store import MemoryStore


# --------------------------------------------------------------------------- #
# A fake connector — no real MCP server needed.
# --------------------------------------------------------------------------- #
class FakeConnector:
    """In-memory read-only connector for tests (no real MCP server)."""

    kind = "fake"

    def __init__(self, name: str, docs: dict[str, str]) -> None:
        self.name = name
        self._docs = docs  # uri -> text

    def list_resources(self) -> list[MCPResource]:
        return [
            MCPResource(uri=uri, name=uri.rsplit("/", 1)[-1], kind=self.kind) for uri in self._docs
        ]

    def read_resource(self, uri: str) -> str:
        return self._docs[uri]

    def write(self, *args: object, **kwargs: object) -> None:
        raise MCPWriteDisabledError("disabled")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def fs_root(tmp_path):
    (tmp_path / "a.txt").write_text("alpha notes about the project", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta meeting summary text", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("gamma nested content here", encoding="utf-8")
    (tmp_path / "ignore.log").write_text("noisy log line", encoding="utf-8")
    return tmp_path


@pytest.fixture
def store(mem_settings, fake_provider):
    s = MemoryStore(db_path=":memory:", settings=mem_settings, provider=fake_provider)
    yield s
    s.close()


# --------------------------------------------------------------------------- #
# Optional-import / graceful degradation
# --------------------------------------------------------------------------- #
def test_mcp_available_is_bool_and_module_imports():
    # The module must import and report availability regardless of whether the
    # optional 'mcp' SDK is installed.
    assert isinstance(mcp_available(), bool)


# --------------------------------------------------------------------------- #
# Registry basics
# --------------------------------------------------------------------------- #
def test_register_and_get():
    reg = MCPRegistry()
    conn = FakeConnector("notes", {"mem://1": "hello"})
    reg.register(conn)
    assert "notes" in reg
    assert len(reg) == 1
    assert reg.get("notes") is conn
    assert reg.names() == ["notes"]


def test_duplicate_registration_rejected():
    reg = MCPRegistry()
    reg.register(FakeConnector("dup", {}))
    with pytest.raises(ValueError):
        reg.register(FakeConnector("dup", {}))


def test_get_missing_raises_keyerror():
    reg = MCPRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_fake_connector_satisfies_protocol():
    assert isinstance(FakeConnector("x", {}), MCPConnector)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def test_from_configs_builds_filesystem_connector(fs_root):
    reg = MCPRegistry.from_configs(
        [ConnectorConfig(name="files", kind="filesystem", root=str(fs_root))]
    )
    assert reg.names() == ["files"]
    conn = reg.get("files")
    assert isinstance(conn, FilesystemMCPConnector)


def test_from_configs_accepts_plain_dicts(fs_root):
    reg = MCPRegistry.from_configs(
        [{"name": "files", "kind": "filesystem", "root": str(fs_root), "junk": 1}]
    )
    assert "files" in reg


def test_from_configs_skips_disabled(fs_root):
    reg = MCPRegistry.from_configs(
        [{"name": "files", "kind": "filesystem", "root": str(fs_root), "enabled": False}]
    )
    assert len(reg) == 0


def test_from_configs_skips_unsupported_kind():
    # An OAuth connector declared in a forward-looking config must not crash the
    # read-only MVP — it is skipped.
    reg = MCPRegistry.from_configs([{"name": "gmail", "kind": "oauth"}])
    assert len(reg) == 0


def test_config_from_dict_requires_name():
    with pytest.raises(ValueError):
        ConnectorConfig.from_dict({"kind": "filesystem"})


def test_filesystem_config_requires_root():
    with pytest.raises(ValueError):
        FilesystemMCPConnector.from_config(ConnectorConfig(name="x", kind="filesystem"))


# --------------------------------------------------------------------------- #
# Filesystem connector
# --------------------------------------------------------------------------- #
def test_filesystem_lists_recursively(fs_root):
    conn = FilesystemMCPConnector("files", fs_root)
    names = sorted(r.name for r in conn.list_resources())
    assert names == ["a.txt", "b.md", "ignore.log", "sub/c.txt"]


def test_filesystem_include_exclude(fs_root):
    conn = FilesystemMCPConnector("files", fs_root, include=["*.txt", "*.md"], exclude=["ignore.*"])
    names = sorted(r.name for r in conn.list_resources())
    assert names == ["a.txt", "b.md", "sub/c.txt"]


def test_filesystem_read_resource_roundtrip(fs_root):
    conn = FilesystemMCPConnector("files", fs_root, include=["a.txt"])
    (res,) = conn.list_resources()
    assert conn.read_resource(res.uri) == "alpha notes about the project"


def test_filesystem_rejects_nonexistent_root(tmp_path):
    with pytest.raises(ValueError):
        FilesystemMCPConnector("files", tmp_path / "does-not-exist")


def test_filesystem_rejects_non_file_uri(fs_root):
    conn = FilesystemMCPConnector("files", fs_root)
    with pytest.raises(ValueError):
        conn.read_resource("http://example.com/x")


def test_filesystem_blocks_path_traversal(fs_root, tmp_path):
    # A file:// URI outside the root must be refused even if it exists.
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("top secret", encoding="utf-8")
    conn = FilesystemMCPConnector("files", fs_root)
    with pytest.raises(ValueError):
        conn.read_resource(outside.as_uri())


def test_read_resource_rejects_excluded_file_via_direct_uri(fs_root):
    # A caller must not be able to read a file that the connector's include/
    # exclude policy hides from list_resources() just by constructing its URI.
    conn = FilesystemMCPConnector("files", fs_root, include=["*.txt", "*.md"], exclude=["ignore.*"])
    listed = {r.name for r in conn.list_resources()}
    assert "ignore.log" not in listed
    excluded_uri = (fs_root / "ignore.log").as_uri()
    with pytest.raises(ValueError):
        conn.read_resource(excluded_uri)


def test_read_resource_rejects_non_included_file_via_direct_uri(fs_root):
    # include is an allowlist: a file not matching include must be unreadable
    # directly, even though it exists in the root.
    conn = FilesystemMCPConnector("files", fs_root, include=["a.txt"])
    with pytest.raises(ValueError):
        conn.read_resource((fs_root / "b.md").as_uri())


def test_read_resource_rejects_oversized_file_via_direct_uri(fs_root):
    # A file larger than max_file_bytes must be refused on direct read, just as
    # _iter_files()/list_resources() would skip it.
    big = fs_root / "big.txt"
    big.write_text("x" * 5000, encoding="utf-8")
    conn = FilesystemMCPConnector("files", fs_root, max_file_bytes=100)
    with pytest.raises(ValueError):
        conn.read_resource(big.as_uri())


def test_read_resource_within_size_limit_succeeds(fs_root):
    conn = FilesystemMCPConnector("files", fs_root, max_file_bytes=1000)
    text = conn.read_resource((fs_root / "a.txt").as_uri())
    assert text == "alpha notes about the project"


def test_read_resource_rejects_symlinked_file_component(fs_root, tmp_path):
    # A symlink *inside* the root pointing outside it must not be followed:
    # even though _relative_in_root() lexically passes, O_NOFOLLOW at open time
    # rejects the symlinked component (closing the TOCTOU/symlink-swap hole).
    secret = tmp_path.parent / "outside_secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    link = fs_root / "link.txt"
    link.symlink_to(secret)
    conn = FilesystemMCPConnector("files", fs_root)
    with pytest.raises(ValueError):
        conn.read_resource(link.as_uri())


def test_read_resource_rejects_symlinked_directory_component(fs_root, tmp_path):
    # A symlinked *directory* component must also be rejected, not traversed.
    outside_dir = tmp_path.parent / "outside_dir"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "leak.txt").write_text("leaked", encoding="utf-8")
    dlink = fs_root / "dlink"
    if not dlink.exists():
        dlink.symlink_to(outside_dir, target_is_directory=True)
    conn = FilesystemMCPConnector("files", fs_root)
    with pytest.raises(ValueError):
        conn.read_resource((dlink / "leak.txt").as_uri())


def test_read_resource_rejects_directory_uri(fs_root):
    conn = FilesystemMCPConnector("files", fs_root)
    with pytest.raises((IsADirectoryError, ValueError)):
        conn.read_resource((fs_root / "sub").as_uri())


def test_write_disabled_on_filesystem(fs_root):
    conn = FilesystemMCPConnector("files", fs_root)
    with pytest.raises(MCPWriteDisabledError):
        conn.write("a.txt", "data")


def test_filesystem_label_is_present():
    assert "MVP" in FilesystemMCPConnector.LABEL


# --------------------------------------------------------------------------- #
# Ingestion into the memory store
# --------------------------------------------------------------------------- #
def test_ingest_filesystem_into_store(fs_root, store):
    reg = MCPRegistry.from_configs(
        [
            {
                "name": "files",
                "kind": "filesystem",
                "root": str(fs_root),
                "include": ["*.txt", "*.md"],
                "exclude": ["ignore.*"],
            }
        ]
    )
    created = reg.ingest(store)
    assert created == 3  # a.txt, b.md, sub/c.txt
    stats = store.stats()
    assert stats["observations"] == 3
    assert stats["blobs"] == 3


def test_ingest_tags_source_and_provenance(store):
    reg = MCPRegistry()
    reg.register(FakeConnector("notes", {"mem://doc1": "unique content for hit"}))
    created = reg.ingest(store)
    assert created == 1
    hits = store.search("unique content", k=5, semantic=False)
    assert hits
    obs = hits[0].observation
    assert obs is not None
    assert obs.source == "mcp"
    assert obs.app == "notes"
    assert obs.url == "mem://doc1"


def test_ingest_skips_empty_resources(store):
    reg = MCPRegistry()
    reg.register(FakeConnector("notes", {"mem://blank": "   \n  "}))
    assert reg.ingest(store) == 0
    assert store.stats()["observations"] == 0


def test_ingest_can_target_subset(store):
    reg = MCPRegistry()
    reg.register(FakeConnector("a", {"mem://a": "content a"}))
    reg.register(FakeConnector("b", {"mem://b": "content b"}))
    created = reg.ingest(store, names=["a"])
    assert created == 1
    assert store.stats()["observations"] == 1
