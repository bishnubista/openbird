"""MCP (Model Context Protocol) client registry and connectors.

MVP scope
---------
* Only a **local FILESYSTEM MCP connector** is provided, and it is clearly
  labeled as such (``kind == "filesystem"``, ``LABEL`` attribute).
* **Write actions are disabled.** Connectors expose read-only resource
  enumeration / reading. Any attempt to mutate a remote resource raises
  :class:`MCPWriteDisabledError`. (Real OAuth read connectors and any write
  capability are fast-follows gated behind auth + confirmation + audit.)
* The real ``mcp`` Python SDK is an **optional extra**. Import is attempted
  lazily; if it is missing, this module still imports and the filesystem
  connector still works (it talks to the local filesystem directly, not to a
  spawned MCP server). :func:`mcp_available` reports availability.

Retrieved file content is treated as **untrusted data** when it later reaches
the LLM; this module only ingests it as observations tagged with
``source="mcp"`` and never executes it.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openbird.memory.store import MemoryStore


def _probe_mcp() -> bool:
    """Return True iff the optional ``mcp`` SDK can be imported.

    Importing ``mcp`` is best-effort and never fatal: the filesystem connector
    does not require it. We probe with :mod:`importlib.util` so we don't pay the
    import cost or pollute the module namespace.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("mcp") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


#: Whether the optional ``mcp`` SDK is importable in this environment.
MCP_AVAILABLE: bool = _probe_mcp()


def mcp_available() -> bool:
    """Return whether the optional ``mcp`` SDK is installed.

    Re-probes each call so the answer stays correct if the extra is installed
    into a long-running process.
    """
    return _probe_mcp()


class MCPWriteDisabledError(RuntimeError):
    """Raised when a write/mutation is attempted through an MCP connector.

    All write actions are disabled in the MVP until OAuth scopes, a confirmation
    UX, and an audit log exist.
    """


@dataclass(frozen=True)
class MCPResource:
    """A read-only resource exposed by a connector.

    Attributes:
        uri: Stable identifier for the resource (e.g. a ``file://`` URI).
        name: Human-readable label (e.g. a relative path).
        kind: The owning connector's kind (e.g. ``"filesystem"``).
    """

    uri: str
    name: str
    kind: str


@dataclass
class ConnectorConfig:
    """Declarative configuration for a single MCP connector.

    Attributes:
        name: Unique registry key for this connector.
        kind: Connector type. Only ``"filesystem"`` is supported in the MVP.
        root: Root directory for a filesystem connector.
        include: Glob patterns (matched against the file name) to include.
            Empty means "all files".
        exclude: Glob patterns to exclude (checked after ``include``).
        max_file_bytes: Skip files larger than this (avoid pulling huge blobs).
        enabled: When False the registry skips this connector.
    """

    name: str
    kind: str = "filesystem"
    root: str | None = None
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    max_file_bytes: int = 1_000_000
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "ConnectorConfig":
        """Build a config from a plain dict (e.g. parsed JSON/TOML).

        Unknown keys are ignored so config files can carry forward-compatible
        fields without breaking older readers.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        if "name" not in filtered:
            raise ValueError("connector config requires a 'name'")
        return cls(**filtered)


@runtime_checkable
class MCPConnector(Protocol):
    """Read-only MCP connector interface.

    Concrete connectors enumerate resources and read their text. They never
    mutate the remote source: :meth:`write` must raise
    :class:`MCPWriteDisabledError`.
    """

    name: str
    kind: str

    def list_resources(self) -> list[MCPResource]:
        """Return the resources currently visible to this connector."""
        ...

    def read_resource(self, uri: str) -> str:
        """Return the text content of a single resource by URI."""
        ...

    def write(self, *args: object, **kwargs: object) -> None:
        """Always raises; write actions are disabled in the MVP."""
        ...


class FilesystemMCPConnector:
    """Local filesystem MCP connector (MVP).

    LABELED MVP: this connector reads files from a local directory tree. It is
    the only connector shipped in the first pass and is intentionally simple
    (it does not spawn a real MCP server process). It exists to exercise the
    registry + ingestion path end-to-end with no external dependencies.

    Write actions are disabled: :meth:`write` raises
    :class:`MCPWriteDisabledError`.
    """

    #: Human-facing label, surfaced in UIs/logs to make the MVP scope explicit.
    LABEL = "Local Filesystem MCP (MVP, read-only)"
    kind = "filesystem"

    def __init__(
        self,
        name: str,
        root: str | os.PathLike[str],
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        max_file_bytes: int = 1_000_000,
    ) -> None:
        """Create a filesystem connector rooted at ``root``.

        Args:
            name: Registry key for this connector.
            root: Directory whose files become readable resources.
            include: Glob patterns (vs the file name); empty means all files.
            exclude: Glob patterns to skip (checked after ``include``).
            max_file_bytes: Files larger than this are skipped.

        Raises:
            ValueError: If ``root`` does not point at an existing directory.
        """
        self.name = name
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"filesystem connector root is not a directory: {self.root}")
        self.include = list(include or [])
        self.exclude = list(exclude or [])
        self.max_file_bytes = max_file_bytes

    @classmethod
    def from_config(cls, config: ConnectorConfig) -> "FilesystemMCPConnector":
        """Build a connector from a :class:`ConnectorConfig`.

        Raises:
            ValueError: If the config kind is not ``"filesystem"`` or no root
                is given.
        """
        if config.kind != "filesystem":
            raise ValueError(f"FilesystemMCPConnector cannot serve kind={config.kind!r}")
        if not config.root:
            raise ValueError(f"connector {config.name!r} requires a 'root'")
        return cls(
            config.name,
            config.root,
            include=config.include,
            exclude=config.exclude,
            max_file_bytes=config.max_file_bytes,
        )

    # -- read paths -----------------------------------------------------------

    def _matches(self, name: str) -> bool:
        """Return whether a file name passes the include/exclude globs."""
        if self.include and not any(fnmatch.fnmatch(name, p) for p in self.include):
            return False
        if any(fnmatch.fnmatch(name, p) for p in self.exclude):
            return False
        return True

    def _iter_files(self) -> Iterator[Path]:
        """Yield matching, in-tree, regular files under the root."""
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if not self._matches(path.name):
                continue
            yield path

    def _relative_in_root(self, path: Path) -> Path:
        """Return ``path``'s components relative to the root, lexically.

        Resolves only ``..``/``.`` segments *lexically* (without touching the
        filesystem, so symlinks are not followed here) and asserts the result
        stays inside the connector root. Returns the relative ``Path``.

        Guards against path traversal (``..``); symlink escapes are blocked
        later at open time via ``O_NOFOLLOW`` on each component
        (see :meth:`_open_in_root`).

        Raises:
            ValueError: If ``path`` is not contained in the root.
        """
        # os.path.normpath collapses '..'/'.' lexically without resolving
        # symlinks, which is exactly what we want: we must not follow links
        # during validation (TOCTOU), only reject obvious traversal.
        if path.is_absolute():
            candidate = Path(os.path.normpath(str(path)))
        else:
            candidate = Path(os.path.normpath(str(self.root / path)))
        try:
            return candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"path escapes connector root: {path}") from exc

    def _open_in_root(self, rel: Path) -> int:
        """Open a regular file at ``rel`` (relative to the root) safely.

        Walks the path one component at a time starting from a directory file
        descriptor for the root, opening each component with ``O_NOFOLLOW`` so
        a symlinked component is rejected rather than followed. This closes the
        TOCTOU/symlink-swap window between validation and open: the final file
        descriptor refers to an object reached entirely through non-symlinked
        components under the root.

        Args:
            rel: Path relative to the connector root (no ``..`` segments).

        Returns:
            An open OS-level file descriptor for the target file. The caller
            owns it and must ``os.close`` it.

        Raises:
            ValueError: If a path component is a symlink (escape attempt).
            FileNotFoundError: If a component does not exist.
            IsADirectoryError: If the target is not a regular file.
        """
        parts = rel.parts
        if not parts:
            raise IsADirectoryError(self.root)

        nofollow = getattr(os, "O_NOFOLLOW", 0)
        dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)

        dir_fd = os.open(self.root, dir_flags)
        try:
            # Descend through intermediate directory components.
            for part in parts[:-1]:
                try:
                    next_fd = os.open(part, dir_flags | nofollow, dir_fd=dir_fd)
                except OSError as exc:
                    # ELOOP indicates O_NOFOLLOW hit a symlink component.
                    raise ValueError(
                        f"path component is a symlink or not a directory: {part!r}"
                    ) from exc
                os.close(dir_fd)
                dir_fd = next_fd
            # Open the final component as a regular file, not following links.
            try:
                file_fd = os.open(parts[-1], os.O_RDONLY | nofollow, dir_fd=dir_fd)
            except OSError as exc:
                raise ValueError(f"path component is a symlink: {parts[-1]!r}") from exc
        finally:
            os.close(dir_fd)
        return file_fd

    def list_resources(self) -> list[MCPResource]:
        """Return one :class:`MCPResource` per matching file under the root."""
        resources: list[MCPResource] = []
        for path in self._iter_files():
            rel = path.relative_to(self.root).as_posix()
            resources.append(MCPResource(uri=path.as_uri(), name=rel, kind=self.kind))
        return resources

    def read_resource(self, uri: str) -> str:
        """Return the UTF-8 text of the file addressed by ``uri``.

        This re-enforces the *full* connector policy on every read — not just
        containment in the root — so a caller cannot bypass
        :meth:`list_resources` by hand-crafting a URI for an excluded or
        oversized file. Specifically it re-checks:

        * the ``file://`` scheme,
        * lexical containment in the root (no ``..`` traversal),
        * the include/exclude globs (same check as :meth:`_matches`),
        * symlink-free path components (via :meth:`_open_in_root`, closing the
          validate-then-open TOCTOU window),
        * regular-file-ness and ``max_file_bytes``, both via ``fstat`` on the
          already-open descriptor (not a re-resolved path).

        Args:
            uri: A ``file://`` URI, normally one returned by
                :meth:`list_resources`.

        Raises:
            ValueError: If the URI is not a ``file://`` URI under the root, a
                path component is a symlink, or the resource is excluded by the
                connector's include/exclude/size policy.
            FileNotFoundError: If the file no longer exists.
            IsADirectoryError: If the URI does not address a regular file.
        """
        import stat as stat_module
        from urllib.parse import unquote, urlparse

        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"filesystem connector only reads file:// URIs, got {uri!r}")

        rel = self._relative_in_root(Path(unquote(parsed.path)))

        # Re-apply the include/exclude policy that gates list_resources()/_iter_files().
        # We match on the final component name, exactly like _iter_files() does.
        if not self._matches(rel.name):
            raise ValueError(f"resource is excluded by connector policy: {rel.as_posix()}")

        fd = self._open_in_root(rel)
        try:
            st = os.fstat(fd)
            if not stat_module.S_ISREG(st.st_mode):
                raise IsADirectoryError(f"not a regular file: {rel.as_posix()}")
            if st.st_size > self.max_file_bytes:
                raise ValueError(
                    f"resource exceeds max_file_bytes "
                    f"({st.st_size} > {self.max_file_bytes}): {rel.as_posix()}"
                )
            with os.fdopen(fd, "rb", closefd=True) as fh:
                fd = -1  # ownership transferred to fh; don't double-close
                data = fh.read()
        finally:
            if fd >= 0:
                os.close(fd)
        return data.decode("utf-8", errors="replace")

    def write(self, *args: object, **kwargs: object) -> None:
        """Write actions are disabled in the MVP."""
        raise MCPWriteDisabledError("MCP write actions are disabled in OpenBird (read-only MVP).")


class MCPRegistry:
    """Registry of MCP connectors, loadable from config.

    The registry owns connector instances by name, can ingest their content
    into a :class:`~openbird.memory.store.MemoryStore` as observations, and
    refuses duplicate registrations.
    """

    #: Connector kinds the MVP knows how to construct from config.
    _BUILDERS = {"filesystem": FilesystemMCPConnector.from_config}

    def __init__(self) -> None:
        self._connectors: dict[str, MCPConnector] = {}

    def __len__(self) -> int:
        return len(self._connectors)

    def __contains__(self, name: object) -> bool:
        return name in self._connectors

    def names(self) -> list[str]:
        """Return registered connector names in insertion order."""
        return list(self._connectors)

    def register(self, connector: MCPConnector) -> None:
        """Register a connector instance.

        Raises:
            ValueError: If a connector with the same name is already registered.
        """
        if connector.name in self._connectors:
            raise ValueError(f"connector already registered: {connector.name!r}")
        self._connectors[connector.name] = connector

    def get(self, name: str) -> MCPConnector:
        """Return a registered connector by name.

        Raises:
            KeyError: If no connector with that name is registered.
        """
        return self._connectors[name]

    # -- config loading -------------------------------------------------------

    @classmethod
    def from_configs(cls, configs: Iterable[ConnectorConfig | dict]) -> "MCPRegistry":
        """Build a registry from connector configs (dataclasses or dicts).

        Disabled and unsupported-kind connectors are skipped (the latter so an
        OAuth connector listed in a forward-looking config does not crash the
        read-only MVP). Filesystem connectors with a missing/invalid root raise.
        """
        registry = cls()
        for raw in configs:
            config = raw if isinstance(raw, ConnectorConfig) else ConnectorConfig.from_dict(raw)
            if not config.enabled:
                continue
            builder = cls._BUILDERS.get(config.kind)
            if builder is None:
                # Unsupported in the MVP (e.g. an OAuth connector). Skip rather
                # than fail so configs can declare future connectors.
                continue
            registry.register(builder(config))
        return registry

    # -- ingestion ------------------------------------------------------------

    def ingest(
        self,
        store: "MemoryStore",
        *,
        names: Iterable[str] | None = None,
    ) -> int:
        """Pull connector content into ``store`` as observations.

        Each readable resource becomes one observation tagged ``source="mcp"``
        with ``app`` set to the connector name and ``window``/``url`` carrying
        the resource name and URI. Returns the number of observations created.
        Unreadable resources (vanished files, decode errors surfaced as
        exceptions) are skipped so one bad file cannot abort an ingest.

        Args:
            store: Destination memory store.
            names: Restrict ingestion to these connector names; default = all.
        """
        targets = list(names) if names is not None else self.names()
        created = 0
        for cname in targets:
            connector = self._connectors[cname]
            for resource in connector.list_resources():
                try:
                    text = connector.read_resource(resource.uri)
                except (OSError, ValueError):
                    continue
                if not text.strip():
                    continue
                store.add_observation(
                    text,
                    app=connector.name,
                    window=resource.name,
                    url=resource.uri,
                    source="mcp",
                )
                created += 1
        return created


__all__ = [
    "MCP_AVAILABLE",
    "ConnectorConfig",
    "FilesystemMCPConnector",
    "MCPConnector",
    "MCPRegistry",
    "MCPResource",
    "MCPWriteDisabledError",
    "mcp_available",
]
