"""OpenBird integrations subsystem.

MVP exposes a labeled **local filesystem MCP connector** and a registry that
loads connectors from config and pulls their content into the memory store as
observations. Write actions are disabled. The real ``mcp`` SDK is an optional
extra; everything here degrades gracefully when it is not installed.
"""

from __future__ import annotations

from openbird.integrations.mcp import (
    MCP_AVAILABLE,
    ConnectorConfig,
    FilesystemMCPConnector,
    MCPConnector,
    MCPRegistry,
    MCPWriteDisabledError,
    mcp_available,
)

__all__ = [
    "MCP_AVAILABLE",
    "ConnectorConfig",
    "FilesystemMCPConnector",
    "MCPConnector",
    "MCPRegistry",
    "MCPWriteDisabledError",
    "mcp_available",
]
