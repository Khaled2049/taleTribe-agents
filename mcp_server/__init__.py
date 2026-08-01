"""NovelSync remote MCP server: OAuth 2.1 authorization server + story tools.

Reads are owner-scoped and need only `stories:read`. Creating stories and
chapters additionally needs `stories:write` and the ENABLE_MCP_WRITES flag; the
whole mutation surface lives in writes.py.

Package name is mcp_server (not mcp) so it never shadows the `mcp` SDK.
"""
