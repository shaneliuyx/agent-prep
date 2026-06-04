"""W3.5.96 — probe: can a plain Python MCP client drive GBrain's MCP server?

Spawns `gbrain serve` over stdio (env-injected — an MCP server is a separate
process that does NOT inherit your shell env) and lists the tools it exposes.
This is the smallest proof that a FUTURE agent (not Claude Code) can use GBrain
as a memory layer over MCP. If this prints put_page/query/etc., the full
ingest+query agent (ingest_agent.py) is just calling these tools.
"""
from __future__ import annotations

import asyncio
import os
import pathlib

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")

_BUN_BIN = os.path.expanduser("~/.bun/bin")
_GBRAIN = os.getenv("GBRAIN_BIN", "gbrain")
if _GBRAIN == "gbrain":
    _GBRAIN = os.path.join(_BUN_BIN, "gbrain")


def _server_env() -> dict[str, str]:
    """Inherit the full env, then add GBrain's engine/embedding vars + bun on PATH.
    The spawned `gbrain serve` reads ~/.gbrain/config.json (HOME) + these vars."""
    env = dict(os.environ)
    env["PATH"] = _BUN_BIN + os.pathsep + env.get("PATH", "")
    for k in ("GBRAIN_DATABASE_URL", "OLLAMA_BASE_URL", "OLLAMA_API_KEY"):
        v = os.getenv(k)
        if v:
            env[k] = v
    return env


async def main() -> None:
    params = StdioServerParameters(command=_GBRAIN, args=["serve"], env=_server_env())
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print(f">>> gbrain MCP exposes {len(tools)} tools:")
            for t in sorted(tools, key=lambda x: x.name):
                print(f"  - {t.name}: {(t.description or '')[:70]}")
            # sanity: the tools our ingest agent needs
            names = {t.name for t in tools}
            need = {"put_page", "add_link", "add_timeline_entry", "query", "search"}
            missing = need - names
            print(f"\n>>> ingest-agent tools present: {sorted(need & names)}")
            if missing:
                print(f"!!! MISSING: {sorted(missing)}")


if __name__ == "__main__":
    asyncio.run(main())
