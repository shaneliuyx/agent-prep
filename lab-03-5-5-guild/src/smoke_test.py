"""Verify Python can talk to guild via MCP stdio. Lists available tools."""
import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


async def main() -> None:
    # Correct args: ['mcp', 'serve'] — guild has no top-level 'serve' verb.
    # Wrong args produce: Error: unknown command "serve" for "guild"
    #                     mcp.shared.exceptions.McpError: Connection closed
    params = StdioServerParameters(command="guild", args=["mcp", "serve"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"guild MCP server reachable. {len(tools.tools)} tools:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description[:80] if t.description else ''}")


if __name__ == "__main__":
    asyncio.run(main())