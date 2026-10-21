"""BOND MCP adapter. Discover schemas; never invent a provider tool name.

Use streamable HTTP with an existing bearer token, or an OAuth-capable stdio
bridge such as mcp-remote. Only explicitly allowlisted read tools are callable.
"""
import asyncio
from contextlib import AsyncExitStack
import os
import jsonschema
from .common import dumps


async def request(config, tool=None, arguments=None):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    async with AsyncExitStack() as stack:
        if config.get("transport") == "stdio":
            streams = await stack.enter_async_context(stdio_client(StdioServerParameters(
                command=config["command"], args=config.get("args", []))))
        elif config.get("transport") == "streamable_http":
            headers = {}
            if config.get("token_env"):
                token = os.environ.get(config["token_env"])
                if not token:
                    raise ValueError("BOND token environment variable is not set")
                headers["Authorization"] = "Bearer " + token
            streams = await stack.enter_async_context(streamablehttp_client(
                config["url"], headers=headers, timeout=30, sse_read_timeout=120))
        else:
            raise ValueError("MCP transport must be stdio or streamable_http")
        session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
        await session.initialize()
        inventory, cursor = [], None
        for _ in range(20):
            page = await session.list_tools(cursor=cursor)
            inventory.extend(t.model_dump(mode="json") for t in page.tools)
            cursor = page.nextCursor
            if not cursor:
                break
        if tool is None:
            return inventory
        if tool not in config.get("allowed_tools", []):
            raise ValueError("MCP tool is not in the explicit read-only allowlist")
        definition = next((t for t in inventory if t["name"] == tool), None)
        if definition is None:
            raise ValueError("Configured MCP tool is absent from the live inventory")
        jsonschema.validate(arguments or {}, definition["inputSchema"])
        result = await session.call_tool(tool, arguments or {})
        if result.isError:
            raise RuntimeError("MCP returned a tool error: " + dumps(result.model_dump(mode="json"))[:2000])
        return result.model_dump(mode="json")


class Bond:
    def __init__(self, config):
        self.config = config

    def tools(self):
        return asyncio.run(request(self.config))

    def call(self, name, args):
        return asyncio.run(request(self.config, name, args))
