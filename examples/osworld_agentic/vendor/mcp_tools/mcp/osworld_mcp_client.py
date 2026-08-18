import asyncio
from fastmcp import Client
import random
import re


def _tokenize(text):
    """Lowercase word tokens, stripping punctuation."""
    return set(re.findall(r'[a-z]+', text.lower()))


def _score_tool(tool, instruction_tokens):
    """Word-overlap score between instruction and tool name+description."""
    tool_tokens = _tokenize(tool['name'] + ' ' + (tool.get('description') or ''))
    return len(instruction_tokens & tool_tokens)


# Tools that are always included regardless of instruction relevance.
_MANDATORY_KEYWORDS = ('save', 'env_info', 'get_workbook_info')


class OsworldMcpClient:
    # Only the Python FastMCP server (no Node.js / uvx required).
    # The filesystem and git MCP servers are omitted to avoid needing
    # npx / uvx inside the Ubuntu VM.
    config = {
        "mcpServers": {
            "osworld_mcp": {
                "url": "http://localhost:9292/mcp",
                "transport": "streamable-http"
            }
        }
    }

    @classmethod
    def list_tools(cls, tool_name, instruction=None, top_k=10, shuffle=False, rag=True):
        async def _list_tools():
            client = Client(cls.config)
            async with client:
                tool_list = await client.list_tools()
                tool_list = [{
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                } for tool in tool_list
                ]

            if rag:
                # Stage 1: app-level filter — only include tools belonging to the
                # active app. If the app has no dedicated MCP tools (e.g. chrome,
                # gimp, thunderbird), return [] rather than falling back to os.*
                # tools: injecting unrelated tools confuses the model and degrades
                # GUI navigation on tasks that can't be solved via MCP anyway.
                result = []
                for tool in tool_list:
                    if (tool_name is not None) and (tool_name in tool['name']):
                        result.append(tool)

                if len(result) == 0:
                    return result

                # Stage 2: task-level filter via instruction word overlap
                if instruction and len(result) > top_k:
                    instruction_tokens = _tokenize(instruction)
                    mandatory, candidates = [], []
                    for tool in result:
                        if any(kw in tool['name'] for kw in _MANDATORY_KEYWORDS):
                            mandatory.append(tool)
                        else:
                            candidates.append(tool)
                    # rank by overlap score, keep top_k slots after mandatory
                    slots = max(top_k - len(mandatory), 0)
                    ranked = sorted(candidates,
                                    key=lambda t: _score_tool(t, instruction_tokens),
                                    reverse=True)
                    result = mandatory + ranked[:slots]

                return result

            else:
                return tool_list

        tool_list = asyncio.run(_list_tools())

        if shuffle:
            random.shuffle(tool_list)

        print(tool_list)
        return tool_list

    @classmethod
    def call_tool(cls, name, params={}):
        async def _call_tool():
            client = Client(cls.config)
            async with client:
                response = await client.call_tool(
                    name,
                    params
                )
            return response

        response = asyncio.run(_call_tool())

        print(response)
        return response


if __name__ == '__main__':
    OsworldMcpClient.call_tool(
        'VSCodeTools_search_text',
        {
            'text': 'files'
        }
    )
