import asyncio
import json
import ollama
from fastmcp import Client

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"
OLLAMA_MODEL = "llama3.2:latest"  # switch to gemma:2b for slightly better tool calling


def convert_tools_for_ollama(mcp_tools) -> list:
    """Convert MCP tool definitions to Ollama-compatible format."""
    tools = []
    for tool in mcp_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        })
    return tools


async def run_agent(client: Client, ollama_tools: list, messages: list) -> str:
    """Single agent turn — handles tool calls recursively until a final response."""
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        tools=ollama_tools,
    )

    message = response.message

    # No tool calls — return the plain response
    if not message.tool_calls:
        return message.content

    # Append assistant message with tool calls
    messages.append({
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": message.tool_calls,
    })

    # Execute each tool call via the MCP server
    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = dict(tool_call.function.arguments)

        # LLMs sometimes pass list/dict args as raw strings — parse them
        for key, value in tool_args.items():
            if isinstance(value, str):
                try:
                    tool_args[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass

        print(f"  [tool call]  {tool_name}({tool_args})")

        result = await client.call_tool(tool_name, tool_args)
        tool_output = result.content[0].text if result.content else "No result"

        print(f"  [tool result] {tool_output}")

        messages.append({
            "role": "tool",
            "content": tool_output,
        })

    # Let the model produce the final answer after seeing tool results
    return await run_agent(client, ollama_tools, messages)


async def main():
    print(f"Connecting to MCP server at {MCP_SERVER_URL} ...")

    async with Client(MCP_SERVER_URL) as client:
        mcp_tools = await client.list_tools()
        ollama_tools = convert_tools_for_ollama(mcp_tools)

        print(f"Tools available: {[t.name for t in mcp_tools]}")
        print(f"Model: {OLLAMA_MODEL}")
        print("Type 'exit' to quit.\n")

        messages = []

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            messages.append({"role": "user", "content": user_input})

            reply = await run_agent(client, ollama_tools, messages)

            messages.append({"role": "assistant", "content": reply})
            print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
