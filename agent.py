import asyncio
import json
import litellm
from fastmcp import Client

# ── Config ────────────────────────────────────────────────────────────────────
MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"
OLLAMA_MODEL   = "ollama_chat/llama3.2"   # or ollama_chat/gemma:2b
OLLAMA_BASE    = "http://localhost:11434"
SYSTEM_PROMPT  = (
    "You are a sorting assistant. "
    "Always use the merge_sort or bubble_sort tools when asked to sort a list. "
    "Never sort manually."
)
# ─────────────────────────────────────────────────────────────────────────────


def mcp_tools_to_litellm(mcp_tools) -> list:
    """Convert FastMCP tool definitions → LiteLLM function-call format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


def parse_args(tool_args: dict) -> dict:
    """Parse any string values that are JSON-encoded (e.g. '[1,2,3]' → list)."""
    out = {}
    for k, v in tool_args.items():
        if isinstance(v, str):
            try:
                out[k] = json.loads(v)
                continue
            except (json.JSONDecodeError, ValueError):
                pass
        out[k] = v
    return out


async def agent_loop(client: Client, litellm_tools: list, messages: list) -> str:
    """Run one agent turn — calls tools as needed, returns final text reply."""
    response = await litellm.acompletion(
        model=OLLAMA_MODEL,
        api_base=OLLAMA_BASE,
        messages=messages,
        tools=litellm_tools,
        tool_choice="auto",
    )

    choice = response.choices[0]
    msg    = choice.message

    # No tool call — plain reply
    if not msg.tool_calls:
        return msg.content or ""

    # Append assistant message with tool_calls
    messages.append(msg.model_dump(exclude_unset=True))

    # Execute every tool call against the MCP server
    for tc in msg.tool_calls:
        name = tc.function.name
        args = parse_args(json.loads(tc.function.arguments))

        print(f"  [tool call]  {name}({args})")

        result       = await client.call_tool(name, args)
        tool_output  = result.content[0].text if result.content else "No result"

        print(f"  [tool result] {tool_output}")

        messages.append({
            "role":         "tool",
            "tool_call_id": tc.id,
            "content":      tool_output,
        })

    # Let LiteLLM produce the final answer now that it has tool results
    return await agent_loop(client, litellm_tools, messages)


async def main():
    print(f"Connecting to MCP server at {MCP_SERVER_URL} ...")

    async with Client(MCP_SERVER_URL) as client:
        mcp_tools     = await client.list_tools()
        litellm_tools = mcp_tools_to_litellm(mcp_tools)

        print(f"Tools     : {[t.name for t in mcp_tools]}")
        print(f"Model     : {OLLAMA_MODEL}")
        print("Type 'exit' to quit.\n")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            messages.append({"role": "user", "content": user_input})
            reply = await agent_loop(client, litellm_tools, messages)
            messages.append({"role": "assistant", "content": reply})
            print(f"Agent: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
