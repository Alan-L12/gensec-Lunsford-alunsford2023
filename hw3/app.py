"""Homework 3: a basic LangChain agent with Python and text-metrics tools.

Environment variables are loaded from a local .env file when present.
OpenAI credentials must be provided through the environment, for example via
OPENAI_API_KEY; this module does not contain API keys.

Author FAU ID: alunsford2023.

Security note: PythonREPLTool executes Python code in the current process.
Only use this application with prompts and code you trust.
"""

# Alunsford2023
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI


load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=15, max_retries=2)
python_repl = PythonREPLTool()


@tool
def calculate_text_metrics(text: str) -> dict[str, int | float]:
    """Calculate basic text statistics for the provided string.

    Character count includes all characters, including whitespace. Word count
    uses whitespace to separate words, and average word length is the number
    of non-whitespace word characters divided by the word count. Empty or
    whitespace-only input has an average word length of 0.0.

    Args:
        text: The string to analyze.

    Returns:
        A mapping containing character_count, word_count, and
        average_word_length.
    """
    words = text.split()
    word_count = len(words)
    average_word_length = (
        sum(len(word) for word in words) / word_count if word_count else 0.0
    )
    return {
        "character_count": len(text),
        "word_count": word_count,
        "average_word_length": average_word_length,
    }


tools = [python_repl, calculate_text_metrics]
tool_map = {t.name: t for t in tools}
model_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a Python REPL tool and a text "
    "statistics tool. Use the text statistics tool for character counts, word "
    "counts, and average word length. Use the Python REPL when other calculation "
    "or executable Python is needed. Explain results clearly and concisely."
)


def run_agent(user_input: str) -> str:
    """Run an input prompt through the model and execute any requested tools."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    response = model_with_tools.invoke(messages)

    if not response.tool_calls:
        return response.content

    # Execute requested tools directly
    tool_messages = []
    for tool_call in response.tool_calls:
        selected_tool = tool_map.get(tool_call["name"])
        if selected_tool:
            tool_output = selected_tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            )

    # Let the model synthesize the final answer from tool outputs
    final_response = llm.invoke(messages + [response] + tool_messages)
    return final_response.content


def main() -> None:
    """Run an interactive command-line session with the tool-calling agent."""
    print("=" * 64, flush=True)
    print("FAU LANGCHAIN TOOL-CALLING ASSIGNMENT | alunsford2023", flush=True)
    print("=" * 64, flush=True)
    print("Loaded tools:", flush=True)
    for configured_tool in tools:
        print(f"  - {configured_tool.name}: {configured_tool.description}", flush=True)
    print("Enter a prompt, or a blank line to exit.\n", flush=True)

    while True:
        try:
            prompt = input("llm>> ")
        except EOFError:
            print(flush=True)
            break

        if not prompt.strip():
            break

        try:
            output = run_agent(prompt)
            print(output, flush=True)
        except ToolException as error:
            print(f"Tool execution failed: {error}", flush=True)
        except Exception as error:
            print(f"Agent request failed ({type(error).__name__}): {error}", flush=True)


if __name__ == "__main__":
    main()