"""Homework 3: a basic LangChain agent with Python and text-metrics tools.

Environment variables are loaded from a local ``.env`` file when present.
OpenAI credentials must be provided through the environment, for example via
``OPENAI_API_KEY``; this module does not contain API keys.

Author FAU ID: alunsford2023.

Security note: ``PythonREPLTool`` executes Python code in the current process.
Only use this application with prompts and code you trust.
"""

from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.tools import ToolException
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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
		A mapping containing ``character_count``, ``word_count``, and
		``average_word_length``.
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

SYSTEM_PROMPT = (
	"You are a helpful assistant with access to a Python REPL tool and a text "
	"statistics tool. Use the text statistics tool for character counts, word "
	"counts, and average word length. Use the Python REPL when other calculation "
	"or executable Python is needed, and explain the result clearly."
)


def build_agent() -> Any:
	"""Create and return the tool-calling agent for this application.

	The agent uses the configured OpenAI chat model, the Python REPL tool,
	and a system prompt that guides when and how the tool should be used.

	Returns:
		Any: A LangChain agent graph ready to receive messages via ``invoke``.
	"""
	return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def main() -> None:
	"""Run an interactive command-line session with the tool-calling agent.

	Each non-empty input is sent to the agent and its final response is
	printed. Enter a blank line or send EOF to exit the session.

	Returns:
		None.
	"""
	agent = build_agent()
	print("=" * 64)
	print("FAU LANGCHAIN TOOL-CALLING ASSIGNMENT | alunsford2023")
	print("=" * 64)
	print("Loaded tools:")
	for configured_tool in tools:
		print(f"  - {configured_tool.name}: {configured_tool.description}")
	print("Enter a prompt, or a blank line to exit.\n")

	while True:
		try:
			prompt = input("llm>> ")
		except EOFError:
			print()
			break

		if not prompt.strip():
			break

		try:
			result = agent.invoke(
				{"messages": [{"role": "user", "content": prompt}]}
			)
			tool_failures = [
				message
				for message in result.get("messages", [])
				if getattr(message, "type", None) == "tool"
				and getattr(message, "status", None) == "error"
			]
			for failure in tool_failures:
				print(
					f"Tool execution failed ({failure.name}): {failure.content}"
				)
			final_message = result.get("messages", [])[-1]
			if not tool_failures or getattr(final_message, "type", None) != "tool":
				print(final_message.content)
		except ToolException as error:
			print(f"Tool execution failed: {error}")
		except Exception as error:
			print(f"Agent request failed ({type(error).__name__}): {error}")


if __name__ == "__main__":
	main()