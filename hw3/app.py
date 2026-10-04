"""Homework 3: a basic LangChain agent with an interactive Python REPL tool.

Environment variables are loaded from a local ``.env`` file when present.
OpenAI credentials must be provided through the environment, for example via
``OPENAI_API_KEY``; this module does not contain API keys.

Security note: ``PythonREPLTool`` executes Python code in the current process.
Only use this application with prompts and code you trust.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
python_repl = PythonREPLTool()
tools = [python_repl]

SYSTEM_PROMPT = (
	"You are a helpful assistant with access to a Python REPL tool. "
	"Use the tool when calculation or executable Python is needed, and "
	"explain the result clearly."
)


def build_agent():
	"""Create and return the tool-calling agent for this application.

	The agent uses the configured OpenAI chat model, the Python REPL tool,
	and a system prompt that guides when and how the tool should be used.

	Returns:
		A LangChain agent graph ready to receive messages via ``invoke``.
	"""
	return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def main():
	"""Run an interactive command-line session with the tool-calling agent.

	Each non-empty input is sent to the agent and its final response is
	printed. Enter a blank line or send EOF to exit the session.
	"""
	agent = build_agent()
	print("Python tool-calling agent ready. Enter a prompt, or a blank line to exit.")
	print(f"Tool: {python_repl.name} - {python_repl.description}")

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
			print(result["messages"][-1].content)
		except Exception as error:
			print(f"Agent error: {error}")


if __name__ == "__main__":
	main()


#alunsford2023