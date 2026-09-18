# Alunsford2023
"""RAG pipeline for UTF-8 text files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextFileLoader(BaseLoader):
	"""Load a text file as a LangChain document."""

	def __init__(self, file_path: str | Path, encoding: str = "utf-8") -> None:
		self.file_path = Path(file_path)
		self.encoding = encoding

	def lazy_load(self) -> Iterator[Document]:
		yield Document(
			page_content=self.file_path.read_text(encoding=self.encoding),
			metadata={"source": str(self.file_path)},
		)


def build_rag_chain(
	file_path: str | Path,
	persist_directory: str = "./chroma_db",
):
	"""Build a callable that answers questions about ``file_path``.

	Set ``OPENAI_API_KEY`` before calling this function.
	"""
	documents = TextFileLoader(file_path).load()
	chunks = RecursiveCharacterTextSplitter(
		chunk_size=1000, chunk_overlap=150
	).split_documents(documents)

	store = Chroma.from_documents(
		documents=chunks,
		embedding=OpenAIEmbeddings(),
		collection_name="text-file-rag",
		persist_directory=persist_directory,
	)
	retriever = store.as_retriever(search_kwargs={"k": 4})
	model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

	def answer(question: str) -> str:
		context = "\n\n".join(
			document.page_content for document in retriever.invoke(question)
		)
		prompt = (
			"Answer using only the context below. If the answer is not present, "
			"say you do not know.\n\nContext:\n{context}\n\nQuestion: {question}"
		).format(context=context, question=question)
		response = model.invoke(prompt)
		return response.content if isinstance(response.content, str) else str(response.content)

	return answer


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Ask questions about a text file.")
	parser.add_argument("file", type=Path)
	parser.add_argument("question")
	parser.add_argument("--persist-directory", default="./chroma_db")
	args = parser.parse_args()
	print(build_rag_chain(args.file, args.persist_directory)(args.question))

