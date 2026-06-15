import warnings
import logging

# Suppress Deprecation Warnings, User Warnings and Hugging Face HTTP warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import Tool
from dataset import docs

bm25_retriever = BM25Retriever.from_documents(docs)

def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation."
)

if __name__ == "__main__":
    # Test the retriever tool with a query
    print("Testing guest_info_tool with query: 'Ada'")
    print("-" * 40)
    print(guest_info_tool.invoke("Ada"))
    print("-" * 40)