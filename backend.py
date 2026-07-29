from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated, Dict, Any, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from groq import RateLimitError
import sys
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
import requests
import threading
import asyncio
import aiosqlite
import tempfile

load_dotenv()

# ************** Config ********************

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

# ******************Model Defination ******************

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0,
    timeout=30,
    max_retries=2
)

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

# *******************Helper Functions ******************

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


def _get_retriever(thread_id : Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None

def ingest_pdf(file_bytes : bytes, thread_id : str, filename : Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000, chunk_overlap = 200, separators = ["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embedding)
        retriever = vector_store.as_retriever(
            search_type = "similarity", search_kwargs = {"k" : 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename" : filename or os.path.basename(temp_path),
            "documents" : len(docs),
            "chunks" : len(chunks)
        }

        return {
            "filename" : filename or os.path.basename(temp_path),
            "documents" : len(docs),
            "chunks" : len(chunks)
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

client = MultiServerMCPClient(
    {
        "calculator" : {
            "transport" : "stdio",
            "command" : sys.executable,
            "args" : [r"C:\Users\DELL\OneDrive\Desktop\chatbot\tools\calculator.py"],
        },
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)

# ************* Tool Definitions *************

search_tools = DuckDuckGoSearchRun(region='us-en')

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []

@tool
def rag_tool(query : str, config : RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    thread_id = config.get("configurable", {}).get("thread_id")

    retriever =  _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


mcp_tools = load_mcp_tools()

tools = [search_tools, get_stock_price, rag_tool, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# ************* State Definitions *************

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ************* Graph Node Definitions **************

async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    try:
        response = await llm_with_tools.ainvoke(messages)
    except RateLimitError as e:
        response = AIMessage(content="⚠️ I've hit my daily usage limit with the model provider. Please try again in a few minutes.")
    return {"messages": [response]}

tool_node = ToolNode(tools) if tools else None

# ************* Checkpointer **************

async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())


# ************* Graph Definition *************

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# ************* Helper Functions *************

async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())