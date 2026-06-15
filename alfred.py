import os
import sys
import warnings
import logging
from dotenv import load_dotenv

# Suppress Deprecation Warnings and user/HF warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Reconfigure stdout to support UTF-8 (emojis/special characters on Windows)
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Import tools from previous sections
from tools import DuckDuckGoSearchRun, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool

# Initialize the web search tool
search_tool = DuckDuckGoSearchRun()

# Retrieve token from environment
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Generate the chat interface, including the tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    # Ensure messages is a list if a single message/string was passed
    msgs = state["messages"]
    if isinstance(msgs, str):
        msgs = [HumanMessage(content=msgs)]
    elif isinstance(msgs, dict):
        msgs = [msgs]
    
    return {
        "messages": [chat_with_tools.invoke(msgs)],
    }

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

# --- End-to-End Examples ---
if __name__ == "__main__":
    print("=========================================")
    print("Starting Alfred's Gala Agent End-to-End Test")
    print("=========================================\n")

    # Example 1: Finding Guest Information
    print("--- Example 1: Finding Guest Information ---")
    response_1 = alfred.invoke({"messages": [HumanMessage(content="Tell me about 'Lady Ada Lovelace'")]})
    print("🎩 Alfred's Response:")
    print(response_1['messages'][-1].content)
    print("-" * 50 + "\n")

    # Example 2: Checking the Weather for Fireworks
    print("--- Example 2: Checking the Weather for Fireworks ---")
    response_2 = alfred.invoke({"messages": [HumanMessage(content="What's the weather like in Paris tonight? Will it be suitable for our fireworks display?")]})
    print("🎩 Alfred's Response:")
    print(response_2['messages'][-1].content)
    print("-" * 50 + "\n")

    # Example 3: Impressing AI Researchers
    print("--- Example 3: Impressing AI Researchers ---")
    response_3 = alfred.invoke({"messages": [HumanMessage(content="One of our guests is from Qwen. What can you tell me about their most popular model?")]})
    print("🎩 Alfred's Response:")
    print(response_3['messages'][-1].content)
    print("-" * 50 + "\n")

    # Example 4: Combining Multiple Tools
    print("--- Example 4: Combining Multiple Tools ---")
    response_4 = alfred.invoke({"messages": [HumanMessage(content="I need to speak with 'Dr. Nikola Tesla' about recent advancements in wireless energy. Can you help me prepare for this conversation?")]})
    print("🎩 Alfred's Response:")
    print(response_4['messages'][-1].content)
    print("-" * 50 + "\n")

    # Example 5: Conversation Memory (Advanced Features)
    print("--- Example 5: Conversation Memory ---")
    # First interaction
    response_5 = alfred.invoke({"messages": [HumanMessage(content="Tell me about 'Lady Ada Lovelace'. What's her background and how is she related to me?")]})
    print("🎩 Alfred's Response (1st turn):")
    print(response_5['messages'][-1].content)
    print()

    # Second interaction (referencing the first)
    response_5_followup = alfred.invoke({"messages": response_5["messages"] + [HumanMessage(content="What projects is she currently working on?")]})
    print("🎩 Alfred's Response (2nd turn):")
    print(response_5_followup['messages'][-1].content)
    print("-" * 50 + "\n")
