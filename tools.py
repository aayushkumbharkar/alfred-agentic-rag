import os
import sys
import warnings
import logging
import random
from dotenv import load_dotenv

# Suppress Deprecation Warnings and user/HF warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Reconfigure stdout to support UTF-8 (emojis/special characters on Windows)
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from huggingface_hub import list_models

# 1. Give Your Agent Access to the Web
search_tool = DuckDuckGoSearchRun()

# 2. Creating a Custom Tool for Weather Information to Schedule the Fireworks
def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the tool
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)

# 3. Creating a Hub Stats Tool for Influential AI Builders
def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads (no direction parameter in latest API)
        models = list(list_models(author=author, sort="downloads", limit=1))

        if models:
            model = models[0]
            downloads = getattr(model, "downloads", None)
            if downloads is not None:
                return f"The most downloaded model by {author} is {model.id} with {downloads:,} downloads."
            else:
                return f"The most downloaded model by {author} is {model.id}."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

# Initialize the tool
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
)

if __name__ == "__main__":
    # Test DuckDuckGoSearchRun
    print("Testing DuckDuckGoSearchRun:")
    print(search_tool.invoke("Who's the current President of France?"))
    print("-" * 50)

    # Test Weather Tool
    print("Testing get_weather_info:")
    print(weather_info_tool.invoke("Paris"))
    print("-" * 50)

    # Test Hub Stats Tool
    print("Testing get_hub_stats:")
    print(hub_stats_tool.invoke("facebook"))
    print("-" * 50)
