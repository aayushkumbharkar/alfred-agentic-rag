import random
import warnings
import logging
import sys
from smolagents import Tool, DuckDuckGoSearchTool
from huggingface_hub import list_models

# Suppress Hugging Face warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Reconfigure stdout to support UTF-8 (emojis/special characters on Windows)
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Search Tool
search_tool = DuckDuckGoSearchTool()

# 2. Weather Info Tool
class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for."
        }
    }
    output_type = "string"

    def forward(self, location: str):
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20}
        ]
        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

weather_info_tool = WeatherInfoTool()

# 3. Hub Stats Tool
class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from."
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
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

hub_stats_tool = HubStatsTool()

if __name__ == "__main__":
    # Test DuckDuckGoSearchTool
    print("Testing DuckDuckGoSearchTool:")
    print(search_tool("Who's the current President of France?"))
    print("-" * 50)

    # Test WeatherInfoTool
    print("Testing WeatherInfoTool:")
    print(weather_info_tool("Paris"))
    print("-" * 50)

    # Test HubStatsTool
    print("Testing HubStatsTool:")
    print(hub_stats_tool("facebook"))
    print("-" * 50)
