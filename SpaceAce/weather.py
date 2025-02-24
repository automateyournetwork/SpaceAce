import os
import json
import time
import logging
import requests
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API Keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)

BASE_API_URL = "https://api.weatherapi.com/v1"

# ---------------------------- API CLASS ---------------------------- #
class WeatherAPI:
    """
    Fetches real-time weather, forecast, historical data, marine conditions, timezone, and astronomy information.
    """
    def __init__(self):
        self.api_key = WEATHER_API_KEY

    def fetch_data(self, endpoint, params):
        """
        Generic function to query any WeatherAPI endpoint.
        """
        url = f"{BASE_API_URL}/{endpoint}.json"
        params["key"] = self.api_key

        retries = 3  # Retry mechanism
        for attempt in range(retries):
            try:
                logging.debug(f"Fetching data from {url} with params {params}")
                response = requests.get(url, params=params, timeout=5)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"API request failed (Attempt {attempt+1}): {e}")
                time.sleep(2)

        return {"error": f"Failed to fetch data from {endpoint} after multiple attempts."}

    # Current Weather
    def get_weather(self, latitude, longitude):
        return self.fetch_data("current", {"q": f"{latitude},{longitude}"})

    # Forecast Weather
    def get_forecast(self, latitude, longitude, days=3):
        return self.fetch_data("forecast", {"q": f"{latitude},{longitude}", "days": days})

    # Historical Weather
    def get_history(self, latitude, longitude, date):
        return self.fetch_data("history", {"q": f"{latitude},{longitude}", "dt": date})

    # Marine Weather
    def get_marine(self, latitude, longitude):
        return self.fetch_data("marine", {"q": f"{latitude},{longitude}"})

    # Timezone Data
    def get_timezone(self, latitude, longitude):
        return self.fetch_data("timezone", {"q": f"{latitude},{longitude}"})

    # Astronomy Data
    def get_astronomy(self, latitude, longitude, date):
        return self.fetch_data("astronomy", {"q": f"{latitude},{longitude}", "dt": date})

# ---------------------------- TOOL FUNCTIONS ---------------------------- #
def parse_input(input_data):
    """
    Ensures input_data is a dictionary.
    If input_data is a JSON string, convert it to a dictionary.
    """
    try:
        if isinstance(input_data, str):  
            logging.debug(f"Parsing input string: {input_data}")
            input_data = json.loads(input_data)  # Convert JSON string to dictionary

        if not isinstance(input_data, dict):
            logging.error(f"Invalid input type: {type(input_data)}. Expected dict.")
            return {"error": "Invalid input format. Expected JSON object."}

        logging.debug(f"Received input data: {input_data}")
        return input_data

    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return {"error": "Invalid JSON format."}


def get_current_weather(input_data):
    """
    Fetches real-time weather based on latitude/longitude.
    """
    input_data = parse_input(input_data)
    if "error" in input_data:
        return input_data  # Return error message

    weather_client = WeatherAPI()
    return weather_client.get_weather(input_data["latitude"], input_data["longitude"])


def get_forecast_weather(input_data):
    """
    Fetches weather forecast for a given location.
    Expects {'latitude': 'xx.xxxx', 'longitude': 'yy.yyyy', 'days': 3}
    """
    input_data = parse_input(input_data)
    if "error" in input_data:
        return input_data  # Return error message

    weather_client = WeatherAPI()
    return weather_client.get_forecast(input_data["latitude"], input_data["longitude"], input_data.get("days", 3))


def get_historical_weather(input_data):
    """
    Fetches past weather for a given date.
    Expects {'latitude': 'xx.xxxx', 'longitude': 'yy.yyyy', 'date': 'YYYY-MM-DD'}
    """
    input_data = parse_input(input_data)
    if "error" in input_data:
        return input_data  # Return error message

    if "date" not in input_data:
        return {"error": "Missing 'date' field."}

    weather_client = WeatherAPI()
    return weather_client.get_history(input_data["latitude"], input_data["longitude"], input_data["date"])


def get_marine_weather(input_data):
    """
    Fetches marine weather data for a given location.
    Expects {'latitude': 'xx.xxxx', 'longitude': 'yy.yyyy'}
    """
    input_data = parse_input(input_data)
    if "error" in input_data:
        return input_data  # Return error message

    weather_client = WeatherAPI()
    return weather_client.get_marine(input_data["latitude"], input_data["longitude"])


def get_timezone_info(input_data):
    """
    Fetches timezone information for a given location.
    Expects {'latitude': 'xx.xxxx', 'longitude': 'yy.yyyy'}
    """
    input_data = parse_input(input_data)
    if "error" in input_data:
        return input_data  # Return error message

    weather_client = WeatherAPI()
    return weather_client.get_timezone(input_data["latitude"], input_data["longitude"])


def get_astronomy_info(input_data):
    """
    Fetches astronomical data for a given date.
    Expects {'latitude': 'xx.xxxx', 'longitude': 'yy.yyyy', 'date': 'YYYY-MM-DD'}
    """
    input_data = parse_input(input_data)
    if "error" in input_data:
        return input_data  # Return error message

    if "date" not in input_data:
        return {"error": "Missing 'date' field."}

    weather_client = WeatherAPI()
    return weather_client.get_astronomy(input_data["latitude"], input_data["longitude"], input_data["date"])

# ---------------------------- LANGCHAIN TOOLS ---------------------------- #
get_weather_tool = Tool(
    name="fetch_weather",
    description="Fetches the current weather at a given latitude and longitude.",
    func=get_current_weather
)

get_forecast_tool = Tool(
    name="fetch_forecast",
    description="Fetches weather forecast for a given location.",
    func=get_forecast_weather
)

get_history_tool = Tool(
    name="fetch_history",
    description="Fetches past weather for a given date.",
    func=get_historical_weather
)

get_marine_tool = Tool(
    name="fetch_marine",
    description="Fetches marine weather data for a given location.",
    func=get_marine_weather
)

get_timezone_tool = Tool(
    name="fetch_timezone",
    description="Fetches timezone information for a given location.",
    func=get_timezone_info
)

get_astronomy_tool = Tool(
    name="fetch_astronomy",
    description="Fetches astronomical data for a given date and location.",
    func=get_astronomy_info
)

# Register all tools in a list
tools = [
    get_weather_tool,
    get_forecast_tool,
    get_history_tool,
    get_marine_tool,
    get_timezone_tool,
    get_astronomy_tool
]

tool_names = ", ".join([tool.name for tool in tools])

tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# ---------------------------- LLM & PROMPT TEMPLATE ---------------------------- #
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

weather_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
    You are a Weather AI Agent that provides real-time weather conditions, forecasts, historical data, marine conditions, timezone information, and astronomical data based on a given latitude and longitude.

    You have access to six tools:

    - **fetch_weather**: Fetches the current weather at a given latitude and longitude.
    - **fetch_forecast**: Retrieves the weather forecast for a location over a number of days.
    - **fetch_history**: Provides historical weather data for a past date.
    - **fetch_marine**: Retrieves marine and ocean weather conditions.
    - **fetch_timezone**: Retrieves timezone information for a given location.
    - **fetch_astronomy**: Retrieves astronomical data (sunrise, sunset, moon phases) for a location and date.

    **FORMAT:**
    Thought: [Your reasoning]  
    Action: [Correct tool to use]  
    Action Input: {{ "latitude": "xx.xxxx", "longitude": "yy.yyyy", "optional_parameters": "..." }}  
    Observation: [Result]  
    Final Answer: [Formatted response based on the retrieved data]  

    **Example Queries and How to Use the Correct Tools:**

    **1️⃣ Current Weather Query**
    - *Question*: What is the weather like at latitude 23.5 and longitude -45.3?  
      Thought: I need to fetch the current weather for that location.  
      Action: fetch_weather  
      Action Input: {{ "latitude": "23.5", "longitude": "-45.3" }}  
      Observation: [Weather details]  
      Final Answer: The current temperature is 24°C with partly cloudy skies.  

    **2️⃣ Weather Forecast Query**
    - *Question*: What is the 5-day forecast for Paris?  
      Thought: I need to get the weather forecast for Paris over the next 5 days.  
      Action: fetch_forecast  
      Action Input: {{ "latitude": "48.8566", "longitude": "2.3522", "days": 5 }}  
      Observation: [Forecast data]  
      Final Answer: Over the next 5 days, Paris will see rain on Tuesday, sunny skies on Wednesday, and cloudy weather the rest of the week.  

    **3️⃣ Historical Weather Query**
    - *Question*: What was the weather like in New York on January 1, 2023?  
      Thought: I need to fetch historical weather data for New York on 2023-01-01.  
      Action: fetch_history  
      Action Input: {{ "latitude": "40.71", "longitude": "-74.01", "date": "2023-01-01" }}  
      Observation: [Historical weather data]  
      Final Answer: On January 1, 2023, New York had a temperature of 5°C with light rain.  

    **4️⃣ Marine Weather Query**
    - *Question*: What is the marine forecast for the Gulf of Mexico?  
      Thought: I need to retrieve marine weather data for the Gulf of Mexico.  
      Action: fetch_marine  
      Action Input: {{ "latitude": "25.0", "longitude": "-90.0" }}  
      Observation: [Marine conditions]  
      Final Answer: The Gulf of Mexico has waves of 1.5 meters with moderate winds from the southeast.  

    **5️⃣ Timezone Information Query**
    - *Question*: What is the timezone for Tokyo?  
      Thought: I need to retrieve timezone data for Tokyo.  
      Action: fetch_timezone  
      Action Input: {{ "latitude": "35.6895", "longitude": "139.6917" }}  
      Observation: [Timezone details]  
      Final Answer: The timezone for Tokyo is JST (Japan Standard Time), UTC+9.  

    **6️⃣ Astronomy Query**
    - *Question*: When is the sunrise and sunset in Los Angeles on July 4, 2025?  
      Thought: I need to retrieve astronomical data for Los Angeles on 2025-07-04.  
      Action: fetch_astronomy  
      Action Input: {{ "latitude": "34.05", "longitude": "-118.25", "date": "2025-07-04" }}  
      Observation: [Sunrise/Sunset details]  
      Final Answer: On July 4, 2025, the sunrise in Los Angeles will be at 5:48 AM and sunset at 8:12 PM.  

    **Begin!**

    Question: {input}

    {agent_scratchpad}
    """
)

# Create the ReAct Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=weather_prompt.partial(tool_names=tool_names, tools=tool_descriptions)
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=5,
    max_execution_time=30
)

# Log initialization
logging.info("🚀 Weather Agent initialized.")

# ---------------------------- TEST EXECUTION ---------------------------- #
if __name__ == "__main__":
    test_input = {"latitude": "23.5", "longitude": "-45.3"}
    response = agent_executor.invoke({"input": test_input})
    print("Weather Agent Response:", response)
