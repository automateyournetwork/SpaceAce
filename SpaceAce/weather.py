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

# API URL for WeatherAPI
WEATHER_API_URL = "https://api.weatherapi.com/v1/current.json"

# ---------------------------- API CLASS ---------------------------- #
class WeatherAPI:
    """
    Fetches real-time weather and geographical location data from WeatherAPI based on the provided latitude and longitude.
    """
    def __init__(self):
        self.api_url = WEATHER_API_URL
        self.api_key = WEATHER_API_KEY

    def get_weather(self, latitude, longitude):
        """
        Fetches the current weather and geographical (city, etc) information at the given latitude & longitude.
        """
        params = {
            "key": self.api_key,
            "q": f"{latitude},{longitude}",
        }

        retries = 3  # Number of retries
        for attempt in range(retries):
            try:
                logging.debug(f"Fetching weather for lat: {latitude}, lon: {longitude}")
                response = requests.get(self.api_url, params=params, timeout=5)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"GET request failed (Attempt {attempt+1}): {e}")
                time.sleep(2)  # Retry delay

        return {"error": "Failed to fetch weather after multiple attempts."}

    def get_current_weather(self, input_data):
        """
        Wrapper function for fetching weather and geography information based on latitude/longitude.
        Accepts input as a dictionary or a JSON string.
        """
        try:
            # Ensure input_data is a dictionary
            if isinstance(input_data, str):  
                logging.debug(f"Parsing input string: {input_data}")
                input_data = json.loads(input_data)  # Convert JSON string to dict

            if not isinstance(input_data, dict):
                logging.error(f"Invalid input type: {type(input_data)}. Expected dict.")
                return {"error": "Invalid input format. Expected JSON object with 'latitude' and 'longitude'."}

            logging.debug(f"Received input data: {input_data}")

            lat = input_data.get("latitude")
            lon = input_data.get("longitude")

            if not lat or not lon:
                return {"error": "Missing latitude and longitude."}

            return self.get_weather(lat, lon)  # ✅ Now correctly calls `get_weather()`

        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            return {"error": "Invalid input format. Expected JSON."}

# ---------------------------- TOOL FUNCTION ---------------------------- #
def get_current_weather(input_data):
    """
    Wrapper function for fetching weather and geographic information based on latitude/longitude.
    Expects input: {'latitude': 'xx.xxxx', 'longitude': 'yy.yyyy'}
    """
    weather_client = WeatherAPI()  # ✅ Instantiate WeatherAPI class
    return weather_client.get_current_weather(input_data)  # ✅ Call the fixed method

# ---------------------------- LANGCHAIN TOOL ---------------------------- #
get_weather_tool = Tool(
    name="fetch_weather",
    description="Fetches the current weather and geographic information at a given latitude and longitude.",
    func=get_current_weather
)

# Define available tools
tools = [get_weather_tool]

tool_names = ", ".join([tool.name for tool in tools])

tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# ---------------------------- LLM & PROMPT TEMPLATE ---------------------------- #
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

weather_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
    You are a Weather AI Agent that provides real-time weather conditions based on a given latitude and longitude.

    **FORMAT:**  
    Thought: [Your reasoning]  
    Action: weather_tool  
    Action Input: {{ "latitude": "xx.xxxx", "longitude": "yy.yyyy" }}  
    Observation: [Result]  
    Final Answer: [Current weather at the given location]  

    **Example Query:**
    - What is the weather like at latitude 23.5 and longitude -45.3?  
      Thought: I need to fetch the weather for that location using the weather_tool.  
      Action: weather_tool  
      Action Input: {{ "latitude": "23.5", "longitude": "-45.3" }}  
      Observation: [Weather details]  
      Final Answer: [Current weather at the given location]  

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
