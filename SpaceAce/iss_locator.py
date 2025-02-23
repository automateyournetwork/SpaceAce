import os
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

# Set API Key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)

# API URL for ISS location
ISS_API_URL = "http://api.open-notify.org/iss-now.json"

# ISS Locator class
class ISSLocator:
    def __init__(self):
        self.api_url = ISS_API_URL

    def get_location(self):
        """
        Fetches the ISS current location with retries.
        """
        retries = 3  # Number of retries
        for attempt in range(retries):
            try:
                response = requests.get(self.api_url, timeout=5)  # Add timeout
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"GET request failed (Attempt {attempt+1}): {e}")
                time.sleep(2)  # Wait before retrying
        return {"error": "Failed to fetch ISS location after multiple attempts."}

# Define a function for the Tool
def get_iss_location(_input):
    """
    Wrapper function for fetching ISS location.
    """
    locator = ISSLocator()
    return locator.get_location()

# Define the LangChain tool
get_iss_location_tool = Tool(
    name="get_iss_location_tool",
    description="Fetches the International Space Station's current location.",
    func=get_iss_location
)

# Define LLM with GPT-4o and low temperature
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

# Define available tools
tools = [get_iss_location_tool]

tool_names = ", ".join([tool.name for tool in tools])

tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# ✅ Define Prompt Template with agent_scratchpad as a variable
iss_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
    You are an International Space Station (ISS) Tracking Agent. Your job is to provide the current location of the ISS when asked.

    **FORMAT:**  
    Thought: [Your reasoning]  
    Action: get_iss_location_tool  
    Action Input: {{}}  
    Observation: [Result]  
    Final Answer: [Answer to the User]  

    **Example Query:**
    - Where is the ISS right now?  
      Thought: I need to find the ISS location.  
      Action: get_iss_location_tool  
      Action Input: {{}}  
      Observation: [Latitude: 23.5, Longitude: -45.3]  
      Final Answer: The ISS is currently at latitude 23.5 and longitude -45.3.  

    **Begin!**
    
    Question: {input}
    
    {agent_scratchpad}
    """
)

# Create the ReAct Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=iss_prompt.partial(tool_names=tool_names, tools=tool_descriptions)
)

# Define the Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=5,
    max_execution_time=30
)

# Log agent initialization
logging.info("🚀 ISS Locator Agent initialized.")

# Example execution
if __name__ == "__main__":
    query = "Where is the ISS right now?"
    response = agent_executor.invoke({"input": query})
    print("Agent Response:", response)
