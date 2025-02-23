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

# API URL for astros location
ASTROS_API_URL = "http://api.open-notify.org/astros.json"

# astros Locator class
class Astros:
    def __init__(self):
        self.api_url = ASTROS_API_URL

    def get_astros(self):
        """
        Fetches the people in space and what spacecraft they are on with retries.
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
        return {"error": "Failed to fetch Astros after multiple attempts."}

# Define a function for the Tool
def get_astros(_input):
    """
    Wrapper function for fetching humans in space and their spacecraft.
    """
    locator = Astros()
    return locator.get_astros()

# Define the LangChain tool
get_astros_tool = Tool(
    name="get_astros_tool",
    description="Fetches the current humans in space and the spacecraft they are on.",
    func=get_astros
)

# Define LLM with GPT-4o and low temperature
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

# Define available tools
tools = [get_astros_tool]

tool_names = ", ".join([tool.name for tool in tools])

tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# ✅ Define Prompt Template with agent_scratchpad as a variable
astros_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
    You are tracking the astronaughts in space along with the spacecraft they are on. Your job is to provide the information about the humans in space.

    **FORMAT:**  
    Thought: [Your reasoning]  
    Action: get_astros_tool  
    Action Input: {{}}  
    Observation: [Result]  
    Final Answer: [Answer to the User]  

    **Example Query:**
    - Who is in space right now?  
      Thought: I need to find the who is in space.  
      Action: get_astros_tool  
      Action Input: {{}}  
      Observation: Oleg Kononenko
      Final Answer: Oleg Kononenko is in space right now.  

    **Begin!**
    
    Question: {input}
    
    {agent_scratchpad}
    """
)

# Create the ReAct Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=astros_prompt.partial(tool_names=tool_names, tools=tool_descriptions)
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
logging.info("🚀 Astros in Space Agent initialized.")

# Example execution
if __name__ == "__main__":
    query = "Who is in space right now?"
    response = agent_executor.invoke({"input": query})
    print("Agent Response:", response)
