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
NASA_API_KEY = os.getenv("NASA_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)

# NASA APOD API URL
APOD_API_URL = "https://api.nasa.gov/planetary/apod"

# Define APOD Fetcher Class
class APOD:
    def __init__(self):
        self.api_url = APOD_API_URL

    def get_apod(self):
        """
        Fetches NASA's Astronomy Picture of the Day (APOD) with retries.
        """
        retries = 3  # Number of retries
        for attempt in range(retries):
            try:
                response = requests.get(self.api_url, params={"api_key": NASA_API_KEY}, timeout=5)
                response.raise_for_status()
                data = response.json()

                if "url" not in data:
                    return {"error": "NASA APOD response is missing an image URL."}

                logging.info("✅ Successfully retrieved NASA APOD.")
                return {
                    "title": data.get("title", "Unknown Title"),
                    "description": data.get("explanation", "No description available."),
                    "date": data.get("date", "Unknown Date"),
                    "media_type": data.get("media_type", "unknown"),
                    "url": data["url"],
                }
            except requests.exceptions.RequestException as e:
                logging.error(f"GET request failed (Attempt {attempt+1}): {e}")
                time.sleep(2)  # Wait before retrying
        return {"error": "Failed to fetch NASA APOD after multiple attempts."}

# Define a function for the Tool
def get_apod(_input):
    """
    Wrapper function for fetching NASA's Astronomy Picture of the Day.
    This function ensures proper formatting to prevent output parsing errors.
    """
    apod_instance = APOD()
    data = apod_instance.get_apod()

    # Ensure response is always properly formatted
    if "error" in data:
        return {"text": "Error: Could not retrieve NASA APOD.", "image_url": None}

    # Check if media type is video and handle it differently
    if data["media_type"] == "video":
        response_text = (
            f"📺 **Title:** {data['title']}\n\n"
            f"🔹 **Description:** {data['description']}\n\n"
            f"🎥 **Watch here:** [Click to View]({data['url']})"
        )
        return {"text": response_text, "image_url": None}  # No image in this case

    # If media type is an image, return it normally
    response_text = (
        f"🖼 **Title:** {data['title']}\n\n"
        f"🔹 **Description:** {data['description']}\n\n"
        f"📅 **Date:** {data['date']}"
    )
    return {"text": response_text, "image_url": data["url"]}

# Define the LangChain tool
get_apod_tool = Tool(
    name="get_apod_tool",
    description="Fetches NASA's Astronomy Picture of the Day (APOD), providing an image, title, and description.",
    func=get_apod
)

# Define LLM with GPT-4o and low temperature
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1, response_format="json")

# Define available tools
tools = [get_apod_tool]

tool_names = ", ".join([tool.name for tool in tools])

tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# ✅ Define Prompt Template with agent_scratchpad as a variable
apod_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
    You are retrieving NASA's Astronomy Picture of the Day (APOD). Your job is to fetch the image, title, and description from NASA using the correct tool.

    ONLY follow this format:
    
    Thought: [Reasoning about next step]
    Action: get_apod_tool
    Action Input: {{}}
    Observation: [Result from API]
    Final Answer: [Formatted response to user]

    If you do not know the answer, say: "I need to use the tool to fetch APOD."

    **Example Query:**
    - Show me the NASA photo of the day.  
      Thought: I need to fetch the NASA Astronomy Picture of the Day (APOD).
      Action: get_apod_tool
      Action Input: {{}}
      Observation: Image URL, title, and description.
      Final Answer: Here is today's NASA APOD.

    **Begin!**
    
    Question: {input}
    
    {agent_scratchpad}
    """
)

# Create the ReAct Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=apod_prompt.partial(tool_names=tool_names, tools=tool_descriptions)
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
logging.info("🚀 NASA APOD Agent initialized.")

# Example execution
if __name__ == "__main__":
    query = "Show me the NASA photo of the day."
    response = agent_executor.invoke({"input": query})
    print("Agent Response:", response)
