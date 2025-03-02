import os
import logging
import streamlit as st
import urllib3
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException

## IMPORT ISS LOCATOR AGENT
from iss_locator import tools as iss_tools, iss_prompt

## IMPORT ISS LOCATOR AGENT
from astros import tools as astros_tools, astros_prompt

# IMPORT THE WEATHER AGENT
from weather import tools as weather_tools, weather_prompt

# IMPORT THE APOD
from apod import tools as apod_tools, apod_prompt

# ============================================================
# **🚀 Load Environment Variables**
# ============================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# **🔧 Configure Logging & Security**
# ============================================================
logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# **🤖 Define the LLM**
# ============================================================
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

# ============================================================
# **🌍 Initialize the ISS Agent**
# ============================================================
iss_agent = initialize_agent(
    tools=iss_tools, 
    llm=llm, 
    agent='zero-shot-react-description', 
    prompt=iss_prompt, 
    verbose=True
)

# ============================================================
# **🌍 Initialize the Astros in Space Agent**
# ============================================================
astros_agent = initialize_agent(
    tools=astros_tools, 
    llm=llm, 
    agent='zero-shot-react-description', 
    prompt=astros_prompt, 
    verbose=True
)

# ============================================================
# **🌍 Initialize the Weather Agent**
# ============================================================
weather_agent = initialize_agent(
    tools=weather_tools, 
    llm=llm, 
    agent='zero-shot-react-description', 
    prompt=weather_prompt, 
    verbose=True
)


# ============================================================
# **🌍 Initialize the Weather Agent**
# ============================================================
apod_agent = initialize_agent(
    tools=apod_tools, 
    llm=llm, 
    agent='zero-shot-react-description', 
    prompt=weather_prompt, 
    verbose=True
)
# Define ISS Agent Function
def iss_agent_func(input_text: str) -> str:
    return iss_agent.invoke(f"ISS: {input_text}")

# Define Astros Agent Function
def astros_agent_func(input_text: str) -> str:
    return astros_agent.invoke(f"Astronauts: {input_text}")

# Define Astros Agent Function
def weather_agent_func(input_text: str) -> str:
    return weather_agent.invoke(f"Weather: {input_text}")

def apod_agent_func(input_text: str) -> str:
    return apod_agent.invoke(input_text)

# Create a LangChain Tool for ISS Agent
iss_tool = Tool(
    name="ISS Locator",
    func=iss_agent_func,
    description="Use this to retrieve information about the International Space Station (ISS)."
)

# Create a LangChain Tool for Astros Agent
astros_tool = Tool(
    name="Astronauts in Space",
    func=astros_agent_func,
    description="Use this to retrieve information about the humans in space and their spacecraft."
)

# Create a LangChain Tool for Astros Agent
weather_tool = Tool(
    name="The Current Weather at a location on Earth",
    func=weather_agent_func,
    description="Use this to retrieve information about the current weather at a given latitude and longitude."
)

# Create a LangChain Tool for NASA APOD Agent
nasa_apod_tool_entry = Tool(
    name="NASA Astronomy Picture of the Day",
    func=apod_agent_func,
    description="Fetches NASA's Astronomy Picture of the Day and provides AI-powered analysis of the image."
)

# ============================================================
# **🤖 Main Parent Routing Agent**
# ============================================================
parent_tools = [iss_tool, astros_tool, *weather_tools, nasa_apod_tool_entry]

parent_agent = initialize_agent(
    tools=parent_tools,
    llm=llm,
    agent="zero-shot-react-description",
    handle_parsing_errors=True,
    verbose=True
)

logging.info(f"🚀 Main Parent Routing Agent Initialized with Tools: {[tool.name for tool in parent_tools]}")

# ============================================================
# **🛰️ Streamlit UI - Chat with the Space Ace**
# ============================================================
st.title("🌌 Chat with the Space Ace")
st.write("Ask real-time questions about Space!")

# User input text area
user_input = st.text_area("🚀 Enter your space-related question:")

# Conversation History (Stored in Session)
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Placeholder for APOD image and text
apod_image_url = None
apod_text = ""

if st.button("Send"):
    if not user_input:
        st.warning("⚠️ Please enter a question.")
    else:
        # 🚀 Invoke the Parent Agent with error handling
        response = parent_agent.invoke(user_input)
        logging.info(f"🛰️ Raw LLM Response Before Parsing:\n{response}")

        # ✅ Ensure response is properly formatted
        if isinstance(response, dict):  
            # 🔥 Fix: Extract the correct output field
            apod_text = response.get("output", "No valid response received.")  
            
            # 🔥 Fix: Extract the correct media URL
            apod_image_url = response.get("url", None)  
        elif isinstance(response, str):  
            apod_text = response
        else:
            apod_text = "No valid response received."

        # ✅ Display AI Agent's response
        st.write(f"### **🛰️ Question:** {user_input}")
        st.write(f"### **📡 Response:** {apod_text}")

        # ✅ Save conversation history
        st.session_state.conversation.append({"role": "user", "content": user_input})
        st.session_state.conversation.append({"role": "assistant", "content": apod_text})

        # ✅ Ensure APOD Image or Video Displays Correctly
        if apod_image_url:
            if "youtube.com" in apod_image_url or "vimeo.com" in apod_image_url:  
                st.video(apod_image_url)  # If it's a video, display it
            else:
                st.image(apod_image_url, caption="📸 NASA Astronomy Picture of the Day", use_column_width=True)

# ============================================================
# **📜 Display Conversation History**
# ============================================================
st.write("### 💬 Conversation History")
for chat in st.session_state.conversation:
    role = "🧑‍🚀 You" if chat["role"] == "user" else "🤖 Space Ace"
    st.write(f"**{role}:** {chat['content']}")
