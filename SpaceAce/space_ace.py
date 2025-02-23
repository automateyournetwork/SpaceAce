import os
import logging
import streamlit as st
import urllib3
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI

## IMPORT ISS LOCATOR AGENT
from iss_locator import tools as iss_tools, iss_prompt

## IMPORT ISS LOCATOR AGENT
from astros import tools as astros_tools, astros_prompt

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

# Define ISS Agent Function
def iss_agent_func(input_text: str) -> str:
    return iss_agent.invoke(f"ISS: {input_text}")

# Define Astros Agent Function
def astros_agent_func(input_text: str) -> str:
    return astros_agent.invoke(f"Astronauts: {input_text}")

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

# ============================================================
# **🤖 Main Parent Routing Agent**
# ============================================================
parent_tools = [iss_tool, astros_tool]

parent_agent = initialize_agent(
    tools=parent_tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

logging.info(f"🚀 Main Parent Routing Agent Initialized with Tools: {[tool.name for tool in parent_tools]}")

# ============================================================
# **🛰️ Streamlit UI - Chat with the ISS Agent**
# ============================================================
st.title("🌌 Chat with the Space Ace")
st.write("Ask real-time questions about the Space!")

# User input text area
user_input = st.text_area("🚀 Enter your space-related question:")

# Conversation History (Stored in Session)
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if st.button("Send"):
    if not user_input:
        st.warning("⚠️ Please enter a question.")
    else:
        # 🚀 Invoke the ISS Agent
        response = parent_agent.invoke(user_input)

        # ✅ Extract response text
        response_text = response.get("output", "No valid response received.")

        # ✅ Display AI Agent's response
        st.write(f"### **🛰️ Question:** {user_input}")
        st.write(f"### **📡 Response:** {response_text}")

        # ✅ Save conversation history
        st.session_state.conversation.append({"role": "user", "content": user_input})
        st.session_state.conversation.append({"role": "assistant", "content": response_text})

# ============================================================
# **📜 Display Conversation History**
# ============================================================
st.write("### 💬 Conversation History")
for chat in st.session_state.conversation:
    role = "🧑‍🚀 You" if chat["role"] == "user" else "🤖 Space Ace"
    st.write(f"**{role}:** {chat['content']}")
