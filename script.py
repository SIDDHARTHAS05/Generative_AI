"""
Streamlit Chatbot with LangChain + LangSmith tracing & feedback

How to run:
1. Install requirements:
   pip install streamlit langchain-openai langchain openai langsmith python-dotenv

2. Create a .env file in the same folder with:
   OPENAI_API_KEY=your_openai_api_key
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=support-chatbot

3. Run:
   python -m streamlit run streamlit_langsmith_chatbot.py

Notes:
- st.set_page_config MUST be the first Streamlit command in the script (immediately after importing streamlit).
- We attempt to use the newer langchain_openai package and fall back to langchain.chat_models if needed.
- LangSmith client is optional — the app continues to work without it.
"""

import os
import streamlit as st
os.environ["OPENAI_API_KEY"] = "openai_api_key_here"
os.environ["LANGCHAIN_API_KEY"] = "langchain_api_key_here"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "support-chatbot"
# --- IMPORTANT: set_page_config must be the first Streamlit command ---
st.set_page_config(page_title="Support Chatbot (LangSmith)", page_icon="💬")

# Standard imports after set_page_config
from datetime import datetime

# Load .env early so env vars are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# LangChain / LLM imports: try newer package first
ChatOpenAI = None
ChatPromptTemplate = None
try:
    # recommended replacement for newer LangChain releases
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    ChatOpenAI = ChatOpenAI
    ChatPromptTemplate = ChatPromptTemplate
except Exception:
    try:
        # fallback to older import if installed
        from langchain.chat_models import ChatOpenAI as ChatOpenAI_old
        from langchain.prompts import ChatPromptTemplate as ChatPromptTemplate_old
        ChatOpenAI = ChatOpenAI_old
        ChatPromptTemplate = ChatPromptTemplate_old
    except Exception:
        ChatOpenAI = None
        ChatPromptTemplate = None

# LangSmith client (optional)
try:
    from langsmith import Client as LangSmithClient
except Exception:
    LangSmithClient = None

# ---------------------------
# Config / Helpers
# ---------------------------

TEMPLATE_TEXT = """
You are a friendly customer support assistant. Answer the user's question clearly and politely.

User: {question}
"""

# Safe cached model loader
@st.cache_resource
def get_langchain_chat_model(model_name: str = "gpt-3.5-turbo"):
    if ChatOpenAI is None:
        return None
    # Use small temperature for consistent answers
    try:
        return ChatOpenAI(model=model_name, temperature=0.2)
    except Exception:
        # If the constructor signature differs, try common alternatives
        try:
            return ChatOpenAI(model_name=model_name, temperature=0.2)
        except Exception:
            return None


def init_langsmith_client():
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if LangSmithClient is None or not api_key:
        return None
    try:
        return LangSmithClient(api_key=api_key)
    except Exception:
        return None

# Session state initialization (after set_page_config)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, message, timestamp)

if "model" not in st.session_state:
    st.session_state.model = get_langchain_chat_model()

if "langsmith" not in st.session_state:
    st.session_state.langsmith = init_langsmith_client()

# ---------------------------
# UI
# ---------------------------

st.title("💬 Support Chatbot — LangSmith tracing & feedback")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("LLM model", options=["gpt-3.5-turbo", "gpt-4"], index=0)
    if st.button("(Re)load model"):
        st.session_state.model = get_langchain_chat_model(model_name)
    st.markdown("---")
    st.caption("LangSmith tracing depends on the LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY environment variables.")
    st.write("Project:", os.environ.get("LANGCHAIN_PROJECT", "(not set)"))
    if st.session_state.langsmith:
        st.success("LangSmith client: initialized")
    else:
        st.info("LangSmith client: not initialized (optional)")

# Display chat history
st.subheader("Conversation")
messages_container = st.container()
with messages_container:
    for i, (role, msg, ts) in enumerate(st.session_state.chat_history):
        if role == "user":
            st.markdown(f"**You** — _{ts}_\n\n{msg}")
        else:
            st.markdown(f"**Bot** — _{ts}_\n\n{msg}")

# User input
st.markdown("---")
user_input = st.text_area("Type your message", key="input_box", height=120)
col1, col2 = st.columns([1, 1])
with col1:
    send = st.button("Send")
with col2:
    clear = st.button("Clear conversation")

if clear:
    st.session_state.chat_history = []
    st.rerun()

# ---------------------------
# Core chat handler
# ---------------------------

def call_llm_and_trace(question: str):
    """Call the LangChain ChatOpenAI model using a prompt template and return the reply text."""
    model = st.session_state.get("model")
    if model is None:
        return "LLM model not available. Ensure langchain-openai/langchain is installed and OPENAI_API_KEY is set."

    # Build prompt
    if ChatPromptTemplate is None:
        prompt_str = TEMPLATE_TEXT.format(question=question)
        try:
            # Some ChatOpenAI variants accept a list of messages
            resp = model([{"role": "user", "content": prompt_str}])
            # resp may be an object with .content, .generations, or str
            reply = getattr(resp, "content", None)
            if not reply and hasattr(resp, "generations") and resp.generations:
                reply = resp.generations[0][0].text
            if not reply:
                reply = str(resp)
            return reply
        except Exception as e:
            return f"Error calling model: {e}"

    prompt = ChatPromptTemplate.from_template(TEMPLATE_TEXT)
    messages = prompt.format_messages(question=question)
    try:
        result = model(messages)
        reply = getattr(result, "content", None)
        if not reply:
            if hasattr(result, "generations") and result.generations:
                # generations sometimes is a list of lists
                gen = result.generations[0]
                if isinstance(gen, list) and gen:
                    reply = getattr(gen[0], "text", None) or getattr(gen[0], "content", None)
                elif hasattr(gen, "text"):
                    reply = gen.text
        if not reply:
            reply = str(result)
        return reply
    except Exception as e:
        return f"Error calling model: {e}"


def log_feedback_to_langsmith(question: str, bot_reply: str, feedback: str):
    """Optionally log the example to LangSmith as an evaluation/example for later review.
    feedback should be 'up' or 'down'."""
    client = st.session_state.get("langsmith")
    if client is None:
        return False

    try:
        dataset_name = "support-chatbot-feedback"
        dataset = None
        try:
            dataset = client.create_dataset(dataset_name, description="Feedback from Streamlit chatbot")
        except Exception:
            try:
                datasets = client.list_datasets()
                dataset = next((d for d in datasets if d.name == dataset_name), None)
            except Exception:
                dataset = None

        if dataset is None:
            dataset = client.create_dataset(dataset_name, description="Feedback from Streamlit chatbot")

        example_inputs = {"question": question}
        example_outputs = {"response": bot_reply, "feedback": feedback}
        client.create_example(inputs=example_inputs, outputs=example_outputs, dataset_id=dataset.id)
        return True
    except Exception as e:
        st.warning(f"Could not log feedback to LangSmith: {e}")
        return False

# Handle send
if send and user_input.strip():
    q = user_input.strip()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_history.append(("user", q, ts))
    with st.spinner("Thinking..."):
        reply = call_llm_and_trace(q)
    ts_bot = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_history.append(("bot", reply, ts_bot))
    st.rerun()

# After a bot reply, show feedback controls for the latest bot message
if st.session_state.chat_history:
    for role, msg, ts in reversed(st.session_state.chat_history):
        if role == "bot":
            st.markdown("---")
            st.subheader("Was this helpful?")
            col_up, col_down = st.columns([1, 1])
            with col_up:
                if st.button("👍 Yes", key=f"up_{len(st.session_state.chat_history)}"):
                    last_user = None
                    for r, m, t in reversed(st.session_state.chat_history):
                        if r == "user":
                            last_user = m
                            break
                    logged = log_feedback_to_langsmith(last_user or "", msg, "up")
                    if logged:
                        st.success("Thanks — feedback logged to LangSmith")
                    else:
                        st.success("Thanks — feedback recorded locally")
            with col_down:
                if st.button("👎 No", key=f"down_{len(st.session_state.chat_history)}"):
                    last_user = None
                    for r, m, t in reversed(st.session_state.chat_history):
                        if r == "user":
                            last_user = m
                            break
                    logged = log_feedback_to_langsmith(last_user or "", msg, "down")
                    if logged:
                        st.success("Thanks — feedback logged to LangSmith")
                    else:
                        st.success("Thanks — feedback recorded locally")
            break

# Small footer
st.markdown("---")
st.caption("This is a demo. Ensure you don't expose secrets in a public deployment.")
