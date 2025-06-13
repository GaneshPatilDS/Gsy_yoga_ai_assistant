import os
import json
import logging
import streamlit as st
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain_together import TogetherEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Set up logging
LOG_FILE = "chat_log.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

# Setup vectorstore
def setup_vectorstore():
    persist_directory = "vector_db_dir_2"
    embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore

# Setup conversational chain
def chat_chain(vectorstore):
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    memory = ConversationBufferWindowMemory(output_key="result", memory_key="chat_history", k=3, return_messages=True)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        verbose=True,
        return_source_documents=True
    )
    return chain

# Log conversation (formatted text log)
def log_conversation(user, assistant, filepath="chat_log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n🕒 {timestamp}\n")
        f.write(f"👤 User: {user}\n")
        f.write(f"🤖 Bot: {assistant}\n")
        f.write("--------------------------------------------------\n")

# Optional: JSON log for structured data
def log_json_conversation(user, assistant, filepath="chat_log.json"):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "assistant": assistant
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

# Streamlit Page Config
st.set_page_config(
    page_title="🧘 Govardhan School of Yoga Support System",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page Title and Intro
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #2E8B57;">🕉️ Welcome to Govardhan School of Yoga 🧘</h1>
        <h3 style="color: #6A5ACD;">Discover the union of body, mind, and soul with our AI support system. 🌸</h3>
    </div>
""", unsafe_allow_html=True)

# Sidebar Content
st.sidebar.title("🙏 Hare Krishna! 🌸")
st.sidebar.markdown("### 🌿 **How Can We Help You Today?**")
st.sidebar.info("Ask anything about yoga, our programs, and well-being. 🧘‍♂️")
st.sidebar.markdown("---")
st.sidebar.title("🌿 **Quick Yoga Tips**")

yoga_tips = [
    "🫁 Start your day with 5 minutes of deep breathing.",
    "🧎‍♂️ Maintain a proper posture while sitting and working.",
    "🙏 Practice mindfulness and gratitude daily.",
    "💧 Stay hydrated—drink plenty of water throughout the day.",
    "🕉️ Chanting 'Om' for a few minutes can help calm your mind.",
    "🌞 A few minutes of sun exposure boosts Vitamin D and positivity."
]

for tip in yoga_tips:
    st.sidebar.markdown(f"- {tip}")

st.sidebar.markdown("---")
st.sidebar.success("✨ Stay balanced, stay healthy! 🕉️")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Display Chat History
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(
                f'<div style="background-color: #E0F7FA; color: #000000; padding: 12px; border-radius: 12px; font-size: 16px;">{message["content"]}</div>',
                unsafe_allow_html=True
            )
    else:
        with st.chat_message("assistant"):
            st.markdown(
                f'<div style="background-color: #E8F5E9; color: #000000; padding: 12px; border-radius: 12px; font-size: 16px;">{message["content"]}</div>',
                unsafe_allow_html=True
            )

# User Input Field
user_input = st.chat_input("🌟 Ask a question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(
            f'<div style="background-color: #E0F7FA; color: #000000; padding: 12px; border-radius: 12px; font-size: 16px;">{user_input}</div>',
            unsafe_allow_html=True
        )

    try:
        response = st.session_state.conversational_chain.invoke({"query": user_input})
        assistant_response = response["result"]

        with st.chat_message("assistant"):
            st.markdown(
                f'<div style="background-color: #E8F5E9; color: #000000; padding: 12px; border-radius: 12px; font-size: 16px;">{assistant_response}</div>',
                unsafe_allow_html=True
            )

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Log conversation both to txt and json
        log_conversation(user_input, assistant_response)
        log_json_conversation(user_input, assistant_response)

    except Exception as e:
        with st.chat_message("assistant"):
            st.markdown(
                '<div style="background-color: #FFEBEE; color: #D32F2F; padding: 12px; border-radius: 12px;">❌ An error occurred. Please try again later.</div>',
                unsafe_allow_html=True
            )
        st.error(f"Error: {e}")
