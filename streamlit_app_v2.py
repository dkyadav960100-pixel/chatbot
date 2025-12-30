#!/usr/bin/env python3
"""
Clean & Modern Streamlit Chat App
User-friendly interface with excellent readability
"""
import streamlit as st
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import load_config
from rag.retriever import RAGRetriever
from llm.llm_handler import LLMHandler
from prompt_handler import PromptHandler

# Page config
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, modern CSS
st.markdown("""
<style>
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Main container */
    .main .block-container {
        padding: 1rem 2rem 6rem 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-title {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    
    .main-title h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-title p {
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Status cards */
    .status-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .status-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        text-align: center;
        min-width: 120px;
    }
    
    .status-card .icon {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }
    
    .status-card .label {
        color: #a0aec0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-card .value {
        color: #4ade80;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* User message */
    .user-msg {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1rem;
    }
    
    .user-msg .bubble {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 75%;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    /* Bot message */
    .bot-msg {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1rem;
    }
    
    .bot-msg .bubble {
        background: rgba(255,255,255,0.95);
        color: #1a1a2e;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        max-width: 85%;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .bot-msg .bubble h1, .bot-msg .bubble h2, .bot-msg .bubble h3 {
        color: #1a1a2e;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .bot-msg .bubble p {
        margin: 0.5rem 0;
    }
    
    .bot-msg .bubble ul, .bot-msg .bubble ol {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .bot-msg .bubble li {
        margin: 0.3rem 0;
    }
    
    .bot-msg .bubble strong {
        color: #6366f1;
    }
    
    /* Avatar styling */
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin: 0 0.5rem;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    }
    
    .bot-avatar {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
    }
    
    /* Quick questions */
    .quick-questions {
        margin-bottom: 1rem;
    }
    
    .quick-questions-title {
        color: #a0aec0;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .quick-btn-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
    }
    
    /* Input area styling */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 25px !important;
        color: white !important;
        padding: 0.8rem 1.2rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.5) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Response time badge */
    .response-time {
        text-align: right;
        color: #a0aec0;
        font-size: 0.75rem;
        margin-top: 0.3rem;
        padding-right: 0.5rem;
    }
    
    /* Welcome message */
    .welcome-box {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .welcome-box h3 {
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .welcome-box p {
        color: #a0aec0;
        margin: 0;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.2);
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.3);
    }
    
    /* Loading animation */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 0.5rem;
    }
    
    .typing-indicator span {
        width: 8px;
        height: 8px;
        background: #6366f1;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out;
    }
    
    .typing-indicator span:nth-child(1) { animation-delay: 0s; }
    .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-8px); }
    }
    
    /* Hide Streamlit branding */
    .viewerBadge_container__1QSob {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state."""
    defaults = {
        'messages': [],
        'rag_retriever': None,
        'llm_handler': None,
        'initialized': False,
        'total_queries': 0,
        'conversation_memory': [],
        'last_context': ''
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def init_rag():
    """Initialize RAG system."""
    try:
        config = load_config()
        retriever = RAGRetriever(
            knowledge_base_path=config.rag.knowledge_base_path,
            vector_store_path=config.rag.vector_store_path,
            embedding_model=config.rag.embedding_model,
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            top_k=config.rag.top_k,
            similarity_threshold=config.rag.similarity_threshold
        )
        retriever.initialize()
        return retriever
    except Exception as e:
        st.error(f"RAG Error: {e}")
        return None


@st.cache_resource
def init_llm():
    """Initialize LLM handler."""
    try:
        config = load_config()
        return LLMHandler(
            ollama_base_url=config.llm.ollama_base_url,
            ollama_model=config.llm.ollama_model,
            openai_api_key=config.llm.openai_api_key,
            max_tokens=256,
            temperature=0.7,
            timeout=15
        )
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return None


def render_header():
    """Render the header."""
    st.markdown("""
    <div class="main-title">
        <h1>🤖 AI Knowledge Assistant</h1>
        <p>Ask me anything about our products, services, and policies</p>
    </div>
    """, unsafe_allow_html=True)


def render_status():
    """Render status indicators."""
    rag_status = "Active" if st.session_state.rag_retriever else "Offline"
    llm_status = "Ready" if st.session_state.llm_handler else "Offline"
    
    st.markdown(f"""
    <div class="status-container">
        <div class="status-card">
            <div class="icon">🔍</div>
            <div class="label">Knowledge Base</div>
            <div class="value">{rag_status}</div>
        </div>
        <div class="status-card">
            <div class="icon">🧠</div>
            <div class="label">AI Engine</div>
            <div class="value">{llm_status}</div>
        </div>
        <div class="status-card">
            <div class="icon">💬</div>
            <div class="label">Queries</div>
            <div class="value">{st.session_state.total_queries}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_chat_messages():
    """Render chat messages with custom styling."""
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-box">
            <h3>👋 Welcome!</h3>
            <p>I'm here to help you find information. Ask me a question or try one of the quick questions below.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    chat_html = '<div class="chat-container">'
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f'''
            <div class="user-msg">
                <div class="bubble">{msg["content"]}</div>
                <div class="avatar user-avatar">👤</div>
            </div>
            '''
        else:
            # Convert markdown-like formatting to HTML
            content = msg["content"]
            content = content.replace('\n\n', '</p><p>')
            content = content.replace('\n', '<br>')
            content = content.replace('**', '<strong>').replace('**', '</strong>')
            
            response_time = msg.get("response_time", 0)
            time_str = f'<div class="response-time">⚡ {response_time:.2f}s</div>' if response_time else ''
            
            chat_html += f'''
            <div class="bot-msg">
                <div class="avatar bot-avatar">🤖</div>
                <div class="bubble"><p>{content}</p></div>
            </div>
            {time_str}
            '''
    
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)


def render_quick_questions():
    """Render quick question buttons."""
    questions = [
        "What services do you offer?",
        "How do I get a refund?",
        "Tell me about pricing",
        "How to contact support?",
        "What are system requirements?"
    ]
    
    st.markdown('<p class="quick-questions-title">💡 Quick Questions</p>', unsafe_allow_html=True)
    
    cols = st.columns(len(questions))
    for i, q in enumerate(questions):
        with cols[i]:
            if st.button(q[:20] + "...", key=f"quick_{i}", use_container_width=True):
                return q
    return None


def process_message(query: str):
    """Process user message and generate response."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.total_queries += 1
    
    start_time = time.time()
    
    # Check for edge cases
    handler = PromptHandler()
    analysis = handler.analyze_query(query)
    
    if not analysis.should_use_rag and analysis.direct_response:
        response = analysis.direct_response
        response_time = time.time() - start_time
    else:
        # RAG processing
        response = generate_response(query)
        response_time = time.time() - start_time
    
    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "response_time": response_time
    })
    
    # Update memory
    st.session_state.conversation_memory.append({
        "query": query,
        "response": response[:500]
    })
    if len(st.session_state.conversation_memory) > 5:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-5:]


def generate_response(query: str) -> str:
    """Generate response using RAG + LLM."""
    context = ""
    
    # Retrieve context
    if st.session_state.rag_retriever:
        try:
            rag_response = st.session_state.rag_retriever.retrieve(query, top_k=3)
            context = rag_response.context
            st.session_state.last_context = context
        except Exception as e:
            pass
    
    # Generate with LLM
    if st.session_state.llm_handler and context:
        try:
            llm_response = st.session_state.llm_handler.generate(
                prompt=query,
                context=context
            )
            if llm_response.success and llm_response.content:
                return llm_response.content
        except Exception:
            pass
    
    # Fallback response
    if context:
        return f"Based on our knowledge base:\n\n{context[:1000]}"
    
    return """I couldn't find specific information about that in our knowledge base.

**I can help you with:**
- Product information and features
- Pricing and plans
- Technical support
- Company policies
- Getting started guides

Please try asking about one of these topics!"""


def main():
    """Main application."""
    init_session_state()
    
    # Initialize systems
    if not st.session_state.initialized:
        with st.spinner("🚀 Starting AI Assistant..."):
            st.session_state.rag_retriever = init_rag()
            st.session_state.llm_handler = init_llm()
            st.session_state.initialized = True
    
    # Render UI components
    render_header()
    render_status()
    
    # Quick questions
    quick_q = render_quick_questions()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Chat messages
    render_chat_messages()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Input area
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="Type your question here...",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_clicked = st.button("Send 📤", use_container_width=True)
    
    # Process input
    if send_clicked and user_input:
        process_message(user_input)
        st.rerun()
    elif quick_q:
        process_message(quick_q)
        st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_memory = []
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 📊 Session Stats")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Queries", st.session_state.total_queries)
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.info("""
        **AI Knowledge Assistant**
        
        Powered by:
        - RAG Technology
        - LLM Integration
        - Smart Responses
        """)


if __name__ == "__main__":
    main()
