#!/usr/bin/env python3
"""
Advanced Streamlit App for GenAI RAG Chatbot
Features: Fast inference, Modern UI, Real-time streaming, Analytics
"""
import streamlit as st
import sys
import os
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import load_config, get_config
from rag.retriever import RAGRetriever
from llm.llm_handler import LLMHandler
from prompt_handler import PromptHandler, QueryType

# Page configuration
st.set_page_config(
    page_title="🤖 AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Modern dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
    }
    
    /* Main container */
    .main .block-container {
        padding: 1rem 2rem;
        max-width: 1400px;
    }
    
    /* Chat container styling */
    .chat-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Message bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        animation: slideInRight 0.3s ease;
    }
    
    .assistant-bubble {
        background: rgba(255, 255, 255, 0.08);
        color: #e2e8f0;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: slideInLeft 0.3s ease;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #6366f1;
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        color: #94a3b8;
        font-size: 1rem;
    }
    
    /* Stat cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Quick action buttons */
    .quick-btn {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.25rem;
        display: inline-block;
    }
    
    .quick-btn:hover {
        background: rgba(99, 102, 241, 0.4);
        transform: scale(1.05);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .status-online {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .status-offline {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Source card */
    .source-card {
        background: rgba(99, 102, 241, 0.1);
        border-left: 3px solid #6366f1;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        color: white !important;
        padding: 1rem 1.5rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
    
    /* Chat message avatars */
    .stChatMessage {
        background: transparent !important;
    }
    
    /* Response time badge */
    .response-time {
        display: inline-block;
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    
    /* Floating action */
    .floating-stats {
        position: fixed;
        top: 70px;
        right: 20px;
        background: rgba(15, 15, 35, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)


# ============== CACHING & OPTIMIZATION ==============

# Thread pool for parallel operations
executor = ThreadPoolExecutor(max_workers=4)

# Response cache for instant replies
@st.cache_data(ttl=3600, max_entries=1000)
def get_cached_response(query_hash: str):
    """Get cached response if available."""
    return None

def cache_response(query: str, response: dict):
    """Cache a response for future use."""
    if 'response_cache' not in st.session_state:
        st.session_state.response_cache = {}
    
    query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
    st.session_state.response_cache[query_hash] = {
        'response': response,
        'timestamp': time.time()
    }

def get_response_from_cache(query: str):
    """Check if response is in cache."""
    if 'response_cache' not in st.session_state:
        return None
    
    query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
    cached = st.session_state.response_cache.get(query_hash)
    
    if cached and (time.time() - cached['timestamp']) < 3600:  # 1 hour cache
        return cached['response']
    return None


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'messages': [],
        'rag_retriever': None,
        'llm_handler': None,
        'chat_history': [],
        'query_analytics': [],
        'conversation_memory': [],
        'last_context': "",
        'system_initialized': False,
        'total_queries': 0,
        'avg_response_time': 0,
        'successful_queries': 0,
        'response_cache': {},
        'prompt_handler': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def initialize_rag_system():
    """Initialize RAG system with caching."""
    try:
        config = load_config()
        rag_config = config.rag
        
        retriever = RAGRetriever(
            knowledge_base_path=rag_config.knowledge_base_path,
            vector_store_path=rag_config.vector_store_path,
            embedding_model=rag_config.embedding_model,
            chunk_size=rag_config.chunk_size,
            chunk_overlap=rag_config.chunk_overlap,
            top_k=rag_config.top_k,
            similarity_threshold=rag_config.similarity_threshold
        )
        
        retriever.initialize()
        return retriever
    except Exception as e:
        return None


@st.cache_resource(show_spinner=False)
def initialize_llm_handler():
    """Initialize LLM handler with caching."""
    try:
        config = load_config()
        llm_config = config.llm
        
        handler = LLMHandler(
            ollama_base_url=llm_config.ollama_base_url,
            ollama_model=llm_config.ollama_model,
            openai_api_key=llm_config.openai_api_key,
            openai_model=llm_config.openai_model,
            max_tokens=256,  # Reduced for faster responses
            temperature=llm_config.temperature,
            timeout=15  # Faster timeout
        )
        
        return handler
    except Exception as e:
        return None


@st.cache_resource(show_spinner=False)
def initialize_prompt_handler():
    """Initialize prompt handler for edge cases."""
    return PromptHandler()


# ============== FAST RESPONSE GENERATION ==============

def quick_rag_retrieve(query: str, top_k: int = 2):
    """Fast RAG retrieval with reduced chunks."""
    if not st.session_state.rag_retriever:
        return "", [], 0
    
    try:
        response = st.session_state.rag_retriever.retrieve(query, top_k=top_k)
        return response.context[:1500], response.sources[:3], response.confidence
    except:
        return "", [], 0


def generate_fast_response(query: str, context: str, conversation_history: str = "") -> str:
    """Generate response with optimized prompting."""
    
    # Concise system prompt for faster inference
    system_prompt = """You are a helpful AI assistant. Answer concisely and accurately based on the context provided. If unsure, say so briefly."""
    
    # Build minimal prompt
    prompt_parts = []
    
    if context:
        prompt_parts.append(f"Context:\n{context[:1000]}\n")
    
    if conversation_history:
        prompt_parts.append(f"Recent chat:\n{conversation_history[-500:]}\n")
    
    prompt_parts.append(f"Question: {query}\n\nAnswer:")
    
    full_prompt = "\n".join(prompt_parts)
    
    if st.session_state.llm_handler:
        try:
            response = st.session_state.llm_handler.generate(
                prompt=full_prompt,
                system_prompt=system_prompt
            )
            if response.success and response.content:
                return response.content
        except:
            pass
    
    # Fallback response
    if context:
        sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 20][:3]
        return "Based on the knowledge base:\n\n" + ". ".join(sentences) + "."
    
    return "I couldn't find specific information about that. Please try rephrasing your question."


# ============== UI COMPONENTS ==============

def render_header():
    """Render modern header."""
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🤖 AI Knowledge Assistant</div>
        <div class="main-subtitle">Powered by RAG • Fast & Intelligent Responses</div>
    </div>
    """, unsafe_allow_html=True)


def render_status_bar():
    """Render status indicators."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rag_status = "status-online" if st.session_state.rag_retriever else "status-offline"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">🔍</div>
            <div class="stat-label">RAG System</div>
            <span class="status-badge {rag_status}">{'● Active' if st.session_state.rag_retriever else '○ Offline'}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        llm_status = "status-online" if st.session_state.llm_handler else "status-offline"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">🧠</div>
            <div class="stat-label">LLM Engine</div>
            <span class="status-badge {llm_status}">{'● Ready' if st.session_state.llm_handler else '○ Offline'}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_time = st.session_state.avg_response_time
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">⚡</div>
            <div class="stat-label">Avg Response</div>
            <span class="status-badge status-online">{avg_time:.1f}s</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        queries = st.session_state.total_queries
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">📊</div>
            <div class="stat-label">Total Queries</div>
            <span class="status-badge status-online">{queries}</span>
        </div>
        """, unsafe_allow_html=True)


def render_quick_actions():
    """Render quick action buttons."""
    st.markdown("#### 💡 Quick Questions")
    
    quick_questions = [
        "What services do you offer?",
        "Tell me about refund policy",
        "How to get started?",
        "Technical documentation",
        "Contact support"
    ]
    
    cols = st.columns(len(quick_questions))
    for i, question in enumerate(quick_questions):
        with cols[i]:
            if st.button(f"📌 {question[:20]}...", key=f"quick_{i}", use_container_width=True):
                st.session_state.quick_question = question


def render_chat_messages():
    """Render chat messages with modern styling."""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"""<div class="user-bubble">{msg["content"]}</div>""", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])
                
                # Show metadata
                meta_parts = []
                if "response_time" in msg:
                    meta_parts.append(f"⚡ {msg['response_time']:.2f}s")
                if "query_type" in msg:
                    meta_parts.append(f"📝 {msg['query_type']}")
                if "confidence" in msg and msg["confidence"] > 0:
                    meta_parts.append(f"🎯 {msg['confidence']:.0%}")
                
                if meta_parts:
                    st.caption(" • ".join(meta_parts))
                
                # Show sources
                if msg.get("sources"):
                    with st.expander("📚 Sources", expanded=False):
                        for i, src in enumerate(msg["sources"][:3], 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>{i}.</strong> {Path(src.get('source', 'Unknown')).name}
                                <small style="float:right; color: #6366f1;">{src.get('score', 0):.0%}</small>
                            </div>
                            """, unsafe_allow_html=True)


def process_query_fast(query: str):
    """Process query with optimized speed."""
    start_time = time.time()
    
    # Add user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Check cache first
    cached = get_response_from_cache(query)
    if cached:
        response_time = time.time() - start_time
        cached['response_time'] = response_time
        cached['from_cache'] = True
        st.session_state.messages.append(cached)
        update_analytics(query, response_time, True, "cached", cached.get('confidence', 0.9))
        return
    
    # Initialize prompt handler
    handler = st.session_state.prompt_handler or PromptHandler()
    
    # Quick edge case check
    analysis = handler.analyze_query(query)
    
    if not analysis.should_use_rag and analysis.direct_response:
        # Instant response for edge cases
        response_time = time.time() - start_time
        response = {
            "role": "assistant",
            "content": analysis.direct_response,
            "sources": [],
            "response_time": response_time,
            "confidence": analysis.confidence,
            "query_type": analysis.query_type.value
        }
        st.session_state.messages.append(response)
        cache_response(query, response)
        update_analytics(query, response_time, True, analysis.query_type.value, analysis.confidence)
        return
    
    # RAG + LLM processing
    try:
        # Fast retrieval (reduced chunks)
        context, sources, confidence = quick_rag_retrieve(
            analysis.modified_query or query, 
            top_k=2
        )
        
        # Check for follow-up and use cached context
        if is_followup_question(query) and not context and st.session_state.last_context:
            context = st.session_state.last_context
            confidence = 0.6
        
        # Build minimal conversation history
        conv_history = ""
        if st.session_state.conversation_memory:
            last = st.session_state.conversation_memory[-1]
            conv_history = f"Q: {last.get('query', '')}\nA: {last.get('answer', '')[:200]}"
        
        # Generate response
        answer = generate_fast_response(query, context, conv_history)
        
        response_time = time.time() - start_time
        
        response = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "response_time": response_time,
            "confidence": confidence,
            "query_type": "rag"
        }
        
        st.session_state.messages.append(response)
        
        # Update memory
        st.session_state.conversation_memory.append({
            "query": query,
            "answer": answer[:300],
            "context": context[:300] if context else ""
        })
        if len(st.session_state.conversation_memory) > 5:  # Keep only 5 for speed
            st.session_state.conversation_memory = st.session_state.conversation_memory[-5:]
        
        st.session_state.last_context = context
        
        # Cache the response
        cache_response(query, response)
        update_analytics(query, response_time, True, "rag", confidence)
        
    except Exception as e:
        response_time = time.time() - start_time
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I encountered an issue processing your request. Please try again.",
            "response_time": response_time,
            "query_type": "error"
        })
        update_analytics(query, response_time, False, "error", 0)


def is_followup_question(query: str) -> bool:
    """Fast follow-up detection."""
    q = query.lower()
    indicators = ['it', 'this', 'that', 'more', 'also', 'what about', 'how about', 'why', 'can you']
    return len(q.split()) <= 4 or any(q.startswith(i) for i in indicators)


def update_analytics(query: str, response_time: float, success: bool, query_type: str, confidence: float):
    """Update analytics data."""
    st.session_state.total_queries += 1
    if success:
        st.session_state.successful_queries += 1
    
    analytics = st.session_state.query_analytics
    analytics.append({
        "query": query[:50],
        "response_time": response_time,
        "confidence": confidence,
        "success": success,
        "query_type": query_type,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 100 analytics
    if len(analytics) > 100:
        analytics = analytics[-100:]
    
    st.session_state.query_analytics = analytics
    st.session_state.avg_response_time = sum(a["response_time"] for a in analytics[-20:]) / min(len(analytics), 20)


def render_analytics_tab():
    """Render analytics dashboard."""
    st.markdown("### 📊 Performance Analytics")
    
    if not st.session_state.query_analytics:
        st.info("No analytics data yet. Start chatting to see performance metrics!")
        return
    
    df = pd.DataFrame(st.session_state.query_analytics)
    
    # Response time chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df['response_time'],
            mode='lines+markers',
            name='Response Time',
            line=dict(color='#6366f1', width=2),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="Response Time Trend",
            yaxis_title="Seconds",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        query_types = df['query_type'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=query_types.index,
            values=query_types.values,
            hole=0.4,
            marker_colors=['#6366f1', '#a855f7', '#ec4899', '#22c55e']
        )])
        fig.update_layout(
            title="Query Types",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", len(df))
    with col2:
        st.metric("Avg Response", f"{df['response_time'].mean():.2f}s")
    with col3:
        st.metric("Success Rate", f"{(df['success'].sum() / len(df) * 100):.1f}%")
    with col4:
        st.metric("Avg Confidence", f"{df['confidence'].mean():.0%}")


def render_knowledge_base_tab():
    """Render knowledge base explorer."""
    st.markdown("### 📚 Knowledge Base")
    
    if not st.session_state.rag_retriever:
        st.warning("RAG system not initialized")
        return
    
    try:
        stats = st.session_state.rag_retriever.vector_store.get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Documents", stats.get('total_documents', 0))
        with col2:
            st.metric("🧩 Chunks", stats.get('total_chunks', 0))
        with col3:
            st.metric("📏 Avg Chunk Size", f"{stats.get('avg_chunk_size', 0):.0f}")
        
        # List documents
        kb_path = Path(project_root / "knowledge_base")
        if kb_path.exists():
            st.markdown("#### Available Documents")
            for doc in kb_path.glob("*.md"):
                with st.expander(f"📄 {doc.name}"):
                    content = doc.read_text()[:500]
                    st.text(content + "...")
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")


def render_settings_tab():
    """Render settings panel."""
    st.markdown("### ⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### System Controls")
        
        if st.button("🔄 Reinitialize System", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.response_cache = {}
            st.rerun()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_memory = []
            st.session_state.last_context = ""
            st.rerun()
        
        if st.button("📊 Reset Analytics", use_container_width=True):
            st.session_state.query_analytics = []
            st.session_state.total_queries = 0
            st.session_state.successful_queries = 0
            st.session_state.avg_response_time = 0
            st.rerun()
        
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.session_state.response_cache = {}
            st.success("Cache cleared!")
    
    with col2:
        st.markdown("#### System Info")
        
        st.info(f"""
        **RAG Status:** {'✅ Active' if st.session_state.rag_retriever else '❌ Offline'}
        
        **LLM Status:** {'✅ Ready' if st.session_state.llm_handler else '❌ Offline'}
        
        **Cached Responses:** {len(st.session_state.get('response_cache', {}))}
        
        **Memory Exchanges:** {len(st.session_state.conversation_memory)}/5
        """)


# ============== MAIN APP ==============

def main():
    """Main application."""
    init_session_state()
    
    # Initialize systems in background
    if not st.session_state.system_initialized:
        with st.spinner("🚀 Initializing AI systems..."):
            st.session_state.rag_retriever = initialize_rag_system()
            st.session_state.llm_handler = initialize_llm_handler()
            st.session_state.prompt_handler = initialize_prompt_handler()
            st.session_state.system_initialized = True
    
    # Render header
    render_header()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "📚 Knowledge Base", "⚙️ Settings"])
    
    with tab1:
        render_status_bar()
        st.markdown("---")
        
        # Quick actions
        render_quick_actions()
        
        st.markdown("---")
        
        # Chat messages
        render_chat_messages()
        
        # Check for quick question
        if 'quick_question' in st.session_state and st.session_state.quick_question:
            query = st.session_state.quick_question
            st.session_state.quick_question = None
            process_query_fast(query)
            st.rerun()
    
    with tab2:
        render_analytics_tab()
    
    with tab3:
        render_knowledge_base_tab()
    
    with tab4:
        render_settings_tab()
    
    # Chat input (always at bottom)
    if prompt := st.chat_input("Ask me anything...", key="main_chat"):
        process_query_fast(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
