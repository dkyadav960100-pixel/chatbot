#!/usr/bin/env python3
"""
Advanced Streamlit App for GenAI RAG Chatbot
Features: Chat interface, RAG visualization, Analytics, Image processing
"""
import streamlit as st
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import load_config, get_config
from rag.retriever import RAGRetriever
from llm.llm_handler import LLMHandler
from vision.vision_handler import VisionHandler

# Page configuration
st.set_page_config(
    page_title="GenAI RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        color: #1a1a2e;
        margin-right: 2rem;
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Source card styling */
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_retriever' not in st.session_state:
        st.session_state.rag_retriever = None
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = None
    if 'vision_handler' not in st.session_state:
        st.session_state.vision_handler = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'query_analytics' not in st.session_state:
        st.session_state.query_analytics = []
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = []  # For follow-up questions
    if 'last_context' not in st.session_state:
        st.session_state.last_context = ""  # Store last retrieved context
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'total_queries' not in st.session_state:
        st.session_state.total_queries = 0
    if 'avg_response_time' not in st.session_state:
        st.session_state.avg_response_time = 0
    if 'successful_queries' not in st.session_state:
        st.session_state.successful_queries = 0


@st.cache_resource
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
        st.error(f"Failed to initialize RAG system: {e}")
        return None


@st.cache_resource
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
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature
        )
        
        return handler
    except Exception as e:
        st.error(f"Failed to initialize LLM handler: {e}")
        return None


@st.cache_resource
def initialize_vision_handler():
    """Initialize Vision handler with caching."""
    try:
        config = load_config()
        vision_config = config.vision
        
        handler = VisionHandler(
            model_type=vision_config.model_name,
            device=vision_config.device,
            max_image_size=vision_config.max_image_size,
            num_tags=vision_config.num_tags
        )
        
        return handler
    except Exception as e:
        st.error(f"Failed to initialize Vision handler: {e}")
        return None


def render_sidebar():
    """Render sidebar with controls and info."""
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        st.markdown("---")
        
        # System Status
        st.markdown("### 📊 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.rag_retriever:
                st.success("RAG ✓")
            else:
                st.error("RAG ✗")
        
        with col2:
            if st.session_state.llm_handler:
                provider = st.session_state.llm_handler.get_available_provider()
                st.success(f"LLM ✓")
            else:
                st.error("LLM ✗")
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        top_k = st.slider("Number of context chunks", 1, 10, 3, key="top_k_slider")
        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.1, key="temp_slider")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### 🚀 Quick Actions")
        
        if st.button("🔄 Reinitialize System", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.system_initialized = False
            st.rerun()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.conversation_memory = []
            st.session_state.last_context = ""
            st.rerun()
        
        if st.button("📊 Reset Analytics", use_container_width=True):
            st.session_state.query_analytics = []
            st.session_state.total_queries = 0
            st.session_state.avg_response_time = 0
            st.session_state.successful_queries = 0
            st.rerun()
        
        st.markdown("---")
        
        # Memory Status
        st.markdown("### 🧠 Conversation Memory")
        memory_count = len(st.session_state.get('conversation_memory', []))
        st.metric("Exchanges Remembered", f"{memory_count}/10")
        if memory_count > 0:
            st.caption("Bot remembers recent conversation for follow-up questions")
        
        st.markdown("---")
        
        # Info Section
        st.markdown("### ℹ️ About")
        st.info("""
        **GenAI RAG Assistant**
        
        A powerful chatbot powered by:
        - 🔍 RAG (Retrieval-Augmented Generation)
        - 🧠 LLM (Ollama/OpenAI)
        - 💬 Conversation Memory
        - 👁️ Vision Processing
        
        Ask questions & follow-ups!
        """)
        
        # Knowledge Base Stats
        if st.session_state.rag_retriever:
            stats = st.session_state.rag_retriever.vector_store.get_stats()
            st.markdown("### 📚 Knowledge Base")
            st.metric("Documents", stats.get('total_documents', 0))
            st.metric("Chunks", stats.get('total_chunks', 0))


def render_metrics_dashboard():
    """Render the metrics dashboard."""
    st.markdown("### 📈 Analytics Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Queries</div>
        </div>
        """.format(st.session_state.total_queries), unsafe_allow_html=True)
    
    with col2:
        success_rate = (st.session_state.successful_queries / max(st.session_state.total_queries, 1)) * 100
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        """.format(success_rate), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.2f}s</div>
            <div class="metric-label">Avg Response Time</div>
        </div>
        """.format(st.session_state.avg_response_time), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Chat Messages</div>
        </div>
        """.format(len(st.session_state.messages)), unsafe_allow_html=True)


def render_chat_interface(prompt=None):
    """Render the main chat interface."""
    st.markdown("### 💬 Chat with Your Knowledge Base")
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message["content"])
                    
                    # Show sources if available
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 View Sources", expanded=False):
                            for i, source in enumerate(message["sources"], 1):
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Source {i}:</strong> {source.get('source', 'Unknown')}<br>
                                    <small>Relevance: {source.get('score', 0):.2%}</small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Show response time
                    if "response_time" in message:
                        st.caption(f"⏱️ Response time: {message['response_time']:.2f}s")
    
    # Process chat input if provided
    if prompt:
        process_query(prompt)


def process_query(query: str):
    """Process user query through RAG pipeline with conversation memory."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)
    
    # Process with RAG
    with st.chat_message("assistant", avatar="🤖"):
        start_time = time.time()
        
        with st.spinner("🔍 Searching knowledge base & generating answer..."):
            try:
                # Build conversation history for context
                conversation_history = build_conversation_history()
                
                # Check if this might be a follow-up question
                is_followup = is_followup_question(query)
                
                # Retrieve relevant context
                context = ""
                sources = []
                confidence = 0
                
                if st.session_state.rag_retriever:
                    # For follow-up questions, also include the original query context
                    search_query = query
                    if is_followup and st.session_state.conversation_memory:
                        # Enhance query with previous context
                        last_query = st.session_state.conversation_memory[-1].get('query', '')
                        search_query = f"{last_query} {query}"
                    
                    rag_response = st.session_state.rag_retriever.retrieve(
                        search_query,
                        top_k=st.session_state.get('top_k_slider', 3)
                    )
                    context = rag_response.context
                    sources = rag_response.sources
                    confidence = rag_response.confidence
                    
                    # Store context for follow-up questions
                    st.session_state.last_context = context
                
                # If low confidence on follow-up, use previous context
                if is_followup and confidence < 0.3 and st.session_state.last_context:
                    context = st.session_state.last_context
                    confidence = 0.5  # Moderate confidence for reused context
                
                # Generate response with LLM (always try to generate an answer)
                answer = generate_answer_with_memory(query, context, conversation_history)
                success = True
                
                response_time = time.time() - start_time
                
                # Display response
                st.markdown(answer)
                
                # Show sources only if they were found
                if sources and confidence > 0.2:
                    with st.expander("📚 View Sources (Optional)", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {i}:</strong> {source.get('source', 'Unknown')}<br>
                                <small>Relevance: {source.get('score', 0):.2%}</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.caption(f"⏱️ Response time: {response_time:.2f}s")
                
                # Store message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "response_time": response_time,
                    "confidence": confidence
                })
                
                # Update conversation memory (keep last 10 exchanges)
                st.session_state.conversation_memory.append({
                    "query": query,
                    "answer": answer,
                    "context": context[:500] if context else ""  # Store truncated context
                })
                if len(st.session_state.conversation_memory) > 10:
                    st.session_state.conversation_memory = st.session_state.conversation_memory[-10:]
                
                # Update analytics
                st.session_state.total_queries += 1
                if success:
                    st.session_state.successful_queries += 1
                
                # Update average response time
                analytics = st.session_state.query_analytics
                analytics.append({
                    "query": query,
                    "response_time": response_time,
                    "confidence": confidence,
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                })
                st.session_state.query_analytics = analytics
                st.session_state.avg_response_time = sum(a["response_time"] for a in analytics) / len(analytics)
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}"
                })


def build_conversation_history() -> str:
    """Build conversation history string from memory."""
    if not st.session_state.conversation_memory:
        return ""
    
    history_parts = []
    for i, exchange in enumerate(st.session_state.conversation_memory[-5:], 1):  # Last 5 exchanges
        history_parts.append(f"User: {exchange['query']}")
        history_parts.append(f"Assistant: {exchange['answer'][:200]}...")  # Truncate long answers
    
    return "\n".join(history_parts)


def is_followup_question(query: str) -> bool:
    """Detect if query is likely a follow-up question."""
    followup_indicators = [
        'it', 'this', 'that', 'they', 'them', 'these', 'those',
        'what about', 'how about', 'and', 'also', 'more', 'else',
        'why', 'can you', 'could you', 'tell me more', 'explain',
        'what is', 'what are', 'elaborate', 'clarify', 'another',
        'same', 'similar', 'related', 'previous', 'last', 'again'
    ]
    
    query_lower = query.lower().strip()
    
    # Short queries are often follow-ups
    if len(query_lower.split()) <= 4:
        return True
    
    # Check for follow-up indicators
    for indicator in followup_indicators:
        if query_lower.startswith(indicator) or f" {indicator} " in query_lower:
            return True
    
    return False


def generate_answer_with_memory(query: str, context: str, conversation_history: str) -> str:
    """Generate answer using LLM with conversation memory."""
    
    # Build the prompt with memory
    system_prompt = """You are a helpful AI assistant with access to a knowledge base. 
Your task is to provide clear, informative, and helpful answers to user questions.

Guidelines:
1. If context from the knowledge base is provided, use it to give accurate answers
2. If no relevant context is found, use your general knowledge to help the user
3. Remember the conversation history to answer follow-up questions
4. Be conversational and friendly
5. If you're not sure about something, say so honestly
6. Provide complete, useful answers - not just references to documents"""

    # Build the full prompt
    prompt_parts = []
    
    if conversation_history:
        prompt_parts.append(f"=== CONVERSATION HISTORY ===\n{conversation_history}\n")
    
    if context:
        prompt_parts.append(f"=== RELEVANT KNOWLEDGE BASE CONTEXT ===\n{context}\n")
    
    prompt_parts.append(f"=== CURRENT QUESTION ===\n{query}")
    prompt_parts.append("\n=== YOUR RESPONSE ===\nProvide a helpful, complete answer:")
    
    full_prompt = "\n".join(prompt_parts)
    
    # Try to use LLM handler
    if st.session_state.llm_handler:
        try:
            response = st.session_state.llm_handler.generate(
                prompt=full_prompt,
                system_prompt=system_prompt
            )
            if response.success and response.content:
                return response.content
        except Exception as e:
            st.warning(f"LLM generation issue: {e}")
    
    # Fallback: Generate a helpful response without LLM
    return generate_fallback_response(query, context, conversation_history)


def generate_fallback_response(query: str, context: str, conversation_history: str) -> str:
    """Generate a response when LLM is not available."""
    
    if context:
        # Extract the most relevant part of context
        context_sentences = context.split('.')
        relevant_sentences = []
        query_words = set(query.lower().split())
        
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            if query_words & sentence_words:  # If there's overlap
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            answer = "Based on the knowledge base, here's what I found:\n\n"
            answer += ". ".join(relevant_sentences[:3]) + "."
            
            if len(context_sentences) > 3:
                answer += "\n\nWould you like me to provide more details on any specific aspect?"
            
            return answer
        else:
            return f"Here's relevant information from the knowledge base:\n\n{context[:800]}"
    
    # No context available
    return """I don't have specific information about that in my knowledge base. 

However, I'd be happy to help if you could:
1. Rephrase your question with more specific terms
2. Ask about a different topic
3. Provide more context about what you're looking for

What would you like to know?"""


def render_analytics_tab():
    """Render the analytics tab with visualizations."""
    st.markdown("### 📊 Query Analytics")
    
    if not st.session_state.query_analytics:
        st.info("No queries yet. Start chatting to see analytics!")
        return
    
    analytics_df = pd.DataFrame(st.session_state.query_analytics)
    
    # Response Time Chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            analytics_df, 
            y="response_time",
            title="Response Time Over Queries",
            labels={"index": "Query #", "response_time": "Time (s)"}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#1a1a2e'
        )
        fig.update_traces(line_color='#667eea', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            analytics_df,
            y="confidence",
            title="Confidence Score Over Queries",
            labels={"index": "Query #", "confidence": "Confidence"}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#1a1a2e'
        )
        fig.update_traces(line_color='#764ba2', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    # Success Rate Pie Chart
    col1, col2 = st.columns(2)
    
    with col1:
        success_counts = analytics_df['success'].value_counts()
        fig = px.pie(
            values=success_counts.values,
            names=['Successful' if x else 'Failed' for x in success_counts.index],
            title="Query Success Rate",
            color_discrete_sequence=['#667eea', '#ff6b6b']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#1a1a2e'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Response Time Distribution
        fig = px.histogram(
            analytics_df,
            x="response_time",
            nbins=20,
            title="Response Time Distribution",
            labels={"response_time": "Time (s)", "count": "Count"}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#1a1a2e'
        )
        fig.update_traces(marker_color='#667eea')
        st.plotly_chart(fig, use_container_width=True)
    
    # Query History Table
    st.markdown("### 📋 Query History")
    display_df = analytics_df[['query', 'response_time', 'confidence', 'success', 'timestamp']].copy()
    display_df['response_time'] = display_df['response_time'].round(2)
    display_df['confidence'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
    display_df.columns = ['Query', 'Response Time (s)', 'Confidence', 'Success', 'Timestamp']
    st.dataframe(display_df, use_container_width=True)


def render_knowledge_base_tab():
    """Render the knowledge base explorer tab."""
    st.markdown("### 📚 Knowledge Base Explorer")
    
    kb_path = Path(__file__).parent / "knowledge_base"
    
    if not kb_path.exists():
        st.warning("Knowledge base directory not found!")
        return
    
    # List documents
    docs = list(kb_path.glob("*.md")) + list(kb_path.glob("*.txt"))
    
    if not docs:
        st.info("No documents in knowledge base. Add .md or .txt files to get started!")
        return
    
    # Document selector
    selected_doc = st.selectbox(
        "Select a document to view",
        docs,
        format_func=lambda x: x.name
    )
    
    if selected_doc:
        with open(selected_doc, 'r', encoding='utf-8') as f:
            content = f.read()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📄 Document Content")
            st.markdown(f"```markdown\n{content[:2000]}{'...' if len(content) > 2000 else ''}\n```")
        
        with col2:
            st.markdown("#### 📊 Document Stats")
            words = len(content.split())
            chars = len(content)
            lines = len(content.split('\n'))
            
            st.metric("Words", words)
            st.metric("Characters", chars)
            st.metric("Lines", lines)
            
            # Word cloud visualization (simplified)
            st.markdown("#### 🏷️ Key Terms")
            # Extract simple word frequency
            words_list = content.lower().split()
            word_freq = {}
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                         'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 
                         'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 
                         'into', 'through', 'during', 'before', 'after', 'above', 'below',
                         'and', 'or', 'but', 'if', 'then', 'else', 'when', 'up', 'down',
                         'out', 'off', 'over', 'under', 'again', 'further', 'once', 'that',
                         'this', 'these', 'those', 'it', 'its', 'you', 'your', 'we', 'our'}
            
            for word in words_list:
                word = ''.join(c for c in word if c.isalnum())
                if word and len(word) > 3 and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if top_words:
                fig = px.bar(
                    x=[w[1] for w in top_words],
                    y=[w[0] for w in top_words],
                    orientation='h',
                    labels={'x': 'Count', 'y': 'Word'}
                )
                fig.update_layout(
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#1a1a2e',
                    showlegend=False
                )
                fig.update_traces(marker_color='#667eea')
                st.plotly_chart(fig, use_container_width=True)


def render_vision_tab():
    """Render the vision/image processing tab."""
    st.markdown("### 👁️ Image Analysis")
    
    st.info("Upload an image to analyze it using the vision model.")
    
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload an image for analysis"
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖼️ Uploaded Image")
            st.image(uploaded_file, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔍 Analysis Results")
            
            if st.button("🔬 Analyze Image", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        if st.session_state.vision_handler:
                            # Read image bytes
                            image_bytes = uploaded_file.getvalue()
                            
                            # Process image
                            result = st.session_state.vision_handler.process_image_bytes(
                                image_bytes,
                                generate_tags=True
                            )
                            
                            if result.success:
                                st.success("Analysis complete!")
                                
                                st.markdown(f"**📝 Caption:** {result.caption}")
                                st.markdown(f"**🏷️ Tags:** {', '.join(f'`{tag}`' for tag in result.tags)}")
                                st.markdown(f"**⏱️ Processing Time:** {result.processing_time:.2f}s")
                                
                                # Image info
                                with st.expander("📊 Image Details"):
                                    for key, value in result.image_info.items():
                                        st.markdown(f"- **{key}:** {value}")
                            else:
                                st.error(f"Analysis failed: {result.error}")
                        else:
                            st.warning("Vision handler not initialized. Please check your configuration.")
                    except Exception as e:
                        st.error(f"Error analyzing image: {e}")


def render_settings_tab():
    """Render the settings tab."""
    st.markdown("### ⚙️ System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 RAG Settings")
        
        config = load_config()
        
        st.text_input("Embedding Model", value=config.rag.embedding_model, disabled=True)
        st.number_input("Chunk Size", value=config.rag.chunk_size, disabled=True)
        st.number_input("Chunk Overlap", value=config.rag.chunk_overlap, disabled=True)
        st.number_input("Top K Results", value=config.rag.top_k, disabled=True)
        st.number_input("Similarity Threshold", value=config.rag.similarity_threshold, disabled=True)
    
    with col2:
        st.markdown("#### 🧠 LLM Settings")
        
        st.text_input("Ollama Base URL", value=config.llm.ollama_base_url, disabled=True)
        st.text_input("Ollama Model", value=config.llm.ollama_model, disabled=True)
        st.number_input("Max Tokens", value=config.llm.max_tokens, disabled=True)
        st.number_input("Temperature", value=config.llm.temperature, disabled=True)
        
        # Check LLM availability
        if st.session_state.llm_handler:
            provider = st.session_state.llm_handler.get_available_provider()
            st.success(f"Active Provider: {provider.value}")
        else:
            st.warning("LLM Handler not initialized")
    
    st.markdown("---")
    
    st.markdown("#### 📁 Paths")
    st.code(f"Knowledge Base: {config.rag.knowledge_base_path}")
    st.code(f"Vector Store: {config.rag.vector_store_path}")
    
    # Export/Import Settings
    st.markdown("---")
    st.markdown("#### 💾 Export Analytics")
    
    if st.session_state.query_analytics:
        analytics_json = json.dumps(st.session_state.query_analytics, indent=2)
        st.download_button(
            "📥 Download Analytics JSON",
            analytics_json,
            file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """Main application entry point."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 GenAI RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Initialize systems
    if not st.session_state.system_initialized:
        with st.spinner("🔄 Initializing AI systems..."):
            st.session_state.rag_retriever = initialize_rag_system()
            st.session_state.llm_handler = initialize_llm_handler()
            st.session_state.vision_handler = initialize_vision_handler()
            st.session_state.system_initialized = True
    
    # Render sidebar
    render_sidebar()
    
    # Chat input MUST be outside tabs - place it at the top level
    prompt = st.chat_input("Ask a question about your knowledge base...")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat",
        "📊 Analytics", 
        "📚 Knowledge Base",
        "👁️ Vision",
        "⚙️ Settings"
    ])
    
    with tab1:
        render_metrics_dashboard()
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        render_chat_interface(prompt)
    
    with tab2:
        render_analytics_tab()
    
    with tab3:
        render_knowledge_base_tab()
    
    with tab4:
        render_vision_tab()
    
    with tab5:
        render_settings_tab()
    
    # Footer
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; padding: 1rem;">
        <p>🤖 GenAI RAG Assistant | Built with Streamlit, LangChain & Love ❤️</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
