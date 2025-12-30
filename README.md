# 🤖 GenAI Telegram Bot

A **Hybrid RAG + Vision** Telegram Bot that can answer questions from a knowledge base and describe uploaded images.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

## 🎯 Features

### 📚 RAG (Retrieval-Augmented Generation)
- Answer questions from your custom knowledge base
- Semantic search using sentence-transformers embeddings
- SQLite vector store for persistent storage
- Context-aware responses with source citations
- Supports Markdown, TXT, JSON, and PDF documents

### 🖼️ Vision/Image Description
- Describe uploaded images using BLIP model
- Generate relevant tags for images
- Support for JPG, PNG, WEBP, GIF formats
- Configurable caption length and tag count

### 💡 Smart Features
- **Conversation History**: Remembers last 3 interactions per user
- **Query Caching**: Fast responses for repeated queries
- **Multi-LLM Support**: Ollama (primary) with OpenAI fallback
- **Session Management**: Per-user context and preferences
- **Source Citations**: Shows which documents were used

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Telegram Bot Interface                      │
│                    (python-telegram-bot)                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
┌───────────────────────┐     ┌───────────────────────┐
│     RAG System        │     │    Vision System      │
│                       │     │                       │
│  ┌─────────────────┐  │     │  ┌─────────────────┐  │
│  │ Document Loader │  │     │  │ Image Processor │  │
│  └────────┬────────┘  │     │  └────────┬────────┘  │
│           ▼           │     │           ▼           │
│  ┌─────────────────┐  │     │  ┌─────────────────┐  │
│  │  Text Chunker   │  │     │  │  BLIP Model     │  │
│  └────────┬────────┘  │     │  │  (Captioning)   │  │
│           ▼           │     │  └────────┬────────┘  │
│  ┌─────────────────┐  │     │           ▼           │
│  │  Embeddings     │  │     │  ┌─────────────────┐  │
│  │ (MiniLM-L6-v2)  │  │     │  │   Tag Extractor │  │
│  └────────┬────────┘  │     │  └─────────────────┘  │
│           ▼           │     │                       │
│  ┌─────────────────┐  │     └───────────────────────┘
│  │  Vector Store   │  │
│  │   (SQLite)      │  │
│  └────────┬────────┘  │
│           ▼           │
│  ┌─────────────────┐  │
│  │    Retriever    │  │
│  └─────────────────┘  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────────────────────────────────┐
│               LLM Handler                          │
│  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  Ollama (Local) │  │  OpenAI API (Fallback)  │ │
│  │  - LLaMA 3      │  │  - GPT-3.5/4            │ │
│  │  - Mistral      │  │                         │ │
│  │  - Phi-3        │  │                         │ │
│  └─────────────────┘  └─────────────────────────┘ │
└───────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────┐
│               Storage Layer                        │
│  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  Query Cache    │  │  Session Manager        │ │
│  │   (SQLite)      │  │  (In-Memory)            │ │
│  └─────────────────┘  └─────────────────────────┘ │
└───────────────────────────────────────────────────┘
```


## App Link : https://dkyadav960100-pixel-chatbot-streamlit-app-mbmm25.streamlit.app/

---
## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- [Ollama](https://ollama.ai/) (recommended) or OpenAI API key

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/genai-telegram-bot.git
cd genai-telegram-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For CPU-only (smaller download):
pip install -r requirements-cpu.txt

# Configure environment
cp .env.example .env
# Edit .env and add your TELEGRAM_BOT_TOKEN

# Start Ollama (in another terminal)
ollama serve
ollama pull llama3.2:1b

# Run the bot
python app.py
```

### Option 2: Docker Compose (Recommended)

```bash
# Clone and configure
git clone https://github.com/yourusername/genai-telegram-bot.git
cd genai-telegram-bot
cp .env.example .env
# Edit .env and add your TELEGRAM_BOT_TOKEN

# Start everything
docker-compose up -d

# View logs
docker-compose logs -f bot
```

## 📋 Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and bot introduction |
| `/help` | Show all available commands |
| `/ask <question>` | Ask a question from the knowledge base |
| `/image` | Get ready to describe an uploaded image |
| `/summarize` | Summarize the last interaction |
| `/status` | Show bot status and statistics |
| `/clear` | Clear conversation history |

### Usage Examples

**RAG Query:**
```
/ask What is the refund policy?
```

**Image Description:**
Send any image directly or use `/image` first.

**Natural Conversation:**
Just type your question without any command!

## 📁 Project Structure

```
genai_telegram_bot/
├── app.py                 # Main entry point
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker build file
├── docker-compose.yml     # Docker Compose configuration
├── .env.example           # Environment variables template
│
├── bot/                   # Telegram bot module
│   ├── __init__.py
│   └── bot.py            # Bot implementation
│
├── rag/                   # RAG system
│   ├── __init__.py
│   ├── document_loader.py # Load documents
│   ├── chunker.py        # Text chunking
│   ├── embeddings.py     # Embedding generation
│   ├── vector_store.py   # SQLite vector storage
│   └── retriever.py      # RAG retrieval logic
│
├── vision/                # Vision system
│   ├── __init__.py
│   ├── image_processor.py # Image preprocessing
│   ├── caption_model.py  # BLIP captioning
│   └── vision_handler.py # Main vision interface
│
├── llm/                   # LLM handling
│   ├── __init__.py
│   └── llm_handler.py    # Ollama/OpenAI integration
│
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── cache.py          # Query caching
│   └── session.py        # User session management
│
├── knowledge_base/        # Your documents go here
│   ├── company_policies.md
│   ├── product_faq.md
│   ├── technical_docs.md
│   ├── refund_policy.md
│   └── getting_started.md
│
└── data/                  # Generated data (auto-created)
    ├── vector_store.db   # Embeddings database
    └── cache.db          # Query cache
```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | ✅ | - | Your Telegram bot token |
| `OLLAMA_BASE_URL` | ❌ | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | ❌ | `llama3.2:1b` | Ollama model name |
| `OPENAI_API_KEY` | ❌ | - | OpenAI API key (fallback) |
| `OPENAI_MODEL` | ❌ | `gpt-3.5-turbo` | OpenAI model |
| `VISION_MODEL` | ❌ | `blip` | Vision model (`blip`, `blip2`, `git`) |
| `DEVICE` | ❌ | `cpu` | Device (`cpu` or `cuda`) |
| `DEBUG` | ❌ | `false` | Enable debug mode |

### Recommended LLM Models

**For Ollama (Local):**
- `llama3.2:1b` - Fast, good quality (recommended)
- `phi3:mini` - Very small, decent quality
- `mistral:7b-instruct` - Best quality, requires more RAM
- `llama3.1:8b` - Good balance of speed and quality

**For OpenAI:**
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-4` - Best quality, higher cost

## 📚 Knowledge Base

Add your documents to the `knowledge_base/` directory:

### Supported Formats
- **Markdown** (`.md`) - Recommended
- **Plain Text** (`.txt`)
- **JSON** (`.json`)
- **PDF** (`.pdf`) - Requires PyPDF2

### Document Tips
1. Use clear headings for better semantic chunking
2. Keep documents focused on specific topics
3. Include relevant keywords and terms
4. Use consistent formatting

### Re-indexing Documents

After adding new documents:
```bash
python app.py --init-kb
```

## 🔧 Advanced Configuration

### Custom Embedding Model

```python
# In config.py or via environment
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Higher quality
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Alternative
```

### Chunking Strategy

```python
# In rag/retriever.py
chunker = TextChunker(
    chunk_size=500,        # Characters per chunk
    chunk_overlap=50,      # Overlap between chunks
    strategy=ChunkingStrategy.PARAGRAPH  # Or SENTENCE, SEMANTIC
)
```

### Vision Model Selection

| Model | Quality | Speed | Memory |
|-------|---------|-------|--------|
| `blip` | Good | Fast | ~1GB |
| `blip2` | Better | Slower | ~3GB |
| `git` | Good | Fast | ~1GB |

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build and start
docker-compose up -d --build

# Scale bot instances (if using webhook)
docker-compose up -d --scale bot=3

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### GPU Support

```yaml
# In docker-compose.yml, add to bot service:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=. tests/

# Specific test
pytest tests/test_rag.py -v
```

## 📊 Monitoring

The bot provides a `/status` command showing:
- Active LLM provider
- Knowledge base statistics
- Active sessions
- Cache hit rate
