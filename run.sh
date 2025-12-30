#!/bin/bash
# GenAI Telegram Bot - Startup Script
# Handles environment setup and bot execution

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "╔═══════════════════════════════════════════════╗"
echo "║        GenAI Telegram Bot Launcher            ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Not running in a virtual environment${NC}"
    
    # Check if venv exists
    if [ -d "venv" ]; then
        echo "Activating existing virtual environment..."
        source venv/bin/activate
    else
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        
        echo "Installing dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
fi

# Check for .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo -e "${YELLOW}No .env file found. Creating from template...${NC}"
        cp .env.example .env
        echo -e "${RED}Please edit .env and add your TELEGRAM_BOT_TOKEN${NC}"
        exit 1
    fi
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check required environment variable
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo -e "${RED}Error: TELEGRAM_BOT_TOKEN is not set${NC}"
    echo "Please set it in your .env file or environment"
    exit 1
fi

# Check if Ollama is running (optional)
echo "Checking Ollama availability..."
if curl -s --connect-timeout 2 ${OLLAMA_BASE_URL:-http://localhost:11434}/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is available${NC}"
else
    echo -e "${YELLOW}⚠ Ollama is not available at ${OLLAMA_BASE_URL:-http://localhost:11434}${NC}"
    echo "  RAG queries will use OpenAI fallback or return raw context"
fi

# Create necessary directories
mkdir -p data logs knowledge_base

# Initialize knowledge base if needed
if [ ! "$(ls -A knowledge_base 2>/dev/null)" ]; then
    echo -e "${YELLOW}Knowledge base is empty. Add documents to knowledge_base/ directory${NC}"
fi

# Run the bot
echo ""
echo -e "${GREEN}Starting GenAI Telegram Bot...${NC}"
echo ""

python app.py "$@"
