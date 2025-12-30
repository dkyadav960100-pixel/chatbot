#!/usr/bin/env python3
"""
GenAI Telegram Bot - Main Entry Point
A hybrid RAG + Vision bot for Telegram.
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import load_config, get_config
from bot import run_bot, GenAIBot


def setup_logging(debug: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('bot.log', mode='a')
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.WARNING)


def check_requirements():
    """Check that all required dependencies are available."""
    required = [
        ('telegram', 'python-telegram-bot'),
        ('sentence_transformers', 'sentence-transformers'),
        ('PIL', 'Pillow'),
        ('numpy', 'numpy'),
    ]
    
    missing = []
    
    for module, package in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    
    return True


def initialize_knowledge_base():
    """Ensure knowledge base has documents."""
    kb_path = Path(__file__).parent / "knowledge_base"
    
    if not kb_path.exists():
        print(f"Creating knowledge base directory: {kb_path}")
        kb_path.mkdir(parents=True, exist_ok=True)
    
    # Check for documents
    docs = list(kb_path.glob("*.md")) + list(kb_path.glob("*.txt"))
    
    if not docs:
        print("⚠️  Warning: No documents found in knowledge base!")
        print(f"   Add .md or .txt files to: {kb_path}")
        print("   The bot will have limited RAG capabilities.")
    else:
        print(f"✓ Found {len(docs)} documents in knowledge base")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='GenAI Telegram Bot - RAG + Vision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                    # Run with default settings
  python app.py --debug            # Run in debug mode
  python app.py --check            # Check configuration only
  
Environment Variables:
  TELEGRAM_BOT_TOKEN     Required: Your Telegram bot token
  OLLAMA_BASE_URL        Ollama API URL (default: http://localhost:11434)
  OLLAMA_MODEL           Ollama model (default: llama3.2:1b)
  OPENAI_API_KEY         Optional: OpenAI API key for fallback
  VISION_MODEL           Vision model type: blip, blip2, git (default: blip)
  DEBUG                  Enable debug mode (true/false)
        """
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Check configuration and exit'
    )
    
    parser.add_argument(
        '--init-kb',
        action='store_true',
        help='Initialize/index knowledge base and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    debug = args.debug or os.getenv('DEBUG', 'false').lower() == 'true'
    setup_logging(debug)
    
    logger = logging.getLogger(__name__)
    
    print("""
╔═══════════════════════════════════════════════╗
║        GenAI Telegram Bot v1.0.0              ║
║        RAG + Vision Hybrid System             ║
╚═══════════════════════════════════════════════╝
    """)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Load configuration
    config = load_config()
    
    # Validate configuration
    if not config.telegram.bot_token:
        print("\n❌ Error: TELEGRAM_BOT_TOKEN environment variable is required!")
        print("   Get your token from @BotFather on Telegram")
        print("   Set it with: export TELEGRAM_BOT_TOKEN='your_token_here'")
        sys.exit(1)
    
    # Show configuration summary
    print("\n📋 Configuration:")
    print(f"   • Telegram Bot Token: {'✓ Set' if config.telegram.bot_token else '✗ Missing'}")
    print(f"   • Ollama URL: {config.llm.ollama_base_url}")
    print(f"   • Ollama Model: {config.llm.ollama_model}")
    print(f"   • OpenAI API Key: {'✓ Set' if config.llm.openai_api_key else '✗ Not set (optional)'}")
    print(f"   • Vision Model: {config.vision.model_name}")
    print(f"   • Embedding Model: {config.rag.embedding_model}")
    print(f"   • Cache Enabled: {config.cache.enabled}")
    print(f"   • Debug Mode: {debug}")
    
    # Initialize knowledge base
    initialize_knowledge_base()
    
    # Configuration check only
    if args.check:
        print("\n✓ Configuration check passed!")
        sys.exit(0)
    
    # Initialize knowledge base only
    if args.init_kb:
        print("\n📚 Initializing knowledge base...")
        from rag import RAGRetriever
        
        rag = RAGRetriever(
            knowledge_base_path=str(Path(__file__).parent / "knowledge_base"),
            vector_store_path=str(Path(__file__).parent / "data" / "vector_store.db"),
            embedding_model=config.rag.embedding_model
        )
        
        success = rag.initialize(force_reload=True)
        
        if success:
            stats = rag.get_stats()
            print(f"✓ Knowledge base initialized!")
            print(f"  Documents: {stats['vector_store']['total_documents']}")
            print(f"  Chunks: {stats['vector_store']['total_chunks']}")
        else:
            print("❌ Failed to initialize knowledge base")
            sys.exit(1)
        
        sys.exit(0)
    
    # Check LLM availability
    print("\n🔍 Checking LLM availability...")
    from llm import LLMHandler
    
    llm = LLMHandler(
        ollama_base_url=config.llm.ollama_base_url,
        ollama_model=config.llm.ollama_model,
        openai_api_key=config.llm.openai_api_key
    )
    
    provider = llm.get_available_provider()
    
    if provider.value == "none":
        print("⚠️  Warning: No LLM provider available!")
        print("   • Start Ollama: ollama serve")
        print("   • Or set OPENAI_API_KEY environment variable")
        print("   RAG queries will return raw context without summarization.")
    else:
        print(f"✓ LLM Provider: {provider.value}")
    
    # Run the bot
    print("\n🚀 Starting bot...")
    
    try:
        run_bot()
    except KeyboardInterrupt:
        print("\n\n👋 Bot stopped. Goodbye!")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
