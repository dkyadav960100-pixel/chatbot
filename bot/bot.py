"""
Telegram Bot Implementation - Hybrid RAG + Vision Bot.
Handles all bot commands and user interactions.
"""
import logging
import asyncio
import os
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile

# Telegram imports
from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from telegram.constants import ParseMode, ChatAction

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config, load_config
from rag import RAGRetriever
from vision import VisionHandler
from llm import LLMHandler
from utils import QueryCache, SessionManager

logger = logging.getLogger(__name__)


class GenAIBot:
    """
    Hybrid GenAI Telegram Bot with RAG and Vision capabilities.
    
    Commands:
        /start - Welcome message
        /help - Show usage instructions
        /ask <query> - Ask a question (RAG)
        /image - Describe an uploaded image
        /summarize - Summarize last interaction
        /status - Show bot status
        /clear - Clear conversation history
    """
    
    HELP_TEXT = """🤖 **GenAI Assistant Bot**

I can help you with:

📚 **RAG Queries** - Ask questions about the knowledge base
   `/ask <your question>`
   Example: `/ask What is the refund policy?`

🖼️ **Image Description** - Upload an image to get a description
   Just send me an image or use `/image` then upload

📝 **Summarize** - Get a summary of our last interaction
   `/summarize`

🗑️ **Clear History** - Reset our conversation
   `/clear`

📊 **Status** - Check bot status
   `/status`

**Tips:**
- I remember our last few conversations
- Cached queries are answered faster
- Image processing may take a few seconds

Send me a message or image to get started! 🚀"""
    
    def __init__(self, config=None):
        """
        Initialize the GenAI bot.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or load_config()
        
        # Validate token
        if not self.config.telegram.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        
        # Initialize components
        self._init_components()
        
        # Application will be set in run()
        self.application: Optional[Application] = None
        
        logger.info("GenAI Bot initialized")
    
    def _init_components(self):
        """Initialize all bot components."""
        # Ensure data directory exists
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RAG
        kb_path = self.config.rag.knowledge_base_path
        if not Path(kb_path).exists():
            kb_path = str(Path(__file__).parent.parent / "knowledge_base")
            Path(kb_path).mkdir(parents=True, exist_ok=True)
        
        self.rag = RAGRetriever(
            knowledge_base_path=kb_path,
            vector_store_path=str(data_dir / "vector_store.db"),
            embedding_model=self.config.rag.embedding_model,
            chunk_size=self.config.rag.chunk_size,
            chunk_overlap=self.config.rag.chunk_overlap,
            top_k=self.config.rag.top_k,
            similarity_threshold=self.config.rag.similarity_threshold
        )
        
        # Initialize Vision
        self.vision = VisionHandler(
            model_type=self.config.vision.model_name,
            device=self.config.vision.device,
            max_image_size=self.config.vision.max_image_size,
            num_tags=self.config.vision.num_tags
        )
        
        # Initialize LLM
        self.llm = LLMHandler(
            ollama_base_url=self.config.llm.ollama_base_url,
            ollama_model=self.config.llm.ollama_model,
            openai_api_key=self.config.llm.openai_api_key,
            openai_model=self.config.llm.openai_model,
            openai_base_url=self.config.llm.openai_base_url,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
            timeout=self.config.llm.timeout
        )
        
        # Initialize Cache
        self.cache = QueryCache(
            db_path=str(data_dir / "cache.db"),
            ttl=self.config.cache.query_cache_ttl,
            max_size=self.config.cache.max_cache_size
        ) if self.config.cache.enabled else None
        
        # Initialize Session Manager
        self.sessions = SessionManager(
            history_length=self.config.session.history_length,
            session_timeout=self.config.session.session_timeout,
            max_sessions=self.config.session.max_sessions
        )
        
        logger.info("All components initialized")
    
    async def _setup_commands(self, application: Application):
        """Set up bot commands for the menu."""
        commands = [
            BotCommand("start", "Welcome message"),
            BotCommand("help", "Show usage instructions"),
            BotCommand("ask", "Ask a question from knowledge base"),
            BotCommand("image", "Upload image for description"),
            BotCommand("summarize", "Summarize last interaction"),
            BotCommand("status", "Show bot status"),
            BotCommand("clear", "Clear conversation history"),
        ]
        await application.bot.set_my_commands(commands)
    
    def _register_handlers(self, application: Application):
        """Register all command and message handlers."""
        # Command handlers
        application.add_handler(CommandHandler("start", self.cmd_start))
        application.add_handler(CommandHandler("help", self.cmd_help))
        application.add_handler(CommandHandler("ask", self.cmd_ask))
        application.add_handler(CommandHandler("image", self.cmd_image))
        application.add_handler(CommandHandler("summarize", self.cmd_summarize))
        application.add_handler(CommandHandler("status", self.cmd_status))
        application.add_handler(CommandHandler("clear", self.cmd_clear))
        
        # Message handlers
        application.add_handler(
            MessageHandler(filters.PHOTO, self.handle_image)
        )
        application.add_handler(
            MessageHandler(filters.Document.IMAGE, self.handle_image_document)
        )
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text)
        )
        
        # Error handler
        application.add_error_handler(self.error_handler)
    
    # ==================== Command Handlers ====================
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        
        welcome_msg = f"""👋 Hello {user.first_name}!

I'm your GenAI Assistant Bot, powered by RAG and Vision AI.

I can:
• 📚 Answer questions from my knowledge base
• 🖼️ Describe and tag images
• 💬 Remember our conversation context

Use /help to see all available commands.

Let's get started! Ask me a question or send an image! 🚀"""
        
        await update.message.reply_text(welcome_msg)
        
        # Initialize session
        self.sessions.get_session(user.id)
        logger.info(f"User {user.id} started the bot")
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        await update.message.reply_text(
            self.HELP_TEXT,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def cmd_ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ask command for RAG queries."""
        user_id = update.effective_user.id
        
        # Get query from command arguments
        query = ' '.join(context.args) if context.args else None
        
        if not query:
            await update.message.reply_text(
                "❓ Please provide a question.\n"
                "Usage: `/ask <your question>`\n"
                "Example: `/ask What is the return policy?`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        await self._process_rag_query(update, user_id, query)
    
    async def cmd_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /image command - prompt for image upload."""
        await update.message.reply_text(
            "📸 Please send me an image and I'll describe it for you!\n\n"
            "Supported formats: JPG, PNG, WEBP, GIF"
        )
        
        # Set context to expect image
        self.sessions.set_context(
            update.effective_user.id,
            'expecting_image',
            True
        )
    
    async def cmd_summarize(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /summarize command."""
        user_id = update.effective_user.id
        
        last_interaction = self.sessions.get_last_interaction(user_id)
        
        if not last_interaction:
            await update.message.reply_text(
                "📝 No previous interaction to summarize.\n"
                "Send me a question or image first!"
            )
            return
        
        # Show typing indicator
        await update.message.chat.send_action(ChatAction.TYPING)
        
        try:
            # Create summary based on interaction type
            if last_interaction['type'] == 'image':
                text_to_summarize = f"Image description: {last_interaction['assistant']}"
            else:
                text_to_summarize = (
                    f"Question: {last_interaction['user']}\n\n"
                    f"Answer: {last_interaction['assistant']}"
                )
            
            summary_response = self.llm.summarize(text_to_summarize)
            
            if summary_response.success:
                await update.message.reply_text(
                    f"📝 **Summary**\n\n{summary_response.content}",
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                await update.message.reply_text(
                    f"📝 **Last Interaction**\n\n{text_to_summarize[:500]}...",
                    parse_mode=ParseMode.MARKDOWN
                )
                
        except Exception as e:
            logger.error(f"Summarize error: {e}")
            await update.message.reply_text(
                "❌ Unable to generate summary. Please try again."
            )
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        # Gather status info
        llm_status = self.llm.get_status()
        rag_stats = self.rag.get_stats()
        session_stats = self.sessions.get_stats()
        cache_stats = self.cache.get_stats() if self.cache else {"enabled": False}
        
        status_text = f"""📊 **Bot Status**

🧠 **LLM Provider:** {llm_status['available_provider']}
   Model: {llm_status.get('ollama_model') or llm_status.get('openai_model')}

📚 **Knowledge Base:**
   Documents: {rag_stats.get('vector_store', {}).get('total_documents', 0)}
   Chunks: {rag_stats.get('vector_store', {}).get('total_chunks', 0)}

💬 **Sessions:**
   Active: {session_stats.get('active_sessions', 0)}
   Total: {session_stats.get('total_sessions', 0)}

💾 **Cache:**
   Enabled: {cache_stats.get('query_cache', {}).get('entries', 0) if cache_stats.get('enabled', True) else 'Disabled'}
   Hits: {cache_stats.get('query_cache', {}).get('total_hits', 0) if cache_stats.get('enabled', True) else 'N/A'}

✅ Bot is running!"""
        
        await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)
    
    async def cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command."""
        user_id = update.effective_user.id
        self.sessions.clear_session(user_id)
        
        await update.message.reply_text(
            "🗑️ Conversation history cleared!\n"
            "Starting fresh. How can I help you?"
        )
    
    # ==================== Message Handlers ====================
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages (treated as RAG queries)."""
        user_id = update.effective_user.id
        query = update.message.text.strip()
        
        if not query:
            return
        
        await self._process_rag_query(update, user_id, query)
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo uploads."""
        user_id = update.effective_user.id
        
        # Show typing indicator
        await update.message.chat.send_action(ChatAction.TYPING)
        
        try:
            # Get the largest photo
            photo = update.message.photo[-1]
            
            # Download photo
            file = await context.bot.get_file(photo.file_id)
            
            # Download to bytes
            image_bytes = await file.download_as_bytearray()
            
            await self._process_image(update, user_id, bytes(image_bytes))
            
        except Exception as e:
            logger.error(f"Image handling error: {e}")
            await update.message.reply_text(
                "❌ Error processing image. Please try again."
            )
    
    async def handle_image_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle image sent as document."""
        user_id = update.effective_user.id
        
        # Show typing indicator
        await update.message.chat.send_action(ChatAction.TYPING)
        
        try:
            document = update.message.document
            
            # Check file size (max 10MB)
            if document.file_size > 10 * 1024 * 1024:
                await update.message.reply_text(
                    "❌ Image too large. Maximum size is 10MB."
                )
                return
            
            # Download file
            file = await context.bot.get_file(document.file_id)
            image_bytes = await file.download_as_bytearray()
            
            await self._process_image(update, user_id, bytes(image_bytes))
            
        except Exception as e:
            logger.error(f"Document image error: {e}")
            await update.message.reply_text(
                "❌ Error processing image document. Please try again."
            )
    
    # ==================== Processing Methods ====================
    
    async def _process_rag_query(self, update: Update, user_id: int, query: str):
        """Process a RAG query."""
        # Show typing indicator
        await update.message.chat.send_action(ChatAction.TYPING)
        
        # Add user message to session
        self.sessions.add_user_message(user_id, query, "query")
        
        try:
            # Check cache first
            cached_response = None
            if self.cache:
                cached = self.cache.get(query)
                if cached:
                    cached_response, metadata = cached
                    logger.info(f"Cache hit for query: {query[:50]}...")
            
            if cached_response:
                response_text = cached_response
                source_info = "📦 (cached response)"
            else:
                # Initialize RAG if needed
                if not self.rag._initialized:
                    await update.message.reply_text(
                        "🔄 Initializing knowledge base... This may take a moment."
                    )
                    self.rag.initialize()
                
                # Retrieve context
                rag_response = self.rag.retrieve(query)
                
                if not rag_response.context:
                    response_text = (
                        "I couldn't find relevant information in my knowledge base "
                        "to answer your question. Could you try rephrasing or ask "
                        "something else?"
                    )
                    source_info = ""
                else:
                    # Get conversation history
                    history = self.sessions.get_conversation_history(user_id)
                    
                    # Generate answer with LLM
                    llm_response = self.llm.generate_with_history(
                        prompt=query,
                        history=history,
                        context=rag_response.context
                    )
                    
                    if llm_response.success:
                        response_text = llm_response.content
                        source_info = f"\n\n📚 Sources:\n{rag_response.get_source_info()}"
                        
                        # Cache the response
                        if self.cache:
                            self.cache.set(query, response_text)
                    else:
                        # Fallback: return raw context
                        response_text = (
                            f"Based on my knowledge base:\n\n"
                            f"{rag_response.context[:1000]}..."
                        )
                        source_info = f"\n\n📚 Sources:\n{rag_response.get_source_info()}"
            
            # Add assistant response to session
            self.sessions.add_assistant_message(user_id, response_text, "query")
            
            # Send response
            full_response = f"💬 **Answer:**\n\n{response_text}{source_info}"
            
            # Split if too long
            if len(full_response) > 4000:
                await update.message.reply_text(
                    full_response[:4000] + "...",
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                await update.message.reply_text(
                    full_response,
                    parse_mode=ParseMode.MARKDOWN
                )
                
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            await update.message.reply_text(
                "❌ Error processing your question. Please try again."
            )
    
    async def _process_image(self, update: Update, user_id: int, image_bytes: bytes):
        """Process an uploaded image."""
        try:
            # Process image
            result = self.vision.process_image_bytes(image_bytes)
            
            if result.success:
                response_text = result.format_response()
                
                # Add timing info
                response_text += f"\n\n⏱️ Processing time: {result.processing_time:.2f}s"
                
                # Add to session
                self.sessions.add_user_message(
                    user_id, 
                    "[Image uploaded]",
                    "image",
                    {"caption": result.caption}
                )
                self.sessions.add_assistant_message(
                    user_id,
                    result.caption,
                    "image",
                    {"tags": result.tags}
                )
                
                await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)
            else:
                await update.message.reply_text(
                    f"❌ Error processing image: {result.error}"
                )
                
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            await update.message.reply_text(
                "❌ Error analyzing image. Please try a different image."
            )
    
    # ==================== Error Handler ====================
    
    async def error_handler(
        self,
        update: object,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle errors in the bot."""
        logger.error(f"Bot error: {context.error}", exc_info=context.error)
        
        if update and isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "⚠️ An error occurred. Please try again later."
            )
    
    # ==================== Bot Lifecycle ====================
    
    async def initialize_systems(self):
        """Initialize RAG and Vision systems."""
        logger.info("Pre-initializing systems...")
        
        try:
            # Initialize RAG
            self.rag.initialize()
            logger.info("RAG system initialized")
        except Exception as e:
            logger.warning(f"RAG initialization warning: {e}")
        
        try:
            # Optionally preload vision model
            if os.getenv("PRELOAD_VISION_MODEL", "false").lower() == "true":
                self.vision.preload_model()
                logger.info("Vision model preloaded")
        except Exception as e:
            logger.warning(f"Vision preload warning: {e}")
    
    def run(self, webhook_url: Optional[str] = None):
        """
        Run the bot.
        
        Args:
            webhook_url: Optional webhook URL for production deployment
        """
        # Build application
        self.application = Application.builder().token(
            self.config.telegram.bot_token
        ).build()
        
        # Register handlers
        self._register_handlers(self.application)
        
        # Run initialization
        async def post_init(application: Application):
            await self._setup_commands(application)
            await self.initialize_systems()
        
        self.application.post_init = post_init
        
        logger.info("Starting GenAI Bot...")
        
        if webhook_url:
            # Webhook mode for production
            self.application.run_webhook(
                listen="0.0.0.0",
                port=int(os.getenv("PORT", 8443)),
                url_path=self.config.telegram.bot_token,
                webhook_url=webhook_url
            )
        else:
            # Polling mode for development
            self.application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
    
    def cleanup(self):
        """Clean up resources."""
        if self.cache:
            self.cache.close()
        self.rag.close()
        logger.info("Bot cleanup complete")


def run_bot():
    """Entry point to run the bot."""
    # Load configuration
    config = load_config()
    
    # Create and run bot
    bot = GenAIBot(config)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        bot.cleanup()


if __name__ == "__main__":
    run_bot()
