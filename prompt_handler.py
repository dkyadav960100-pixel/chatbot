#!/usr/bin/env python3
"""
Prompt Handler - Edge Cases, Guardrails, and Smart Response Generation
Handles greetings, out-of-topic queries, safety guardrails, and special cases
"""
import re
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import random


class QueryType(Enum):
    """Types of user queries."""
    GREETING = "greeting"
    FAREWELL = "farewell"
    GRATITUDE = "gratitude"
    HELP_REQUEST = "help_request"
    ABOUT_BOT = "about_bot"
    SMALL_TALK = "small_talk"
    OUT_OF_SCOPE = "out_of_scope"
    INAPPROPRIATE = "inappropriate"
    HARMFUL = "harmful"
    PERSONAL_INFO = "personal_info"
    CODING_REQUEST = "coding_request"
    MATH_CALCULATION = "math_calculation"
    EMPTY_QUERY = "empty_query"
    GIBBERISH = "gibberish"
    NORMAL = "normal"


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    query_type: QueryType
    should_use_rag: bool
    direct_response: Optional[str]
    modified_query: Optional[str]
    confidence: float
    flags: List[str]


class PromptHandler:
    """
    Handles edge cases, guardrails, and smart query processing.
    """
    
    # Greeting patterns
    GREETINGS = {
        'hi', 'hello', 'hey', 'hii', 'hiii', 'hiiii', 'hola', 'greetings',
        'good morning', 'good afternoon', 'good evening', 'good night',
        'morning', 'afternoon', 'evening', 'howdy', 'sup', 'whats up',
        "what's up", 'yo', 'namaste', 'salaam', 'bonjour', 'ciao',
        'konnichiwa', 'aloha', 'hallo', 'heya', 'heyyy', 'heyy'
    }
    
    # Farewell patterns
    FAREWELLS = {
        'bye', 'goodbye', 'see you', 'see ya', 'later', 'take care',
        'farewell', 'cya', 'ttyl', 'gtg', 'got to go', 'gotta go',
        'good night', 'goodnight', 'nite', 'signing off', 'logging off',
        'catch you later', 'peace', 'peace out', 'adios', 'au revoir',
        'sayonara', 'ciao', 'bye bye', 'byebye', 'tata'
    }
    
    # Gratitude patterns
    GRATITUDE = {
        'thanks', 'thank you', 'thx', 'thanx', 'ty', 'tysm', 'tyvm',
        'thank u', 'thanks a lot', 'thanks so much', 'much appreciated',
        'appreciate it', 'grateful', 'cheers', 'ta', 'merci', 'gracias',
        'danke', 'arigato', 'dhanyavaad', 'shukriya'
    }
    
    # About bot queries
    ABOUT_BOT_PATTERNS = [
        r'who are you', r'what are you', r'what can you do',
        r'your name', r'introduce yourself', r'about you',
        r'what is your purpose', r'how do you work', r'your capabilities',
        r'what do you know', r'are you (a |an )?(ai|bot|robot|assistant)',
        r'tell me about yourself', r'who made you', r'who created you'
    ]
    
    # Help request patterns
    HELP_PATTERNS = [
        r'^help$', r'^help me$', r'how to use', r'how does this work',
        r'what can i ask', r'give me examples', r'show me examples',
        r'how to start', r'getting started', r'usage guide'
    ]
    
    # Small talk patterns
    SMALL_TALK_PATTERNS = [
        r'how are you', r"how's it going", r'how do you do',
        r'are you there', r'you there', r'anyone there',
        r"what's new", r'what is new', r'how have you been',
        r'nice to meet you', r'pleased to meet you',
        r"how's your day", r'hows your day', r'good to see you',
        r'long time no see', r'whats happening', r"what's happening"
    ]
    
    # Harmful/inappropriate content indicators
    HARMFUL_PATTERNS = [
        r'how to (make|create|build) (a |an )?(bomb|weapon|explosive|gun)',
        r'how to (kill|murder|harm|hurt|attack|poison)',
        r'how to (hack|steal|break into|crack)',
        r'suicide|self.?harm|end my life',
        r'illegal (drugs|substances|activities)',
        r'(child|minor).*(abuse|exploitation|pornography)',
        r'terrorism|terrorist|extremist',
        r'how to (get away with|commit) (crime|murder|fraud)',
    ]
    
    # Personal information requests
    PERSONAL_INFO_PATTERNS = [
        r'(my |your )?(credit card|ssn|social security|password|bank)',
        r'(what is|tell me) (my|your) (address|phone|email)',
        r'personal (data|information|details)',
        r'share.*private.*information'
    ]
    
    # Out of scope topics (not related to knowledge base)
    OUT_OF_SCOPE_PATTERNS = [
        r'weather (today|tomorrow|forecast)',
        r'stock (price|market)',
        r'sports (score|news|results)',
        r'latest news',
        r'celebrity gossip',
        r'(today\'s|current) date',
        r'what time is it',
        r'play (music|song|video)',
        r'set (alarm|timer|reminder)',
        r'call|dial|phone',
        r'order (food|pizza|uber)',
        r'book (ticket|flight|hotel)'
    ]
    
    # Greeting responses
    GREETING_RESPONSES = [
        "Hello! 👋 I'm your AI assistant. How can I help you today? Feel free to ask me questions about our knowledge base!",
        "Hi there! 😊 Welcome! I'm here to help answer your questions. What would you like to know?",
        "Hey! 👋 Great to see you! I can help you find information from our knowledge base. What's on your mind?",
        "Hello! I'm ready to assist you. Ask me anything about our products, policies, or services!",
        "Hi! 🤖 I'm your friendly AI assistant. How may I help you today?",
        "Greetings! I'm here to help. Feel free to ask me any questions!"
    ]
    
    # Farewell responses
    FAREWELL_RESPONSES = [
        "Goodbye! 👋 It was nice chatting with you. Feel free to come back anytime!",
        "See you later! Take care and don't hesitate to return if you have more questions! 😊",
        "Bye! Have a great day! I'll be here whenever you need assistance.",
        "Farewell! Thanks for chatting. Come back soon! 🌟",
        "Take care! Looking forward to helping you again! 👋",
        "Goodbye! Wishing you all the best! Feel free to return anytime."
    ]
    
    # Gratitude responses
    GRATITUDE_RESPONSES = [
        "You're welcome! 😊 Happy to help! Let me know if you have any other questions.",
        "My pleasure! Don't hesitate to ask if you need anything else!",
        "Glad I could help! 🌟 Is there anything else you'd like to know?",
        "You're welcome! Feel free to ask more questions anytime!",
        "Happy to assist! Let me know if there's anything else I can do for you.",
        "No problem at all! I'm here to help whenever you need! 😊"
    ]
    
    # About bot responses
    ABOUT_BOT_RESPONSES = [
        """I'm an AI-powered assistant designed to help you find information! 🤖

**What I can do:**
- Answer questions about our knowledge base (products, policies, FAQs)
- Have natural conversations with follow-up questions
- Process and analyze images (coming soon!)
- Provide helpful, accurate information

**How I work:**
I use RAG (Retrieval-Augmented Generation) to search through our knowledge base and provide relevant answers backed by real documents.

Feel free to ask me anything!""",
    ]
    
    # Help responses
    HELP_RESPONSES = [
        """Here's how to get the most out of me! 📚

**Getting Started:**
1. Simply type your question in the chat box
2. I'll search our knowledge base for relevant information
3. Ask follow-up questions - I remember our conversation!

**Example Questions:**
- "What products do you offer?"
- "How do I get a refund?"
- "Tell me about your pricing plans"
- "What are the system requirements?"

**Tips:**
- Be specific in your questions for better answers
- Ask follow-up questions for more details
- I work best with questions related to our products and services

What would you like to know?"""
    ]
    
    # Small talk responses
    SMALL_TALK_RESPONSES = [
        "I'm doing great, thanks for asking! 😊 I'm here and ready to help you. What can I assist you with today?",
        "I'm functioning well! Always happy to chat and help out. What's on your mind?",
        "Doing wonderful! Thank you for asking. How can I help you today?",
        "I'm great! Ready to assist you with any questions. What would you like to know?",
        "All systems operational! 🤖 How can I make your day better?",
    ]
    
    # Out of scope responses
    OUT_OF_SCOPE_RESPONSE = """I appreciate your question, but that's a bit outside my expertise! 🤔

**I'm specialized in helping with:**
- Product information and FAQs
- Company policies and procedures
- Technical documentation
- Getting started guides
- Refund and support policies

**I can't help with:**
- Real-time information (weather, news, stocks)
- External services (ordering food, booking tickets)
- General knowledge outside our knowledge base

Is there something related to our products or services I can help you with instead?"""

    # Inappropriate content response
    INAPPROPRIATE_RESPONSE = """I'm sorry, but I can't help with that request. 🚫

I'm designed to be helpful, harmless, and honest. That type of content goes against my guidelines.

**I'm here to help with:**
- Product questions
- Technical support
- Policy information
- General assistance

Is there something else I can help you with today?"""

    # Harmful content response
    HARMFUL_RESPONSE = """I'm not able to assist with that request. 

If you're experiencing a crisis or having thoughts of self-harm, please reach out to professional help:
- **National Suicide Prevention Lifeline:** 988 (US)
- **Crisis Text Line:** Text HOME to 741741
- **International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/

I'm here to help with questions about our products and services. Is there something else I can assist you with?"""

    # Empty query response
    EMPTY_QUERY_RESPONSE = """It looks like you didn't type anything! 😊

Feel free to ask me questions about:
- Our products and services
- Pricing and plans
- Technical support
- Policies and procedures

Just type your question and I'll do my best to help!"""

    # Gibberish response
    GIBBERISH_RESPONSE = """I'm not quite sure I understood that. 🤔

Could you please rephrase your question? Here are some examples of what I can help with:
- "What products do you offer?"
- "How do I contact support?"
- "Tell me about your refund policy"

What would you like to know?"""

    def __init__(self):
        """Initialize the prompt handler."""
        # Compile regex patterns for efficiency
        self.harmful_regex = [re.compile(p, re.IGNORECASE) for p in self.HARMFUL_PATTERNS]
        self.personal_info_regex = [re.compile(p, re.IGNORECASE) for p in self.PERSONAL_INFO_PATTERNS]
        self.out_of_scope_regex = [re.compile(p, re.IGNORECASE) for p in self.OUT_OF_SCOPE_PATTERNS]
        self.about_bot_regex = [re.compile(p, re.IGNORECASE) for p in self.ABOUT_BOT_PATTERNS]
        self.help_regex = [re.compile(p, re.IGNORECASE) for p in self.HELP_PATTERNS]
        self.small_talk_regex = [re.compile(p, re.IGNORECASE) for p in self.SMALL_TALK_PATTERNS]
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a query and determine how to handle it.
        
        Args:
            query: The user's input query
            
        Returns:
            QueryAnalysis with handling instructions
        """
        if not query:
            return QueryAnalysis(
                query_type=QueryType.EMPTY_QUERY,
                should_use_rag=False,
                direct_response=self.EMPTY_QUERY_RESPONSE,
                modified_query=None,
                confidence=1.0,
                flags=["empty_input"]
            )
        
        # Clean and normalize query
        cleaned = query.strip().lower()
        cleaned_no_punct = re.sub(r'[^\w\s]', '', cleaned)
        
        # Check for gibberish (very short with no real words)
        if self._is_gibberish(cleaned_no_punct):
            return QueryAnalysis(
                query_type=QueryType.GIBBERISH,
                should_use_rag=False,
                direct_response=self.GIBBERISH_RESPONSE,
                modified_query=None,
                confidence=0.8,
                flags=["possible_gibberish"]
            )
        
        # Check for harmful content FIRST (highest priority)
        if self._contains_harmful_content(query):
            return QueryAnalysis(
                query_type=QueryType.HARMFUL,
                should_use_rag=False,
                direct_response=self.HARMFUL_RESPONSE,
                modified_query=None,
                confidence=1.0,
                flags=["harmful_content_detected", "blocked"]
            )
        
        # Check for personal information requests
        if self._requests_personal_info(query):
            return QueryAnalysis(
                query_type=QueryType.PERSONAL_INFO,
                should_use_rag=False,
                direct_response=self.INAPPROPRIATE_RESPONSE,
                modified_query=None,
                confidence=0.9,
                flags=["personal_info_request", "blocked"]
            )
        
        # Check for greetings
        if self._is_greeting(cleaned_no_punct):
            return QueryAnalysis(
                query_type=QueryType.GREETING,
                should_use_rag=False,
                direct_response=random.choice(self.GREETING_RESPONSES),
                modified_query=None,
                confidence=1.0,
                flags=["greeting_detected"]
            )
        
        # Check for farewells
        if self._is_farewell(cleaned_no_punct):
            return QueryAnalysis(
                query_type=QueryType.FAREWELL,
                should_use_rag=False,
                direct_response=random.choice(self.FAREWELL_RESPONSES),
                modified_query=None,
                confidence=1.0,
                flags=["farewell_detected"]
            )
        
        # Check for gratitude
        if self._is_gratitude(cleaned_no_punct):
            return QueryAnalysis(
                query_type=QueryType.GRATITUDE,
                should_use_rag=False,
                direct_response=random.choice(self.GRATITUDE_RESPONSES),
                modified_query=None,
                confidence=1.0,
                flags=["gratitude_detected"]
            )
        
        # Check for about bot queries
        if self._is_about_bot(query):
            return QueryAnalysis(
                query_type=QueryType.ABOUT_BOT,
                should_use_rag=False,
                direct_response=random.choice(self.ABOUT_BOT_RESPONSES),
                modified_query=None,
                confidence=0.9,
                flags=["about_bot_query"]
            )
        
        # Check for help requests
        if self._is_help_request(query):
            return QueryAnalysis(
                query_type=QueryType.HELP_REQUEST,
                should_use_rag=False,
                direct_response=random.choice(self.HELP_RESPONSES),
                modified_query=None,
                confidence=0.9,
                flags=["help_request"]
            )
        
        # Check for small talk
        if self._is_small_talk(query):
            return QueryAnalysis(
                query_type=QueryType.SMALL_TALK,
                should_use_rag=False,
                direct_response=random.choice(self.SMALL_TALK_RESPONSES),
                modified_query=None,
                confidence=0.85,
                flags=["small_talk"]
            )
        
        # Check for out of scope queries
        if self._is_out_of_scope(query):
            return QueryAnalysis(
                query_type=QueryType.OUT_OF_SCOPE,
                should_use_rag=False,
                direct_response=self.OUT_OF_SCOPE_RESPONSE,
                modified_query=None,
                confidence=0.8,
                flags=["out_of_scope"]
            )
        
        # Normal query - proceed with RAG
        return QueryAnalysis(
            query_type=QueryType.NORMAL,
            should_use_rag=True,
            direct_response=None,
            modified_query=self._enhance_query(query),
            confidence=1.0,
            flags=["normal_query", "use_rag"]
        )
    
    def _is_gibberish(self, text: str) -> bool:
        """Check if text appears to be gibberish."""
        if len(text) < 2:
            return True
        
        # Check for repeated characters
        if re.match(r'^(.)\1+$', text):
            return True
        
        # Check for random character strings
        words = text.split()
        if len(words) == 1 and len(text) > 10:
            # Single long "word" with no vowels
            if not re.search(r'[aeiou]', text):
                return True
        
        # Check for keyboard mashing patterns
        keyboard_patterns = [
            r'^[asdfghjkl]+$', r'^[qwertyuiop]+$', r'^[zxcvbnm]+$',
            r'^[asdfjkl;]+$', r'^[123456789]+$'
        ]
        for pattern in keyboard_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _is_greeting(self, text: str) -> bool:
        """Check if text is a greeting."""
        words = text.split()
        
        # Single word greeting
        if text in self.GREETINGS:
            return True
        
        # Multi-word greeting
        for greeting in self.GREETINGS:
            if ' ' in greeting and greeting in text:
                return True
        
        # First word is greeting and query is short
        if words and words[0] in self.GREETINGS and len(words) <= 3:
            return True
        
        return False
    
    def _is_farewell(self, text: str) -> bool:
        """Check if text is a farewell."""
        words = text.split()
        
        if text in self.FAREWELLS:
            return True
        
        for farewell in self.FAREWELLS:
            if ' ' in farewell and farewell in text:
                return True
        
        if words and words[0] in self.FAREWELLS and len(words) <= 3:
            return True
        
        return False
    
    def _is_gratitude(self, text: str) -> bool:
        """Check if text expresses gratitude."""
        if text in self.GRATITUDE:
            return True
        
        for thanks in self.GRATITUDE:
            if thanks in text:
                return True
        
        return False
    
    def _is_about_bot(self, text: str) -> bool:
        """Check if query is about the bot itself."""
        for pattern in self.about_bot_regex:
            if pattern.search(text):
                return True
        return False
    
    def _is_help_request(self, text: str) -> bool:
        """Check if query is a help request."""
        for pattern in self.help_regex:
            if pattern.search(text):
                return True
        return False
    
    def _is_small_talk(self, text: str) -> bool:
        """Check if query is small talk."""
        for pattern in self.small_talk_regex:
            if pattern.search(text):
                return True
        return False
    
    def _is_out_of_scope(self, text: str) -> bool:
        """Check if query is out of scope."""
        for pattern in self.out_of_scope_regex:
            if pattern.search(text):
                return True
        return False
    
    def _contains_harmful_content(self, text: str) -> bool:
        """Check for harmful or dangerous content."""
        for pattern in self.harmful_regex:
            if pattern.search(text):
                return True
        return False
    
    def _requests_personal_info(self, text: str) -> bool:
        """Check if query requests personal information."""
        for pattern in self.personal_info_regex:
            if pattern.search(text):
                return True
        return False
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query for better RAG retrieval."""
        # Remove filler words
        filler_words = {'um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally'}
        words = query.split()
        enhanced_words = [w for w in words if w.lower() not in filler_words]
        
        # Clean up query
        enhanced = ' '.join(enhanced_words)
        
        # Remove excessive punctuation
        enhanced = re.sub(r'[!?]{2,}', '?', enhanced)
        
        return enhanced if enhanced else query
    
    def get_query_type_emoji(self, query_type: QueryType) -> str:
        """Get emoji for query type."""
        emoji_map = {
            QueryType.GREETING: "👋",
            QueryType.FAREWELL: "👋",
            QueryType.GRATITUDE: "🙏",
            QueryType.HELP_REQUEST: "❓",
            QueryType.ABOUT_BOT: "🤖",
            QueryType.SMALL_TALK: "💬",
            QueryType.OUT_OF_SCOPE: "🚫",
            QueryType.INAPPROPRIATE: "⚠️",
            QueryType.HARMFUL: "🚨",
            QueryType.PERSONAL_INFO: "🔒",
            QueryType.EMPTY_QUERY: "📭",
            QueryType.GIBBERISH: "❓",
            QueryType.NORMAL: "✅",
        }
        return emoji_map.get(query_type, "❓")


# Create singleton instance
prompt_handler = PromptHandler()


def analyze_and_respond(query: str) -> Tuple[bool, Optional[str], str]:
    """
    Convenience function to analyze query and get response.
    
    Returns:
        Tuple of (should_use_rag, direct_response, query_type_name)
    """
    analysis = prompt_handler.analyze_query(query)
    return (
        analysis.should_use_rag,
        analysis.direct_response,
        analysis.query_type.value
    )
