"""
Session management module for tracking user interactions.
Maintains conversation history per user.
"""
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading
import json

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    message_type: str = "text"  # 'text', 'image', 'query'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserSession:
    """User session with conversation history."""
    user_id: int
    created_at: float
    last_activity: float
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired (default 1 hour)."""
        return time.time() - self.last_activity > 3600
    
    def add_message(
        self,
        role: str,
        content: str,
        message_type: str = "text",
        metadata: Optional[Dict] = None
    ):
        """Add a message to the session."""
        self.messages.append(Message(
            role=role,
            content=content,
            timestamp=time.time(),
            message_type=message_type,
            metadata=metadata or {}
        ))
        self.last_activity = time.time()
    
    def get_history(self, limit: int = 3) -> List[Dict[str, str]]:
        """
        Get recent conversation history in LLM-compatible format.
        
        Args:
            limit: Number of message pairs to return
            
        Returns:
            List of {"role": "user/assistant", "content": "..."} dicts
        """
        history = []
        
        for msg in self.messages[-(limit * 2):]:
            history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return history
    
    def get_last_interaction(self) -> Optional[Dict[str, Any]]:
        """Get the last user-assistant interaction."""
        if len(self.messages) < 2:
            return None
        
        # Find last user message and its response
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == "user":
                user_msg = self.messages[i]
                # Look for assistant response
                if i + 1 < len(self.messages) and self.messages[i + 1].role == "assistant":
                    return {
                        "user": user_msg.content,
                        "assistant": self.messages[i + 1].content,
                        "type": user_msg.message_type,
                        "metadata": user_msg.metadata
                    }
        
        return None
    
    def clear_history(self):
        """Clear conversation history."""
        self.messages.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "message_count": len(self.messages),
            "context": self.context
        }


class SessionManager:
    """
    Manages user sessions with conversation history.
    Thread-safe implementation.
    """
    
    def __init__(
        self,
        history_length: int = 3,
        session_timeout: int = 3600,
        max_sessions: int = 1000
    ):
        """
        Initialize session manager.
        
        Args:
            history_length: Number of messages to keep per session
            session_timeout: Session expiry time in seconds
            max_sessions: Maximum number of concurrent sessions
        """
        self.history_length = history_length
        self.session_timeout = session_timeout
        self.max_sessions = max_sessions
        
        self._sessions: Dict[int, UserSession] = {}
        self._lock = threading.RLock()
        
        logger.info(f"SessionManager initialized (max sessions: {max_sessions})")
    
    def get_session(self, user_id: int) -> UserSession:
        """
        Get or create a session for a user.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            UserSession object
        """
        with self._lock:
            if user_id in self._sessions:
                session = self._sessions[user_id]
                
                # Check if expired
                if session.is_expired:
                    logger.debug(f"Session expired for user {user_id}")
                    session = self._create_session(user_id)
                else:
                    session.last_activity = time.time()
                
                return session
            
            return self._create_session(user_id)
    
    def _create_session(self, user_id: int) -> UserSession:
        """Create a new session for a user."""
        with self._lock:
            # Enforce max sessions
            self._cleanup_if_needed()
            
            session = UserSession(
                user_id=user_id,
                created_at=time.time(),
                last_activity=time.time()
            )
            
            self._sessions[user_id] = session
            logger.debug(f"Created session for user {user_id}")
            
            return session
    
    def add_user_message(
        self,
        user_id: int,
        content: str,
        message_type: str = "text",
        metadata: Optional[Dict] = None
    ):
        """Add a user message to session."""
        session = self.get_session(user_id)
        session.add_message("user", content, message_type, metadata)
        
        # Trim history if needed
        self._trim_history(session)
    
    def add_assistant_message(
        self,
        user_id: int,
        content: str,
        message_type: str = "text",
        metadata: Optional[Dict] = None
    ):
        """Add an assistant response to session."""
        session = self.get_session(user_id)
        session.add_message("assistant", content, message_type, metadata)
        
        # Trim history if needed
        self._trim_history(session)
    
    def _trim_history(self, session: UserSession):
        """Trim conversation history to configured length."""
        max_messages = self.history_length * 2  # User + assistant pairs
        
        if len(session.messages) > max_messages:
            session.messages = session.messages[-max_messages:]
    
    def get_conversation_history(
        self,
        user_id: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a user.
        
        Args:
            user_id: User ID
            limit: Override default limit
            
        Returns:
            List of message dictionaries
        """
        session = self.get_session(user_id)
        return session.get_history(limit or self.history_length)
    
    def get_last_interaction(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get the last interaction for a user."""
        session = self.get_session(user_id)
        return session.get_last_interaction()
    
    def set_context(self, user_id: int, key: str, value: Any):
        """Set context data for a session."""
        session = self.get_session(user_id)
        session.context[key] = value
    
    def get_context(self, user_id: int, key: str, default: Any = None) -> Any:
        """Get context data from a session."""
        session = self.get_session(user_id)
        return session.context.get(key, default)
    
    def clear_session(self, user_id: int):
        """Clear a user's session."""
        with self._lock:
            if user_id in self._sessions:
                del self._sessions[user_id]
                logger.debug(f"Cleared session for user {user_id}")
    
    def clear_all(self):
        """Clear all sessions."""
        with self._lock:
            self._sessions.clear()
            logger.info("All sessions cleared")
    
    def _cleanup_if_needed(self):
        """Cleanup expired sessions if at capacity."""
        with self._lock:
            # Remove expired sessions
            current_time = time.time()
            expired = [
                uid for uid, session in self._sessions.items()
                if current_time - session.last_activity > self.session_timeout
            ]
            
            for uid in expired:
                del self._sessions[uid]
            
            if expired:
                logger.debug(f"Cleaned up {len(expired)} expired sessions")
            
            # If still at capacity, remove oldest
            if len(self._sessions) >= self.max_sessions:
                oldest = sorted(
                    self._sessions.items(),
                    key=lambda x: x[1].last_activity
                )[:len(self._sessions) - self.max_sessions + 1]
                
                for uid, _ in oldest:
                    del self._sessions[uid]
                
                logger.debug(f"Removed {len(oldest)} oldest sessions")
    
    def cleanup_expired(self) -> int:
        """Manual cleanup of expired sessions."""
        with self._lock:
            current_time = time.time()
            expired = [
                uid for uid, session in self._sessions.items()
                if current_time - session.last_activity > self.session_timeout
            ]
            
            for uid in expired:
                del self._sessions[uid]
            
            return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        with self._lock:
            current_time = time.time()
            
            total_messages = sum(
                len(s.messages) for s in self._sessions.values()
            )
            
            active_sessions = sum(
                1 for s in self._sessions.values()
                if current_time - s.last_activity < 300  # Active in last 5 min
            )
            
            return {
                "total_sessions": len(self._sessions),
                "active_sessions": active_sessions,
                "total_messages": total_messages,
                "session_timeout": self.session_timeout,
                "max_sessions": self.max_sessions,
                "history_length": self.history_length
            }
    
    def get_session_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific session."""
        with self._lock:
            if user_id in self._sessions:
                return self._sessions[user_id].to_dict()
        return None
