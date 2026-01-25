"""
Base Agent class with OpenAI GPT-5.2 and Gemini 3 Flash Preview support.

Features:
- Automatic history truncation when context overflows
- Token usage tracking
- Streaming support
- Auto-retry on errors
"""

import os
import json
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from rich.console import Console

console = Console()


# =============================================================================
# Token Metrics
# =============================================================================

@dataclass
class TokenMetrics:
    """Accumulated token metrics from API responses."""
    message_count: int = 0
    max_context_size: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_tokens": self.total_tokens,
            "message_count": self.message_count,
            "max_context_size": self.max_context_size,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


# =============================================================================
# Model Pricing (per 1M tokens, USD)
# =============================================================================

MODEL_PRICING = {
    # OpenAI
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    # Gemini
    # NOTE: pricing/context for preview models may change; adjust as needed.
    "gemini-3-flash-preview": {"input": 0.15, "cached_input": 0.0375, "output": 0.60},
    "gemini-2.5-flash": {"input": 0.15, "cached_input": 0.0375, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gemini-1.5-pro": {"input": 1.25, "cached_input": 0.3125, "output": 5.00},
}

# Context window limits (tokens)
MODEL_CONTEXT_LIMITS = {
    "gpt-5.2": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gemini-3-flash-preview": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
}


def get_model_pricing(model: str) -> dict:
    """Get pricing for a model, handling version suffixes."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    
    for known_model in MODEL_PRICING:
        if model.startswith(known_model):
            return MODEL_PRICING[known_model]
    
    # Default fallback (warn but don't crash)
    console.print(f"[yellow]Warning: Unknown model '{model}', using gpt-4o pricing[/yellow]")
    return MODEL_PRICING["gpt-4o"]


def get_context_limit(model: str) -> int:
    """Get context window limit for a model."""
    if model in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model]
    
    for known_model in MODEL_CONTEXT_LIMITS:
        if model.startswith(known_model):
            return MODEL_CONTEXT_LIMITS[known_model]
    
    return 128_000  # Conservative default


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    model: str
) -> float:
    """Calculate USD cost from token counts."""
    pricing = get_model_pricing(model)
    uncached_input = max(input_tokens - cached_tokens, 0)
    
    input_cost = (uncached_input / 1_000_000) * pricing["input"]
    cached_cost = (cached_tokens / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + cached_cost + output_cost


# =============================================================================
# History Manager (Auto-truncation)
# =============================================================================

class HistoryManager:
    """
    Manages conversation history with automatic truncation.
    
    When context window overflows:
    1. Keep system message (always)
    2. Keep first user message (initial context)
    3. Truncate middle messages, keeping recent ones
    4. Add summary of truncated content
    """
    
    def __init__(self, max_tokens: int = 100_000, chars_per_token: float = 4.0):
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self.messages: List[Dict[str, str]] = []
        self.truncation_count = 0
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ~ 1 token)."""
        return int(len(text) / self.chars_per_token)
    
    def estimate_message_tokens(self, msg: Dict[str, str]) -> int:
        """Estimate tokens for a single message."""
        content = msg.get("content", "")
        # Add overhead for role, formatting
        return self.estimate_tokens(content) + 10
    
    def total_tokens(self) -> int:
        """Estimate total tokens in history."""
        return sum(self.estimate_message_tokens(m) for m in self.messages)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to history."""
        self.messages.append({"role": role, "content": content})
        self._maybe_truncate()
    
    def add_messages(self, messages: List[Dict[str, str]]) -> None:
        """Add multiple messages to history."""
        self.messages.extend(messages)
        self._maybe_truncate()
    
    def set_messages(self, messages: List[Dict[str, str]]) -> None:
        """Replace all messages."""
        self.messages = list(messages)
        self._maybe_truncate()
    
    def _maybe_truncate(self) -> None:
        """Truncate history if it exceeds max_tokens."""
        if self.total_tokens() <= self.max_tokens:
            return
        
        # Keep system message (index 0) and first user message (index 1)
        if len(self.messages) <= 2:
            return
        
        system_msg = self.messages[0] if self.messages[0]["role"] == "system" else None
        first_user_idx = 1 if system_msg else 0
        first_user_msg = self.messages[first_user_idx] if first_user_idx < len(self.messages) else None
        
        # Find how many messages to remove from middle
        target_tokens = int(self.max_tokens * 0.8)  # Leave 20% headroom
        
        # Start from index after first user message
        start_idx = first_user_idx + 1 if first_user_msg else 1
        
        # Keep removing oldest middle messages until under limit
        truncated_messages = []
        while self.total_tokens() > target_tokens and len(self.messages) > start_idx + 2:
            # Remove message after the protected initial messages
            removed = self.messages.pop(start_idx)
            truncated_messages.append(removed)
        
        if truncated_messages:
            self.truncation_count += len(truncated_messages)
            
            # Insert truncation notice
            notice = {
                "role": "system",
                "content": f"[History truncated: {len(truncated_messages)} messages removed to fit context window. Total truncations: {self.truncation_count}]"
            }
            self.messages.insert(start_idx, notice)
            
            console.print(f"[yellow]History truncated: removed {len(truncated_messages)} messages[/yellow]")
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get current message history."""
        return list(self.messages)
    
    def clear(self) -> None:
        """Clear history."""
        self.messages = []
        self.truncation_count = 0


# =============================================================================
# LLM Client Factory
# =============================================================================

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Send chat completion request.
        
        Returns:
            Tuple of (response_text, usage_dict)
            usage_dict contains: input_tokens, output_tokens, cached_tokens
        """
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client with truncation support."""
    
    def __init__(self):
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
            self.client = OpenAI()
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Enable auto-truncation for supported models
        if model.startswith("gpt-5") or model.startswith("gpt-4o"):
            kwargs["truncation"] = "auto"
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        response = self.client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content or ""
        
        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "cached_tokens": 0,
        }
        
        # Extract cached tokens if available
        if response.usage and hasattr(response.usage, "prompt_tokens_details"):
            details = response.usage.prompt_tokens_details
            if details and hasattr(details, "cached_tokens"):
                usage["cached_tokens"] = details.cached_tokens or 0
        
        return content, usage


class GeminiClient(LLMClient):
    """Google Gemini API client."""
    
    def __init__(self):
        try:
            import google.generativeai as genai  # type: ignore[import-not-found]
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            genai.configure(api_key=api_key)
            self.genai = genai
        except ImportError:
            raise RuntimeError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-3-flash-preview",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        # Convert messages to Gemini format
        gemini_model = self.genai.GenerativeModel(model)
        
        # Combine system + user messages
        system_msg = ""
        conversation = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_msg = content
            elif role == "user":
                conversation.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                conversation.append({"role": "model", "parts": [content]})
        
        # Prepend system message to first user message
        if system_msg and conversation:
            first_content = conversation[0]["parts"][0]
            conversation[0]["parts"][0] = f"{system_msg}\n\n---\n\n{first_content}"
        
        # Generation config
        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        # Send request
        chat = gemini_model.start_chat(history=conversation[:-1] if len(conversation) > 1 else [])
        response = chat.send_message(
            conversation[-1]["parts"][0] if conversation else "",
            generation_config=generation_config,
        )
        
        content = response.text or ""
        
        # Estimate tokens (Gemini doesn't always return usage)
        usage = {
            "input_tokens": sum(len(m["parts"][0]) // 4 for m in conversation),
            "output_tokens": len(content) // 4,
            "cached_tokens": 0,
        }
        
        # Try to get actual usage if available
        if hasattr(response, "usage_metadata"):
            meta = response.usage_metadata
            if hasattr(meta, "prompt_token_count"):
                usage["input_tokens"] = meta.prompt_token_count
            if hasattr(meta, "candidates_token_count"):
                usage["output_tokens"] = meta.candidates_token_count
            if hasattr(meta, "cached_content_token_count"):
                usage["cached_tokens"] = meta.cached_content_token_count or 0
        
        return content, usage


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """
    Get LLM client based on provider or environment.
    
    Args:
        provider: "openai" or "gemini". If None, auto-detect from env vars.
    """
    if provider == "openai" or (provider is None and os.environ.get("OPENAI_API_KEY")):
        return OpenAIClient()
    
    if provider == "gemini" or (provider is None and os.environ.get("GEMINI_API_KEY")):
        return GeminiClient()
    
    raise RuntimeError("No LLM API key found. Set OPENAI_API_KEY or GEMINI_API_KEY")


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """
    Base agent class with automatic history management and retry logic.
    
    Features:
    - Automatic history truncation when context overflows
    - Token usage tracking
    - Auto-retry on API errors
    - Support for OpenAI GPT-5.2 and Gemini 3 Flash Preview
    """
    
    # Default models
    DEFAULT_OPENAI_MODEL = "gpt-5.2"
    DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
    
    def __init__(
        self,
        name: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        temperature: float = 0.7,
        max_retries: int = 5,
        retry_delay: float = 1.0,
    ):
        self.name = name
        self.provider = provider
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Auto-detect provider and model
        if provider is None:
            if os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
            elif os.environ.get("GEMINI_API_KEY"):
                self.provider = "gemini"
            else:
                raise RuntimeError("No LLM API key found")
        
        if model is None:
            self.model = self.DEFAULT_OPENAI_MODEL if self.provider == "openai" else self.DEFAULT_GEMINI_MODEL
        else:
            self.model = model
        
        # Get context limit
        context_limit = max_context_tokens or get_context_limit(self.model)
        # Use 80% of limit to leave room for response
        self.history = HistoryManager(max_tokens=int(context_limit * 0.8))
        
        # Metrics
        self.metrics = TokenMetrics()
        
        # Client (lazy init)
        self._client: Optional[LLMClient] = None
    
    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_llm_client(self.provider)
        return self._client
    
    def _update_metrics(self, usage: Dict[str, int]) -> None:
        """Update token metrics from API response."""
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)
        
        self.metrics.total_input_tokens += input_tokens
        self.metrics.total_output_tokens += output_tokens
        self.metrics.total_cached_tokens += cached_tokens
        self.metrics.message_count += 1
        
        if input_tokens > self.metrics.max_context_size:
            self.metrics.max_context_size = input_tokens
        
        cost = calculate_cost(input_tokens, output_tokens, cached_tokens, self.model)
        self.metrics.total_cost_usd += cost
    
    def call_llm(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        use_history: bool = True,
    ) -> str:
        """
        Call LLM with automatic retry and history management.
        
        Args:
            messages: Messages to send. If None, uses history.
            use_history: If True, adds messages to history and uses it.
        
        Returns:
            Response text from LLM.
        """
        if messages is None:
            msgs_to_send = self.history.get_messages()
        elif use_history:
            self.history.add_messages(messages)
            msgs_to_send = self.history.get_messages()
        else:
            msgs_to_send = messages
        
        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response, usage = self.client.chat(
                    messages=msgs_to_send,
                    model=self.model,
                    temperature=self.temperature,
                )
                
                self._update_metrics(usage)
                
                # Add assistant response to history
                if use_history:
                    self.history.add_message("assistant", response)
                
                # Log usage
                console.print(
                    f"[dim]{self.name} | "
                    f"In: {usage.get('input_tokens', 0):,} | "
                    f"Out: {usage.get('output_tokens', 0):,} | "
                    f"Cost: ${self.metrics.total_cost_usd:.4f}[/dim]"
                )
                
                return response
                
            except Exception as e:
                last_error = e
                wait_time = self.retry_delay * (2 ** attempt)
                console.print(
                    f"[yellow]{self.name} error (attempt {attempt + 1}/{self.max_retries}): {e}[/yellow]"
                )
                console.print(f"[yellow]Retrying in {wait_time:.1f}s...[/yellow]")
                time.sleep(wait_time)
        
        raise RuntimeError(f"{self.name} failed after {self.max_retries} retries: {last_error}")
    
    def set_system_prompt(self, content: str) -> None:
        """Set or replace system prompt."""
        # Remove existing system message if any
        self.history.messages = [m for m in self.history.messages if m["role"] != "system"]
        # Insert system message at beginning
        self.history.messages.insert(0, {"role": "system", "content": content})
    
    def add_user_message(self, content: str) -> None:
        """Add user message to history."""
        self.history.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> None:
        """Add assistant message to history."""
        self.history.add_message("assistant", content)
    
    def clear_history(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        system_msg = None
        for m in self.history.messages:
            if m["role"] == "system":
                system_msg = m
                break
        
        self.history.clear()
        if system_msg:
            self.history.messages.append(system_msg)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current token metrics."""
        return self.metrics.to_dict()
    
    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with given state.
        
        Args:
            state: Current solver state
        
        Returns:
            Updated state
        """
        pass


# =============================================================================
# JSON Response Parser
# =============================================================================

def parse_json_response(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response."""
    # Try to find JSON block
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        json_str = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        json_str = text[start:end].strip()
    else:
        # Try to find JSON object directly
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
        else:
            json_str = text.strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        console.print(f"[red]JSON parse error: {e}[/red]")
        console.print(f"[dim]Raw text: {text[:500]}...[/dim]")
        return {}
