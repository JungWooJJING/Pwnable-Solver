"""
Agent module for PWN Solver.

Provides agents with:
- OpenAI GPT-5.2 and Gemini 3 Flash Preview support
- Automatic history truncation when context overflows
- Token usage tracking
- Auto-retry on errors
"""

from Agent.base import (
    BaseAgent,
    TokenMetrics,
    HistoryManager,
    LLMClient,
    OpenAIClient,
    GeminiClient,
    get_llm_client,
    parse_json_response,
    get_model_pricing,
    get_context_limit,
    calculate_cost,
    MODEL_PRICING,
    MODEL_CONTEXT_LIMITS,
)

from Agent.plan import PlanAgent
from Agent.instruction import InstructionAgent
from Agent.parsing import ParsingAgent
from Agent.feedback import FeedbackAgent
from Agent.exploit import ExploitAgent, ExploitRefinerAgent


__all__ = [
    # Base
    "BaseAgent",
    "TokenMetrics",
    "HistoryManager",
    "LLMClient",
    "OpenAIClient",
    "GeminiClient",
    "get_llm_client",
    "parse_json_response",
    "get_model_pricing",
    "get_context_limit",
    "calculate_cost",
    "MODEL_PRICING",
    "MODEL_CONTEXT_LIMITS",
    # Agents
    "PlanAgent",
    "InstructionAgent",
    "ParsingAgent",
    "FeedbackAgent",
    "ExploitAgent",
    "ExploitRefinerAgent",
]
