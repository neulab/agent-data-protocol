"""
Convert STD (Standardized) format to OWL (dual-agent conversation) format.

This script transforms single-agent trajectories into role-playing conversations
between a user (instruction-giver) and assistant (instruction-follower).
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
try:
    import litellm
    HAS_LITELLM = True
    # Suppress LiteLLM's verbose logging
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
except ImportError:
    HAS_LITELLM = False
    logging.warning("LiteLLM library not found.")

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Add project root to Python path for schema imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Schema imports
from schema.action.api import ApiAction
from schema.action.code import CodeAction
from schema.action.message import MessageAction
from schema.observation.text import TextObservation
from schema.observation.web import WebObservation
from schema.trajectory import Trajectory

# Type aliases for clarity
Event = Union[ApiAction, CodeAction, MessageAction, TextObservation, WebObservation, Any]
Action = Union[ApiAction, CodeAction, MessageAction]
Observation = Union[TextObservation, WebObservation, Any]
EventGroup = Tuple[List[Observation], List[Action]]

# New data structures for improved pipeline
@dataclass
class ActionSet:
    """Represents a consecutive sequence of agent actions."""
    actions: List[Action]
    start_index: int
    end_index: int

@dataclass
class ObservationSet:
    """Represents a consecutive sequence of observations."""
    observations: List[Observation]
    start_index: int
    end_index: int

@dataclass
class ProcessingGroup:
    """Combines an action set with processing results."""
    action_set: ActionSet
    related_obs_set: Optional[ObservationSet]
    task_type: str
    instruction: str
    assistant_response: str
    tool_calls: List[Dict[str, Any]]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global LLM model (LiteLLM handles client internally)
llm_model = ""

# Global rate limiting for LLM API calls (seconds between calls)
LLM_RATE_LIMIT_SECONDS = 0.5

# Global spending limit (USD) - breaks processing if exceeded
LLM_SPENDING_LIMIT = 5.0

# Global truncation statistics
truncation_stats = {
    'total_truncations': 0,
    'truncated_chars': 0,
    'longest_original': 0,
    'longest_truncated': 0
}

# Global cost tracking
cost_stats = {
    'total_cost': 0.0,
    'average_cost_per_trajectory': 0.0,
    'llm_calls': 0,
}

# Template dictionaries for rule-based generation
USER_TEMPLATES = {
    # alfworld functions
    "go": "Move to the following location: {location}",
    "take": "Pick up {item} from {source}",
    "put": "Place {item} onto {target}",
    "open": "Open the {obj}",
    "close": "Close the {obj}",
    "examine": "Look closely at {obj}",
    "heat": "Heat {item} using {appliance}",
    "cool": "Cool {item} using {appliance}",
    "clean": "Clean {item} using {appliance}",
    "use": "Use {obj}",
    "report_problem": "Report a problem with {obj}",
    "inventory": "Check what items you are currently holding",
    "look": "Look around and describe the current environment",
    "look_at_under": "Look at {item} under {reference}",
    
    # agenttuning_kg functions
    "get_relations": "Get all relations connected to entity or variable: {variable}",
    "get_neighbors": "Get all entities connected to {variable} via relation: {relation}",
    "intersection": "Compute the intersection of variables {variable1} and {variable2}",
    "get_attributes": "Get all numerical attributes of variable: {variable}",
    "argmax": "Return the entity with maximum value of attribute {attribute} in variable {variable}",
    "argmin": "Return the entity with minimum value of attribute {attribute} in variable {variable}",
    "count": "Count the number of entities in variable: {variable}",
    
    # agenttuning_webshop functions  
    "search": "Search for products using keywords: {keywords}",
    # "click": "Click on the element: {element}",  # DUPLICATE with multiple datasets
    
    # codeactinstruct functions
    "wikipedia_search": "Search Wikipedia for query: {query}",
    # "put": "Put {object} in/on {receptacle}",  # DUPLICATE with alfworld (different param names)
    "goto": "Go to the location: {receptacle}",  # NOTE: param name varies across datasets
    "take_from": "Take {object} from {receptacle}",
    "open_receptacle": "Open the receptacle: {receptacle}",
    "toggle": "Toggle the object or receptacle: {object_or_receptacle}",
    "close_receptacle": "Close the receptacle: {receptacle}",
    # "clean": "Clean {object} with {receptacle}",  # DUPLICATE with alfworld (different param names)
    # "heat": "Heat {object} with {receptacle}",  # DUPLICATE with alfworld (different param names)
    # "cool": "Cool {object} with {receptacle}",  # DUPLICATE with alfworld (different param names)  
    # "use": "Use the receptacle: {receptacle}",  # DUPLICATE with alfworld (different param names)
    # "look": "Look around the room",  # DUPLICATE with alfworld
    
    # go-browse-wa functions (browser automation)
    "noop": "Wait for {wait_ms} milliseconds",
    "scroll": "Scroll horizontally {delta_x} pixels and vertically {delta_y} pixels",
    "fill": "Fill form field {bid} with value: {value}",
    "select_option": "Select option {options} from dropdown {bid}",
    # "click": "Click element {bid}",  # DUPLICATE - browser version with bid param
    "dblclick": "Double click element {bid}",
    "hover": "Hover over element {bid}",
    "press": "Press key combination {key_comb} on element {bid}",
    "focus": "Focus on element {bid}",
    "clear": "Clear the input field {bid}",
    "drag_and_drop": "Drag element {from_bid} and drop onto element {to_bid}",
    "upload_file": "Upload file {file} via element {bid}",
    "go_back": "Navigate to the previous page in browser history",
    "go_forward": "Navigate to the next page in browser history",
    # "goto": "Navigate to URL: {url}",  # DUPLICATE - browser version
    
    # mind2web functions
    "select": "Select option {value} from dropdown at xpath {xpath}",
    # "click": "Click on element at xpath {xpath}",  # DUPLICATE - xpath version
    "type": "Type text {value} into element at xpath {xpath}",
    # "goto": "Navigate to URL: {url}",  # DUPLICATE
    
    # openhands functions
    "initialize": "Set environment variables: {env_vars}",
    "change_agent_state": "Change agent state to: {agent_state}",
    "delegate_to_agent": "Delegate task {task} to agent: {agent}",
    "delegate_to_CrawlAgent": "Delegate task {task} to CrawlAgent for link: {link}",
    "delegate_to_RagAgent": "Delegate task {task} to RagAgent with query: {query}",
    "finish": "Finish the task with output: {output}",
    "add_task": "Add task with goal: {goal}",
    "modify_task": "Modify task {task_id} to state: {state}",
    "save_plan": "Save the plan: {plan}",
    "task_plan": "Plan task {task} with plan: {plan}",
    "edit": "Edit file {path} from line {start} to {end} with content: {content}",
    "read": "Read file {path} from line {start} to {end}",
    "crawl": "Crawl webpage at link: {link}",
    "rag_search": "Search using RAG model with query: {query}",
    "send_msg_to_user": "Send message to user: {msg}",
}


def load_dotenv_config(custom_env_path: Optional[str] = None) -> bool:
    """Load environment variables from .env file.
    
    Args:
        custom_env_path: Custom path to .env file (optional)
        
    Returns:
        True if .env file was loaded, False otherwise
    """
    if not HAS_DOTENV:
        logger.debug("python-dotenv not available, skipping .env file loading")
        return False
    
    # Determine .env file paths to try
    env_paths = []
    
    if custom_env_path:
        # Use custom path if provided
        env_paths.append(Path(custom_env_path))
    else:
        # Try standard locations
        env_paths.extend([
            script_dir / ".env",      # scripts/.env
            project_root / ".env"     # project root .env
        ])
    
    for env_path in env_paths:
        if env_path.exists():
            try:
                load_dotenv(env_path, override=False)  # Don't override existing env vars
                logger.info(f"Loaded .env file from: {env_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load .env file {env_path}: {e}")
                continue
    
    logger.debug("No .env file found or loaded")
    return False


def initialize_llm_client(model: str = llm_model, env_file: Optional[str] = None) -> bool:
    """Initialize LiteLLM for proxy usage.
    
    Args:
        model: Model name to use
        env_file: Custom .env file path (optional)
        
    Returns:
        True if initialization succeeded, False otherwise
    """
    global llm_model
    
    # Store the model name globally
    llm_model = model
    
    if not HAS_LITELLM:
        logger.error("LiteLLM library not available. Install with: pip install litellm")
        return False
        
    try:
        # Load .env file if available
        load_dotenv_config(env_file)
        
        # Get proxy credentials from environment (now includes .env variables)
        proxy_key = os.getenv("LITELLM_PROXY_API_KEY")
        proxy_base = os.getenv("LITELLM_PROXY_API_BASE")
        
        if not proxy_key:
            logger.error("LITELLM_PROXY_API_KEY environment variable is required")
            return False
            
        if not proxy_base:
            logger.error("LITELLM_PROXY_API_BASE environment variable is required")
            return False
        
        # Configure LiteLLM global settings
        litellm.api_key = proxy_key
        litellm.api_base = proxy_base
        
        logger.info(f"Initialized LiteLLM with proxy_base: {proxy_base}, model: {model}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize LiteLLM: {e}")
        return False


class STDToOWLError(Exception):
    """Base exception for STD to OWL conversion errors."""
    pass


class GroupingError(STDToOWLError):
    """Error in event grouping logic."""
    pass


class LLMExtractionError(STDToOWLError):
    """Error in LLM-based instruction extraction."""
    pass


class ValidationError(STDToOWLError):
    """Error in validation logic."""
    pass


def truncate_content(content: str, max_length: int = 5000) -> str:
    """Truncate content if it exceeds max_length, preserving head and tail.
    
    Args:
        content: The content to potentially truncate
        max_length: Maximum allowed length
        
    Returns:
        Original content if under limit, otherwise truncated with marker
    """
    original_length = len(content)
    
    # Update statistics
    global truncation_stats
    truncation_stats['longest_original'] = max(truncation_stats['longest_original'], original_length)
    
    if original_length <= max_length:
        return content
    
    # Track truncation
    truncation_stats['total_truncations'] += 1
    truncated_chars = original_length - max_length
    truncation_stats['truncated_chars'] += truncated_chars
    truncation_stats['longest_truncated'] = max(truncation_stats['longest_truncated'], max_length)
    
    # Keep first and last portions
    head_size = max_length // 2
    tail_size = max_length - head_size - 50  # Reserve space for marker
    
    logger.debug(f"Truncating content from {original_length} to {max_length} characters ({truncated_chars} removed)")
    
    return (content[:head_size] + 
            f"\n\n... [truncated {truncated_chars} characters] ...\n\n" +
            content[-tail_size:])


def check_trajectory_length(trajectory: Trajectory, max_tokens: int) -> Tuple[bool, int]:
    """Check if trajectory is within acceptable length.
    
    Args:
        trajectory: The trajectory to check
        max_tokens: Maximum allowed total characters/tokens
        
    Returns:
        Tuple of (is_within_limit, total_length)
    """
    total_length = 0
    for event in trajectory.content:
        if hasattr(event, 'content'):
            total_length += len(str(event.content))
        else:
            # For events without content, estimate size
            total_length += len(str(event))
    
    return total_length <= max_tokens, total_length


def load_prompt_config(prompt_name: str) -> Dict[str, Any]:
    """Load LLM prompt configuration from JSON file.
    
    Args:
        prompt_name: Name of the prompt file (without .json extension)
        
    Returns:
        Dictionary containing prompt configuration
        
    Raises:
        STDToOWLError: If prompt file cannot be loaded
    """
    try:
        prompt_path = Path(__file__).parent / "owl_prompts" / f"{prompt_name}.json"
        if not prompt_path.exists():
            raise STDToOWLError(f"Prompt file not found: {prompt_path}")
            
        with open(prompt_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Validate required fields for new format
        required_fields = ["messages"]
        for field in required_fields:
            if field not in config:
                raise STDToOWLError(f"Missing required field '{field}' in {prompt_name}.json")
        
        # Validate messages structure
        if not isinstance(config["messages"], list) or len(config["messages"]) == 0:
            raise STDToOWLError(f"'messages' must be a non-empty list in {prompt_name}.json")
            
        for i, msg in enumerate(config["messages"]):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise STDToOWLError(f"Message {i} must have 'role' and 'content' fields in {prompt_name}.json")
                
        return config
        
    except json.JSONDecodeError as e:
        raise STDToOWLError(f"Invalid JSON in prompt file {prompt_name}.json: {e}")
    except Exception as e:
        raise STDToOWLError(f"Error loading prompt config {prompt_name}: {e}")


def identify_main_task_endpoints(trajectory: Trajectory, endpoint_count: int = 2, short_threshold: int = 500, max_obs_length: int = 5000) -> str:
    """Identify the main task from trajectory endpoints using LLM.

    Uses only the first and last few events instead of the entire trajectory for efficiency.
    For very short trajectories with simple Q&A patterns, extracts task directly from first observation.

    Args:
        trajectory: The trajectory to analyze
        endpoint_count: Number of events to take from start and end (default: 2)
        short_threshold: Max chars in first observation to treat as direct task (default: 500)
        max_obs_length: Maximum length for observation content

    Returns:
        Main task description string

    Raises:
        LLMExtractionError: If task identification fails
    """
    global llm_model, _last_api_call_ts

    try:
        events = trajectory.content

        # Handle very short trajectories (likely simple Q&A)
        if len(events) <= 3:
            first_event = events[0]
            if (hasattr(first_event, 'content') and
                len(str(first_event.content)) <= short_threshold):
                # Direct task extraction for simple Q&A
                logger.debug(f"Step 0: Using direct task extraction for short trajectory {trajectory.id}")
                return str(first_event.content).strip()

        # Extract endpoints for efficiency
        start_events = events[:endpoint_count]
        end_events = events[-endpoint_count:] if len(events) > endpoint_count else []

        # Combine start and end events (avoid duplicates if trajectory is very short)
        relevant_events = start_events
        for event in end_events:
            if event not in start_events:
                relevant_events.append(event)

        logger.debug(f"Step 0: Using endpoint extraction for trajectory {trajectory.id} ({len(relevant_events)}/{len(events)} events)")

        config = load_prompt_config("0_identify_main_task_endpoints")

        # Format interaction sequence for the prompt (same as original but with fewer events)
        interaction_lines = []
        for event in relevant_events:
            if hasattr(event, 'content'):
                content = truncate_content(str(event.content), max_obs_length)
                interaction_lines.append(f"- {event.class_}: {content}")
            else:
                event_str = truncate_content(str(event), max_obs_length)
                interaction_lines.append(f"- {event.class_}: {event_str}")

        interaction_sequence = "\\n".join(interaction_lines)

        if HAS_LITELLM:
            try:
                model_to_use = llm_model or "gpt-3.5-turbo"
                logger.debug(f"Step 0: Starting endpoint-based task identification with model {model_to_use}")

                # Build messages with template variables
                messages = []
                for msg in config["messages"]:
                    content = msg["content"].format(interaction_sequence=interaction_sequence)
                    messages.append({"role": msg["role"], "content": content})

                logger.debug(f"Step 0: Built {len(messages)} messages, total chars: {sum(len(m['content']) for m in messages)}")

                # Rate limiting
                now = time.time()
                if now - _last_api_call_ts < LLM_RATE_LIMIT_SECONDS:
                    time.sleep(LLM_RATE_LIMIT_SECONDS - (now - _last_api_call_ts))
                _last_api_call_ts = now

                params = config.get("parameters", {})
                logger.debug(f"Step 0: Making LLM request with max_tokens={params.get('max_tokens', 500)}")

                response = litellm.completion(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=params.get("max_tokens", 500),
                    temperature=params.get("temperature", 0)
                )

                logger.debug(f"Step 0: LLM request completed successfully")

                # Track cost
                track_llm_cost(response, "endpoint_task_identification")

                result = response.choices[0].message.content.strip()

                # Extract task from markdown code block if present
                if result.startswith("```") and result.endswith("```"):
                    result = result[3:-3].strip()

                if not result:
                    raise LLMExtractionError("Empty task description returned")

                return result

            except Exception as e:
                raise LLMExtractionError(f"LLM endpoint-based task identification failed: {e}")
        else:
            raise LLMExtractionError("LiteLLM not available for task identification")

    except Exception as e:
        raise LLMExtractionError(f"Failed to identify main task from endpoints: {e}")


def identify_main_task_llm(trajectory: Trajectory, max_obs_length: int = 5000) -> str:
    """Identify the main task from trajectory using LLM.
    
    Args:
        trajectory: The trajectory to analyze
        max_obs_length: Maximum length for observation content
        
    Returns:
        Main task description string
        
    Raises:
        LLMExtractionError: If task identification fails
    """
    global llm_model, _last_api_call_ts
    
    try:
        config = load_prompt_config("0_identify_main_task")
        
        # Format interaction sequence for the prompt
        interaction_lines = []
        for event in trajectory.content:
            if hasattr(event, 'content'):
                content = truncate_content(str(event.content), max_obs_length)
                interaction_lines.append(f"- {event.class_}: {content}")
            else:
                event_str = truncate_content(str(event), max_obs_length)
                interaction_lines.append(f"- {event.class_}: {event_str}")
        
        interaction_sequence = "\\n".join(interaction_lines)
        
        if HAS_LITELLM:
            try:
                model_to_use = llm_model or "gpt-3.5-turbo"
                logger.debug(f"Step 0: Starting main task identification with model {model_to_use}")
                
                # Build messages with template variables
                messages = []
                for msg in config["messages"]:
                    content = msg["content"].format(interaction_sequence=interaction_sequence)
                    messages.append({"role": msg["role"], "content": content})
                
                logger.debug(f"Step 0: Built {len(messages)} messages, total chars: {sum(len(m['content']) for m in messages)}")
                
                # Rate limiting
                now = time.time()
                if now - _last_api_call_ts < LLM_RATE_LIMIT_SECONDS:
                    time.sleep(LLM_RATE_LIMIT_SECONDS - (now - _last_api_call_ts))
                _last_api_call_ts = now
                
                params = config.get("parameters", {})
                logger.debug(f"Step 0: Making LLM request with max_tokens={params.get('max_tokens', 500)}")
                
                response = litellm.completion(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=params.get("max_tokens", 500),
                    temperature=params.get("temperature", 0)
                )

                logger.debug(f"Step 0: LLM request completed successfully")

                # Track cost
                track_llm_cost(response, "task_identification")

                result = response.choices[0].message.content.strip()

                # Extract task from markdown code block if present
                if result.startswith("```") and result.endswith("```"):
                    result = result[3:-3].strip()

                if not result:
                    raise LLMExtractionError("Empty task description returned")

                return result
                
            except Exception as e:
                raise LLMExtractionError(f"LLM task identification failed: {e}")
        else:
            raise LLMExtractionError("LiteLLM not available for task identification")
            
    except Exception as e:
        raise LLMExtractionError(f"Failed to identify main task: {e}")


def find_task_type_llm(action_set: ActionSet, obs_set: Optional[ObservationSet], max_obs_length: int = 5000) -> str:
    """Find task type using LLM.
    
    Args:
        action_set: The action set to analyze
        obs_set: Following observation set (if any)
        max_obs_length: Maximum length for observation content
        
    Returns:
        Task type string (one of: code_execution, information_retrieval, other_tool, task_completion)
        
    Raises:
        LLMExtractionError: If task type identification fails
    """
    global llm_model, _last_api_call_ts
    
    try:
        config = load_prompt_config("1_find_task_type")
        
        # Format actions for the prompt
        action_lines = []
        for action in action_set.actions:
            if hasattr(action, 'content'):
                content = truncate_content(str(action.content), max_obs_length)
                action_lines.append(f"- {action.class_}: {content}")
            elif hasattr(action, 'function'):
                function_name = getattr(action, 'function', 'unknown')
                kwargs = getattr(action, 'kwargs', {})
                action_lines.append(f"- {action.class_}: {function_name}({kwargs})")
            else:
                action_str = truncate_content(str(action), max_obs_length)
                action_lines.append(f"- {action.class_}: {action_str}")
        
        actions_text = "\\n".join(action_lines)
        
        if HAS_LITELLM:
            try:
                model_to_use = llm_model or "gpt-3.5-turbo"
                logger.debug(f"Step 1: Starting task type identification with model {model_to_use}")
                
                messages = []
                for msg in config["messages"]:
                    content = msg["content"].format(actions=actions_text)
                    messages.append({"role": msg["role"], "content": content})
                
                logger.debug(f"Step 1: Built {len(messages)} messages, actions text length: {len(actions_text)}")
                
                # Rate limiting
                now = time.time()
                if now - _last_api_call_ts < LLM_RATE_LIMIT_SECONDS:
                    time.sleep(LLM_RATE_LIMIT_SECONDS - (now - _last_api_call_ts))
                _last_api_call_ts = now
                
                params = config.get("parameters", {})
                logger.debug(f"Step 1: Making LLM request with max_tokens={params.get('max_tokens', 100)}")
                
                response = litellm.completion(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=params.get("max_tokens", 100),
                    temperature=params.get("temperature", 0)
                )

                logger.debug(f"Step 1: LLM request completed successfully")

                # Track cost
                track_llm_cost(response, "task_type_identification")
                
                result = response.choices[0].message.content.strip()

                # Remove backticks (1 or 3) if present.
                result = result.strip('`').strip().lower()
                
                valid_types = ["code_execution", "information_retrieval", "other_tool", "task_completion"]
                if result not in valid_types:
                    raise LLMExtractionError(f"Invalid task type returned: {result}")
                
                return result
                
            except Exception as e:
                raise LLMExtractionError(f"LLM task type identification failed: {e}")
        else:
            raise LLMExtractionError("LiteLLM not available for task type identification")
            
    except Exception as e:
        raise LLMExtractionError(f"Failed to identify task type: {e}")


def check_relevance_llm(action_set: ActionSet, obs_set: Optional[ObservationSet], context_obs: Optional[ObservationSet] = None, max_obs_length: int = 5000) -> str:
    """Check if observations are caused by actions using LLM.
    
    Args:
        action_set: The action set
        obs_set: The observation set to check
        context_obs: Previous observation set for context
        max_obs_length: Maximum length for observation content
        
    Returns:
        "YES" if causal, "NO" if not causal
        
    Raises:
        LLMExtractionError: If relevance check fails
    """
    global llm_model, _last_api_call_ts
    
    if obs_set is None:
        return "NO"
    
    try:
        config = load_prompt_config("2_check_relevance")
        
        # Format context
        context_text = ""
        if context_obs:
            context_lines = []
            for obs in context_obs.observations:
                content = truncate_content(str(getattr(obs, 'content', str(obs))), max_obs_length)
                context_lines.append(f"- {obs.class_}: {content}")
            context_text = "\\n".join(context_lines)
        
        # Format actions
        action_lines = []
        for action in action_set.actions:
            if hasattr(action, 'content'):
                content = truncate_content(str(action.content), max_obs_length)
                action_lines.append(f"- {action.class_}: {content}")
            elif hasattr(action, 'function'):
                function_name = getattr(action, 'function', 'unknown')
                kwargs = getattr(action, 'kwargs', {})
                action_lines.append(f"- {action.class_}: {function_name}({kwargs})")
            else:
                action_str = truncate_content(str(action), max_obs_length)
                action_lines.append(f"- {action.class_}: {action_str}")
        actions_text = "\\n".join(action_lines)
        
        # Format observations
        obs_lines = []
        for obs in obs_set.observations:
            content = truncate_content(str(getattr(obs, 'content', str(obs))), max_obs_length)
            obs_lines.append(f"- {obs.class_}: {content}")
        observations_text = "\\n".join(obs_lines)
        
        if HAS_LITELLM:
            try:
                model_to_use = llm_model or "gpt-3.5-turbo"
                logger.debug(f"Step 2: Starting relevance check with model {model_to_use}")
                
                messages = []
                for msg in config["messages"]:
                    content = msg["content"].format(
                        context=context_text,
                        actions=actions_text,
                        observations=observations_text
                    )
                    messages.append({"role": msg["role"], "content": content})
                
                logger.debug(f"Step 2: Built {len(messages)} messages, obs length: {len(observations_text)}")
                
                # Rate limiting
                now = time.time()
                if now - _last_api_call_ts < LLM_RATE_LIMIT_SECONDS:
                    time.sleep(LLM_RATE_LIMIT_SECONDS - (now - _last_api_call_ts))
                _last_api_call_ts = now
                
                params = config.get("parameters", {})
                logger.debug(f"Step 2: Making LLM request with max_tokens={params.get('max_tokens', 50)}")
                
                response = litellm.completion(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=params.get("max_tokens", 50),
                    temperature=params.get("temperature", 0)
                )

                logger.debug(f"Step 2: LLM request completed successfully")

                # Track cost
                track_llm_cost(response, "relevance_check")
                
                result = response.choices[0].message.content.strip().upper()
                
                if result not in ["YES", "NO"]:
                    raise LLMExtractionError(f"Invalid relevance response: {result}")
                
                return result
                
            except Exception as e:
                raise LLMExtractionError(f"LLM relevance check failed: {e}")
        else:
            raise LLMExtractionError("LiteLLM not available for relevance check")
            
    except Exception as e:
        raise LLMExtractionError(f"Failed to check relevance: {e}")


def generate_instruction_template(action_set: ActionSet, related_obs: Optional[ObservationSet] = None) -> str:
    """Generate user instruction using rule-based templates.
    
    Args:
        action_set: The action set
        related_obs: Related observations (unused in template version)
        
    Returns:
        Formatted instruction string
        
    Raises:
        STDToOWLError: If instruction generation fails
    """
    try:
        # Handle single action for now (can extend to multiple later)
        if not action_set.actions:
            raise STDToOWLError("Empty action set")
        
        action = action_set.actions[0]
        
        if isinstance(action, MessageAction):
            # MessageActions with <finish> should terminate - no instruction needed
            if has_finish_tag(action):
                return "<CAMEL_TASK_DONE>"
            else:
                raise STDToOWLError("MessageAction without <finish> tags - trajectory should be rejected")
                
        elif isinstance(action, CodeAction):
            # Use description field with template
            description = getattr(action, 'description', '').strip()
            if not description:
                raise STDToOWLError("CodeAction missing description field")
            # Clean up description (remove trailing colons, etc.)
            description = description.rstrip(':').strip()
            return f"Instruction: Take action based on this plan: {description}\nInput: None"
            
        elif isinstance(action, ApiAction):
            # Check if description exists first
            description = getattr(action, 'description', None)
            if description and description.strip():
                description = description.strip().rstrip(':').strip()
                return f"Instruction: Take action based on this plan: {description}\nInput: None"
            
            # Fall back to template-based generation
            function_name = getattr(action, 'function', 'unknown')
            kwargs = getattr(action, 'kwargs', {})
            
            if function_name not in USER_TEMPLATES:
                logger.warning(f"No template found for function: {function_name}")
                return f"Instruction: Execute the {function_name} function\nInput: None"
            
            # Extract and format parameters
            template = USER_TEMPLATES[function_name]
            try:
                # Clean parameter values (remove quotes)
                clean_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        clean_kwargs[key] = value.strip('"').strip("'")
                    else:
                        clean_kwargs[key] = value
                
                formatted_instruction = template.format(**clean_kwargs)
                return f"Instruction: {formatted_instruction}\nInput: None"
                
            except KeyError as e:
                logger.warning(f"Missing parameter {e} for template {template}")
                return f"Instruction: Execute the {function_name} function\nInput: None"
        
        else:
            raise STDToOWLError(f"Unknown action type: {type(action)}")
            
    except Exception as e:
        raise STDToOWLError(f"Failed to generate template instruction: {e}")


def generate_instruction_llm(action_set: ActionSet, related_obs: Optional[ObservationSet], max_obs_length: int = 5000) -> str:
    """Generate user instruction using LLM.
    
    Args:
        action_set: The action set
        related_obs: Related observations (if causal)
        max_obs_length: Maximum length for observation content
        
    Returns:
        Formatted instruction string
        
    Raises:
        LLMExtractionError: If instruction generation fails
    """
    global llm_model, _last_api_call_ts
    
    try:
        config = load_prompt_config("3_gen_instruction")
        
        # Format actions
        action_lines = []
        for action in action_set.actions:
            if hasattr(action, 'content'):
                content = truncate_content(str(action.content), max_obs_length)
                action_lines.append(f"- {action.class_}: {content}")
            elif hasattr(action, 'function'):
                function_name = getattr(action, 'function', 'unknown')
                kwargs = getattr(action, 'kwargs', {})
                action_lines.append(f"- {action.class_}: {function_name}({kwargs})")
            else:
                action_str = truncate_content(str(action), max_obs_length)
                action_lines.append(f"- {action.class_}: {action_str}")
        actions_text = "\\n".join(action_lines)
        
        # Format observations if present
        observations_text = ""
        if related_obs:
            obs_lines = []
            for obs in related_obs.observations:
                content = truncate_content(str(getattr(obs, 'content', str(obs))), max_obs_length)
                obs_lines.append(f"- {obs.class_}: {content}")
            observations_text = "\\n".join(obs_lines)
        
        if HAS_LITELLM:
            try:
                model_to_use = llm_model or "gpt-3.5-turbo"
                logger.debug(f"Step 3: Starting instruction generation with model {model_to_use}")
                
                messages = []
                for msg in config["messages"]:
                    content = msg["content"].format(
                        actions=actions_text,
                        observations=observations_text
                    )
                    messages.append({"role": msg["role"], "content": content})
                
                logger.debug(f"Step 3: Built {len(messages)} messages")
                
                # Rate limiting
                now = time.time()
                if now - _last_api_call_ts < LLM_RATE_LIMIT_SECONDS:
                    time.sleep(LLM_RATE_LIMIT_SECONDS - (now - _last_api_call_ts))
                _last_api_call_ts = now
                
                params = config.get("parameters", {})
                logger.debug(f"Step 3: Making LLM request with max_tokens={params.get('max_tokens', 500)}")
                
                response = litellm.completion(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=params.get("max_tokens", 500),
                    temperature=params.get("temperature", 0)
                )

                logger.debug(f"Step 3: LLM request completed successfully")

                # Track cost
                track_llm_cost(response, "instruction_generation")
                
                result = response.choices[0].message.content.strip()
                
                # Validate format
                if not ("Instruction:" in result and "Input:" in result):
                    raise LLMExtractionError(f"Invalid instruction format: {result}")
                
                return result
                
            except Exception as e:
                raise LLMExtractionError(f"LLM instruction generation failed: {e}")
        else:
            raise LLMExtractionError("LiteLLM not available for instruction generation")
            
    except Exception as e:
        raise LLMExtractionError(f"Failed to generate instruction: {e}")


def generate_response_template(action_set: ActionSet, related_obs: Optional[ObservationSet] = None, max_obs_length: int = 5000) -> str:
    """Generate assistant response using rule-based templates.
    
    Args:
        action_set: The action set
        related_obs: Related observations (if causal)
        max_obs_length: Maximum length for observation content
        
    Returns:
        Assistant response string (without "Solution:" prefix or "Next request." suffix)
        
    Raises:
        STDToOWLError: If response generation fails
    """
    try:
        if not action_set.actions:
            raise STDToOWLError("Empty action set")
        
        action = action_set.actions[0]
        
        if isinstance(action, MessageAction):
            # MessageActions with <finish> should not generate responses - conversation ends
            if has_finish_tag(action):
                return ""  # No response needed, conversation terminates
            else:
                raise STDToOWLError("MessageAction without <finish> tags - trajectory should be rejected")
                
        elif isinstance(action, CodeAction):
            # Generate response based on observations
            if related_obs and related_obs.observations:
                obs_content = []
                for obs in related_obs.observations:
                    content = truncate_content(str(getattr(obs, 'content', str(obs))), max_obs_length).rstrip('.')  # Remove trailing period
                    obs_content.append(content)
                obs_text = "\n".join(obs_content)
                return f"Code was executed and resulted in the following:\n{obs_text}"
            else:
                return "Code was executed"
                
        elif isinstance(action, ApiAction):
            function_name = getattr(action, 'function', 'unknown')
            
            if related_obs and related_obs.observations:
                obs_content = []
                for obs in related_obs.observations:
                    content = truncate_content(str(getattr(obs, 'content', str(obs))), max_obs_length).rstrip('.')  # Remove trailing period
                    obs_content.append(content)
                obs_text = "\n".join(obs_content)
                return f"{function_name} was executed and resulted in the following:\n{obs_text}"
            else:
                return f"{function_name} was executed"
        
        else:
            raise STDToOWLError(f"Unknown action type: {type(action)}")
            
    except Exception as e:
        raise STDToOWLError(f"Failed to generate template response: {e}")


def generate_response_llm(action_set: ActionSet, related_obs: Optional[ObservationSet], max_obs_length: int = 5000) -> str:
    """Generate assistant response using LLM.
    
    Args:
        action_set: The action set
        related_obs: Related observations (if causal)
        max_obs_length: Maximum length for observation content
        
    Returns:
        Assistant response string
        
    Raises:
        LLMExtractionError: If response generation fails
    """
    global llm_model, _last_api_call_ts
    
    try:
        config = load_prompt_config("5_gen_response")
        
        # Format actions
        action_lines = []
        for action in action_set.actions:
            if hasattr(action, 'content'):
                content = truncate_content(str(action.content), max_obs_length)
                action_lines.append(f"- {action.class_}: {content}")
            elif hasattr(action, 'function'):
                function_name = getattr(action, 'function', 'unknown')
                kwargs = getattr(action, 'kwargs', {})
                action_lines.append(f"- {action.class_}: {function_name}({kwargs})")
            else:
                action_str = truncate_content(str(action), max_obs_length)
                action_lines.append(f"- {action.class_}: {action_str}")
        actions_text = "\\n".join(action_lines)
        
        # Format observations if present
        observations_text = ""
        if related_obs:
            obs_lines = []
            for obs in related_obs.observations:
                content = truncate_content(str(getattr(obs, 'content', str(obs))), max_obs_length)
                obs_lines.append(f"- {obs.class_}: {content}")
            observations_text = "\\n".join(obs_lines)
        
        if HAS_LITELLM:
            try:
                model_to_use = llm_model or "gpt-3.5-turbo"
                logger.debug(f"Step 5: Starting response generation with model {model_to_use}")
                
                messages = []
                for msg in config["messages"]:
                    content = msg["content"].format(
                        actions=actions_text,
                        observations=observations_text
                    )
                    messages.append({"role": msg["role"], "content": content})
                
                logger.debug(f"Step 5: Built {len(messages)} messages")
                
                # Rate limiting
                now = time.time()
                if now - _last_api_call_ts < LLM_RATE_LIMIT_SECONDS:
                    time.sleep(LLM_RATE_LIMIT_SECONDS - (now - _last_api_call_ts))
                _last_api_call_ts = now
                
                params = config.get("parameters", {})
                logger.debug(f"Step 5: Making LLM request with max_tokens={params.get('max_tokens', 1000)}")
                
                response = litellm.completion(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=params.get("max_tokens", 1000),
                    temperature=params.get("temperature", 0)
                )

                logger.debug(f"Step 5: LLM request completed successfully")

                # Track cost
                track_llm_cost(response, "response_generation")
                
                result = response.choices[0].message.content.strip()
                
                # Validate markdown format
                if not (result.startswith("```") and result.endswith("```")):
                    raise LLMExtractionError(f"Invalid response format (should be markdown block): {result}")
                
                return result
                
            except Exception as e:
                raise LLMExtractionError(f"LLM response generation failed: {e}")
        else:
            raise LLMExtractionError("LiteLLM not available for response generation")
            
    except Exception as e:
        raise LLMExtractionError(f"Failed to generate response: {e}")


def validate_trajectory(trajectory: Trajectory) -> bool:
    """Validate that trajectory has required structure for conversion.
    
    Args:
        trajectory: The trajectory to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If trajectory is invalid
    """
    if not trajectory.id:
        raise ValidationError("Trajectory missing required 'id' field")
        
    if not trajectory.content or len(trajectory.content) == 0:
        raise ValidationError("Trajectory has empty content")
        
    # Check MessageAction sequence validity first
    if not validate_message_action_sequence(trajectory.content):
        raise ValidationError("Invalid MessageAction sequence - trajectory should be rejected")
        
    # Check each event has class_ field
    for i, event in enumerate(trajectory.content):
        if not hasattr(event, 'class_'):
            raise ValidationError(f"Event {i} missing 'class_' field: {event}")
            
        # Validate known event types
        known_classes = [
            'api_action', 'code_action', 'message_action',
            'text_observation', 'web_observation', 'image_observation'
        ]
        if event.class_ not in known_classes:
            logger.warning(f"Unknown event class: {event.class_}")
            
    return True


def is_action(event: Event) -> bool:
    """Check if event is an action type."""
    return hasattr(event, 'class_') and event.class_.endswith('_action')


def is_observation(event: Event) -> bool:
    """Check if event is an observation type.""" 
    return hasattr(event, 'class_') and event.class_.endswith('_observation')


def has_finish_tag(message_action: MessageAction) -> bool:
    """Check if MessageAction contains <finish> tags."""
    content = getattr(message_action, 'content', '')
    return '<finish>' in content and '</finish>' in content


def should_skip_initial_message_action(action: MessageAction, position: int, total_actions: int) -> bool:
    """Check if this is an initial acknowledgment MessageAction that should be skipped."""
    if position > 2:  # Not at the very beginning
        return False
    
    content = getattr(action, 'content', '').lower().strip()
    
    # Acknowledgment phrases that are clearly boilerplate responses
    acknowledgment_phrases = [
        "ok", "okay", "understood", "got it", "sure", "yes",
        "i'll follow", "i will follow", "i'll do that", "i will do that",
        "alright", "will do"
    ]
    
    # Check if content is just an acknowledgment phrase (possibly with punctuation)
    # Remove common punctuation for matching
    clean_content = content.strip('.,!').strip()
    return (clean_content in acknowledgment_phrases or 
            any(clean_content.startswith(phrase) for phrase in [
                "ok", "okay", "sure", "alright", "understood",
                "i'll follow", "i will follow"
            ]))


def validate_message_action_sequence(events: List[Event]) -> bool:
    """Validate MessageAction sequence according to MOD.md rules.
    
    Returns True if valid, False if trajectory should be rejected.
    Rules:
    - Initial acknowledgment MessageActions are allowed and will be skipped
    - Finish MessageActions should only appear at the end and must contain <finish> tags
    - No non-MessageActions should appear after finish MessageActions
    """
    finish_message_indices = []
    for i, event in enumerate(events):
        if isinstance(event, MessageAction):
            # Skip initial acknowledgment messages - they're valid but will be filtered out
            if should_skip_initial_message_action(event, i, len(events)):
                logger.debug(f"Skipping validation for initial acknowledgment MessageAction at index {i}")
                continue
                
            # This is a non-initial MessageAction - it must have finish tags
            if not has_finish_tag(event):
                logger.warning(f"Non-initial MessageAction at index {i} lacks <finish> tags")
                return False
                
            finish_message_indices.append(i)
    
    if not finish_message_indices:
        return True  # No finish MessageActions is fine (initial ones were skipped)
    
    # Check that no non-MessageActions appear after any finish MessageAction
    first_finish_message_idx = finish_message_indices[0]
    for i in range(first_finish_message_idx + 1, len(events)):
        if not isinstance(events[i], MessageAction):
            logger.warning(f"Non-MessageAction found after finish MessageAction at index {i}")
            return False
    
    return True


def group_events_alternating(events: List[Event]) -> List[Union[ObservationSet, ActionSet]]:
    """Group events into alternating observation and action sets.
    
    Creates alternating ObservationSet and ActionSet objects following the pattern:
    obs-action-obs-action... Every action set must have a following observation set.
    Initial acknowledgment MessageActions are filtered out during grouping.
    
    Args:
        events: List of events from STD trajectory
        
    Returns:
        List alternating between ObservationSet and ActionSet objects
        
    Raises:
        GroupingError: If events cannot be properly grouped or pattern is violated
    """
    if not events:
        raise GroupingError("Cannot group empty event list")
        
    if not all(hasattr(e, 'class_') for e in events):
        raise GroupingError("All events must have 'class_' field for grouping")
    
    groups = []
    current_set = []
    current_type = None
    start_index = 0
    
    for i, event in enumerate(events):
        try:
            # Skip initial acknowledgment MessageActions
            if isinstance(event, MessageAction) and should_skip_initial_message_action(event, i, len(events)):
                logger.debug(f"Skipping initial acknowledgment MessageAction at position {i}")
                continue
                
            event_is_obs = is_observation(event)
            event_is_action = is_action(event)
            
            if not (event_is_obs or event_is_action):
                logger.warning(f"Unknown event type at position {i}: {event.class_}")
                continue
                
            event_type = 'obs' if event_is_obs else 'action'
            
            # If this is the first event or same type as current, add to current set
            if current_type is None or current_type == event_type:
                if current_type is None:
                    current_type = event_type
                    start_index = i
                current_set.append(event)
            else:
                # Type changed - finalize current set and start new one
                if current_type == 'obs':
                    groups.append(ObservationSet(current_set.copy(), start_index, i - 1))
                else:
                    groups.append(ActionSet(current_set.copy(), start_index, i - 1))
                
                # Start new set
                current_set = [event]
                current_type = event_type
                start_index = i
                
        except Exception as e:
            raise GroupingError(f"Error processing event {i}: {e}")
    
    # Handle final set
    if current_set:
        if current_type == 'obs':
            groups.append(ObservationSet(current_set, start_index, len(events) - 1))
        else:
            groups.append(ActionSet(current_set, start_index, len(events) - 1))
    
    if not groups:
        raise GroupingError("No valid groups found")
    
    # Validate alternating pattern and ensure each action set has following observation set
    action_indices = [i for i, g in enumerate(groups) if isinstance(g, ActionSet)]
    obs_indices = [i for i, g in enumerate(groups) if isinstance(g, ObservationSet)]
    
    for action_idx in action_indices:
        # Check if there's an observation set after this action set
        next_obs_exists = any(obs_idx > action_idx for obs_idx in obs_indices)
        # If it's the last group, then it's fine if no next observation
        next_obs_exists = next_obs_exists or (action_idx == len(groups) - 1)
        
        if not next_obs_exists:
            raise GroupingError(f"Action set at index {action_idx} has no following observation set")
    
    logger.debug(f"Grouped {len(events)} events into {len(groups)} alternating sets (skipped initial acknowledgments)")
    return groups


_last_api_call_ts = 0.0


def track_llm_cost(response, step_name: str = "unknown"):
    """Track the cost of an LLM API call using response cost information.

    Args:
        response: LiteLLM completion response object
        step_name: Name of the processing step for logging
    """
    try:
        # Primary: Use built-in cost from response hidden params
        cost = None
        if hasattr(response, '_hidden_params') and response._hidden_params:
            cost = response._hidden_params.get("response_cost", None)

        # Fallback: Calculate cost from response
        if cost is None and HAS_LITELLM:
            from litellm import completion_cost
            cost = completion_cost(completion_response=response)

        if cost is not None:
            # Update global cost_stats
            global cost_stats
            cost_stats['total_cost'] += cost
            cost_stats['llm_calls'] += 1

            logger.debug(f"Cost for {step_name}: ${cost:.6f}")

        else:
            logger.debug(f"Could not determine cost for {step_name}")

    except Exception as e:
        # Log cost tracking errors but continue processing
        logger.debug(f"Could not track cost for {step_name}: {e}")

def generate_tool_call_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:20]}"


def is_user_feedback_request(task_type: str, action: MessageAction) -> bool:
    """Detect if message action represents a user feedback request.
    
    Args:
        task_type: The task type from LLM analysis
        action: The message action to check
        
    Returns:
        True if this should be a user_feedback tool call
    """
    # Per DETAILED_EDIT.md: user_feedback can be identified from task type
    if task_type == "information_retrieval":
        # Check if this is asking the user for information
        content = getattr(action, 'content', '').lower()
        feedback_indicators = [
            'what would you like', 'please specify', 'can you clarify', 'would you prefer',
            'could you provide', 'please tell me', 'what do you think', 'which option',
            'do you want', 'would you like me to', 'should i', 'how would you like'
        ]
        return any(indicator in content for indicator in feedback_indicators)
    
    return False


def convert_std_action_to_tool_call(action: Action, task_type: str = "") -> Dict[str, Any]:
    """Convert STD action to OpenAI tool call format.
    
    Args:
        action: STD action to convert
        task_type: Task type for user_feedback detection
        
    Returns:
        Tool call dictionary in OpenAI format
    """
    tool_call = {
        "id": generate_tool_call_id(),
        "type": "function",
        "function": {}
    }
    
    if isinstance(action, CodeAction):
        # Map CodeAction to execute_code function
        tool_call["function"]["name"] = "execute_code"
        tool_call["function"]["arguments"] = json.dumps({
            "code": getattr(action, 'content', ''),
            "language": getattr(action, 'language', 'python')
        })
        
    elif isinstance(action, ApiAction):
        # Map ApiAction to its function name
        tool_call["function"]["name"] = getattr(action, 'function', 'unknown_function')
        tool_call["function"]["arguments"] = json.dumps(getattr(action, 'kwargs', {}))
        
    elif isinstance(action, MessageAction):
        # Check if this should be a user_feedback tool call
        if is_user_feedback_request(task_type, action):
            tool_call["function"]["name"] = "user_feedback"
            tool_call["function"]["arguments"] = json.dumps({
                "query": action.content
            })
        else:
            # Handle other message actions (finish, send_message)
            content = getattr(action, 'content', '')
            if '<finish>' in content and '</finish>' in content:
                import re
                match = re.search(r'<finish>(.*?)</finish>', content, re.DOTALL)
                if match:
                    tool_call["function"]["name"] = "finish"
                    tool_call["function"]["arguments"] = json.dumps({
                        "message": match.group(1).strip(),
                        "task_completed": True
                    })
                else:
                    # Generic message action
                    tool_call["function"]["name"] = "send_message"
                    tool_call["function"]["arguments"] = json.dumps({"message": content})
            else:
                # Generic message action
                tool_call["function"]["name"] = "send_message"
                tool_call["function"]["arguments"] = json.dumps({"message": content})
    
    return tool_call


def convert_action_set_to_tool_calls(action_set: ActionSet, task_type: str) -> List[Dict[str, Any]]:
    """Convert action set to OpenAI tool calls format.
    
    Args:
        action_set: The action set to convert
        task_type: Task type for user_feedback detection
        
    Returns:
        List of OpenAI tool call dictionaries
    """
    tool_calls = []
    for action in action_set.actions:
        tool_call = convert_std_action_to_tool_call(action, task_type)
        tool_calls.append(tool_call)
    return tool_calls




def process_action_set(action_set: ActionSet, obs_set: Optional[ObservationSet], 
                       main_task: str, context_obs: Optional[ObservationSet] = None, 
                       use_templates: bool = True, use_llm_relevance: bool = False, max_obs_length: int = 5000) -> ProcessingGroup:
    """Process an action set through template-based or LLM pipeline.
    
    Args:
        action_set: The action set to process
        obs_set: The observation set following the action set
        main_task: The main task description
        context_obs: Previous observation set for context
        use_templates: Whether to use template-based generation (default True)
        max_obs_length: Maximum length for observation content
        
    Returns:
        ProcessingGroup with all derived information
        
    Raises:
        STDToOWLError: If any step fails
    """
    if use_templates:
        # Template-based pipeline (new approach)
        # Step 1: Check causality 
        if use_llm_relevance and HAS_LITELLM:
            relevance = check_relevance_llm(action_set, obs_set, context_obs, max_obs_length)
        else:
            relevance = "YES"  # Default to assuming all observations are relevant
        related_obs = obs_set if relevance == "YES" else None
        
        # Step 2: Generate instruction using templates
        instruction = generate_instruction_template(action_set, related_obs)
        
        # Step 3: Convert to tool calls
        task_type = "template_generated"  # Simplified since we're not using LLM classification
        tool_calls = convert_action_set_to_tool_calls(action_set, task_type)
        
        # Step 4: Generate assistant response using templates
        assistant_response = generate_response_template(action_set, related_obs, max_obs_length)
        
    else:
        # Original LLM-based pipeline
        # Step 1: Find task type
        task_type = find_task_type_llm(action_set, obs_set, max_obs_length)
        
        # Step 2: Check causality 
        relevance = check_relevance_llm(action_set, obs_set, context_obs, max_obs_length)
        related_obs = obs_set if relevance == "YES" else None
        
        # Step 3: Generate instruction
        instruction = generate_instruction_llm(action_set, related_obs, max_obs_length)
        
        # Step 4: Convert to tool calls
        tool_calls = convert_action_set_to_tool_calls(action_set, task_type)
        
        # Step 5: Generate assistant response
        assistant_response = generate_response_llm(action_set, related_obs, max_obs_length)
    
    return ProcessingGroup(
        action_set=action_set,
        related_obs_set=related_obs,
        task_type=task_type,
        instruction=instruction,
        assistant_response=assistant_response,
        tool_calls=tool_calls
    )


def build_parallel_conversations(processing_groups: List[ProcessingGroup], main_task: str, trajectory_id: str, initial_context: str = "") -> Tuple[Dict, Dict]:
    """Build both user and assistant conversations simultaneously to guarantee parallelism.
    
    Args:
        processing_groups: List of processed action sets
        main_task: Main task description
        trajectory_id: Trajectory ID for conversation ID
        initial_context: Initial context from first observations to embed in system prompt
        
    Returns:
        Tuple of (user_conversation, assistant_conversation) dictionaries
    """
    try:
        # Load system prompts
        user_config = load_prompt_config("sysprompt_owl_user")
        assistant_config = load_prompt_config("sysprompt_owl_assistant")
        
        user_messages = []
        assistant_messages = []
        
        # Add system messages
        for msg in user_config["messages"]:
            if msg["role"] == "system":
                content = msg["content"].format(task_description=main_task, initial_context=initial_context)
                user_messages.append({"role": "system", "content": content})
                break
                
        for msg in assistant_config["messages"]:
            if msg["role"] == "system":
                content = msg["content"].format(task_description=main_task, initial_context=initial_context)
                assistant_messages.append({"role": "system", "content": content})
                break
        
        # Add initial user message (user conversation only)
        for msg in user_config["messages"]:
            if msg["role"] == "user":
                user_messages.append({"role": "user", "content": msg["content"]})
                break
        
        # Process each group - building both conversations in parallel
        for group in processing_groups:
            # Handle MessageAction with <finish> tags - terminate conversation
            if (group.action_set.actions and 
                isinstance(group.action_set.actions[0], MessageAction) and 
                has_finish_tag(group.action_set.actions[0])):
                # Add <CAMEL_TASK_DONE> to user conversation (user agent terminates)
                user_messages.append({
                    "role": "assistant", 
                    "content": "<CAMEL_TASK_DONE>"
                })
                # Add <CAMEL_TASK_DONE> to assistant conversation (user terminates)
                assistant_messages.append({
                    "role": "user",
                    "content": "<CAMEL_TASK_DONE>",
                    "refusal": None,
                    "reasoning": None
                })
                break  # Terminate conversation building
            
            # Extract and format response content once
            response_content = group.assistant_response
            if response_content.startswith("```") and response_content.endswith("```"):
                response_content = response_content[3:-3].strip()
            
            # Skip empty responses (e.g., from MessageActions)
            if not response_content:
                continue
                
            # Format response properly (template already excludes "Solution:" and "Next request.")
            formatted_response = f"Solution: {response_content}. Next request."
            
            # USER CONVERSATION:
            # 1. Add instruction as assistant message (user agent gives instructions)
            user_messages.append({
                "role": "assistant",
                "content": group.instruction,
                "refusal": None,
                "reasoning": None
            })
            
            # 2. Add formatted response as user message  
            user_messages.append({
                "role": "user",
                "content": formatted_response
            })
            
            # ASSISTANT CONVERSATION:
            # 1. Add instruction as user message (user gives instruction to assistant)
            assistant_messages.append({
                "role": "user",
                "content": group.instruction
            })
            
            # 2. Add assistant response with tool calls
            if group.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": "",
                    "refusal": None,
                    "reasoning": None,
                    "tool_calls": group.tool_calls
                }
                assistant_messages.append(assistant_msg)

                if group.related_obs_set:
                    for i, obs in enumerate(group.related_obs_set.observations):
                        # Use the corresponding tool call ID, or last one if not enough
                        tool_call_id = group.tool_calls[min(i, len(group.tool_calls) - 1)]["id"]
                        
                        assistant_messages.append({
                            "role": "tool",
                            "content": getattr(obs, 'content', str(obs)),
                            "tool_call_id": tool_call_id
                        })
                else:
                    # If none, create boilerplate response.
                    assistant_messages.append({
                        "role": "tool",
                        "content": "Tool calls completed.",
                        "tool_call_id": "NONE"
                    })

            assistant_msg = {
                "role": "assistant",
                "content": formatted_response,
                "refusal": None,
                "reasoning": None,
            }

            assistant_messages.append(assistant_msg)
        
        # Now, we check for the very specific case where there is one input (observation) set
        # followed by one finishing action set, because that would lead to no conversation (the finish
        # would turn into <CAMEL_TASK_DONE> without the relevant information being conveyed.)
        first_action_is_finish = (
            len(processing_groups) == 1 and
            processing_groups[0].action_set.actions and
            isinstance(processing_groups[0].action_set.actions[0], MessageAction) and
            has_finish_tag(processing_groups[0].action_set.actions[0])
        )
        if first_action_is_finish:
            # Before the <CAMEL_TASK_DONE> (a user-message)
            # Add "Complete the task: {task}" as user message, then:
            # Add the processing_groups[0].action_set.actions[0].content (stripped of tags) as assistant message
            user_messages.insert(-1, {
                "role": "assistant",
                "content": f"Complete the task: {main_task}"
            })
            assistant_messages.insert(-1, {
                "role": "user",
                "content": f"Complete the task: {main_task}"
            })
            finish_content = processing_groups[0].action_set.actions[0].content
            import re
            match = re.search(r'<finish>(.*?)</finish>', finish_content, re.DOTALL)
            if match:
                finish_message = match.group(1).strip()
                user_messages.insert(-1, {
                    "role": "user",
                    "content": finish_message
                })
                assistant_messages.insert(-1, {
                    "role": "assistant",
                    "content": finish_message,
                    "refusal": None,
                    "reasoning": None
                })

        user_conv = {
            "conversation_id": f"{trajectory_id}_user",
            "messages": user_messages
        }
        
        assistant_conv = {
            "conversation_id": f"{trajectory_id}_assistant", 
            "messages": assistant_messages
        }
        
        return user_conv, assistant_conv
        
    except Exception as e:
        raise STDToOWLError(f"Failed to build parallel conversations: {e}")


def normalize_conversation_endings(user_conv: Dict, assistant_conv: Dict) -> Tuple[Dict, Dict]:
    """Normalize conversation endings to ensure proper role alternation and <CAMEL_TASK_DONE> completion.
    
    Ensures both conversations end properly with parallel structure. The conversations must mirror
    each other - the same instruction appears in both but with different role perspectives.
    
    Args:
        user_conv: User conversation dictionary (user agent perspective)
        assistant_conv: Assistant conversation dictionary (assistant agent perspective) 
        
    Returns:
        Tuple of (normalized_user_conv, normalized_assistant_conv)
    """
    # Work on copies to avoid mutating originals
    user_messages = user_conv["messages"].copy()
    assistant_messages = assistant_conv["messages"].copy()
    
    # Helper function to get last non-system message role
    def get_last_message_role(messages):
        for msg in reversed(messages):
            if msg["role"] != "system":
                return msg["role"]
        return None
    
    # 1. Handle assistant conversation completion first (it may be incomplete)
    assistant_last_role = get_last_message_role(assistant_messages)
    user_last_role = get_last_message_role(user_messages)
    
    def add_assistant_message(content, refusal=None, reasoning=None):
        assistant_messages.append({
            "role": "assistant",
            "content": content,
            "refusal": refusal,
            "reasoning": reasoning
        })
        user_messages.append({
            "role": "user",
            "content": content
        })
        assistant_last_role = "assistant"
        user_last_role = "user"

    def add_user_message(content):
        assistant_messages.append({
            "role": "user",
            "content": content,
            "refusal": None,
            "reasoning": None
        })
        user_messages.append({
            "role": "assistant",
            "content": content
        })
        assistant_last_role = "user"
        user_last_role = "assistant"
    
    # Basic error checks
    if assistant_last_role is None or user_last_role is None:
        raise STDToOWLError("One of the conversations has no non-system messages")
    elif assistant_last_role == "system" or user_last_role == "system":
        raise STDToOWLError("One of the conversations ends with a system message")

    # Make sure the assistant conversation ends with an assistant message.
    if assistant_last_role == "tool" and user_last_role == "user":
        raise STDToOWLError("Assistant ends with tool but user ends with user - cannot normalize")
    elif assistant_last_role == "tool":
        add_assistant_message("Solution: The action has been completed. Next request.")
    
    # Now check if we're ending on an assistant or user message
    if assistant_last_role == "user" and user_last_role == "assistant" and assistant_messages[-1]["content"] == "<CAMEL_TASK_DONE>":
        pass
    elif assistant_last_role == "user" and user_last_role == "assistant":
        # Ending on user message - add boilerplate assistant and then user <CAMEL_TASK_DONE>
        add_assistant_message("Solution: The task has been completed successfully. Next request.")
        add_user_message("<CAMEL_TASK_DONE>")

    elif assistant_last_role == "assistant" and user_last_role == "user":
        # Ending on assistant message - add user <CAMEL_TASK_DONE> directly
        add_user_message("<CAMEL_TASK_DONE>")
    else:
        raise STDToOWLError("Conversations are out of sync in roles - cannot normalize")
    
    # Return normalized conversations
    normalized_user = user_conv.copy()
    normalized_user["messages"] = user_messages
    
    normalized_assistant = assistant_conv.copy() 
    normalized_assistant["messages"] = assistant_messages
    
    return normalized_user, normalized_assistant


def process_trajectory_conversations(trajectory: Trajectory, use_templates: bool = True, use_llm_relevance: bool = False, max_obs_length: int = 5000, max_trajectory_tokens: int = 50000, skip_long: bool = False, use_endpoint_task_id: bool = True, endpoint_count: int = 2, short_task_threshold: int = 500) -> Tuple[Dict, Dict]:
    """Process trajectory using template-based or LLM pipeline.
    
    Args:
        trajectory: STD trajectory to convert
        use_templates: Whether to use template-based generation (default True)
        use_llm_relevance: Whether to use LLM for relevance checking
        max_obs_length: Maximum length for observation content
        max_trajectory_tokens: Skip trajectories exceeding this length
        skip_long: Skip long trajectories instead of processing with truncation
        
    Returns:
        Tuple of (user_conversation, assistant_conversation) dictionaries
        
    Raises:
        STDToOWLError: If conversion fails
    """
    # Validate input
    validate_trajectory(trajectory)
    
    # Check trajectory length
    is_within_limit, total_length = check_trajectory_length(trajectory, max_trajectory_tokens)
    if not is_within_limit:
        if skip_long:
            raise STDToOWLError(f"Trajectory {trajectory.id} exceeds length limit ({total_length} > {max_trajectory_tokens}), skipping.")
        else:
            logger.warning(f"Trajectory {trajectory.id} exceeds length limit ({total_length} > {max_trajectory_tokens}), processing with truncation")
    
    # Step 0: Identify main task
    if use_endpoint_task_id:
        main_task = identify_main_task_endpoints(trajectory, endpoint_count, short_task_threshold, max_obs_length)
    else:
        main_task = identify_main_task_llm(trajectory, max_obs_length)
    
    # Step 1: Group into alternating sets
    grouped_sets = group_events_alternating(trajectory.content)
    
    # Step 2: Extract initial context from first observation set and skip it
    initial_context = ""
    if grouped_sets and isinstance(grouped_sets[0], ObservationSet):
        first_obs_set = grouped_sets[0]
        # Format the observations as context with truncation
        context_lines = []
        for obs in first_obs_set.observations:
            content = truncate_content(str(getattr(obs, 'content', str(obs))), max_obs_length)
            context_lines.append(f"- {obs.class_}: {content}")
        initial_context = "\n".join(context_lines)
        grouped_sets = grouped_sets[1:]  # Skip for processing
    
    # Step 3-7: Process each action set
    processing_groups = []
    context_obs = None
    
    i = 0
    while i < len(grouped_sets):
        if isinstance(grouped_sets[i], ActionSet):
            action_set = grouped_sets[i]
            # Find the next observation set if it exists
            obs_set = None
            if i + 1 < len(grouped_sets) and isinstance(grouped_sets[i + 1], ObservationSet):
                obs_set = grouped_sets[i + 1]
            
            # Process this action set
            group = process_action_set(action_set, obs_set, main_task, context_obs, use_templates, use_llm_relevance, max_obs_length)
            processing_groups.append(group)
            
            # Update context for next iteration
            context_obs = obs_set
            i += 2  # Skip both action set and observation set
        else:
            i += 1  # Skip observation sets that don't follow action sets
    
    if not processing_groups:
        raise STDToOWLError("No action sets found to process")
    
    # Step 8: Build OWL conversations in parallel
    user_conv, assistant_conv = build_parallel_conversations(processing_groups, main_task, trajectory.id, initial_context)
    
    # Step 9: Normalize conversation endings
    user_conv, assistant_conv = normalize_conversation_endings(user_conv, assistant_conv)
    
    return user_conv, assistant_conv




def validate_owl_output(user_conv: Dict, assistant_conv: Dict) -> bool:
    """Validate OWL conversation format.
    
    Args:
        user_conv: User conversation dictionary
        assistant_conv: Assistant conversation dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    # Check required fields
    for conv, name in [(user_conv, "user"), (assistant_conv, "assistant")]:
        if "conversation_id" not in conv:
            raise ValidationError(f"{name} conversation missing conversation_id")
        if "messages" not in conv:
            raise ValidationError(f"{name} conversation missing messages")
        if not isinstance(conv["messages"], list):
            raise ValidationError(f"{name} conversation messages must be list")
        if len(conv["messages"]) == 0:
            raise ValidationError(f"{name} conversation has no messages")
    
    # Check message format
    for conv, name in [(user_conv, "user"), (assistant_conv, "assistant")]:
        for i, msg in enumerate(conv["messages"]):
            if "role" not in msg:
                raise ValidationError(f"{name} message {i} missing role")
            if "content" not in msg:
                raise ValidationError(f"{name} message {i} missing content")
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValidationError(f"{name} message {i} invalid role: {msg['role']}")
    
    # Check system message is first
    for conv, name in [(user_conv, "user"), (assistant_conv, "assistant")]:
        if conv["messages"][0]["role"] != "system":
            raise ValidationError(f"{name} conversation must start with system message")
    
    # Check user conversation ends with CAMEL_TASK_DONE
    last_user_msg = user_conv["messages"][-1]
    if "<CAMEL_TASK_DONE>" not in last_user_msg["content"]:
        logger.warning("User conversation doesn't end with <CAMEL_TASK_DONE>")
    
    return True


def process_trajectory(line: str, output_dir: Path, use_templates: bool = True, use_llm_relevance: bool = False, max_obs_length: int = 5000, max_trajectory_tokens: int = 100000, skip_long: bool = False, use_endpoint_task_id: bool = False, endpoint_count: int = 2, short_task_threshold: int = 500, current_idx: int = 0, total_count: int = 0) -> bool:
    """Process a single trajectory line.
    
    Args:
        line: JSON line containing STD trajectory
        output_dir: Directory to write output files
        use_templates: Whether to use template-based generation
        use_llm_relevance: Whether to use LLM for relevance checking
        max_obs_length: Maximum length for observation content
        max_trajectory_tokens: Skip trajectories exceeding this length
        skip_long: Skip long trajectories instead of processing with truncation
        
    Returns:
        True if processing succeeded
    """
    trajectory_id = "unknown"
    try:
        # Parse trajectory
        logger.debug(f"Parsing JSON line: {line[:100]}...")
        data = json.loads(line.strip())
        trajectory = Trajectory(**data)
        trajectory_id = trajectory.id

        # Create progress prefix if we have counting info
        progress_prefix = f"[{current_idx}/{total_count}] " if total_count > 0 else ""
        logger.info(f"{progress_prefix}Processing {trajectory_id}")
        logger.debug(f"Trajectory has {len(trajectory.content)} events")
        
        # Convert to OWL format using pipeline
        pipeline_type = "template-based" if use_templates else "LLM-based"
        logger.debug(f"Converting trajectory {trajectory_id} to OWL format using {pipeline_type} pipeline")
        user_conv, assistant_conv = process_trajectory_conversations(trajectory, use_templates, use_llm_relevance, max_obs_length, max_trajectory_tokens, skip_long, use_endpoint_task_id, endpoint_count, short_task_threshold)
        
        logger.debug(f"User conversation has {len(user_conv['messages'])} messages")
        logger.debug(f"Assistant conversation has {len(assistant_conv['messages'])} messages")
        
        # Validate output
        logger.debug(f"Validating OWL output for trajectory {trajectory_id}")
        validate_owl_output(user_conv, assistant_conv)
        
        # Write output files
        user_file = output_dir / f"{trajectory_id}_user.json"
        assistant_file = output_dir / f"{trajectory_id}_assistant.json"
        
        logger.debug(f"Writing output files for trajectory {trajectory_id}")
        with open(user_file, 'w', encoding='utf-8') as f:
            json.dump(user_conv, f, indent=2, ensure_ascii=False)
            
        with open(assistant_file, 'w', encoding='utf-8') as f:
            json.dump(assistant_conv, f, indent=2, ensure_ascii=False)

        logger.info(f"{progress_prefix}✓ {trajectory_id} converted successfully")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"{progress_prefix}✗ {trajectory_id}: Invalid JSON - {e}")
        logger.debug(f"Problematic line: {line}")
        return False
    except ValidationError as e:
        logger.error(f"{progress_prefix}✗ {trajectory_id}: {e}")
        logger.debug(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"{progress_prefix}✗ {trajectory_id}: Unexpected error - {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return False


def main():
    """Main CLI interface for STD to OWL conversion."""
    parser = argparse.ArgumentParser(
        description="Convert STD (Standardized) trajectories to OWL (dual-agent) format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert from stdin to output directory
  cat trajectories.jsonl | python std_to_owl.py --output_dir ./owl_output
  
  # Process specific file
  python std_to_owl.py --input_file data.jsonl --output_dir ./results
        """
    )
    
    parser.add_argument(
        "--input_file", 
        type=str,
        help="Input JSONL file (default: read from stdin)"
    )
    parser.add_argument(
        "--output_dir",
        type=str, 
        required=True,
        help="Output directory for OWL conversation files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-3.5-turbo",
        help="LLM model to use for instruction extraction (accessed via LiteLLM proxy)"
    )
    parser.add_argument(
        "--env_file",
        type=str,
        help="Custom .env file path for LiteLLM proxy credentials (default: searches scripts/.env and .env)"
    )
    parser.add_argument(
        "--disable_llm",
        action="store_true",
        help="Disable LLM instruction extraction, use fallback only"
    )
    parser.add_argument(
        "--use_templates",
        action="store_true",
        help="Use rule-based templates instead of LLM for instruction/response generation"
    )
    parser.add_argument(
        "--use_llm_relevance",
        action="store_true",
        help="Use LLM for relevance checking (default: assume all observations are relevant)"
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        default=5000,
        help="Maximum characters per observation before truncation (default: 5000)"
    )
    parser.add_argument(
        "--max_trajectory_tokens",
        type=int,
        default=100000,
        help="Skip trajectories exceeding this total character limit (default: 100000)"
    )
    parser.add_argument(
        "--skip_long",
        action="store_true",
        help="Skip long trajectories instead of processing with truncation"
    )
    parser.add_argument(
        "--use_endpoint_task_id",
        action="store_true",
        help="Use endpoint-based task identification (first/last events only) for efficiency"
    )
    parser.add_argument(
        "--endpoint_count",
        type=int,
        default=2,
        help="Number of events to take from start and end for endpoint task identification (default: 2)"
    )
    parser.add_argument(
        "--short_task_threshold",
        type=int,
        default=1000,
        help="Max characters in first observation to treat as direct task for short trajectories (default: 1000)"
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        help="Ratio of trajectories to process (0.0-1.0, randomly sampled)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for reproducible sampling"
    )
    
    args = parser.parse_args()
    
    # Validate output directory first
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Test write permissions
        test_file = output_dir / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        print(f"Cannot write to output directory {output_dir}: {e}")
        sys.exit(1)
    
    # Configure logging with file handler
    log_file = output_dir / "std_to_owl.log"
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up new handlers
    handlers = [
        logging.StreamHandler(sys.stdout),  # Console output
        logging.FileHandler(log_file, mode='w', encoding='utf-8')  # File output
    ]
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)

    # Suppress LiteLLM's verbose logging
    if HAS_LITELLM:
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)  # Also suppress httpx used by LiteLLM

    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Command line args: {vars(args)}")

    # Reset cost tracking for this run
    global cost_stats, truncation_stats
    cost_stats = {
        'total_cost': 0.0,
        'average_cost_per_trajectory': 0.0,
        'llm_calls': 0,
    }

    # Initialize LLM client if not disabled
    if not args.disable_llm:
        if not initialize_llm_client(model=args.llm_model, env_file=args.env_file):
            logger.error("Failed to initialize LiteLLM. LLM features will be disabled.")
            args.disable_llm = True
    
    # Determine input source and format
    trajectories = []
    
    if args.input_file:
        if not Path(args.input_file).exists():
            logger.error(f"Input file not found: {args.input_file}")
            sys.exit(1)
            
        # Try to detect input format
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if content.startswith('['):
            # JSON array format
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    trajectories = [json.dumps(item) for item in data]
                    logger.info(f"Detected JSON array format with {len(trajectories)} trajectories")
                else:
                    trajectories = [content]
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON array format: {e}")
                sys.exit(1)
        else:
            # JSONL format - each line is a JSON object
            trajectories = [line.strip() for line in content.split('\n') if line.strip()]
            logger.info(f"Detected JSONL format with {len(trajectories)} trajectories")
    else:
        # Read from stdin (assume JSONL format)
        trajectories = [line.strip() for line in sys.stdin if line.strip()]
        logger.info(f"Read {len(trajectories)} trajectories from stdin")

    # Calculate target sample size and apply random shuffling if sampling is requested
    target_sample_size = None
    if args.sample_ratio is not None:
        target_sample_size = int(len(trajectories) * args.sample_ratio)
        if args.random_seed is not None:
            random.seed(args.random_seed)
            logger.info(f"Set random seed to {args.random_seed}")

        random.shuffle(trajectories)
        logger.info(f"Shuffled {len(trajectories)} trajectories for random sampling (ratio: {args.sample_ratio}, target: {target_sample_size} successful)")

    # Process trajectories
    processed = 0
    successful = 0
    failed = 0
    total_trajectories = len(trajectories)

    try:
        for i, trajectory_json in enumerate(trajectories):
            if not trajectory_json:
                continue

            # Check if we've successfully processed enough trajectories
            if target_sample_size and successful >= target_sample_size:
                logger.info(f"Reached target sample size: {target_sample_size} successful trajectories")
                logger.info(f"(Processed {processed} total, {failed} failed)")
                break

            try:
                # Pass trajectory count info to process_trajectory
                result = process_trajectory(
                    trajectory_json, output_dir, args.use_templates, args.use_llm_relevance,
                    args.max_obs_length, args.max_trajectory_tokens, args.skip_long,
                    args.use_endpoint_task_id, args.endpoint_count, args.short_task_threshold,
                    processed + 1, total_trajectories  # Add progress info
                )
                if result:
                    successful += 1
                else:
                    failed += 1

                processed += 1

                # Check spending limit after each trajectory
                if cost_stats['total_cost'] > LLM_SPENDING_LIMIT:
                    logger.error(f"LLM spending limit exceeded: ${cost_stats['total_cost']:.6f} > ${LLM_SPENDING_LIMIT:.2f}")
                    logger.info(f"Stopping processing after {processed} trajectories due to spending limit")
                    break

                if processed % 100 == 0:
                    logger.info(f"Progress: {processed}/{total_trajectories} trajectories ({successful} successful, {failed} failed)")

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error on trajectory {i}: {e}")
                failed += 1
                processed += 1
                
    finally:
        pass

    # Check if we couldn't reach the target sample size
    if target_sample_size and successful < target_sample_size:
        logger.warning(f"Could not reach target sample size!")
        logger.info(f"  Target sample ratio: {args.sample_ratio}")
        logger.info(f"  Target sample size: {target_sample_size}")
        logger.info(f"  Successfully processed: {successful}")
        logger.info(f"  Failed/invalid trajectories skipped: {failed}")
        logger.info(f"  Total trajectories attempted: {processed}")
        logger.info(f"  Total trajectories available: {len(trajectories)}")
        logger.info(f"  Trajectories not attempted: {len(trajectories) - processed}")
        logger.info(f"  Still needed: {target_sample_size - successful}")

        if len(trajectories) - processed > 0:
            logger.info(f"Note: There were {len(trajectories) - processed} trajectories not attempted. "
                       f"The high failure rate ({failed}/{processed} = {failed/processed*100:.1f}%) "
                       f"prevented reaching the target.")
        else:
            logger.info(f"All available trajectories were attempted. "
                       f"Failure rate: {failed}/{processed} = {failed/processed*100:.1f}%")

    # Final statistics
    logger.info(f"Conversion complete: {processed} total, {successful} successful, {failed} failed")

    if target_sample_size:
        if successful >= target_sample_size:
            logger.info(f"Successfully reached target sample size of {target_sample_size} ({args.sample_ratio:.1%} of {len(trajectories)})")
        else:
            logger.info(f"Only processed {successful}/{target_sample_size} target samples ({args.sample_ratio:.1%} of {len(trajectories)})")

    logger.info(f"Success rate: {successful/processed*100:.1f}%" if processed > 0 else "No trajectories processed")
    
    # Truncation statistics
    if truncation_stats['total_truncations'] > 0:
        logger.info(f"Truncation statistics:")
        logger.info(f"  Total truncations: {truncation_stats['total_truncations']}")
        logger.info(f"  Total characters truncated: {truncation_stats['truncated_chars']:,}")
        logger.info(f"  Longest original content: {truncation_stats['longest_original']:,} characters")
        logger.info(f"  Average truncation per event: {truncation_stats['truncated_chars'] / truncation_stats['total_truncations']:.0f} characters")
    else:
        logger.info("No content was truncated during processing")

    # Cost statistics
    if cost_stats['llm_calls'] > 0:
        avg_cost_per_call = cost_stats['total_cost'] / cost_stats['llm_calls']
        logger.info(f"LLM cost statistics:")
        logger.info(f"  Total spending: ${cost_stats['total_cost']:.6f}")
        logger.info(f"  Total API calls: {cost_stats['llm_calls']}")
        logger.info(f"  Average cost per call: ${avg_cost_per_call:.6f}")
        if processed > 0:
            cost_stats['average_cost_per_trajectory'] = cost_stats['total_cost'] / processed
            logger.info(f"  Average cost per trajectory: ${cost_stats['average_cost_per_trajectory']:.6f}")
    else:
        logger.info("No LLM costs tracked (LLM disabled or no calls made)")

    logger.info(f"Log file saved to: {log_file}")
    
    if failed > 0:
        logger.error(f"Conversion completed with {failed} failures. Check log file for details.")
        sys.exit(1)
    else:
        logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()

# Create a .env in root, with the following:
# LITELLM_PROXY_API_KEY="API_KEY"
# LITELLM_PROXY_API_BASE="https://cmu.litellm.ai"

# Then run something like:
# python scripts/std_to_owl.py --input_file datasets/orca_agentinstruct/sample_std.json --output_dir ./test_owl_output --llm_model "litellm_proxy/gpt-4o-mini" --use_templates