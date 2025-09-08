"""
Convert STD (Standardized) format to OWL (dual-agent conversation) format.

This script transforms single-agent trajectories into role-playing conversations
between a user (instruction-giver) and assistant (instruction-follower).
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logging.warning("OpenAI library not found.")

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

# Global LLM client and model
llm_client = None
llm_model = ""

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


def initialize_llm_client(api_key: str = None, base_url: str = None, model: str = llm_model) -> Optional[OpenAI]:
    """Initialize OpenAI-compatible LLM client.
    
    Args:
        api_key: API key for the service
        base_url: Base URL for OpenAI-compatible endpoint
        model: Model name to use
        
    Returns:
        Initialized OpenAI client or None if not available
    """
    global llm_client, llm_model
    
    # Store the model name globally
    llm_model = model
    
    if not HAS_OPENAI:
        logger.warning("OpenAI library not available for LLM integration")
        return None
        
    try:
        # Default to OpenRouter if no base_url provided
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"
            
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            
        if not api_key:
            logger.warning("No API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable")
            return None
        
        llm_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info(f"Initialized LLM client with base_url: {base_url}, model: {model}")
        return llm_client
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return None


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


def identify_main_task_llm(trajectory: Trajectory) -> str:
    """Identify the main task from trajectory using LLM.
    
    Args:
        trajectory: The trajectory to analyze
        
    Returns:
        Main task description string
        
    Raises:
        LLMExtractionError: If task identification fails
    """
    global llm_client, llm_model, _last_api_call_ts
    
    try:
        config = load_prompt_config("0_identify_main_task")
        
        # Format interaction sequence for the prompt
        interaction_lines = []
        for event in trajectory.content:
            if hasattr(event, 'content'):
                content = str(event.content)
                interaction_lines.append(f"- {event.class_}: {content}")
            else:
                interaction_lines.append(f"- {event.class_}: {str(event)}")
        
        interaction_sequence = "\\n".join(interaction_lines)
        
        if llm_client:
            try:
                model_to_use = llm_model or "gpt-3.5-turbo"
                logger.info(f"Step 0: Starting main task identification with model {model_to_use}")
                
                # Build messages with template variables
                messages = []
                for msg in config["messages"]:
                    content = msg["content"].format(interaction_sequence=interaction_sequence)
                    messages.append({"role": msg["role"], "content": content})
                
                logger.debug(f"Step 0: Built {len(messages)} messages, total chars: {sum(len(m['content']) for m in messages)}")
                
                # Rate limiting
                now = time.time()
                if now - _last_api_call_ts < 1.0:
                    time.sleep(1.0 - (now - _last_api_call_ts))
                _last_api_call_ts = now
                
                params = config.get("parameters", {})
                logger.info(f"Step 0: Making LLM request with max_tokens={params.get('max_tokens', 500)}")
                
                response = llm_client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=params.get("max_tokens", 500),
                    temperature=params.get("temperature", 0)
                )
                
                logger.info(f"Step 0: LLM request completed successfully")
                
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
            raise LLMExtractionError("No LLM client available for task identification")
            
    except Exception as e:
        raise LLMExtractionError(f"Failed to identify main task: {e}")


def find_task_type_llm(action_set: ActionSet, obs_set: Optional[ObservationSet]) -> str:
    """Find task type using LLM.
    
    Args:
        action_set: The action set to analyze
        obs_set: Following observation set (if any)
        
    Returns:
        Task type string (one of: code_execution, information_retrieval, other_tool, task_completion)
        
    Raises:
        LLMExtractionError: If task type identification fails
    """
    global llm_client, llm_model, _last_api_call_ts
    
    try:
        config = load_prompt_config("1_find_task_type")
        
        # Format actions for the prompt
        action_lines = []
        for action in action_set.actions:
            if hasattr(action, 'content'):
                content = str(action.content)
                action_lines.append(f"- {action.class_}: {content}")
            elif hasattr(action, 'function'):
                function_name = getattr(action, 'function', 'unknown')
                kwargs = getattr(action, 'kwargs', {})
                action_lines.append(f"- {action.class_}: {function_name}({kwargs})")
            else:
                action_lines.append(f"- {action.class_}: {str(action)}")
        
        actions_text = "\\n".join(action_lines)
        
        if llm_client:
            try:
                model_to_use = llm_model or "gpt-3.5-turbo"
                logger.info(f"Step 1: Starting task type identification with model {model_to_use}")
                
                messages = []
                for msg in config["messages"]:
                    content = msg["content"].format(actions=actions_text)
                    messages.append({"role": msg["role"], "content": content})
                
                logger.debug(f"Step 1: Built {len(messages)} messages, actions text length: {len(actions_text)}")
                
                # Rate limiting
                now = time.time()
                if now - _last_api_call_ts < 1.0:
                    time.sleep(1.0 - (now - _last_api_call_ts))
                _last_api_call_ts = now
                
                params = config.get("parameters", {})
                logger.info(f"Step 1: Making LLM request with max_tokens={params.get('max_tokens', 100)}")
                
                response = llm_client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=params.get("max_tokens", 100),
                    temperature=params.get("temperature", 0)
                )
                
                logger.info(f"Step 1: LLM request completed successfully")
                
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
            raise LLMExtractionError("No LLM client available for task type identification")
            
    except Exception as e:
        raise LLMExtractionError(f"Failed to identify task type: {e}")


def check_relevance_llm(action_set: ActionSet, obs_set: Optional[ObservationSet], context_obs: Optional[ObservationSet] = None) -> str:
    """Check if observations are caused by actions using LLM.
    
    Args:
        action_set: The action set
        obs_set: The observation set to check
        context_obs: Previous observation set for context
        
    Returns:
        "YES" if causal, "NO" if not causal
        
    Raises:
        LLMExtractionError: If relevance check fails
    """
    global llm_client, llm_model, _last_api_call_ts
    
    if obs_set is None:
        return "NO"
    
    try:
        config = load_prompt_config("2_check_relevance")
        
        # Format context
        context_text = ""
        if context_obs:
            context_lines = []
            for obs in context_obs.observations:
                content = getattr(obs, 'content', str(obs))
                context_lines.append(f"- {obs.class_}: {content}")
            context_text = "\\n".join(context_lines)
        
        # Format actions
        action_lines = []
        for action in action_set.actions:
            if hasattr(action, 'content'):
                action_lines.append(f"- {action.class_}: {action.content}")
            elif hasattr(action, 'function'):
                function_name = getattr(action, 'function', 'unknown')
                kwargs = getattr(action, 'kwargs', {})
                action_lines.append(f"- {action.class_}: {function_name}({kwargs})")
            else:
                action_lines.append(f"- {action.class_}: {str(action)}")
        actions_text = "\\n".join(action_lines)
        
        # Format observations
        obs_lines = []
        for obs in obs_set.observations:
            content = str(getattr(obs, 'content', str(obs)))
            obs_lines.append(f"- {obs.class_}: {content}")
        observations_text = "\\n".join(obs_lines)
        
        if llm_client:
            try:
                model_to_use = llm_model or "gpt-3.5-turbo"
                logger.info(f"Step 2: Starting relevance check with model {model_to_use}")
                
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
                if now - _last_api_call_ts < 1.0:
                    time.sleep(1.0 - (now - _last_api_call_ts))
                _last_api_call_ts = now
                
                params = config.get("parameters", {})
                logger.info(f"Step 2: Making LLM request with max_tokens={params.get('max_tokens', 50)}")
                
                response = llm_client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=params.get("max_tokens", 50),
                    temperature=params.get("temperature", 0)
                )
                
                logger.info(f"Step 2: LLM request completed successfully")
                
                result = response.choices[0].message.content.strip().upper()
                
                if result not in ["YES", "NO"]:
                    raise LLMExtractionError(f"Invalid relevance response: {result}")
                
                return result
                
            except Exception as e:
                raise LLMExtractionError(f"LLM relevance check failed: {e}")
        else:
            raise LLMExtractionError("No LLM client available for relevance check")
            
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


def generate_instruction_llm(action_set: ActionSet, related_obs: Optional[ObservationSet]) -> str:
    """Generate user instruction using LLM.
    
    Args:
        action_set: The action set
        related_obs: Related observations (if causal)
        
    Returns:
        Formatted instruction string
        
    Raises:
        LLMExtractionError: If instruction generation fails
    """
    global llm_client, llm_model, _last_api_call_ts
    
    try:
        config = load_prompt_config("3_gen_instruction")
        
        # Format actions
        action_lines = []
        for action in action_set.actions:
            if hasattr(action, 'content'):
                action_lines.append(f"- {action.class_}: {action.content}")
            elif hasattr(action, 'function'):
                function_name = getattr(action, 'function', 'unknown')
                kwargs = getattr(action, 'kwargs', {})
                action_lines.append(f"- {action.class_}: {function_name}({kwargs})")
            else:
                action_lines.append(f"- {action.class_}: {str(action)}")
        actions_text = "\\n".join(action_lines)
        
        # Format observations if present
        observations_text = ""
        if related_obs:
            obs_lines = []
            for obs in related_obs.observations:
                content = str(getattr(obs, 'content', str(obs)))
                obs_lines.append(f"- {obs.class_}: {content}")
            observations_text = "\\n".join(obs_lines)
        
        if llm_client:
            try:
                model_to_use = llm_model or "gpt-3.5-turbo"
                logger.info(f"Step 3: Starting instruction generation with model {model_to_use}")
                
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
                if now - _last_api_call_ts < 1.0:
                    time.sleep(1.0 - (now - _last_api_call_ts))
                _last_api_call_ts = now
                
                params = config.get("parameters", {})
                logger.info(f"Step 3: Making LLM request with max_tokens={params.get('max_tokens', 500)}")
                
                response = llm_client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=params.get("max_tokens", 500),
                    temperature=params.get("temperature", 0)
                )
                
                logger.info(f"Step 3: LLM request completed successfully")
                
                result = response.choices[0].message.content.strip()
                
                # Validate format
                if not ("Instruction:" in result and "Input:" in result):
                    raise LLMExtractionError(f"Invalid instruction format: {result}")
                
                return result
                
            except Exception as e:
                raise LLMExtractionError(f"LLM instruction generation failed: {e}")
        else:
            raise LLMExtractionError("No LLM client available for instruction generation")
            
    except Exception as e:
        raise LLMExtractionError(f"Failed to generate instruction: {e}")


def generate_response_template(action_set: ActionSet, related_obs: Optional[ObservationSet] = None) -> str:
    """Generate assistant response using rule-based templates.
    
    Args:
        action_set: The action set
        related_obs: Related observations (if causal)
        
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
                    content = getattr(obs, 'content', str(obs)).rstrip('.')  # Remove trailing period
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
                    content = getattr(obs, 'content', str(obs)).rstrip('.')  # Remove trailing period
                    obs_content.append(content)
                obs_text = "\n".join(obs_content)
                return f"{function_name} was executed and resulted in the following:\n{obs_text}"
            else:
                return f"{function_name} was executed"
        
        else:
            raise STDToOWLError(f"Unknown action type: {type(action)}")
            
    except Exception as e:
        raise STDToOWLError(f"Failed to generate template response: {e}")


def generate_response_llm(action_set: ActionSet, related_obs: Optional[ObservationSet]) -> str:
    """Generate assistant response using LLM.
    
    Args:
        action_set: The action set
        related_obs: Related observations (if causal)
        
    Returns:
        Assistant response string
        
    Raises:
        LLMExtractionError: If response generation fails
    """
    global llm_client, llm_model, _last_api_call_ts
    
    try:
        config = load_prompt_config("5_gen_response")
        
        # Format actions
        action_lines = []
        for action in action_set.actions:
            if hasattr(action, 'content'):
                action_lines.append(f"- {action.class_}: {action.content}")
            elif hasattr(action, 'function'):
                function_name = getattr(action, 'function', 'unknown')
                kwargs = getattr(action, 'kwargs', {})
                action_lines.append(f"- {action.class_}: {function_name}({kwargs})")
            else:
                action_lines.append(f"- {action.class_}: {str(action)}")
        actions_text = "\\n".join(action_lines)
        
        # Format observations if present
        observations_text = ""
        if related_obs:
            obs_lines = []
            for obs in related_obs.observations:
                content = str(getattr(obs, 'content', str(obs)))
                obs_lines.append(f"- {obs.class_}: {content}")
            observations_text = "\\n".join(obs_lines)
        
        if llm_client:
            try:
                model_to_use = llm_model or "gpt-3.5-turbo"
                logger.info(f"Step 5: Starting response generation with model {model_to_use}")
                
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
                if now - _last_api_call_ts < 1.0:
                    time.sleep(1.0 - (now - _last_api_call_ts))
                _last_api_call_ts = now
                
                params = config.get("parameters", {})
                logger.info(f"Step 5: Making LLM request with max_tokens={params.get('max_tokens', 1000)}")
                
                response = llm_client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=params.get("max_tokens", 1000),
                    temperature=params.get("temperature", 0)
                )
                
                logger.info(f"Step 5: LLM request completed successfully")
                
                result = response.choices[0].message.content.strip()
                
                # Validate markdown format
                if not (result.startswith("```") and result.endswith("```")):
                    raise LLMExtractionError(f"Invalid response format (should be markdown block): {result}")
                
                return result
                
            except Exception as e:
                raise LLMExtractionError(f"LLM response generation failed: {e}")
        else:
            raise LLMExtractionError("No LLM client available for response generation")
            
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
    
    logger.info(f"Grouped {len(events)} events into {len(groups)} alternating sets (skipped initial acknowledgments)")
    return groups


_last_api_call_ts = 0.0

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
                       use_templates: bool = True, use_llm_relevance: bool = False) -> ProcessingGroup:
    """Process an action set through template-based or LLM pipeline.
    
    Args:
        action_set: The action set to process
        obs_set: The observation set following the action set
        main_task: The main task description
        context_obs: Previous observation set for context
        use_templates: Whether to use template-based generation (default True)
        
    Returns:
        ProcessingGroup with all derived information
        
    Raises:
        STDToOWLError: If any step fails
    """
    if use_templates:
        # Template-based pipeline (new approach)
        # Step 1: Check causality 
        if use_llm_relevance and llm_client:
            relevance = check_relevance_llm(action_set, obs_set, context_obs)
        else:
            relevance = "YES"  # Default to assuming all observations are relevant
        related_obs = obs_set if relevance == "YES" else None
        
        # Step 2: Generate instruction using templates
        instruction = generate_instruction_template(action_set, related_obs)
        
        # Step 3: Convert to tool calls
        task_type = "template_generated"  # Simplified since we're not using LLM classification
        tool_calls = convert_action_set_to_tool_calls(action_set, task_type)
        
        # Step 4: Generate assistant response using templates
        assistant_response = generate_response_template(action_set, related_obs)
        
    else:
        # Original LLM-based pipeline
        # Step 1: Find task type
        task_type = find_task_type_llm(action_set, obs_set)
        
        # Step 2: Check causality 
        relevance = check_relevance_llm(action_set, obs_set, context_obs)
        related_obs = obs_set if relevance == "YES" else None
        
        # Step 3: Generate instruction
        instruction = generate_instruction_llm(action_set, related_obs)
        
        # Step 4: Convert to tool calls
        tool_calls = convert_action_set_to_tool_calls(action_set, task_type)
        
        # Step 5: Generate assistant response
        assistant_response = generate_response_llm(action_set, related_obs)
    
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


def process_trajectory_conversations(trajectory: Trajectory, use_templates: bool = True, use_llm_relevance: bool = False) -> Tuple[Dict, Dict]:
    """Process trajectory using template-based or LLM pipeline.
    
    Args:
        trajectory: STD trajectory to convert
        use_templates: Whether to use template-based generation (default True)
        
    Returns:
        Tuple of (user_conversation, assistant_conversation) dictionaries
        
    Raises:
        STDToOWLError: If conversion fails
    """
    # Validate input
    validate_trajectory(trajectory)
    
    # Step 0: Identify main task
    main_task = identify_main_task_llm(trajectory)
    
    # Step 1: Group into alternating sets
    grouped_sets = group_events_alternating(trajectory.content)
    
    # Step 2: Extract initial context from first observation set and skip it
    initial_context = ""
    if grouped_sets and isinstance(grouped_sets[0], ObservationSet):
        first_obs_set = grouped_sets[0]
        # Format the observations as context
        context_lines = []
        for obs in first_obs_set.observations:
            content = getattr(obs, 'content', str(obs))
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
            group = process_action_set(action_set, obs_set, main_task, context_obs, use_templates, use_llm_relevance)
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


def process_trajectory(line: str, output_dir: Path, use_templates: bool = True, use_llm_relevance: bool = False) -> bool:
    """Process a single trajectory line.
    
    Args:
        line: JSON line containing STD trajectory
        output_dir: Directory to write output files
        
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
        
        logger.info(f"Processing trajectory {trajectory_id}")
        logger.debug(f"Trajectory has {len(trajectory.content)} events")
        
        # Convert to OWL format using pipeline
        pipeline_type = "template-based" if use_templates else "LLM-based"
        logger.debug(f"Converting trajectory {trajectory_id} to OWL format using {pipeline_type} pipeline")
        user_conv, assistant_conv = process_trajectory_conversations(trajectory, use_templates, use_llm_relevance)
        
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
            
        logger.info(f"Successfully wrote {user_file.name} and {assistant_file.name}")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input line for trajectory {trajectory_id}: {e}")
        logger.debug(f"Problematic line: {line}")
        return False
    except ValidationError as e:
        logger.error(f"Validation error for trajectory {trajectory_id}: {e}")
        logger.debug(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing trajectory {trajectory_id}: {e}")
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
        "--max_trajectories",
        type=int,
        help="Maximum number of trajectories to process (for testing)"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="qwen/qwen-2.5-72b-instruct:free",
        help="LLM model to use for instruction extraction"
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        help="Base URL for OpenAI-compatible LLM endpoint (default: OpenRouter)"
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        help="API key for LLM service (default: from environment)"
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
    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Command line args: {vars(args)}")
    
    # Initialize LLM client if not disabled
    if not args.disable_llm:
        initialize_llm_client(
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model
        )
    
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
    
    # Process trajectories
    processed = 0
    successful = 0
    failed = 0
    
    try:
        for i, trajectory_json in enumerate(trajectories):
            if not trajectory_json:
                continue
                
            if args.max_trajectories and processed >= args.max_trajectories:
                logger.info(f"Reached maximum trajectories limit: {args.max_trajectories}")
                break
                
            try:
                if process_trajectory(trajectory_json, output_dir, args.use_templates, args.use_llm_relevance):
                    successful += 1
                else:
                    failed += 1
                    
                processed += 1
                
                if processed % 100 == 0:
                    logger.info(f"Processed {processed} trajectories ({successful} successful, {failed} failed)")
                    
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error on trajectory {i}: {e}")
                failed += 1
                processed += 1
                
    finally:
        pass
    
    # Final statistics
    logger.info(f"Conversion complete: {processed} total, {successful} successful, {failed} failed")
    logger.info(f"Success rate: {successful/processed*100:.1f}%" if processed > 0 else "No trajectories processed")
    logger.info(f"Log file saved to: {log_file}")
    
    if failed > 0:
        logger.error(f"Conversion completed with {failed} failures. Check log file for details.")
        sys.exit(1)
    else:
        logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()

# Experimental personal key below, please replace.

# python scripts/std_to_owl.py --input_file scripts/owl_example/test.json --output_dir ./test_owl_output --llm_api_key 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 --llm_model "qwen/qwen-2.5-72b-instruct" --use_templates

# python scripts/std_to_owl.py --input_file datasets/SWE-smith_5kTrajectories/sample_std.json --output_dir ./swesmith_sample --llm_api_key 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 --llm_model "qwen/qwen-2.5-72b-instruct" --use_templates