"""
Utility script to apply context condensation to OpenHands SFT trajectories.

This script:
1. Loads sample_sft_openhands.json trajectories
2. Initializes an OpenHands context condenser
3. Applies condensation at appropriate timing to trajectories
4. Splits trajectories when condensation occurs (since prefix changes)
5. Outputs condensed trajectories in the same format

Input: sample_sft_openhands.json trajectories (N total)
Output: sample_sft_openhands_condensed.json trajectories (N*M where M is average condensations + 1)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import SecretStr

# Add scripts directory to path for mock_condenser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openhands.sdk import LLM
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context.condenser.base import CondenserBase
from openhands.sdk.context.view import View
from openhands.sdk.event import Condensation
from openhands.sdk.event.llm_convertible import MessageEvent
from openhands.sdk.llm import Message, TextContent
from mock_condenser import MockCondenser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_sft_conversation_to_events(
    conversations: list[dict[str, str]], trajectory_id: str
) -> list[MessageEvent]:
    """Convert SFT conversation format to MessageEvent objects.
    
    Args:
        conversations: List of conversation turns with 'from' and 'value' keys
        trajectory_id: Base ID for the trajectory
        
    Returns:
        List of MessageEvent objects
    """
    events = []
    for idx, conv in enumerate(conversations):
        role = "user" if conv["from"] == "human" else "assistant"
        content = conv["value"]
        
        message = Message(role=role, content=[TextContent(text=content)])
        
        # Determine source based on role
        source = "user" if role == "user" else "agent"
        
        event = MessageEvent(
            source=source,
            llm_message=message,
        )
        events.append(event)
        
    return events


def events_to_sft_conversations(events: list[MessageEvent]) -> list[dict[str, str]]:
    """Convert MessageEvent objects back to SFT conversation format.
    
    Args:
        events: List of MessageEvent objects
        
    Returns:
        List of conversation dictionaries with 'from' and 'value' keys
    """
    conversations = []
    for event in events:
        message = event.to_llm_message()
        role_mapping = {"user": "human", "assistant": "gpt"}
        from_role = role_mapping.get(message.role, message.role)
        
        # Extract text content
        text_parts = []
        for content in message.content:
            if isinstance(content, TextContent):
                text_parts.append(content.text)
        
        value = "".join(text_parts)
        conversations.append({"from": from_role, "value": value})
        
    return conversations


def condense_trajectory(
    trajectory: dict[str, Any],
    condenser: CondenserBase,
    trajectory_idx: int,
) -> list[dict[str, Any]]:
    """Apply condensation to a single trajectory and split when condensation occurs.
    
    Args:
        trajectory: Original trajectory dict with id, system, and conversations
        condenser: Initialized condenser instance
        trajectory_idx: Index of the trajectory for logging
        
    Returns:
        List of trajectory dicts (split by condensation points)
    """
    trajectory_id = trajectory["id"]
    system_prompt = trajectory["system"]
    conversations = trajectory["conversations"]
    
    logger.info(
        f"Processing trajectory {trajectory_idx} (id={trajectory_id}) "
        f"with {len(conversations)} conversation turns"
    )
    
    # Convert conversations to events
    events = parse_sft_conversation_to_events(conversations, trajectory_id)
    
    # Track all events as we process them
    all_events = []
    split_trajectories = []
    current_segment_start = 0
    condensation_count = 0
    
    # Process events iteratively, checking for condensation after each addition
    for event_idx, event in enumerate(events):
        all_events.append(event)
        
        # Create a view from current events
        view = View.from_events(all_events)
        
        # Try to condense
        result = condenser.condense(view)
        
        if isinstance(result, Condensation):
            condensation_count += 1
            logger.info(
                f"  Condensation {condensation_count} triggered after event {event_idx + 1}/{len(events)}"
            )
            logger.info(
                f"    Forgetting {len(result.forgotten_event_ids)} events"
            )
            if result.summary:
                logger.info(
                    f"    Summary (first 100 chars): {result.summary[:100]}..."
                )
            
            # Add the condensation event to all_events
            all_events.append(result)
            
            # Create a new trajectory segment up to this point (before condensation)
            segment_events = all_events[current_segment_start:event_idx + 1]
            segment_conversations = events_to_sft_conversations(
                [e for e in segment_events if isinstance(e, MessageEvent)]
            )
            
            if segment_conversations:
                segment_id = f"{trajectory_id}_seg{len(split_trajectories)}"
                split_trajectories.append(
                    {
                        "id": segment_id,
                        "system": system_prompt,
                        "conversations": segment_conversations,
                    }
                )
                logger.info(
                    f"    Created segment {segment_id} with {len(segment_conversations)} conversations"
                )
            
            # Get the new view after condensation
            view_after = View.from_events(all_events)
            
            # Start a new segment from the condensed view
            # The system prompt should now include the summary
            current_segment_start = len(all_events)
    
    # Add the final segment (or the entire trajectory if no condensation occurred)
    if condensation_count == 0:
        # No condensation occurred, return the original trajectory
        logger.info(
            f"  No condensation occurred for trajectory {trajectory_id}"
        )
        split_trajectories.append(trajectory)
    else:
        # Add remaining events as final segment
        segment_events = all_events[current_segment_start:]
        segment_message_events = [
            e for e in segment_events if isinstance(e, MessageEvent)
        ]
        
        if segment_message_events:
            segment_conversations = events_to_sft_conversations(segment_message_events)
            segment_id = f"{trajectory_id}_seg{len(split_trajectories)}"
            
            # Get the final view to include any summary
            final_view = View.from_events(all_events)
            
            split_trajectories.append(
                {
                    "id": segment_id,
                    "system": system_prompt,
                    "conversations": segment_conversations,
                }
            )
            logger.info(
                f"    Created final segment {segment_id} with {len(segment_conversations)} conversations"
            )
    
    logger.info(
        f"  Completed trajectory {trajectory_id}: "
        f"{condensation_count} condensations, "
        f"{len(split_trajectories)} segments"
    )
    
    return split_trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Apply context condensation to OpenHands SFT trajectories"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input sample_sft_openhands.json file",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output condensed trajectories file",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=10,
        help="Maximum number of events before condensation (default: 10)",
    )
    parser.add_argument(
        "--keep-first",
        type=int,
        default=2,
        help="Number of initial events to always keep (default: 2)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model to use (default: from LLM_MODEL env var)",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="LLM base URL (default: from LLM_BASE_URL env var)",
    )
    parser.add_argument(
        "--use-mock-condenser",
        action="store_true",
        help="Use mock condenser instead of LLM-based condenser (for testing)",
    )
    
    args = parser.parse_args()
    
    # Initialize condenser based on type
    if args.use_mock_condenser:
        logger.info(
            f"Initializing mock condenser with max_size={args.max_size}, "
            f"keep_first={args.keep_first}"
        )
        condenser = MockCondenser(max_size=args.max_size, keep_first=args.keep_first)
    else:
        # Check for required environment variables
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            logger.error("LLM_API_KEY environment variable is not set")
            return 1
        
        model = args.llm_model or os.getenv("LLM_MODEL", "anthropic/claude-3-5-sonnet-20241022")
        base_url = args.llm_base_url or os.getenv("LLM_BASE_URL")
        
        logger.info(f"Initializing LLM with model: {model}")
        if base_url:
            logger.info(f"Using base URL: {base_url}")
        
        # Initialize LLM
        llm = LLM(
            usage_id="condenser",
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
        )
        
        # Initialize condenser
        logger.info(
            f"Initializing LLM condenser with max_size={args.max_size}, "
            f"keep_first={args.keep_first}"
        )
        condenser = LLMSummarizingCondenser(
            llm=llm, max_size=args.max_size, keep_first=args.keep_first
        )
    
    # Load input trajectories
    logger.info(f"Loading trajectories from {args.input_file}")
    with open(args.input_file, "r") as f:
        trajectories = json.load(f)
    
    logger.info(f"Loaded {len(trajectories)} trajectories")
    
    # Process each trajectory
    all_output_trajectories = []
    for idx, trajectory in enumerate(trajectories):
        try:
            split_trajectories = condense_trajectory(trajectory, condenser, idx)
            all_output_trajectories.extend(split_trajectories)
        except Exception as e:
            logger.error(
                f"Error processing trajectory {idx} (id={trajectory.get('id', 'unknown')}): {e}",
                exc_info=True,
            )
            # Include original trajectory on error
            all_output_trajectories.append(trajectory)
    
    # Write output
    logger.info(
        f"Writing {len(all_output_trajectories)} trajectories to {args.output_file}"
    )
    with open(args.output_file, "w") as f:
        json.dump(all_output_trajectories, f, indent=2)
    
    logger.info(
        f"Condensation complete: {len(trajectories)} input trajectories -> "
        f"{len(all_output_trajectories)} output trajectories"
    )
    logger.info(f"Average split factor: {len(all_output_trajectories) / len(trajectories):.2f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
