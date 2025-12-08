"""Mock condenser for testing condensation logic without requiring LLM calls."""

from openhands.sdk.context.condenser.base import RollingCondenser
from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation
from pydantic import Field


class MockCondenser(RollingCondenser):
    """A mock condenser that triggers condensation based on a simple threshold."""

    max_size: int = Field(default=10, gt=0)
    keep_first: int = Field(default=2, ge=0)

    def handles_condensation_requests(self) -> bool:
        return False

    def should_condense(self, view: View) -> bool:
        return len(view) > self.max_size

    def get_condensation(self, view: View) -> Condensation:
        """Create a condensation event with a mock summary."""
        head = view[: self.keep_first]
        target_size = self.max_size // 2
        events_from_tail = target_size - len(head) - 1

        # Identify events to be forgotten
        forgotten_events = view[self.keep_first : -events_from_tail]

        # Create a simple summary
        summary = (
            f"[Summary: Condensed {len(forgotten_events)} events from the conversation history]"
        )

        return Condensation(
            forgotten_event_ids=[event.id for event in forgotten_events],
            summary=summary,
            summary_offset=self.keep_first,
        )
