from typing import Any


def web_agent(task: str) -> dict:
    """Use a web browser agent to complete a web task."""
    pass


def file_agent(task: str, file_path_dict: dict | None = None) -> dict:
    """Use a file-analysis agent to answer a task over local files."""
    pass


def stop(
    output: Any = None,
    log: Any = None,
    answer: Any = None,
    summary: Any = None,
) -> dict:
    """Finalize a task with either source-specific final-answer signature."""
    pass


def ask_llm(query: str) -> str:
    """Ask a language model for tasks that need no external tools."""
    pass


def simple_web_search(query: str) -> str:
    """Run a quick web search for straightforward information needs."""
    pass


def load_file(file_name: str) -> str:
    """Load a local file into the CognitiveKernel file environment."""
    pass


def read_text(file_name: str, page_id_list: list) -> str:
    """Read selected file pages as text."""
    pass


def read_screenshot(file_name: str, page_id_list: list) -> str:
    """Read selected file pages with screenshot-based processing."""
    pass


def search(file_name: str, key_word_list: list) -> str:
    """Search a file for keywords and return matching pages."""
    pass
