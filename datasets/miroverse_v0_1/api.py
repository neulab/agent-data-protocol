from typing import Any


def use_mcp_tool(server_name: str, tool_name: str, arguments: dict[str, Any] | str):
    """Fallback MCP wrapper for rows whose XML arguments cannot be parsed.

    Args:
        server_name: Name of the MCP server that provides the tool.
        tool_name: Name of the tool to execute.
        arguments: Tool arguments as a JSON object, or the raw argument string when parsing fails.

    """
    return None


def browsing_agent__search_and_browse(subtask: str) -> dict:
    """[browsing-agent/search_and_browse] This tool is an agent that performs the subtask of searching and browsing the web for specific missing information and generating the desired answer. The subtask should be clearly defined, include relevant background, and focus on factual gaps. It does not perform vague or speculative subtasks.
    Args:
            subtask: the subtask to be performed.
    Returns:
            the result of the subtask.
    """
    return {}


def tool_google_search__google_search(
    q: str,
    gl: str | None = None,
    hl: str | None = None,
    location: str | None = None,
    num: float | None = None,
    tbs: str | None = None,
    page: float | None = None,
    autocorrect: bool | None = None,
) -> dict:
    """[tool-google-search/google_search] Tool to perform web searches via Serper API and retrieve rich results. It is able to retrieve organic search results, people also ask, related searches, and knowledge graph.

    Args:
        q: Search query string
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        location: Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')
        num: Number of results to return (default: 10)
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year)
        page: Page number of results to return (default: 1)
        autocorrect: Whether to autocorrect spelling in query
    """
    return {}


def tool_google_search__scrape(url: str, includeMarkdown: bool | None = None) -> dict:
    """[tool-google-search/scrape] Tool to scrape a webpage and retrieve the text and, optionally, the markdown content. It will retrieve also the JSON-LD metadata and the head metadata.

    Args:
        url: The URL of the webpage to scrape.
        includeMarkdown: Whether to include markdown content.
    """
    return {}


def tool_serper_search__google_search(
    q: str,
    gl: str | None = None,
    hl: str | None = None,
    location: str | None = None,
    num: float | None = None,
    tbs: str | None = None,
    page: float | None = None,
    autocorrect: bool | None = None,
) -> dict:
    """[tool-serper-search/google_search] Tool to perform web searches via Serper API and retrieve rich results. It is able to retrieve organic search results, people also ask, related searches, and knowledge graph.

    Args:
        q: Search query string
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        location: Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')
        num: Number of results to return (default: 10)
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year)
        page: Page number of results to return (default: 1)
        autocorrect: Whether to autocorrect spelling in query
    """
    return {}


def tool_serper_search__scrape(url: str, includeMarkdown: bool | None = None) -> dict:
    """[tool-serper-search/scrape] Tool to scrape a webpage and retrieve the text and, optionally, the markdown content. It will retrieve also the JSON-LD metadata and the head metadata.

    Args:
        url: The URL of the webpage to scrape.
        includeMarkdown: Whether to include markdown content.
    """
    return {}


def tool_code__create_sandbox(timeout: int = 300) -> dict:
    """[tool-code/create_sandbox] Create a linux sandbox.

    Args:
        timeout: Time in seconds before the sandbox is automatically shutdown. The default is 300 seconds.

    Returns:
        The id of the newly created sandbox. You should use this sandbox_id to run other tools in the sandbox.
    """
    return {}


def tool_code__download_internet_file_to_sandbox(
    sandbox_id: str, url: str, sandbox_file_path: str = "/home/user"
) -> dict:
    """[tool-code/download_internet_file_to_sandbox] Download a file from the internet to the `/home/user` dir of the remote python interpreter.
    You should use this tool to download files from the internet.

    Args:
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.
        url: The URL of the file to download.
        sandbox_file_path: The path of directory to download the file to in the sandbox. Default is `/home/user/`.

    Returns:
        The path of the downloaded file in the python interpreter if the download is successful.
    """
    return {}


def tool_code__run_command(command: str, sandbox_id: str) -> dict:
    """[tool-code/run_command] Execute a command in the linux sandbox.

    Args:
        command: The command to execute
        sandbox_id: The id of the sandbox to execute the command in. To create a new sandbox, use tool `create_sandbox`.

    Returns:
        A result of the command execution, format like (stderr=..., stdout=..., exit_code=..., error=...)
    """
    return {}


def tool_code__run_python_code(code_block: str, sandbox_id: str) -> dict:
    """[tool-code/run_python_code] Run python code in an interpreter and return the execution result.

    Args:
        code_block: The python code to run.
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.

    Returns:
        A result of the command execution, format like (stderr=..., stdout=..., exit_code=..., error=...)
    """
    return {}


def tool_code__upload_local_file_to_sandbox(
    sandbox_id: str, local_file_path: str, sandbox_file_path: str = "/home/user"
) -> dict:
    """[tool-code/upload_local_file_to_sandbox] Upload a local file to the `/home/user` dir of the remote python interpreter.

    Args:
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.
        local_file_path: The path of the file on local machine to upload.
        sandbox_file_path: The path of directory to upload the file to in the sandbox. Default is `/home/user/`.

    Returns:
        The path of the uploaded file in the remote python interpreter if the upload is successful.
    """
    return {}


def tool_python__create_sandbox(timeout: int = 300) -> dict:
    """[tool-python/create_sandbox] Create a linux sandbox.

    Args:
        timeout: Time in seconds before the sandbox is automatically shutdown. The default is 300 seconds.

    Returns:
        The id of the newly created sandbox. You should use this sandbox_id to run other tools in the sandbox.
    """
    return {}


def tool_python__download_internet_file_to_python_interpreter(
    url: str, sandbox_id: str = None
) -> dict:
    """[tool-python/download_internet_file_to_python_interpreter] Download a file from the internet to the `/home/user` dir of the remote python interpreter.

    Args:
        url: The URL of the file to download.
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. Only create new ones if this is the first time running code in this sandbox.

    Returns:
        The path of the downloaded file in the python interpreter.
    """
    return {}


def tool_python__download_internet_file_to_sandbox(
    sandbox_id: str, url: str, sandbox_file_path: str = "/home/user"
) -> dict:
    """[tool-python/download_internet_file_to_sandbox] Download a file from the internet to the `/home/user` dir of the remote python interpreter.

    Args:
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.
        url: The URL of the file to download.
        sandbox_file_path: The path of directory to download the file to in the sandbox. Default is `/home/user/`.

    Returns:
        The path of the downloaded file in the python interpreter if the download is successful.
    """
    return {}


def tool_python__run_command(command: str, sandbox_id: str) -> dict:
    """[tool-python/run_command] Execute a command in the linux sandbox.

    Args:
        command: The command to execute
        sandbox_id: The id of the sandbox to execute the command in. To create a new sandbox, use tool `create_sandbox`.

    Returns:
        A CommandResult object containing the result of the command execution, format like CommandResult(stderr=..., stdout=..., exit_code=..., error=...)
    """
    return {}


def tool_python__run_python_code(
    code_block: str, timeout: int = 300, sandbox_id: str = None
) -> dict:
    """[tool-python/run_python_code] Run python code in an interperter and return the execution result.

    Args:
        code_block: The python code to run.
        timeout: Time in seconds before the sandbox is automatically shutdown. The default is 300 seconds.
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. Only create new ones if this is the first time running code in this sandbox.

    Returns:
        An object containing the sandbox id and the execution result object including results, logs and errors.
    """
    return {}


def tool_python__upload_local_file_to_python_interpreter(
    local_file_path: str, sandbox_id: str = None
) -> dict:
    """[tool-python/upload_local_file_to_python_interpreter] Upload a local file to the `/home/user` dir of the remote python interpreter.

    Args:
        file_path: The path of the file on local machine to upload.
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. Only create new ones if this is the first time running code in this sandbox.

    Returns:
        The path of the uploaded file in the remote python interpreter.
    """
    return {}


def tool_python__upload_local_file_to_sandbox(
    sandbox_id: str, local_file_path: str, sandbox_file_path: str = "/home/user"
) -> dict:
    """[tool-python/upload_local_file_to_sandbox] Upload a local file to the `/home/user` dir of the remote python interpreter.

    Args:
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.
        local_file_path: The path of the file on local machine to upload.
        sandbox_file_path: The path of directory to upload the file to in the sandbox. Default is `/home/user/`.

    Returns:
        The path of the uploaded file in the remote python interpreter if the upload is successful.
    """
    return {}


def tool_reader__convert_to_markdown(uri: str) -> dict:
    """[tool-reader/convert_to_markdown] Convert a resource described by an http:, https:, file: or data: URI to markdown

    Args:
        uri: Uri
    """
    return {}


def tool_reading__convert_to_markdown(uri: str) -> dict:
    """[tool-reading/convert_to_markdown] Convert various types of resources (doc, ppt, pdf, excel, csv, zip file etc.)
    described by an file: or data: URI to markdown.

    Args:
        uri: Required. The URI of the resource to convert. Need to start with 'file:' or 'data:' schemes.

    Returns:
        str: The converted markdown content, or an error message if conversion fails.
    """
    return {}


def tool_reasoning__reasoning(question: str) -> dict:
    """[tool-reasoning/reasoning] You can use this tool use solve hard math problem, puzzle, riddle and IQ test question that requries a lot of chain of thought efforts.
    DO NOT use this tool for simple and obvious question.

    Args:
        question: The hard question.

    Returns:
        The answer to the question.
    """
    return {}


def tool_transcribe__audio_transcription(audio_path_or_url: str) -> dict:
    """[tool-transcribe/audio_transcription] Transcribe audio file to text and return the transcription.
    Args:
        audio_path_or_url: The path of the audio file locally or its URL.

    Returns:
        The transcription of the audio file.
    """
    return {}


def tool_vqa__visual_question_answering(image_path_or_url: str, question: str) -> dict:
    """[tool-vqa/visual_question_answering] This tool is used to ask question about an image or a video and get the answer with both Claude and OpenAI vision language models. It also automatically performs OCR (text extraction) on the image for additional context.

    Args:
        image_path_or_url: The path of the image file locally or its URL.
        question: The question to ask about the image.

    Returns:
        The concatenated answers from both Claude and OpenAI vision models, including both VQA responses and OCR results.
    """
    return {}
