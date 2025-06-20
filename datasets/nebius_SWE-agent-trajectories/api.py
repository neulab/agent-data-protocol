def open(path: str, line_number: int = None):
    """Open the file at the given path in the editor. If line_number is provided, the window will be moved to include that line.

    Args:
    ----
        path (str): The path to the file to open.
        line_number (int, optional): The line number to move the window to (if not provided, the window will start at the top of the file).

    """
    pass


def goto(line_number: int):
    """Move the window to show the specified line number.

    Args:
    ----
        line_number (int): The line number to move the window to.

    """
    pass


def scroll_down():
    """Move the window down {WINDOW} lines."""
    pass


def scroll_up():
    """Move the window up {WINDOW} lines."""
    pass


def create(filename: str):
    """Create and open a new file with the given name.

    Args:
    ----
        filename (str): The name of the file to create.

    """
    pass


def submit():
    """Submit your current code and terminate the interactive session."""
    pass


def search_dir(search_term: str, dir: str = None):
    """Search for the search_term in all files in the specified directory. If dir is not provided, search in the current directory.

    Args:
    ----
        search_term (str): The term to search for.
        dir (str, optional): The directory to search in (if not provided, search in the current directory).

    """
    pass


def search_file(search_term: str, file: str = None):
    """Search for the search_term in the specified file. If file is not provided, search in the current open file.

    Args:
    ----
        search_term (str): The term to search for.
        file (str, optional): The file to search in (if not provided, search in the current open file).

    """
    pass


def find_file(file_name: str, dir: str = None):
    """Find all files with the given name in the specified directory. If dir is not provided, search in the current directory.

    Args:
    ----
        file_name (str): The name of the file to search for.
        dir (str, optional): The directory to search in (if not provided, search in the current directory).

    """
    pass


def edit(start_line: int, end_line: int, replacement_text: str):
    """Replace lines from start_line to end_line (inclusive) with the given text in the open file. The replacement text is terminated by a line with only 'end_of_edit' on it. All of the replacement text will be entered, so make sure your indentation is formatted properly. Python files will be checked for syntax errors after the edit. If the system detects a syntax error, the edit will not be executed. Simply try to edit the file again, but make sure to read the error message and modify the edit command you issue accordingly. Issuing the same command a second time will just lead to the same error message again. Please note that THE EDIT COMMAND REQUIRES PROPER INDENTATION. If you'd like to add the line '        pr int(x)' you must fully write that out, with all those spaces before the code! Indentation is important and code that is not indented correctly will fail and require fixing before it can be run.

    Args:
    ----
        start_line (int): The line number to start the edit at.
        end_line (int): The line number to end the edit at (inclusive).
        replacement_text (str): The text to replace the current selection with.

    """
    pass
