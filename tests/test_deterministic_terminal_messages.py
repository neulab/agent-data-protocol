import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dataset_converters_do_not_use_unseeded_random_choice():
    offenders = []
    converter_paths = [
        *(REPO_ROOT / "datasets").glob("*/raw_to_atif.py"),
        *(REPO_ROOT / "datasets").glob("*/atif_to_std.py"),
    ]
    for converter_path in sorted(converter_paths):
        tree = ast.parse(converter_path.read_text(), filename=str(converter_path))
        random_module_names = {"random"}
        random_choice_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "random":
                        random_module_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "random":
                for alias in node.names:
                    if alias.name == "choice":
                        random_choice_names.add(alias.asname or alias.name)
                    elif alias.name == "*":
                        random_choice_names.add("choice")

        for node in ast.walk(tree):
            uses_random_choice = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "choice"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in random_module_names
            )
            uses_imported_choice = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in random_choice_names
            )
            if uses_random_choice or uses_imported_choice:
                offenders.append(f"{converter_path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not offenders, "Use a per-trajectory seeded RNG instead of random.choice: " + ", ".join(
        offenders
    )
