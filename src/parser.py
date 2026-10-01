from pathlib import Path
import argparse
import json
import os

from .models import Prompts, FuncDefs
from pydantic import ValidationError


def parse_pydantic_model(
        value: str, model: type[Prompts | FuncDefs]) -> Prompts | FuncDefs:
    """Validates provided JSON files using a Pydantic model
    raising argparse errors to be displayed.
    Returns models if valid to be kept in argparse namespace.
    """
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentError(
            None, f"No such file or directory: {value}")
    try:
        json_data = json.loads(path.read_text(encoding='utf-8'))
        return model.model_validate(json_data)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Input must be a valid JSON file.")
    except ValidationError as e:
        raise argparse.ArgumentTypeError(
            f"Validation failed for {path.name}. Missing or invalid fields: "
            f"{e}")
    except IOError:
        raise argparse.ArgumentTypeError(f"Could not read file: {value}")


def check_output_path(path: str) -> Path:
    """Checks if the provided output path is valid and writable.
    Raises argparse errors to be displayed.
    Returns Path object if valid to be kept in argparse namespace.
    """
    try:
        output_path = Path(path)
        if output_path.exists() and not output_path.is_file():
            raise argparse.ArgumentTypeError(
                f"Output path must be a file, not a directory: {path}")
        else:
            os.makedirs(output_path.parent, exist_ok=True)
        output_path.touch(exist_ok=True)
        with open(output_path, 'a'):
            pass
    except IOError:
        raise argparse.ArgumentTypeError(f"Cannot write to file: {path}")
    return output_path


def read_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns
    --------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Hey I just met you",
        prog="python -m src"
    )
    parser.add_argument(
        "-fd", "--functions_definition",
        type=lambda x: parse_pydantic_model(x, FuncDefs),
        dest="defs",
        metavar="path_to_definitions",
        default="data/input/functions_definition.json",
        help="Path to JSON file with function definitions"
                "(default: data/input/functions_definition.json)"
    )
    parser.add_argument(
        "-i", "--input",
        type=lambda x: parse_pydantic_model(x, Prompts),
        dest="prompts",
        default="data/input/function_calling_tests.json",
        help="Path to JSON file with propmts"
                "(default: data/input/function_calling_tests.json)"
    )
    parser.add_argument(
        "-o", "--output",
        type=check_output_path,
        default="data/output/function_calls.json",
        help="Path to output JSON file with results"
                "(default: data/output/function_calls.json)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name (default: Qwen/Qwen3-0.6B)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable visualisation output"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=4242,
        help="Port number for visualisation (default: 4242)"
    )
    parser.add_argument(
        "-d", "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help=("Chose device for model "
              "(default: cuda or mps if available else cpu)")
    )
    return parser.parse_args()
