from pathlib import Path
import argparse
import json

from .models import Prompts, FuncDefs
from pydantic import ValidationError


def parse_pydantic_model(
        value, model: Prompts | FuncDefs) -> Prompts | FuncDefs:
    """Validates provided JSON files using a Pydantic model
    raising argparse errors to be displayed.
    Returns models if valid to be kept in argparse namespace.
    """
    path = Path(value)
    if not path.exists:
        raise argparse.ArgumentError(f"No such file or directory: {value}")
    try:
        json_data = json.loads(path.read_text())
        return model.model_validate(json_data)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Input must be a valid JSON file.")
    except ValidationError as e:
        failed_fields = [str(err['loc'][0]) for err in e.errors()]
        raise argparse.ArgumentTypeError(
            "Validation failed. Missing or invalid fields: "
            f"{', '.join(failed_fields)}"
        )
    except IOError:
        raise argparse.ArgumentTypeError(f"Could not read file: {value}")


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
        type=lambda x:
            FuncDefs.model_validate(
                json.loads(Path(x).read_text(encoding='utf-8'))),
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
        type=Path,
        default="data/output/function_calling_results.json",
        help="Path to output JSON file with results"
                "(default: data/output/function_calling_results.json)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name (default: Qwen/Qwen3-0.6B)"
    )
    return parser.parse_args()
