from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from mtsc_train.cli import main
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise SystemExit(
            "PyTorch is required for training. Install dependencies first, e.g. `pip install -e .`."
        ) from exc
    raise


if __name__ == "__main__":
    main()
