try:
    from .cli import main
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise SystemExit(
            "PyTorch is required for training. Install dependencies first, e.g. `pip install -e .`."
        ) from exc
    raise


if __name__ == "__main__":
    main()
