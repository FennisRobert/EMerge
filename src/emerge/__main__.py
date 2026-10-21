# src/emerge/__main__.py
# Allows running the CLI as `python -m emerge ...`, which also works on Windows
# where the `emerge` console script may not be on PATH.
from .cli import main

if __name__ == "__main__":
    main()
