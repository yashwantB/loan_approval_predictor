from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


NOTEBOOK = Path("notebooks/loan_approval_system.ipynb")


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK)
    print(f"Executed notebook: {NOTEBOOK}")


if __name__ == "__main__":
    main()
