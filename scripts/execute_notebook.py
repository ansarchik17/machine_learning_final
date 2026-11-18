"""Execute every notebook cell in a fresh local project kernel and save outputs."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    path = ROOT / "notebooks/students_dropout_project.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=180, kernel_name="students-dropout", allow_errors=False,
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    nbformat.write(notebook, path)
    print(f"Executed and saved {len(code_cells)} code cells without errors: {path.relative_to(ROOT)}")
