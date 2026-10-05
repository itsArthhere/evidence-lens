"""Execute every notebook cell in a fresh kernel and fail on any error."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import sys

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/01_evidence_retrieval.ipynb'
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
NotebookClient(notebook, km=manager, timeout=600, resources={'metadata': {'path': str(root)}}).execute()
nbformat.write(notebook, path)
code = [c for c in notebook.cells if c.cell_type == 'code']
assert all(c.execution_count is not None for c in code)
assert not any(o.output_type == 'error' for c in code for o in c.outputs)
print(f'Executed {len(code)} code cells; zero notebook errors: {path}')
