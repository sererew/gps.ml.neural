# Reproducible Python Environment

The active project stack is Python 3.11.x with TensorFlow 2.21.0 and Keras 3.14.1. Use this stack for all live scripts: v3, context_v1, context_tcn_v1, filters, comparison, and remote training.

`python/requirements.txt` contains the active direct dependencies. The lock files capture exact environments:

- `python/requirements-context-tf221-win-lock.txt`: exact local Windows environment.
- `python/requirements-context-tf221-a40-lock.txt`: exact A40/Linux GPU environment, including CUDA Python packages.

## Windows/local machines

Create the local virtual environment from the repository root:

```powershell
python -m venv .venv_keras3
.\.venv_keras3\Scripts\python.exe -m pip install --upgrade pip
.\.venv_keras3\Scripts\python.exe -m pip install -r python\requirements.txt
```

For exact reproduction of this machine, install the Windows lock instead:

```powershell
.\.venv_keras3\Scripts\python.exe -m pip install -r python\requirements-context-tf221-win-lock.txt
```

Run scripts with:

```powershell
.\.venv_keras3\Scripts\python.exe -X utf8 python\pipeline\7_apply_all_filters.py --filtros nn_context_v1,nn_context_tcn_v1
```

## A40/Linux GPU

The remote script uses `python/requirements-context-tf221-a40-lock.txt` automatically. Manual setup is:

```bash
python3.11 -m venv venv_context_v1_py311
. venv_context_v1_py311/bin/activate
python -m pip install --upgrade pip
pip install -r python/requirements-context-tf221-a40-lock.txt
```

## Notes

- Do not use the old TensorFlow 2.15 / Keras 2.15 global environment for active model work.
- `model_final_*.keras` is the model artifact consumed by filters. `model_best_*` is intentionally no longer produced.
- Local virtual environments are ignored by `.gitignore` via `.venv*/`.