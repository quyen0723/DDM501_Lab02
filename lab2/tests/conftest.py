"""
Pytest configuration for the ML Pipeline test suite.

Purpose
-------
Two environmental dependencies make the tests fragile on a fresh clone:

1. **MLflow server.** ``pipeline/registry.py`` constructs ``MlflowClient()``
   without an explicit URI, so it falls back to the ``MLFLOW_TRACKING_URI`` env
   var. The project default (``pipeline/config.py``) is ``http://localhost:5000``;
   if a student exports that URI but no server is running, tests that touch the
   registry (e.g. ``test_list_registered_models``) raise a connection error
   instead of passing.

2. **MovieLens 100K download.** ``Dataset.load_builtin('ml-100k')`` prompts
   "Y/n" on first download, which hangs / EOFErrors in a non-interactive shell
   or CI.

This conftest makes the suite self-contained:

* It pins ``MLFLOW_TRACKING_URI`` to a local file store (``file:./mlruns``) so
  registry tests run without a server. ``setdefault`` keeps an explicit value
  the user/exported env already provided.
* It pre-downloads MovieLens 100K with ``prompt=False`` (no prompt) so data
  tests do not block.

It is additive: it touches no application code and only sets environment +
downloads a public dataset.
"""

import os

# Use a local MLflow file store for tests (no server required).
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")


def _ensure_movielens() -> None:
    """Download MovieLens 100K non-interactively if it is not already cached."""
    try:
        from surprise import Dataset

        Dataset.load_builtin("ml-100k", prompt=False)
    except TypeError:
        # Older surprise without the `prompt` kwarg: nothing we can do here
        # without prompting; let the data tests surface a clear error instead.
        pass
    except Exception:
        # Network/permission errors: don't fail collection; tests that need
        # the dataset will report the real problem.
        pass


_ensure_movielens()