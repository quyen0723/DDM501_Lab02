# =============================================================================
# Dockerfile for the ML Pipeline application (supports the bonus
# `ml-pipeline` service in docker-compose.yml)
# =============================================================================
FROM python:3.10-slim

WORKDIR /app

# build-essential is required to compile scikit-surprise from source.
# libgomp1 is needed at runtime by numpy/scikit-learn on slim images.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# --- scikit-surprise build prerequisites -------------------------------------
# scikit-surprise 1.1.3 predates Cython 3 and numpy 2, and imports the removed
# `pkg_resources` module at load time. Install Cython<3 + numpy 1.26.2 +
# setuptools<81 first, then build surprise with --no-build-isolation so it
# reuses these instead of pulling incompatible build deps.
RUN pip install --no-cache-dir "cython<3" "numpy==1.26.2" "setuptools<81"

# Install the pipeline requirements. --no-build-isolation makes scikit-surprise
# compile against the pinned numpy above (other packages ship as wheels).
COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir --no-build-isolation -r requirements-pipeline.txt

# Pre-download the MovieLens 100K dataset so the pipeline never blocks on the
# surprise "download? (Y/n)" prompt at runtime (which hangs non-interactive
# containers). prompt=False downloads without asking.
RUN python -c "from surprise import Dataset; Dataset.load_builtin('ml-100k', prompt=False)"

COPY pipeline/ ./pipeline/
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/

CMD ["python", "-m", "pipeline.run_pipeline"]