# Ĥolmes: Streamlit UI (port 8501) and REST API (port 8000) in one CPU-only image.
#
#   docker build -t holmes .
#   docker run -p 8501:8501 holmes                                                  # Streamlit app
#   docker run -p 8000:8000 holmes uvicorn api:app --host 0.0.0.0 --port 8000       # REST API (docs at /docs)
#   docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-... holmes                      # enable the Claude explainer
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg STREAMLIT_SERVER_HEADLESS=true STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libgomp: OpenMP runtime used by the torch and scipy wheels; curl: container health checks / smoke tests
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code edits do not invalidate the (slow) pip layer.
# torch comes from the CPU wheel index only when requirements.txt asks for it: the default index pulls ~2 GB of CUDA libraries.
COPY requirements.txt pyproject.toml ./
RUN if grep -qiE '^torch' requirements.txt; then pip install torch --index-url https://download.pytorch.org/whl/cpu; fi \
    && pip install -r requirements.txt

# The code (see .dockerignore), registered as a package so `import circuitsleuth` works from any directory.
COPY . .
RUN pip install --no-deps -e .

EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
