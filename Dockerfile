# Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# JaamCTRL — AI Adaptive Traffic Signal Control
# HuggingFace Spaces Docker deployment
#
# Exposes port 7860 (HF Spaces default).
# Runs Streamlit dashboard by default.
# Switch CMD to inference mode via INFERENCE_MODE=1 env var.
#
# Build locally:
#   docker build -t jaamctrl .
#   docker run -p 7860:7860 -e MOCK_SUMO=1 jaamctrl
#
# On HF Spaces:
#   Push this repo to a Space with sdk: docker
#   Set MOCK_SUMO=1 in Space secrets if SUMO install is not desired.
# ─────────────────────────────────────────────────────────────────────────────

FROM ubuntu:22.04

# ── System deps ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

RUN apt-get update && apt-get install -y --no-install-recommends \
    # SUMO traffic simulator
    sumo \
    sumo-tools \
    sumo-doc \
    # Python runtime
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-distutils \
    # Build tools (needed by some pip packages)
    build-essential \
    git \
    curl \
    # Streamlit needs these for browser rendering checks
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# ── SUMO environment variable ─────────────────────────────────────────────────
# HF Spaces standard SUMO install path on Ubuntu 22.04
ENV SUMO_HOME=/usr/share/sumo
ENV PATH="${SUMO_HOME}/bin:${PATH}"

# ── Python alias ──────────────────────────────────────────────────────────────
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && ln -sf /usr/bin/python3.11 /usr/bin/python \
 && python3 -m pip install --upgrade pip

# ── Create non-root user (HF Spaces requirement: user ID 1000) ────────────────
RUN useradd -m -u 1000 user
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY --chown=user . /app

# ── Create writable directories for logs and model outputs ───────────────────
RUN mkdir -p /app/logs /app/agents/models \
 && chown -R user:user /app/logs /app/agents/models

# ── Switch to non-root user ───────────────────────────────────────────────────
USER user

# ── Environment defaults ──────────────────────────────────────────────────────
# MOCK_SUMO=1  → skip real SUMO; use synthetic observations (safe on HF Spaces)
# MOCK_SUMO=0  → use real SUMO (requires SUMO installed, which it is here)
ENV MOCK_SUMO=0
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:${PATH}
ENV PYTHONPATH=/app

# ── Port ─────────────────────────────────────────────────────────────────────
# HF Spaces Docker must expose 7860
# OpenEnv API uses 5000 (redirected to 7860 by HF Spaces)
EXPOSE 7860 5000

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || curl -f http://localhost:5000/health || exit 1

# ── Startup ───────────────────────────────────────────────────────────────────
# Default: Streamlit dashboard on port 7860
# INFERENCE_MODE=1: Run inference.py inference pipeline
# OPENENV_API=1: Run OpenEnv HTTP API server on port 5000 (for submission checker)
CMD ["sh", "-c", \
    "if [ \"$OPENENV_API\" = '1' ]; then \
        python openenv_api.py --port 7860; \
     elif [ \"$INFERENCE_MODE\" = '1' ]; then \
        python inference.py --mock; \
     else \
        streamlit run app.py \
            --server.port 7860 \
            --server.address 0.0.0.0 \
            --server.headless true \
            --server.enableCORS false \
            --server.enableXsrfProtection false; \
     fi"]
