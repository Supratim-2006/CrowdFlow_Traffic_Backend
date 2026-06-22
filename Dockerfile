# ── Base image: slim Python 3.11 ──────────────────────────────────────────────
FROM python:3.11-slim

# HF Spaces runs as a non-root user (uid 1000). Declare it early.
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Keeps HF hub cache inside the writable home
    HF_HOME=/home/user/.cache/huggingface \
    # OpenCV headless – no display needed
    OPENCV_IO_ENABLE_OPENEXR=0

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libspatialindex-dev \
        gdal-bin \
        libgdal-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (matches HF Spaces uid) ──────────────────────────────
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY --chown=user requirements.txt .

# Install CPU-only torch first (saves ~1.5 GB vs default CUDA wheel)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.1.0+cpu \
        torchvision==0.16.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source ────────────────────────────────────────────────────
# Copy everything; .dockerignore keeps the image clean
COPY --chown=user . .

# ── Pre-download HF models at build time (optional but speeds up cold start) ──
# Comment this block out if you prefer lazy loading at runtime
RUN python - <<'EOF'
from huggingface_hub import hf_hub_download
import os
for repo, fname in [
    ("SupratimKukri/road-closure-model",      "road_closure_model.pkl"),
    ("SupratimKukri/traffic-disruption-model","traffic_disruption_model.pkl"),
]:
    try:
        hf_hub_download(repo_id=repo, filename=fname)
        print(f"Cached {repo}/{fname}")
    except Exception as e:
        print(f"Skipping {repo}/{fname}: {e}")
EOF

# ── Expose port 7860 (required by HF Spaces) ──────────────────────────────────
EXPOSE 7860

# ── Start the API ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]