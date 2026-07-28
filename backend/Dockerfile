# SafeWatch Backend — Hugging Face Docker Space
# Optimized for Hugging Face Spaces (16GB RAM, 2 vCPUs Free Tier)

FROM python:3.11-slim

# Install system dependencies (OpenCV requires libGL, but we use opencv-python-headless to save space)
# We still install basic libs just in case ONNX or Ultralytics needs them
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs as non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirement files first for layer caching
COPY --chown=user:user requirements-local.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user:user . .

# IMPORTANT: In Hugging Face, the build context is the root of the space repo.
# If you pushed the `backend/` folder contents to the HF repo root, the models 
# must be placed in the `/models/` folder inside the HF repo.
# Note: HF Spaces does not allow COPY ../ outside of context. 
# Make sure to copy the actual model files into the same directory before pushing.

# Expose the mandatory Hugging Face port
EXPOSE 7860

# Run Uvicorn on 0.0.0.0:7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
