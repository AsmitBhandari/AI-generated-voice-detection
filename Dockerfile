FROM python:3.10.11-slim

WORKDIR /app

# Install system dependencies for audio (ffmpeg, libsndfile)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only pytorch first to avoid downloading huge CUDA wheels
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Remove torch from requirements to avoid re-installing (or keep it and trust pip to skip)
# Safest is to let pip check but it might try to upgrade if versions mismatch. 
# We'll run pip install but hopefully it respects the installed version if compatible.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
