# ============================================================
# VeriFeed Backend Dockerfile
# Compatible with face_recognition + dlib
# Python 3.11 (Stable for all dependencies)
# ============================================================

FROM python:3.11-slim

# ------------------------------------------------------------
# Install system dependencies required by dlib and face_recognition
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libx11-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Set working directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# Copy requirements first for optimized layer caching
# ------------------------------------------------------------
COPY requirements.txt .

# ------------------------------------------------------------
# Install Python packages
# ------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Copy project files
# ------------------------------------------------------------
COPY . .

# ------------------------------------------------------------
# Expose the port your Flask backend uses
# ------------------------------------------------------------
EXPOSE 5000

# ------------------------------------------------------------
# Start the Flask backend using Waitress (production server)
# ------------------------------------------------------------
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "src.app:app"]
