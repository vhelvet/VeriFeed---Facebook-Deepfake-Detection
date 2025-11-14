FROM python:3.11-slim

# ------------------------------------------------------------
# Install dependencies required by dlib & face_recognition
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
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
# Copy requirements first
# ------------------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Copy application files
# ------------------------------------------------------------
COPY . .

# ------------------------------------------------------------
# Expose port
# ------------------------------------------------------------
EXPOSE 5000

# ------------------------------------------------------------
# Run app with Waitress
# ------------------------------------------------------------
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "src.app:app"]
