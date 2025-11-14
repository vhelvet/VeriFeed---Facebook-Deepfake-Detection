# Use a Python base image that supports your version and apt (Debian/Ubuntu)
FROM python:3.13-slim

# Install the missing system dependencies for OpenCV (libgl1)
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install your requirements (including your custom PyTorch URL)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files (including src/ and your models)
COPY . .

# Define the command to start your Waitress server
CMD ["python", "src/app.py"]
