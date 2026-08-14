FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python 3.10 and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "serve.py", "--weights", "checkpoints/best_model.pt", "--port", "8000"]
