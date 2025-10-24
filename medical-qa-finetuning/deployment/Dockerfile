FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    git \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the package
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p data models results logs

# Expose ports
EXPOSE 7860 8080

# Default command
CMD ["python3", "scripts/deploy_gradio.py", "--host", "0.0.0.0", "--port", "7860"]
