FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY hf_spaces/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY hf_spaces/app.py .
COPY model.py .

# Create checkpoints folder
RUN mkdir -p checkpoints

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]