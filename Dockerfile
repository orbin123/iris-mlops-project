# Base Image
FROM python:3.11-slim


# Metadata
LABEL maintainer="orbinsunny@gmail.com" \
      description="Iris ML Model - Training and Evaluation" \
      version="1.0"

# Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Working Directory
WORKDIR /app

# Install Dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Project Files
COPY train.py evaluate.py ./

# Create Directories
RUN mkdir -p /app/models

# Default Command
CMD ["python", "train.py"]