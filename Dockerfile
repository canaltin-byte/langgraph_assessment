# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/langgraph_assessment

# Set working directory
WORKDIR /langgraph_assessment

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies and spaCy model
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_trf

# Copy only necessary files
COPY api.py .
COPY main.py .
COPY data_retrieval.py .
COPY evaluation.py .
COPY text_analyze.py .
COPY utils.py .
COPY templates ./templates/
COPY source ./source/

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
