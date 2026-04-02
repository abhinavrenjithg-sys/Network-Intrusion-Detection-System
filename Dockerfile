# Use lightweight official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy requirement and install dependencies
COPY requirements.txt .

# Install dependencies required for machine learning ops
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project repository
COPY . .

# Expose port (if you were deploying a FastAPI web app on top of this, e.g. port 8000)
EXPOSE 8000

# Set Python Path to include src/ for absolute imports
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# By default, trigger the Malware detection pipeline
# This can be overridden explicitly via docker run command
CMD ["python", "src/run_malware_detection.py"]
