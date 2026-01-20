# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy model files and code
COPY ./model /app/model
COPY app.py /app/app.py

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn transformers

# Install PyTorch (CPU version) from the specific PyTorch repository
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Expose the port Cloud Run expects (PORT env var)
ENV PORT=8080

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]