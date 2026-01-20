# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy model files and code
COPY ./distillated_student_model /app/distillated_student_model
COPY app.py /app/app.py

# Install dependencies
# PRO TIP: We install the CPU-only version of PyTorch to save space (700MB vs 4GB)
RUN pip install --no-cache-dir fastapi uvicorn transformers \
    torch --index-url https://download.pytorch.org/whl/cpu

# Expose the port Cloud Run expects (PORT env var)
ENV PORT=8080

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]