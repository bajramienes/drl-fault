# ---------------------------------------------------------------------
# Base image
# ---------------------------------------------------------------------
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# ---------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-setuptools \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------
# GPUtil will skip gracefully if no GPU is detected
RUN pip install --no-cache-dir psutil GPUtil py-cpuinfo

# ---------------------------------------------------------------------
# Copy project files
# ---------------------------------------------------------------------
COPY . /app

# ---------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------
# Root folder for result logs (bind this to local drive)
ENV RESULTS_ROOT=/app/results

# ---------------------------------------------------------------------
# Container entry point
# ---------------------------------------------------------------------
CMD ["python", "runner.py"]
