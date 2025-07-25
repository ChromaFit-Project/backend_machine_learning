# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.3
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

<<<<<<< Updated upstream
# Efficient pip install (with BuildKit caching support)
=======
# Efficient pip install with cache if BuildKit enabled
>>>>>>> Stashed changes
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install --no-cache-dir -r requirements.txt

<<<<<<< Updated upstream
# Copy Models and Dataset folders
COPY Models /app/Models
COPY Dataset /app/Dataset

# Copy main.py and any other app files
=======
# Copy model files and dataset (structured copy)
COPY Models /app/Models
COPY Dataset /app/Dataset

# Copy rest of the source code
>>>>>>> Stashed changes
COPY . .

EXPOSE 8080

<<<<<<< Updated upstream
# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
=======
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
>>>>>>> Stashed changes
