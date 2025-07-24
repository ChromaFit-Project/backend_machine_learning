# Use the official Python 3.11.3 slim image
FROM python:3.11.3-slim

# Set the working directory inside the container
WORKDIR /app

# Copy everything from your local folder into the container
COPY . .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your FastAPI app will run on (optional but good practice)
EXPOSE 8080

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
