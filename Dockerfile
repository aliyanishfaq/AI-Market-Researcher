# Use an official Python 3.10 slim image
FROM python:3.10-slim

# Create a working directory in the container
WORKDIR /app

# Copy requirements.txt first, then install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /app

# Expose the port that your FastAPI app will listen on
EXPOSE 8080

# This command runs your FastAPI server with Uvicorn on port 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
