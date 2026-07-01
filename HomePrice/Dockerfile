# ------------------------------------------------------------
# Dockerfile — Home Price Prediction API
# ------------------------------------------------------------

# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy only dependency file first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire server folder (Flask app + util + artifacts)
COPY server/ ./server

# Set environment variables for Flask
ENV FLASK_APP=server.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Expose Flask port
EXPOSE 5000

# Run Flask server
CMD ["python", "server/server.py"]
