# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the repository contents into the container
COPY . /app

# Install dependencies (update requirements.txt as needed)
RUN pip install --no-cache-dir -r requirements.txt

# Default command (replace 'app.py' when your application is ready)
CMD ["python", "app.py"]
