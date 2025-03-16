FROM python:3.8-slim

WORKDIR /app

# Create a simple file to verify the image works
RUN echo "print('Hello from Docker container')" > test.py

CMD ["python", "/app/test.py"]