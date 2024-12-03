# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y gcc \
    && apt-get clean

# Install Python dependencies
COPY requirements.txt /usr/src/app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Add the rest of the code
COPY . /usr/src/app/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Make port for gRPC
EXPOSE 50051

# Run the application:
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
