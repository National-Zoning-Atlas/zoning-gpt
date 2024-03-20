# Use an official Python runtime as a parent image, specifying the version
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the local directory contents into the container at /usr/src/app
COPY templates /app/templates
COPY ui /app/ui
COPY pdm.lock /app
COPY pyproject.toml /app
COPY zoning /app/zoning
COPY data/ground_truth.csv /app/data/
RUN mkdir /app/data/logs
RUN mkdir /app/data/logs/included_context_phrases
RUN mkdir /app/data/logs/included_context_phrases/found-multiple-times
RUN mkdir /app/data/logs/included_context_phrases/found-once
RUN mkdir /app/data/logs/included_context_phrases/not-found
RUN mkdir /app/data/results/
RUN mkdir /app/data/results/snapshots

# Required by pdm
COPY README.md /app

# Required by zoning-gpt
COPY .git /app/.git

# Install pdm for Python dependency management
RUN apt-get update
RUN apt-get install -y --no-install-recommends curl
RUN pip install -U pip setuptools wheel
RUN pip install pdm

# Install any needed packages specified in pdm.lock
RUN pdm install

# Streamlit uses port 8501 by default, so we expose it
EXPOSE 8501

# Define environment variable, if needed
ENV ENVIRONMENT=Production

# Run the Streamlit application
CMD ["pdm", "run", "streamlit", "run", "ui/main.py"]
