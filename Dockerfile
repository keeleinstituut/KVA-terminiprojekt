# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

EXPOSE 5006

ENV PANEL_PORT=5006
ENV APP_CONFIG=../config/config.json
ENV LOGGER_CONFIG=../config/logging.config
ENV LOG_FILE=../logs/app.log  

COPY config /app/config
COPY db /app/db
COPY utils /app/utils
COPY app /app/app

RUN mkdir -p /app/data/temp /app/data/finished /app/logs 
RUN ls -l

WORKDIR /app/app

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:5006 || exit 1

# Run the command to start the application
CMD ["panel", "serve", "main.py", "--port", "5006", "--allow-websocket-origin", "*"]


