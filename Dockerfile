# Use the official Python image as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

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

# Run the command to start the application
ENTRYPOINT ["panel", "serve", "main.py", "--allow-websocket-origin", "*", "--num-procs", "1", "--websocket-max-message-size", "157286400"]
CMD ["--cookie-secret", "my_super_safe_cookie_secret_2", "--basic-auth", "../config/credentials.json"]
