FROM python:3.13-slim

# Use build arguments and also set them as environment variables
ARG MYSQL_HOST
ARG REDIS_HOST
ARG MYSQL_USER
ARG MYSQL_PASSWORD
ARG MYSQL_DATABASE
ARG MYSQL_PORT
ARG REDIS_PORT

ENV MYSQL_HOST=${MYSQL_HOST} \
    REDIS_HOST=${REDIS_HOST} \
    MYSQL_USER=${MYSQL_USER} \
    MYSQL_PASSWORD=${MYSQL_PASSWORD} \
    MYSQL_DATABASE=${MYSQL_DATABASE} \
    MYSQL_PORT=${MYSQL_PORT} \
    REDIS_PORT=${REDIS_PORT} \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
