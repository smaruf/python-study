# Deployment Guide

This guide covers deployment options for the FastAPI Fintech Application.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Checklist](#production-checklist)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Scaling](#scaling)
8. [Security](#security)

## Prerequisites

- Python 3.9 or higher
- PostgreSQL 12 or higher (for production)
- Docker and Docker Compose (for containerized deployment)
- Reverse proxy (Nginx or similar)
- SSL certificate

## Environment Configuration

### Production Environment Variables

Create a `.env` file with the following production settings:

```bash
# Application
APP_NAME=FastAPI Fintech Application
DEBUG=False
API_V1_PREFIX=/api/v1
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=postgresql+asyncpg://user:password@db-host:5432/fintech_db

# Security
SECRET_KEY=<generate-with-openssl-rand-hex-32>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS
BACKEND_CORS_ORIGINS=["https://yourdomain.com"]

# Logging
LOG_LEVEL=INFO
```

### Generate Secret Key

```bash
openssl rand -hex 32
```

## Docker Deployment

### Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: fintech
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: fintech_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fintech"]
      interval: 10s
      timeout: 5s
      retries: 5

  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://fintech:${DB_PASSWORD}@db:5432/fintech_db
      SECRET_KEY: ${SECRET_KEY}
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
```

### Deploy with Docker Compose

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Cloud Deployment

### AWS Deployment

#### Using Elastic Beanstalk

1. Install EB CLI:
```bash
pip install awsebcli
```

2. Initialize EB application:
```bash
eb init -p python-3.11 fastapi-fintech
```

3. Create environment:
```bash
eb create production-env
```

4. Deploy:
```bash
eb deploy
```

#### Using ECS (Elastic Container Service)

1. Build and push Docker image to ECR
2. Create ECS cluster
3. Define task definition
4. Create ECS service
5. Configure Application Load Balancer

### Google Cloud Platform

#### Using Cloud Run

```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/fastapi-fintech

# Deploy to Cloud Run
gcloud run deploy fastapi-fintech \
  --image gcr.io/PROJECT_ID/fastapi-fintech \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create fastapi-fintech

# Add PostgreSQL addon
heroku addons:create heroku-postgresql:hobby-dev

# Deploy
git push heroku main

# Run migrations
heroku run alembic upgrade head
```

## Production Checklist

### Security
- [ ] Change default SECRET_KEY
- [ ] Disable DEBUG mode
- [ ] Set up HTTPS/SSL
- [ ] Configure CORS properly
- [ ] Enable rate limiting
- [ ] Set up firewall rules
- [ ] Implement API authentication
- [ ] Regular security audits

### Database
- [ ] Use PostgreSQL (not SQLite)
- [ ] Enable connection pooling
- [ ] Set up automated backups
- [ ] Configure read replicas
- [ ] Implement database migrations
- [ ] Index optimization

### Application
- [ ] Configure proper logging
- [ ] Set up error tracking (Sentry)
- [ ] Enable application monitoring
- [ ] Configure health checks
- [ ] Set up CI/CD pipeline
- [ ] Document API endpoints
- [ ] Write tests (>80% coverage)

### Infrastructure
- [ ] Use reverse proxy (Nginx)
- [ ] Configure load balancer
- [ ] Set up auto-scaling
- [ ] Configure CDN
- [ ] Implement caching (Redis)
- [ ] Set up monitoring and alerts

## Monitoring and Logging

### Application Logging

Configure structured logging:

```python
# app/core/logging_config.py
import logging
from loguru import logger

logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time} {level} {message}"
)
```

### Monitoring Tools

1. **Prometheus + Grafana**: Metrics collection and visualization
2. **Sentry**: Error tracking and monitoring
3. **New Relic**: Application performance monitoring
4. **ELK Stack**: Log aggregation and analysis

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/db
```

## Scaling

### Horizontal Scaling

Run multiple application instances:

```bash
# Using uvicorn with multiple workers
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000

# Using gunicorn
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Load Balancing

Configure Nginx as load balancer:

```nginx
upstream fastapi_backend {
    least_conn;
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Caching

Implement Redis caching:

```python
# Cache frequently accessed data
from redis import asyncio as aioredis

redis = aioredis.from_url("redis://localhost")

# Cache stock prices
await redis.setex(f"stock:{symbol}", 60, price)
```

## Security

### HTTPS Configuration

Nginx SSL configuration:

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/data")
@limiter.limit("100/minute")
async def get_data():
    return {"data": "value"}
```

### Database Security

1. Use environment variables for credentials
2. Enable SSL for database connections
3. Implement row-level security
4. Regular security patches

## Backup and Recovery

### Automated Backups

```bash
# PostgreSQL backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump -U user fintech_db > backup_${TIMESTAMP}.sql
aws s3 cp backup_${TIMESTAMP}.sql s3://backups/
```

### Restore

```bash
psql -U user fintech_db < backup_20240101_120000.sql
```

## Continuous Deployment

### GitHub Actions

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build and push Docker image
        run: |
          docker build -t fastapi-fintech .
          docker push registry/fastapi-fintech:latest
      - name: Deploy to server
        run: |
          ssh user@server 'docker pull registry/fastapi-fintech:latest'
          ssh user@server 'docker-compose up -d'
```

## Troubleshooting

### Common Issues

1. **Database connection errors**: Check DATABASE_URL and network
2. **Import errors**: Verify all dependencies are installed
3. **Port conflicts**: Ensure port 8000 is available
4. **Memory issues**: Increase worker count or instance size

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m app.main
```

## Support

For deployment assistance:
- Check logs: `docker-compose logs -f`
- Review documentation
- Open an issue on GitHub
