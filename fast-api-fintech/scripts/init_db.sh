#!/bin/bash
# Database initialization script

set -e

echo "🔧 Initializing FastAPI Fintech Database..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file. Please update it with your configuration."
fi

# Source environment variables
export $(cat .env | xargs)

# Check if database URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL not set in .env file"
    exit 1
fi

echo "📦 Installing dependencies..."
pip install -r requirements.txt

# For PostgreSQL, create database if it doesn't exist
if [[ $DATABASE_URL == postgresql* ]]; then
    echo "🐘 Setting up PostgreSQL database..."
    
    # Extract database name from URL
    DB_NAME=$(echo $DATABASE_URL | sed 's/.*\/\([^?]*\).*/\1/')
    
    # Check if database exists, create if not
    psql -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
        psql -U postgres -c "CREATE DATABASE $DB_NAME"
    
    echo "✅ PostgreSQL database ready"
fi

# Run database migrations (when implemented with Alembic)
# echo "🔄 Running database migrations..."
# alembic upgrade head

# Seed initial data (optional)
if [ -f "scripts/seed_data.py" ]; then
    echo "🌱 Seeding initial data..."
    python scripts/seed_data.py
fi

echo "✅ Database initialization complete!"
echo "🚀 You can now start the application with: python -m app.main"
