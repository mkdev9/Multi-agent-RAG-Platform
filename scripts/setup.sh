#!/bin/bash

# RAG Knowledge Platform Setup Script

set -e

echo "🚀 Setting up RAG Knowledge Platform..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/files data/chroma logs ssl monitoring/grafana/{dashboards,datasources}

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before running the application"
else
    echo "✅ .env file already exists"
fi

# Generate SSL certificates (self-signed for development)
echo "🔐 Generating SSL certificates..."
if [ ! -f ssl/cert.pem ]; then
    mkdir -p ssl
    openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem \
        -days 365 -nodes -subj "/CN=localhost"
    echo "✅ SSL certificates generated"
else
    echo "✅ SSL certificates already exist"
fi

# Set up Grafana datasources
echo "📊 Setting up Grafana..."
cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    access: proxy
    isDefault: true
EOF

# Create Grafana dashboard for RAG Platform
cat > monitoring/grafana/dashboards/rag-platform.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "RAG Platform Metrics",
    "tags": ["rag", "platform"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "API Requests",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF

# Build and start services
echo "🐳 Building and starting services..."
docker-compose build
docker-compose up -d postgres redis

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Run database migrations
echo "🗃️  Running database migrations..."
docker-compose run --rm api python -m alembic upgrade head

# Start all services
echo "🚀 Starting all services..."
docker-compose up -d

echo ""
echo "✅ Setup completed!"
echo ""
echo "🌐 Services:"
echo "  API: https://localhost (or http://localhost:8000)"
echo "  Database: localhost:5432"
echo "  Redis: localhost:6379"
echo "  Grafana: http://localhost:3000 (admin/admin)"
echo "  Prometheus: http://localhost:9090"
echo ""
echo "📚 Next steps:"
echo "  1. Edit .env file with your API keys and configuration"
echo "  2. Restart services: docker-compose restart"
echo "  3. Create your first user via API"
echo "  4. Upload documents and start querying!"
echo ""
echo "📖 View logs: docker-compose logs -f"
echo "🛑 Stop services: docker-compose down"