# RAG Knowledge Platform

A production-ready, scalable RAG (Retrieval-Augmented Generation) knowledge platform with multi-agent orchestration capabilities.

## 🚀 Features

### Core Capabilities
- **Multi-format Document Ingestion**: PDF, text, markdown, CSV, JSON, and URL content
- **Advanced RAG System**: Vector similarity search with metadata filtering
- **Multi-Agent Orchestration**: LangChain-based agent workflows for complex tasks
- **Multiple LLM Providers**: OpenAI, Azure OpenAI, AWS Bedrock, Hugging Face
- **Flexible Vector Storage**: ChromaDB and FAISS support with abstraction for Pinecone/Weaviate

### Enterprise Features
- **Production-Ready Architecture**: FastAPI, PostgreSQL, Redis, containerized deployment
- **Security**: JWT authentication, RBAC, rate limiting, CORS protection
- **Scalability**: Kubernetes deployment, horizontal scaling, cloud-native design
- **Monitoring**: Prometheus metrics, Grafana dashboards, structured logging
- **CI/CD**: GitHub Actions, Infrastructure as Code (Terraform)

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- OpenAI API key (or other LLM provider)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd rag-knowledge-platform
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 3. Start Services
```bash
docker-compose up -d
```

### 4. Create First User and Start Using
```bash
# The setup script will show you all available services
# Access API documentation at http://localhost:8000/docs
```

## 📖 API Documentation

Access the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🌩️ Deployment Options

- **Local Development**: `python -m uvicorn src.main:app --reload`
- **Docker Compose**: `docker-compose up -d`
- **Kubernetes**: `kubectl apply -f k8s/`
- **AWS (Terraform)**: `cd terraform && terraform apply`

## 🛡️ Security & Production Ready

- JWT authentication with refresh tokens
- Role-based access control
- Rate limiting and CORS protection
- Comprehensive monitoring and logging
- Infrastructure as Code
- Container security best practices

## 🧪 Multi-Agent Workflows

Support for sophisticated workflows:
- **Research**: Planning → Research → Synthesis → Validation
- **Analysis**: Data gathering → Analysis
- **Summarization**: Content extraction → Summary
- **Content Generation**: Research → Creation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run code quality checks
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.