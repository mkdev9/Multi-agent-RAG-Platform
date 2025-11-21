#!/bin/bash

# Production deployment script for RAG Knowledge Platform

set -e

ENVIRONMENT=${1:-staging}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-your-registry.com}
IMAGE_TAG=${IMAGE_TAG:-latest}

echo "🚀 Deploying RAG Knowledge Platform to ${ENVIRONMENT}..."

# Build and tag Docker image
echo "🐳 Building Docker image..."
docker build -t rag-platform:${IMAGE_TAG} .
docker tag rag-platform:${IMAGE_TAG} ${DOCKER_REGISTRY}/rag-platform:${IMAGE_TAG}

# Push to registry
echo "📤 Pushing to registry..."
docker push ${DOCKER_REGISTRY}/rag-platform:${IMAGE_TAG}

# Deploy based on environment
case ${ENVIRONMENT} in
    "kubernetes"|"k8s")
        echo "☸️  Deploying to Kubernetes..."
        ./scripts/deploy-k8s.sh ${IMAGE_TAG}
        ;;
    "aws")
        echo "☁️  Deploying to AWS ECS..."
        ./scripts/deploy-aws.sh ${IMAGE_TAG}
        ;;
    "azure")
        echo "☁️  Deploying to Azure..."
        ./scripts/deploy-azure.sh ${IMAGE_TAG}
        ;;
    *)
        echo "🐳 Deploying with Docker Compose..."
        export IMAGE_TAG=${IMAGE_TAG}
        docker-compose -f docker-compose.prod.yml up -d
        ;;
esac

echo "✅ Deployment completed!"