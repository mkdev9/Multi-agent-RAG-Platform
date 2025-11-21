# RAG Knowledge Platform Makefile

.PHONY: help install setup dev test lint format docker-build docker-run k8s-deploy clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt

setup: ## Initial setup for development
	chmod +x scripts/setup.sh
	./scripts/setup.sh

dev: ## Start development server
	python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests
	pytest tests/ -v

test-coverage: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html

lint: ## Run linting
	flake8 src/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

docker-build: ## Build Docker image
	docker build -t rag-platform:latest .

docker-run: ## Run with Docker Compose
	docker-compose up -d

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-stop: ## Stop Docker services
	docker-compose down

k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f k8s/

k8s-delete: ## Delete from Kubernetes
	kubectl delete -f k8s/

db-migrate: ## Run database migrations
	alembic upgrade head

db-revision: ## Create new database revision
	alembic revision --autogenerate -m "$(MSG)"

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

monitoring: ## Start monitoring stack
	docker-compose up -d prometheus grafana

ssl: ## Generate SSL certificates for development
	mkdir -p ssl
	openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem \
		-days 365 -nodes -subj "/CN=localhost"

backup-db: ## Backup database
	docker-compose exec postgres pg_dump -U rag_user rag_db > backup_$(shell date +%Y%m%d_%H%M%S).sql

restore-db: ## Restore database (FILE=backup.sql)
	docker-compose exec -T postgres psql -U rag_user rag_db < $(FILE)

logs: ## View application logs
	docker-compose logs -f api

shell: ## Open shell in API container
	docker-compose exec api bash

docs: ## Generate API documentation
	@echo "API documentation available at:"
	@echo "  Swagger UI: http://localhost:8000/docs"
	@echo "  ReDoc: http://localhost:8000/redoc"

health: ## Check system health
	curl -f http://localhost:8000/health || exit 1
	curl -f http://localhost:8000/api/v1/admin/health || exit 1

# Development helpers
req-update: ## Update requirements.txt
	pip freeze > requirements.txt

env-example: ## Update .env.example from current .env
	grep -v "^#" .env | sed 's/=.*/=your_value_here/' > .env.example.new

# Production targets
prod-deploy: ## Deploy to production
	./scripts/deploy.sh production

staging-deploy: ## Deploy to staging
	./scripts/deploy.sh staging