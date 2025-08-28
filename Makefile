# Makefile for RAG/MCP and Graph RAG Platform

.PHONY: help setup dev prod test clean migrate shell logs

# Default target
help:
	@echo "Available commands:"
	@echo "  setup     - Initial project setup"
	@echo "  dev       - Start development environment"
	@echo "  prod      - Start production environment"
	@echo "  test      - Run tests"
	@echo "  migrate   - Run database migrations"
	@echo "  shell     - Open Django shell"
	@echo "  logs      - Show application logs"
	@echo "  clean     - Clean up containers and volumes"
	@echo "  build     - Build Docker images"
	@echo "  stop      - Stop all services"

# Initial setup
setup:
	@echo "Setting up the project..."
	cp .env.example .env
	@echo "Please edit .env file with your configuration"
	docker-compose -f docker-compose.dev.yml build
	docker-compose -f docker-compose.dev.yml up -d postgres redis
	@echo "Waiting for database to be ready..."
	sleep 10
	docker-compose -f docker-compose.dev.yml exec web python create_migrations.py
	docker-compose -f docker-compose.dev.yml exec web python manage.py migrate
	@echo "Setup complete! Run 'make dev' to start development server"

# Development environment
dev:
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml up --build

# Start MCP server only
mcp:
	@echo "Starting MCP server..."
	python start_mcp_server.py

# Start Django app with local MCP server
dev-local:
	@echo "Starting development with local MCP server..."
	@echo "Make sure to set MCP_SERVER_URL=http://localhost:8080 in .env"
	python start_mcp_server.py &
	sleep 5
	python manage.py runserver

# Production environment
prod:
	@echo "Starting production environment..."
	docker-compose -f docker-compose.prod.yml up -d --build

# Run tests
test:
	@echo "Running tests..."
	docker-compose -f docker-compose.dev.yml exec web python -m pytest

# Run migrations
migrate:
	@echo "Running migrations..."
	docker-compose -f docker-compose.dev.yml exec web python manage.py makemigrations
	docker-compose -f docker-compose.dev.yml exec web python manage.py migrate

# Django shell
shell:
	docker-compose -f docker-compose.dev.yml exec web python manage.py shell

# View logs
logs:
	docker-compose -f docker-compose.dev.yml logs -f web

logs-prod:
	docker-compose -f docker-compose.prod.yml logs -f

# Clean up
clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose -f docker-compose.dev.yml down -v
	docker-compose -f docker-compose.prod.yml down -v
	docker system prune -f

# Build images
build:
	docker-compose -f docker-compose.dev.yml build
	docker-compose -f docker-compose.prod.yml build

# Stop services
stop:
	docker-compose -f docker-compose.dev.yml down
	docker-compose -f docker-compose.prod.yml down

# Database backup
backup-db:
	@echo "Creating database backup..."
	docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U $(DB_USER) $(DB_NAME) > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Restore database
restore-db:
	@echo "Restoring database from backup..."
	@read -p "Enter backup file name: " backup_file; \
	docker-compose -f docker-compose.prod.yml exec -T postgres psql -U $(DB_USER) $(DB_NAME) < $$backup_file

# Create superuser
createsuperuser:
	docker-compose -f docker-compose.dev.yml exec web python manage.py createsuperuser

# Collect static files
collectstatic:
	docker-compose -f docker-compose.dev.yml exec web python manage.py collectstatic --noinput

# Load sample data (if you create fixtures)
loaddata:
	docker-compose -f docker-compose.dev.yml exec web python manage.py loaddata sample_data.json

# Performance monitoring
monitor:
	@echo "Opening monitoring dashboard..."
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"

# Security scan
security-scan:
	@echo "Running security scan..."
	docker run --rm -v $(PWD):/app pyupio/safety safety check -r /app/requirements.txt