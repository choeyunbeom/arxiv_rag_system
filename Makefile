.PHONY: up down build logs test lint fmt seed dev api ui health clean

# ── Docker Compose ────────────────────────────────────────────
up:                ## Start all services (ChromaDB + API + UI)
	docker compose up --build -d

down:              ## Stop all services
	docker compose down

build:             ## Rebuild containers without cache
	docker compose build --no-cache

logs:              ## Tail logs from all services
	docker compose logs -f

# ── Local Development ─────────────────────────────────────────
dev:               ## Start ChromaDB + API + UI locally (no Docker for API/UI)
	docker compose up chromadb -d
	@echo "ChromaDB running on :8200"
	@echo "Run 'make api' and 'make ui' in separate terminals"

api:               ## Start FastAPI dev server with hot-reload
	uvicorn src.api.main:app --reload

ui:                ## Start Streamlit frontend
	streamlit run ui/app.py

# ── Data Pipeline ─────────────────────────────────────────────
seed:              ## Run full data pipeline: crawl → parse → index
	python -m src.ingestion.arxiv_crawler
	python -m src.ingestion.pdf_parser
	python -m src.ingestion.indexer

# ── Quality ───────────────────────────────────────────────────
test:              ## Run all tests
	pytest tests/ -v

lint:              ## Run linter
	ruff check src/ tests/

fmt:               ## Auto-fix lint issues
	ruff check --fix src/ tests/

# ── Utilities ─────────────────────────────────────────────────
health:            ## Check API health
	@curl -s http://localhost:8000/health | python -m json.tool

clean:             ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

help:              ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
