# Finance Analysis AI ENGINE API

FastAPI based finance analysis ai engine

## Last Version v0.2.2 Input Preprocessing Advanced

- PDF page relevance detector
- Image and Pdf finance region detector

## Features

#  Stage 1 Classification

- PDF vs image detection
- Real PDF vs scanned PDF classification
- Screenshot vs camera photo classification
- PostgreSQL metadata persistence
- Redis/Celery pipeline dispatch
- Local filesystem object storage
- Docker Compose local environment
- Health checks for API, PostgreSQL and Redis
- uv based modern Python package management

#  Stage 2 Preprocessing

- PDF to Image 
- Image quailty updates

## Local development

```bash
uv sync --all-groups
uv run uvicorn app.main:app --reload