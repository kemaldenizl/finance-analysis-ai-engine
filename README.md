# Finance Analysis AI ENGINE API

FastAPI based finance analysis ai engine

## Last Version v0.2.1 Input Preprocessing


## Features

- PDF vs image detection
- Real PDF vs scanned PDF classification
- Screenshot vs camera photo classification
- PostgreSQL metadata persistence
- Redis/Celery pipeline dispatch
- Local filesystem object storage
- Docker Compose local environment
- Health checks for API, PostgreSQL and Redis
- uv based modern Python package management

## Local development

```bash
uv sync --all-groups
uv run uvicorn app.main:app --reload