FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /dashboard

WORKDIR /dashboard
RUN uv sync --frozen --no-cache

# Launch streamlit application on start up
CMD ["/dashboard/.venv/bin/streamlit", "run", "app.py"]
