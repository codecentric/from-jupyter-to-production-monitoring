FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configure default user name and ID
# ARG USER_ID="1337"
# ARG USER_NAME="fastapi"

# RUN useradd -l -m ${USER_NAME} -u ${USER_ID} && \
#     chown -R ${USER_NAME} /home
# USER ${USER_NAME}

COPY api pyproject.toml uv.lock /api
COPY data /data
COPY models /models

WORKDIR /api
RUN uv sync --frozen --no-cache

# Launch uvicorn server with FastAPI application on start up
CMD ["/api/.venv/bin/fastapi", "run", "app.py", "--port", "5010", "--host", "0.0.0.0"]
