FROM python:3.12-slim

# Install system packages
# RUN apt-get update && \
#     apt-get install -y \
#     apt-transport-https \
#     build-essential \
#     cmake \
#     curl \
#     gcc \
#     g++ \
#     git \
#     libpq-dev \
#     python3-dev \
#     tree \
#     sudo \
#     unzip \
#     wget \
#     && apt-get autoremove -yqq --purge \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*


# Install Python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r /requirements.txt

# Configure default user name and ID
ARG USER_ID="1337"
ARG USER_NAME="fastapi"

RUN useradd -l -m ${USER_NAME} -u ${USER_ID} && \
    chown -R ${USER_NAME} /home
USER ${USER_NAME}

# Configure workdir
WORKDIR /home/api
# Configure PYTHONPATH environment variable
ENV PYTHONPATH=/home/api

# Launch uvicorn server with FastAPI application on start up
CMD ["uvicorn", "app:app", "--app-dir", "api", "--reload", "--host", "0.0.0.0", "--port", "5010"]
