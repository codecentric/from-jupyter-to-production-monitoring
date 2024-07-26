FROM python:3.12-slim

# Install Python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Configure workdir
WORKDIR /home/dashboard
# Configure PYTHONPATH environment variable
ENV PYTHONPATH=/home/dashboard

# Launch streamlit application on start up
CMD ["streamlit", "run", "app.py"]
