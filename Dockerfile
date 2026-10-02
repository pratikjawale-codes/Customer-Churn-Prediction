FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Train once while building the image so each container starts quickly.
RUN python run_pipeline.py \
    && rm -f data/telecom_churn_data.csv mlflow.db mlflow.db-shm mlflow.db-wal \
    && rm -rf catboost_info mlruns mlartifacts

# Expose API port
EXPOSE 5000

CMD ["python", "src/app.py"]
