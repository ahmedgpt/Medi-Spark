FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps for numpy/xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home medispark
USER medispark

COPY --chown=medispark:medispark requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=medispark:medispark . .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/')" || exit 1

CMD ["python", "run.py"]
