FROM python:3.13-slim
WORKDIR /app
COPY ../runner.py /app/runner.py
RUN pip install --no-cache-dir pandas numpy pika psutil
CMD ["python", "runner.py"]
