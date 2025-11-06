FROM python:3.13-slim
WORKDIR /app
COPY ../env.py /app/env.py
RUN pip install --no-cache-dir docker psutil GPUtil pandas numpy pika
CMD ["python", "env.py"]
