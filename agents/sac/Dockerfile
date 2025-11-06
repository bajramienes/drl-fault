FROM python:3.13-slim
WORKDIR /app
COPY ../runner.py /app/runner.py
RUN pip install --no-cache-dir stable-baselines3 gymnasium numpy pandas pika
CMD ["python", "runner.py", "--agent=sac"]
