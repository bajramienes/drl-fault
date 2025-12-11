FROM python:3.11-slim
RUN pip install flask gunicorn psutil
ADD workload.py /app/workload.py
WORKDIR /app
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "workload:app"]