Write-Host "Starting all containers..."
docker-compose up -d
Write-Host "Containers are running."
python run.py
