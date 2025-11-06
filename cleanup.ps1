Write-Host "Removing old containers and dangling images..."
docker system prune -f
Write-Host "Cleanup complete."
