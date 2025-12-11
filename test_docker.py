import docker

try:
    client = docker.from_env(timeout=90)
    client.ping()
    print("SUCCESS: Connected to Docker Desktop!")
    print("Docker version:", client.version()["Version"])
except Exception as e:
    print("FAILED:", e)