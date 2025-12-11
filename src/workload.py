# workload.py
from flask import Flask, request
import time, os, threading

app = Flask(__name__)

@app.route("/compute")
def compute():
    size_mb = int(request.args.get("size", 5))
    start = time.time()
    # Real CPU + disk work
    with open("/dev/null", "wb") as f:
        f.write(os.urandom(size_mb * 1024 * 1024))
    time.sleep(0.05)
    return {"latency_ms": (time.time() - start)*1000}