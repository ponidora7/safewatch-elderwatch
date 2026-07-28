import os
import sys
import time
import requests
from dotenv import load_dotenv

# Resolve paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

RAILWAY_DOMAIN = os.environ.get("RAILWAY_STATIC_URL")
HF_API_URL = os.environ.get("HF_API_URL")
HF_TOKEN = os.environ.get("HF_TOKEN")

print("--- SafeWatch Keep-Alive Routine ---")
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# 1. Ping Local Backend
local_backend_url = "http://localhost:8000/health"
print(f"\n[INFO] Pinging local backend: {local_backend_url}")
try:
    start = time.time()
    response = requests.get(local_backend_url, timeout=5)
    elapsed = time.time() - start
    print(f"   Status Code: {response.status_code}")
    print(f"   Response Time: {elapsed:.4f}s")
    if response.status_code == 200:
        print("   [SUCCESS] Local backend is ACTIVE and responding.")
    else:
        print("   [WARNING] Local backend returned non-200 response.")
except Exception as e:
    print(f"   [ERROR] Local backend ping failed: {e}")

# 2. Ping Railway Production Backend (if configured)
if RAILWAY_DOMAIN:
    if not RAILWAY_DOMAIN.startswith("http"):
        backend_url = f"https://{RAILWAY_DOMAIN}/health"
    else:
        backend_url = f"{RAILWAY_DOMAIN}/health"
    print(f"\n[INFO] Pinging production backend: {backend_url}")
    try:
        start = time.time()
        response = requests.get(backend_url, timeout=5)
        elapsed = time.time() - start
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {elapsed:.4f}s")
        if response.status_code == 200:
            print("   [SUCCESS] Production backend is ACTIVE and responding.")
        else:
            print("   [WARNING] Production backend returned non-200 response.")
    except Exception as e:
        print(f"   [ERROR] Production backend ping failed: {e}")

# 3. Ping Hugging Face Endpoint
if HF_API_URL:
    print(f"\n[INFO] Pinging Hugging Face model: {HF_API_URL}")
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
    try:
        start = time.time()
        # Send empty post to trigger container wakeup (Hugging Face Inference API uses POST)
        response = requests.post(HF_API_URL, headers=headers, json={}, timeout=5)
        elapsed = time.time() - start
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {elapsed:.4f}s")
        if response.status_code in [200, 400]:
            print("   [SUCCESS] Hugging Face service is ACTIVE.")
        elif response.status_code == 503:
            print("   [WAIT] Hugging Face service is WAKING UP (loading model)...")
        else:
            print(f"   [WARNING] Hugging Face service returned unexpected status: {response.status_code}")
    except Exception as e:
        print(f"   [ERROR] Hugging Face ping failed: {e}")
else:
    print("\n[INFO] Hugging Face model URL (HF_API_URL) is not configured in .env.")

print("\n--- Keep-Alive Routine Completed ---")
