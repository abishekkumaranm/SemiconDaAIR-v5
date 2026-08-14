"""
test_api.py — Quick Python Script to Test REST Ingestion Endpoint (POST /uploadImage).
"""

import json
import requests

url = "http://localhost:8000/uploadImage"
image_path = "data/clean_images/img_0.png"

metadata = {
    "wafer_id": "WAF_300MM_8921",
    "lot_id": "LOT_EUV_9942",
    "layer_id": "M1_INTERCONNECT",
    "magnification": "50000X",
    "acquisition_mode": "SEM_SECONDARY_ELECTRON",
    "resolution": "256x256",
    "sensor_settings": "1.2kV",
    "nm_per_pixel": 1.5
}

print(f"Connecting to REST API endpoint: {url}...")

try:
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/png")}
        data = {"metadata": json.dumps(metadata)}
        response = requests.post(url, files=files, data=data)

    print(f"\nHTTP Status Code: {response.status_code}")
    print("\n--- Response Telemetry Headers ---")
    for k, v in response.headers.items():
        if k.lower().startswith("x-") or k.lower() == "content-type":
            print(f"  {k}: {v}")

    print("\n--- Response Payload (JSON) ---")
    print(json.dumps(response.json(), indent=2))

except Exception as e:
    print(f"\nError: Could not connect to microservice at {url}.")
    print("Make sure you started serve.py in another terminal window first:")
    print("  python serve.py --port 8000")
    print(f"Details: {e}")
