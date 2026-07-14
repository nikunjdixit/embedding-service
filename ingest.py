import json
import requests
import os

# ---- Config ----
PRODUCTS_JSON = "../products.json"
EMBEDDING_URL = "http://localhost:8000/embed"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "products"

# ---- Load product metadata ----
with open(PRODUCTS_JSON) as f:
    products = json.load(f)

print(f"📦 Found {len(products)} products to ingest\n")

success_count = 0
fail_count = 0

# ---- Loop through each product ----
for p in products:
    product_id = p["id"]
    local_path = os.path.join("..", p["localPath"])

    if not os.path.exists(local_path):
        print(f"❌ [{product_id}] Image not found: {local_path}")
        fail_count += 1
        continue

    try:
        # 1. Send image to embedding service
        with open(local_path, "rb") as img_file:
            files = {"file": (os.path.basename(local_path), img_file, "image/jpeg")}
            resp = requests.post(EMBEDDING_URL, files=files, timeout=30)
            resp.raise_for_status()
            vector = resp.json()["vector"]

        # 2. Insert into Qdrant
        qdrant_payload = {
            "points": [
                {
                    "id": product_id,
                    "vector": vector,
                    "payload": {
                        "url": p["url"],
                        "productName": p["productName"],
                        "category": p["category"]
                    }
                }
            ]
        }
        qdrant_resp = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
            json=qdrant_payload,
            timeout=10
        )
        qdrant_resp.raise_for_status()

        print(f"✅ [{product_id}] {p['productName']}")
        success_count += 1

    except Exception as e:
        print(f"❌ [{product_id}] Failed: {str(e)}")
        fail_count += 1

# ---- Summary ----
print(f"\n📊 Summary: {success_count} success, {fail_count} failed")