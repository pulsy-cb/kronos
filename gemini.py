import requests

GAMMA = "https://gamma-api.polymarket.com"

# Test 1 — tri décroissant
resp = requests.get(f"{GAMMA}/events", params={
    "closed": "true",
    "limit": 5,
    "offset": 0,
    "order": "endDate",
    "ascending": "false",
})
print("=== TEST 1 (tri décroissant) ===")
for e in resp.json():
    print(e.get("slug"), "|", e.get("endDate"))

# Test 2 — filtre tag crypto
resp2 = requests.get(f"{GAMMA}/events", params={
    "closed": "true",
    "limit": 5,
    "tag": "crypto",
})
print("\n=== TEST 2 (tag=crypto) ===")
for e in resp2.json():
    print(e.get("slug"), "|", e.get("endDate"))

# Test 3 — recherche par slug partiel
resp3 = requests.get(f"{GAMMA}/events", params={
    "slug": "btc",
    "limit": 5,
})
print("\n=== TEST 3 (slug=btc) ===")
for e in resp3.json():
    print(e.get("slug"), "|", e.get("endDate"))