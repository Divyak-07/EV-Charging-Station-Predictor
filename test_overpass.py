import requests
mirrors = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
q = '[out:json][timeout:10];node(around:500,28.6139,77.2090)["amenity"];out tags;'
for url in mirrors:
    name = url.split("/")[2]
    try:
        r = requests.post(url, data={"data": q}, timeout=20)
        data = r.json() if r.status_code == 200 else {}
        els = len(data.get("elements", []))
        print(f"  {name:45s}  status={r.status_code}  elements={els}")
    except Exception as e:
        print(f"  {name:45s}  ERROR: {e}")
