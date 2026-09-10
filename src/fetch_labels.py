"""Download pole point labels from public ArcGIS layers + OSM (Overpass) into labels/*.geojson.gz"""
import json, gzip, os, sys, time, requests

CA = "/root/.ccr/ca-bundle.crt"
OUT = os.path.join(os.path.dirname(__file__), "..", "labels")
S = requests.Session(); S.verify = CA
LAYERS = {
    # name: (url, keep_fields)
    "scl_poles": ("https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/Seattle_City_Light_Poles_PROD/FeatureServer/1", ["FACILITYTYPE", "HEIGHT", "HAS_STREETLIGHT"]),
    "scl_poles_full": ("https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/services/SCL_Poles/FeatureServer/0", ["HEIGHT", "IS_STREETLIGHT", "DATEUPDATED"]),
    "redmond": ("https://maps.redmond.gov/server/rest/services/Web/RedmondDataForArcGISOnline/MapServer/31", ["d_Ownership", "d_PoleCategory", "d_PoleMaterial", "HEIGHT", "d_Status"]),
    "renton": ("https://gismaps.rentonwa.gov/as03/rest/services/Operational/TransportationSystems/MapServer/1", ["POLETYPE", "POLEHEIGHT", "LIGHTTYPE"]),
    "federalway": ("https://geoportal.cityoffederalway.com/res/rest/services/FW_Land/street_lights/FeatureServer/0", ["OWNER", "POLE_TYPE", "LAMP_TYPE"]),
    "kirkland": ("https://services2.arcgis.com/loGMwowmR0OPlOQb/arcgis/rest/services/Street_Lights_view/FeatureServer/0", ["OWNERSHIP", "MAINT"]),
    "shoreline_scl": ("https://services3.arcgis.com/q5Jezm9AgzqyE7Q6/arcgis/rest/services/SCL_Poles_in_Shoreline/FeatureServer/0", ["SUBTYPECD", "HEIGHT", "HasStreetlight"]),
    "kc_traffic_poles": ("https://services.arcgis.com/Ej0PsM5Aw677QF1W/arcgis/rest/services/POLE_POINT_1472/FeatureServer/0", ["Material", "Height_Ft", "Owner", "CurrentStatus"]),
}


def fetch_arcgis(url, fields):
    feats, off, page = [], 0, 1000
    while True:
        for attempt in range(5):
            try:
                r = S.get(url + "/query", params=dict(where="1=1", outFields="*", outSR=4326, resultOffset=off, resultRecordCount=page, f="geojson"), timeout=120)
                r.raise_for_status()
                j = r.json(); break
            except Exception as ex:
                print("retry", attempt, ex); time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError("arcgis fetch failed " + url)
        fs = j.get("features", [])
        for f in fs:
            if f.get("geometry") and f["geometry"]["type"] == "Point":
                p = f.get("properties", {})
                feats.append(dict(type="Feature", geometry=f["geometry"], properties={k: p.get(k) for k in fields if k in p}))
        off += len(fs)
        print(f"  {off}", end="\r", flush=True)
        if not fs or not (j.get("properties", {}).get("exceededTransferLimit") or j.get("exceededTransferLimit") or len(fs) == page):
            break
    return feats


def fetch_osm():
    q = '[out:json][timeout:600];(node["highway"="street_lamp"](47.08,-122.54,47.78,-121.07);node["power"="pole"](47.08,-122.54,47.78,-121.07);node["man_made"="utility_pole"](47.08,-122.54,47.78,-121.07);node["man_made"="mast"](47.08,-122.54,47.78,-121.07);node["man_made"="flagpole"](47.08,-122.54,47.78,-121.07);node["power"="tower"](47.08,-122.54,47.78,-121.07););out body;'
    for host in ["https://overpass.private.coffee/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]:
        for attempt in range(3):
            try:
                r = S.post(host, data={"data": q}, timeout=900)
                if r.status_code == 200:
                    els = r.json()["elements"]
                    feats = []
                    for e in els:
                        t = e.get("tags", {})
                        cls = t.get("highway") or t.get("power") or t.get("man_made")
                        feats.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[e["lon"], e["lat"]]), properties=dict(osm_id=e["id"], cls=cls, tags={k: v for k, v in t.items() if k in ("lamp_mount", "lamp_type", "height", "material", "support", "operator")})))
                    return feats
                print(host, r.status_code, r.text[:200])
            except Exception as ex:
                print(host, "err", ex)
            time.sleep(10)
    raise RuntimeError("overpass failed")


def save(name, feats):
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(os.path.join(OUT, name + ".geojson.gz"), "wt") as f:
        json.dump(dict(type="FeatureCollection", features=feats), f)
    print(f"{name}: {len(feats)} features")


if __name__ == "__main__":
    which = sys.argv[1:] or list(LAYERS) + ["osm"]
    for name in which:
        if os.path.exists(os.path.join(OUT, name + ".geojson.gz")):
            print(name, "exists"); continue
        if name == "osm":
            save("osm", fetch_osm())
        else:
            url, fields = LAYERS[name]
            print(name, flush=True)
            save(name, fetch_arcgis(url, fields))
