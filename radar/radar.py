#!/usr/bin/env python3
"""
Eigener Regenradar aus den offenen Daten des Deutschen Wetterdienstes (DWD).

Quelle:  https://opendata.dwd.de/weather/radar/composite/rv/  (RADVOR-RV: Messung + 2-h-Prognose, 1 km, 5 Min.)
Ergebnis: farbige PNG-Karten (Web-Mercator-Raster, passend zu den HD-Karten der App) + manifest.json

Karten:
  radar        Regenradar in mm/h, 1 h zurück bis 2 h voraus (5-Minuten-Bilder)
  sum_past1h   Regensumme der letzten Stunde (mm)
  sum_next2h   prognostizierte Regensumme der nächsten 2 Stunden (mm)
  max_next2h   höchste Regenintensität in den nächsten 2 Stunden (mm/h)
"""
import argparse, datetime as dt, io, json, math, os, re, sys, tarfile, tempfile

import numpy as np
import requests
import wradlib as wrl
from PIL import Image
from scipy.spatial import cKDTree

BASE = "https://opendata.dwd.de/weather/radar/composite/rv/"
BOUNDS = (46.5, 4.5, 55.5, 16.5)      # Süd, West, Nord, Ost
WIDTH = 1000                          # Pixelbreite der Bilder
PAST_MIN = 60                         # so weit zurück werden Messbilder geholt
UTC = dt.timezone.utc
HEADERS = {"User-Agent": "Wetterapp-Regenradar/1.0 (GitHub Actions)"}

# Farbskala (Radar-typisch): hellblau -> blau -> grün -> gelb -> orange -> rot -> violett
STOPS = [[0.00, "#A5E3FF"], [0.10, "#5BB8FF"], [0.20, "#2F7BFF"], [0.33, "#22C55E"],
         [0.45, "#FACC15"], [0.61, "#FB923C"], [0.80, "#EF4444"], [1.00, "#C026D3"]]


def param(title, unit, vmax, gamma):
    return {"title": title, "unit": unit, "min": 0, "max": vmax, "gamma": gamma,
            "decimals": 1, "stops": STOPS}


PARAMS = {
    "radar": param("Regenradar", "mm/h", 60, 0.45),
    "sum_past1h": param("Regen letzte Stunde", "mm", 50, 0.5),
    "sum_next2h": param("Regen nächste 2 h", "mm", 60, 0.5),
    "max_next2h": param("Starkregen (max. nächste 2 h)", "mm/h", 60, 0.45),
}
THRESHOLD = 0.1   # darunter: transparent


def log(*a):
    print(*a, flush=True)


def get(url):
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r


def parse_stamp(s):
    return dt.datetime.strptime(s, "%y%m%d%H%M").replace(tzinfo=UTC)


# ---------- Farbpalette (identisch zur Dekodierung in der App) ----------
def hexrgba(c):
    c = c.lstrip("#")
    v = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    v.append(int(c[6:8], 16) if len(c) >= 8 else 255)
    return np.array(v, float)


def make_palette(stops):
    st = [(t, hexrgba(c)) for t, c in stops]
    pal = np.zeros((256, 4), np.uint8)          # Index 0 = transparent
    for i in range(1, 256):
        t = (i - 1) / 254
        for (t0, c0), (t1, c1) in zip(st, st[1:]):
            if t0 <= t <= t1:
                f = 0 if t1 == t0 else (t - t0) / (t1 - t0)
                pal[i] = np.round(c0 + (c1 - c0) * f)
                break
        else:
            pal[i] = st[-1][1]
    return pal


PALETTE = make_palette(STOPS)


def encode(values, p):
    """Messwerte -> RGBA-Bild. Index = 1 + 254 * ((v-min)/(max-min))^gamma"""
    v = np.asarray(values, float)
    ok = np.isfinite(v) & (v >= THRESHOLD)
    t = np.clip((np.where(ok, v, 0) - p["min"]) / (p["max"] - p["min"]), 0, 1) ** p["gamma"]
    idx = np.where(ok, 1 + np.round(254 * t), 0).astype(np.uint8)
    return PALETTE[idx]


# ---------- Daten holen ----------
def read_rv(raw):
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(raw)
        path = f.name
    try:
        data, attrs = wrl.io.read_radolan_composite(path)
    finally:
        os.unlink(path)
    data = np.asarray(data, float)
    data[data < 0] = np.nan                       # Fehlwerte (-9999)
    seconds = attrs.get("intervalduration", 300) or 300
    return data * 3600.0 / seconds                 # mm je Intervall -> mm/h


def load_tar(stamp, only_lead0=False):
    raw = get(f"{BASE}DE1200_RV{stamp}.tar.bz2").content
    out = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:bz2") as tf:
        for m in tf.getmembers():
            hit = re.search(r"_(\d{3})(?:\.\w+)?$", m.name)
            if not m.isfile() or not hit:
                continue
            lead = int(hit.group(1))
            if only_lead0 and lead != 0:
                continue
            out[lead] = read_rv(tf.extractfile(m).read())
    return out


def radolan_lonlat(shape):
    nr, nc = shape
    try:
        g = wrl.georef.get_radolan_grid(nr, nc, wgs84=True)
    except TypeError:
        g = wrl.georef.get_radolan_grid(nr, nc)
        g = wrl.georef.reproject(g, src_crs=wrl.georef.get_radolan_projection(), trg_crs=4326) \
            if hasattr(wrl.georef, "get_radolan_projection") else g
    return g[..., 0], g[..., 1]


# ---------- Resampling auf ein Web-Mercator-Raster ----------
def build_remapper(shape):
    lon, lat = radolan_lonlat(shape)
    S, W, N, E = BOUNDS
    my = lambda la: np.log(np.tan(np.pi / 4 + np.radians(la) / 2))
    height = int(round(WIDTH * (my(N) - my(S)) / np.radians(E - W)))
    lons = np.linspace(W, E, WIDTH)
    ys = np.linspace(my(N), my(S), height)
    lats = np.degrees(2 * np.arctan(np.exp(ys)) - np.pi / 2)
    LON, LAT = np.meshgrid(lons, lats)
    k = math.cos(math.radians(51))
    tree = cKDTree(np.c_[lon.ravel() * k, lat.ravel()])
    dist, idx = tree.query(np.c_[LON.ravel() * k, LAT.ravel()], distance_upper_bound=0.03)
    mask = idx < lon.size

    def remap(arr):
        out = np.full(LON.size, np.nan)
        out[mask] = np.asarray(arr).ravel()[idx[mask]]
        return out.reshape(height, WIDTH)
    return remap


def save_png(arr, p, path):
    Image.fromarray(encode(arr, p), "RGBA").save(path, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/radar")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    listing = get(BASE).text
    stamps = sorted(set(re.findall(r"DE1200_RV(\d{10})\.tar\.bz2", listing)))
    if not stamps:
        log("Keine RV-Dateien im DWD-Verzeichnis gefunden. Anfang der Antwort:")
        log(listing[:1500])
        sys.exit(2)
    latest = stamps[-1]
    t_latest = parse_stamp(latest)
    log("Neuester RV-Lauf:", t_latest.isoformat(), "| Dateien im Verzeichnis:", len(stamps))

    # Messbilder der letzten Stunde (Analyse = Vorhersagezeit 000) + neuester Lauf mit Prognose
    obs = {}
    for s in stamps:
        t = parse_stamp(s)
        if t_latest - dt.timedelta(minutes=PAST_MIN) <= t < t_latest:
            try:
                obs[t] = load_tar(s, only_lead0=True)[0]
            except Exception as e:                  # einzelne Lücken sind okay
                log("  übersprungen", s, e)
    cur = load_tar(latest)
    if 0 not in cur:
        sys.exit("Im neuesten RV-Lauf fehlt die Analyse (Vorhersagezeit 000).")
    frames = [(t, "obs", a) for t, a in obs.items()]
    for lead, a in sorted(cur.items()):
        frames.append((t_latest + dt.timedelta(minutes=lead), "obs" if lead == 0 else "fc", a))
    frames.sort(key=lambda f: f[0])
    log("Bilder:", len(frames), "(", sum(1 for f in frames if f[1] == "obs"), "Messung,",
        sum(1 for f in frames if f[1] == "fc"), "Prognose )")

    remap = build_remapper(frames[0][2].shape)

    manifest = {
        "run": t_latest.isoformat().replace("+00:00", "Z"),
        "updated": dt.datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": "DWD RADVOR-RV · 1 km · Messung und 2-h-Prognose",
        "bounds": [[BOUNDS[0], BOUNDS[1]], [BOUNDS[2], BOUNDS[3]]],
        "params": {},
    }

    # 1) Animation
    fl = []
    for t, kind, a in frames:
        name = f"rv_{t:%Y%m%d%H%M}.png"
        save_png(remap(a), PARAMS["radar"], os.path.join(args.out, name))
        fl.append({"valid": t.isoformat().replace("+00:00", "Z"), "file": name, "kind": kind})
    manifest["params"]["radar"] = {**PARAMS["radar"], "frames": fl}

    # 2) Summen / Spitzen
    def stack(sel):
        s = np.stack(sel)
        allnan = np.all(~np.isfinite(s), axis=0)
        return s, allnan

    past = [a for t, k, a in frames if k == "obs" and t > t_latest - dt.timedelta(minutes=PAST_MIN)]
    nxt = [a for t, k, a in frames if k == "fc"]
    extra = {}
    if past:
        s, nanm = stack(past)
        tot = np.nansum(s / 12.0, axis=0) * (12.0 / len(past))   # fehlende Bilder hochrechnen
        tot[nanm] = np.nan
        extra["sum_past1h"] = tot
    if nxt:
        s, nanm = stack(nxt)
        tot = np.nansum(s / 12.0, axis=0)
        tot[nanm] = np.nan
        extra["sum_next2h"] = tot
        mx = np.nanmax(np.where(np.isfinite(s), s, -1), axis=0)
        mx[nanm] = np.nan
        extra["max_next2h"] = mx
    for key, arr in extra.items():
        name = f"{key}.png"
        save_png(remap(arr), PARAMS[key], os.path.join(args.out, name))
        t = t_latest if key == "sum_past1h" else t_latest + dt.timedelta(minutes=60)
        manifest["params"][key] = {**PARAMS[key], "frames": [
            {"valid": t.isoformat().replace("+00:00", "Z"), "file": name}]}

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    log("Fertig:", args.out, "->", ", ".join(manifest["params"]))


if __name__ == "__main__":
    main()
