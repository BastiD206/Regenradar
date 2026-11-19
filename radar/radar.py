#!/usr/bin/env python3
"""
Eigener Regenradar aus den offenen Daten des Deutschen Wetterdienstes (DWD).

Quelle:  https://opendata.dwd.de/weather/radar/composite/rv/
         Dateien "composite_rv_JJJJMMTT_HHMM.tar" mit HDF5-Dateien (ODIM): Messung + 2-h-Prognose, 1 km, 5 Min.
Ergebnis: farbige PNG-Karten (Web-Mercator-Raster, passend zu den HD-Karten der App) + manifest.json

Karten:
  radar        Regenradar in mm/h, 1 h zurück bis 2 h voraus (5-Minuten-Bilder)
  sum_past1h   Regensumme der letzten Stunde (mm)
  sum_next2h   prognostizierte Regensumme der nächsten 2 Stunden (mm)
  max_next2h   höchste Regenintensität in den nächsten 2 Stunden (mm/h)
"""
import argparse, datetime as dt, io, json, math, os, re, sys, tarfile

import h5py
import numpy as np
import requests
from PIL import Image
from pyproj import CRS, Transformer

BASE = "https://opendata.dwd.de/weather/radar/composite/rv/"
BOUNDS = (46.5, 4.5, 55.5, 16.5)      # Süd, West, Nord, Ost
WIDTH = 1000                          # Pixelbreite der Bilder
PAST_MIN = 60                         # so weit zurück werden Messbilder geholt
UTC = dt.timezone.utc
HEADERS = {"User-Agent": "Wetterapp-Regenradar/1.0 (GitHub Actions)"}
DEFAULT_PROJ = "+proj=stere +lat_0=90 +lon_0=10 +lat_ts=60 +a=6378137 +b=6356752.3142 +x_0=0 +y_0=0"

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
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r


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


# ---------- HDF5 (ODIM) lesen ----------
def attr(a, k, default=None):
    if k not in a:
        return default
    v = a[k]
    if isinstance(v, np.ndarray):
        v = v.item() if v.size == 1 else v.tolist()
    if isinstance(v, (bytes, np.bytes_)):
        v = v.decode("utf-8", "replace")
    elif isinstance(v, np.generic):
        v = v.item()
    return v


def to_rate(a, qty):
    """in mm/h umrechnen"""
    q = (qty or "").upper()
    if q == "ACRR":                                   # mm je 5 Minuten
        return a * 12.0
    if q.startswith("DBZ"):                           # Reflektivität -> Z-R-Beziehung
        with np.errstate(all="ignore"):
            return np.where(np.isfinite(a), (10 ** (a / 10.0) / 200.0) ** (1 / 1.6), np.nan)
    return a                                          # RATE: schon mm/h


def read_hdf(raw):
    """-> (Liste von Datensätzen, Geo-Angaben, Beschreibung für das Log)"""
    sets, info = [], []
    with h5py.File(io.BytesIO(raw), "r") as h:
        w = h["where"].attrs if "where" in h else {}
        geo = {k: attr(w, k) for k in ("projdef", "xsize", "ysize", "xscale", "yscale",
                                       "UL_lon", "UL_lat", "LL_lon", "LL_lat")}
        info.append("where: " + json.dumps({k: v for k, v in geo.items()}, default=str))
        for dname in sorted(k for k in h.keys() if k.startswith("dataset")):
            ds = h[dname]
            dw = ds["what"].attrs if "what" in ds else {}
            for gname in sorted(k for k in ds.keys() if k.startswith("data")):
                g = ds[gname]
                if "data" not in g:
                    continue
                gw = g["what"].attrs if "what" in g else {}
                raw_arr = g["data"][()]
                gain, offset = float(attr(gw, "gain", 1.0)), float(attr(gw, "offset", 0.0))
                nodata, undetect = attr(gw, "nodata"), attr(gw, "undetect")
                qty = str(attr(gw, "quantity", "") or attr(dw, "quantity", ""))
                a = raw_arr.astype(float) * gain + offset
                if nodata is not None:
                    a[raw_arr == nodata] = np.nan
                if undetect is not None:
                    a[raw_arr == undetect] = 0.0
                a[a < -1e3] = np.nan
                a = to_rate(a, qty)
                a = np.where(np.isfinite(a), np.clip(a, 0, None), np.nan)
                end = (attr(dw, "enddate"), attr(dw, "endtime"))
                sets.append({"arr": a, "end": end, "qty": qty})
                fin = a[np.isfinite(a)]
                mx = f"{fin.max():.2f}" if fin.size else "nur Fehlwerte"
                info.append(f"{dname}/{gname}: qty={qty} gain={gain} offset={offset} nodata={nodata} "
                            f"undetect={undetect} shape={a.shape} end={end} max={mx}")
    return sets, geo, info


def lead_minutes(member, sets, k, base):
    """Vorhersagezeit in Minuten (aus Dateiname, sonst aus der Gültigkeitszeit)"""
    m = re.search(r"_(\d{3})(?:[-.]\w+)?$", os.path.basename(member))
    if m and len(sets) == 1:
        return int(m.group(1))
    e = sets[k]["end"]
    if e[0] and e[1]:
        try:
            t = dt.datetime.strptime(str(e[0]) + str(e[1])[:4], "%Y%m%d%H%M").replace(tzinfo=UTC)
            return int(round((t - base).total_seconds() / 60))
        except ValueError:
            pass
    return 5 * k


def load_tar(stamp, base, only_analysis=False, verbose=False):
    raw = get(f"{BASE}composite_rv_{stamp}.tar").content
    out, geo = [], None
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        if verbose:
            log(f"Inhalt von composite_rv_{stamp}.tar: {len(members)} Dateien")
            for m in members[:6]:
                log("   ", m.name, m.size)
        for m in members:
            data = tf.extractfile(m).read()
            if not data.startswith(b"\x89HDF"):
                if verbose:
                    log("   (übersprungen, kein HDF5):", m.name)
                continue
            sets, g, info = read_hdf(data)
            if verbose and m is members[0]:
                for line in info:
                    log("   ", line)
            geo = geo or g
            for k, s in enumerate(sets):
                lead = lead_minutes(m.name, sets, k, base)
                if only_analysis and lead != 0:
                    continue
                out.append((lead, s["arr"]))
    out.sort(key=lambda x: x[0])
    return out, geo


# ---------- Resampling auf ein Web-Mercator-Raster ----------
def build_remapper(geo, shape):
    S, W, N, E = BOUNDS
    my = lambda la: np.log(np.tan(np.pi / 4 + np.radians(la) / 2))
    height = int(round(WIDTH * (my(N) - my(S)) / np.radians(E - W)))
    lons = np.linspace(W, E, WIDTH)
    ys_m = np.linspace(my(N), my(S), height)
    lats = np.degrees(2 * np.arctan(np.exp(ys_m)) - np.pi / 2)
    LON, LAT = np.meshgrid(lons, lats)

    proj = geo.get("projdef") or DEFAULT_PROJ
    tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(proj), always_xy=True)
    ysize, xsize = shape
    xscale, yscale = float(geo.get("xscale") or 1000.0), float(geo.get("yscale") or 1000.0)
    if geo.get("UL_lon") is not None and geo.get("UL_lat") is not None:
        xl, yt = tr.transform(float(geo["UL_lon"]), float(geo["UL_lat"]))
    elif geo.get("LL_lon") is not None and geo.get("LL_lat") is not None:
        xl, yb = tr.transform(float(geo["LL_lon"]), float(geo["LL_lat"]))
        yt = yb + ysize * yscale
    else:
        raise RuntimeError("Keine Eckkoordinaten in den Radardaten gefunden: " + json.dumps(geo, default=str))
    X, Y = tr.transform(LON, LAT)
    j = np.floor((X - xl) / xscale).astype(int)
    i = np.floor((yt - Y) / yscale).astype(int)
    mask = (i >= 0) & (i < ysize) & (j >= 0) & (j < xsize)
    log(f"Ausschnitt: {WIDTH}x{height} Pixel, davon {mask.mean() * 100:.0f} % mit Radardaten")
    if mask.mean() < 0.3:
        log("WARNUNG: Nur wenig Überdeckung - Georeferenzierung prüfen!")

    def remap(arr):
        out = np.full(mask.shape, np.nan)
        out[mask] = np.asarray(arr)[i[mask], j[mask]]
        return out
    return remap


def save_png(arr, p, path):
    Image.fromarray(encode(arr, p), "RGBA").save(path, optimize=True)


def iso(t):
    return t.isoformat().replace("+00:00", "Z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/radar")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    listing = get(BASE).text
    stamps = sorted(set(re.findall(r"composite_rv_(\d{8}_\d{4})\.tar", listing)))
    if not stamps:
        log("Keine RV-Dateien im DWD-Verzeichnis gefunden. Anfang der Antwort:")
        log(listing[:1500])
        sys.exit(2)
    parse = lambda s: dt.datetime.strptime(s, "%Y%m%d_%H%M").replace(tzinfo=UTC)
    log("Dateien im DWD-Verzeichnis:", len(stamps), "| neueste:", stamps[-1])

    # neueste vollständige Datei (die allerletzte ist manchmal noch im Upload)
    cur, geo, latest = None, None, None
    for s in reversed(stamps[-4:]):
        try:
            cur, geo = load_tar(s, parse(s), verbose=True)
            if cur and any(l == 0 for l, _ in cur):
                latest = s
                break
            log("  keine Analyse in", s)
        except Exception as e:
            log("  Datei", s, "nicht lesbar:", repr(e))
    if latest is None:
        sys.exit("Keine lesbare RV-Datei gefunden.")
    t_latest = parse(latest)
    log("Verwendeter Lauf:", t_latest.isoformat())

    # Messbilder der letzten Stunde: Analyse (Vorhersagezeit 0) der vorherigen Dateien
    obs = {}
    for s in stamps:
        t = parse(s)
        if t_latest - dt.timedelta(minutes=PAST_MIN) <= t < t_latest:
            try:
                res, _ = load_tar(s, t, only_analysis=True)
                if res:
                    obs[t] = res[0][1]
            except Exception as e:
                log("  übersprungen", s, repr(e))
    frames = [(t, "obs", a) for t, a in obs.items()]
    for lead, a in cur:
        frames.append((t_latest + dt.timedelta(minutes=lead), "obs" if lead == 0 else "fc", a))
    frames.sort(key=lambda f: f[0])
    log("Bilder:", len(frames), "(", sum(1 for f in frames if f[1] == "obs"), "Messung,",
        sum(1 for f in frames if f[1] == "fc"), "Prognose )")

    remap = build_remapper(geo, frames[0][2].shape)

    manifest = {
        "run": iso(t_latest),
        "updated": iso(dt.datetime.now(UTC)),
        "source": "DWD RADVOR-RV · 1 km · Messung und 2-h-Prognose",
        "bounds": [[BOUNDS[0], BOUNDS[1]], [BOUNDS[2], BOUNDS[3]]],
        "params": {},
    }

    # 1) Animation
    fl = []
    for t, kind, a in frames:
        name = f"rv_{t:%Y%m%d%H%M}.png"
        save_png(remap(a), PARAMS["radar"], os.path.join(args.out, name))
        fl.append({"valid": iso(t), "file": name, "kind": kind})
    manifest["params"]["radar"] = {**PARAMS["radar"], "frames": fl}

    # 2) Summen / Spitzen
    def stack(sel):
        s = np.stack(sel)
        return s, np.all(~np.isfinite(s), axis=0)

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
        manifest["params"][key] = {**PARAMS[key], "frames": [{"valid": iso(t), "file": name}]}

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    log("Fertig:", args.out, "->", ", ".join(manifest["params"]))


if __name__ == "__main__":
    main()
