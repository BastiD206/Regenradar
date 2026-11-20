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

ROOT = "https://opendata.dwd.de/weather/radar/"
BASE = ROOT + "composite/rv/"
BOUNDS = (46.5, 4.5, 55.5, 16.5)      # Süd, West, Nord, Ost
WIDTH = 1000                          # Pixelbreite der Bilder
PAST_MIN = 60                         # so weit zurück werden Messbilder geholt
UTC = dt.timezone.utc
HEADERS = {"User-Agent": "Wetterapp-Regenradar/1.0 (GitHub Actions)"}
DEFAULT_PROJ = "+proj=stere +lat_0=90 +lon_0=10 +lat_ts=60 +a=6378137 +b=6356752.3142 +x_0=0 +y_0=0"

# Farbskala (Radar-typisch): hellblau -> blau -> grün -> gelb -> orange -> rot -> violett
STOPS = [[0.00, "#A5E3FF"], [0.10, "#5BB8FF"], [0.20, "#2F7BFF"], [0.33, "#22C55E"],
         [0.45, "#FACC15"], [0.61, "#FB923C"], [0.80, "#EF4444"], [1.00, "#C026D3"]]


# Regenintensität: unten grob, oben fein, bis 500 mm/h (extreme Gewitter)
EDGES = [0.1, 0.5, 1, 2, 4, 7, 12, 20, 30, 45, 65, 90, 120, 160, 210, 280, 370]
COLORS = ["#8FD8FF", "#3FA9F5", "#1E6FE8", "#22B04A", "#8BD02E", "#F4E21F", "#F7B718", "#F58A1F", "#EF5A1E",
          "#E0261F", "#B3121B", "#8A0C25", "#D81B8C", "#E65CCB", "#C77DFF", "#9B51E0", "#7B2FBE"]
RMAX, RGAMMA = 500.0, 0.3
STOPS_R = [[0.0, COLORS[0]]] + [[round((e / RMAX) ** RGAMMA, 4), c] for e, c in zip(EDGES, COLORS)] + [[1.0, "#4B1A8A"]]


def param(title, unit, vmax, gamma, stops=None, vmin=0, **extra):
    d = {"title": title, "unit": unit, "min": vmin, "max": vmax, "gamma": gamma,
         "decimals": 1, "stops": stops or STOPS}
    d.update(extra)
    return d


def stops_from(edges, colors, vmin, vmax, gamma, last):
    st = [[round(((e - vmin) / (vmax - vmin)) ** gamma, 4), c] for e, c in zip(edges, colors)]
    if st[0][0] > 0:
        st.insert(0, [0.0, colors[0]])
    return st + [[1.0, last]]


RATE = dict(stops=STOPS_R, edges=EDGES, colors=COLORS)
# Hagel-Indikator aus der maximalen Reflektivität (dBZ)
HAIL_EDGES = [30, 35, 40, 45, 50, 55, 60, 65, 70]
HAIL_COLORS = ["#3FA9F5", "#22B04A", "#8BD02E", "#F4E21F", "#F7B718", "#EF5A1E", "#E0261F", "#D81B8C", "#7B2FBE"]
# Gewitterkern: vertikal integrierter Flüssigwassergehalt (kg/m²)
VIL_EDGES = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]
VIL_COLORS = ["#8FD8FF", "#3FA9F5", "#1E6FE8", "#22B04A", "#8BD02E", "#F4E21F", "#F7B718", "#F58A1F", "#EF5A1E", "#E0261F", "#B3121B", "#D81B8C"]
# Doppler: Windgeschwindigkeit entlang der Blickrichtung des Radars (m/s), negativ = auf den Radar zu
DOP_EDGES = list(range(-40, 40, 4))
DOP_COLORS = ["#0B6E3A", "#138A45", "#1EA654", "#2DC066", "#4CD57F", "#6FE49B", "#96EFB7", "#B9F5CF", "#D5FAE3", "#EAFCF1",
              "#FDEEEE", "#FBD5D5", "#F9B8B8", "#F69595", "#F27070", "#EC4C4C", "#E02A2A", "#CC1717", "#AD0D0D", "#820808"]
PARAMS = {
    "radar": param("Regenradar", "mm/h", RMAX, RGAMMA, **RATE),
    "radar250": param("Radar 250 m", "mm/h", RMAX, RGAMMA, **RATE,
                      note="250-m-Raster aus der maximalen Reflektivität, in mm/h umgerechnet (Z-R-Beziehung)"),
    "hail250": param("Hagel-Indikator", "dBZ", 75, 1.0, vmin=20, thr=30, edges=HAIL_EDGES, colors=HAIL_COLORS,
                     stops=stops_from(HAIL_EDGES, HAIL_COLORS, 20, 75, 1.0, "#4B1A8A"),
                     note="Ab etwa 55 dBZ ist Hagel möglich – Indikator aus der Radarreflektivität, keine Hagelmessung"),
    "doppler": param("Dopplerradar", "m/s", 40, 1.0, vmin=-40, thr=-1e9, edges=DOP_EDGES, colors=DOP_COLORS,
                     stops=stops_from(DOP_EDGES, DOP_COLORS, -40, 40, 1.0, "#5E0505"),
                     note="Wind entlang der Radarstrahlen: grün = auf den nächsten Radar zu, rot = vom Radar weg"),
    "vil": param("Gewitterkern (VIL)", "kg/m²", 80, 0.6, thr=1, edges=VIL_EDGES, colors=VIL_COLORS,
                 stops=stops_from(VIL_EDGES, VIL_COLORS, 0, 80, 0.6, "#8A0C6B"),
                 note="Vertikal integrierter Flüssigwassergehalt: hohe Werte zeigen kräftige Gewitterzellen"),
    "sum_past1h": param("Regen letzte Stunde", "mm", 150, 0.4),
    "sum_next2h": param("Regen nächste 2 h", "mm", 200, 0.4),
    "max_next2h": param("Starkregen (max. nächste 2 h)", "mm/h", RMAX, RGAMMA, **RATE),
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


PALETTES = {}


def palette_for(p):
    key = id(p)
    if key not in PALETTES:
        PALETTES[key] = make_palette(p["stops"])
    return PALETTES[key]


def encode_idx(values, p):
    """Messwerte -> Palettenindex (0 = transparent). Index = 1 + 254 * ((v-min)/(max-min))^gamma"""
    v = np.asarray(values, np.float32)
    ok = np.isfinite(v) & (v >= p.get("thr", THRESHOLD))
    t = np.clip((np.where(ok, v, p["min"]) - p["min"]) / (p["max"] - p["min"]), 0, 1) ** p["gamma"]
    return np.where(ok, 1 + np.round(254 * t), 0).astype(np.uint8)


def encode(values, p):
    return palette_for(p)[encode_idx(values, p)]


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
def build_remapper(geo, shape, bounds=BOUNDS, width=WIDTH, chunk=400):
    S, W, N, E = bounds
    my = lambda la: np.log(np.tan(np.pi / 4 + np.radians(la) / 2))
    height = int(round(width * (my(N) - my(S)) / np.radians(E - W)))
    lons = np.linspace(W, E, width)
    ys_m = np.linspace(my(N), my(S), height)
    lats = np.degrees(2 * np.arctan(np.exp(ys_m)) - np.pi / 2)

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
    flat = np.full(height * width, -1, np.int64)
    for r0 in range(0, height, chunk):
        LON, LAT = np.meshgrid(lons, lats[r0:r0 + chunk])
        X, Y = tr.transform(LON, LAT)
        j = np.floor((X - xl) / xscale).astype(np.int64)
        i = np.floor((yt - Y) / yscale).astype(np.int64)
        ok = (i >= 0) & (i < ysize) & (j >= 0) & (j < xsize)
        flat[r0 * width:r0 * width + ok.size] = np.where(ok, i * xsize + j, -1).ravel()
    sel = np.flatnonzero(flat >= 0)
    src = flat[sel]
    cover = sel.size / flat.size
    log(f"Ausschnitt: {width}x{height} Pixel, davon {cover * 100:.0f} % mit Radardaten")
    if cover < 0.3:
        log("WARNUNG: Nur wenig Überdeckung - Georeferenzierung prüfen!")

    def remap(arr):
        out = np.full(height * width, np.nan, np.float32)
        out[sel] = np.asarray(arr).ravel()[src]
        return out.reshape(height, width)
    return remap


def save_png(arr, p, path):
    """Als Palettenbild mit Transparenz speichern (deutlich kleiner als RGBA)"""
    pal = palette_for(p)
    im = Image.fromarray(encode_idx(arr, p))
    im.putpalette(pal[:, :3].astype(np.uint8).flatten().tolist())
    im.save(path, optimize=True, transparency=bytes(int(a) for a in pal[:, 3]))


def iso(t):
    return t.isoformat().replace("+00:00", "Z")



# ---------- Zusatzprodukte: Radar 250 m, Hagel-Indikator, Gewitterkern, Doppler ----------
BOUNDS250 = (47.1, 5.5, 55.2, 15.7)   # engerer Ausschnitt für das 250-m-Raster
WIDTH250 = 3000
N250 = 7                              # Bilder für 250 m (30 Minuten)
DBZ_MIN = 8.0                         # darunter kein Regen (Rauschen)
WIDTH_DOP = 1400
SITE_RANGE_M = 150000.0


def read_composite(raw):
    """Einzelnes ODIM-Komposit -> (Werte als float32, Geo-Angaben, Zeitpunkt). Fehlwerte = NaN"""
    with h5py.File(io.BytesIO(raw), "r") as h:
        w = h["where"].attrs
        geo = {k: attr(w, k) for k in ("projdef", "xsize", "ysize", "xscale", "yscale", "UL_lon", "UL_lat", "LL_lon", "LL_lat")}
        wh = h["what"].attrs
        when = None
        try:
            when = dt.datetime.strptime(str(attr(wh, "date")) + str(attr(wh, "time"))[:4], "%Y%m%d%H%M").replace(tzinfo=UTC)
        except Exception:
            pass
        g = h["dataset1"]["data1"]
        gw = g["what"].attrs
        raw_arr = g["data"][()]
        gain, offset = float(attr(gw, "gain", 1.0)), float(attr(gw, "offset", 0.0))
        nodata, undetect = attr(gw, "nodata"), attr(gw, "undetect")
        a = raw_arr.astype(np.float32) * np.float32(gain) + np.float32(offset)
        if nodata is not None:
            a[raw_arr == nodata] = np.nan
        if undetect is not None:
            a[raw_arr == undetect] = np.nan
        return a, geo, when, str(attr(gw, "quantity", ""))


def dbz_to_rate(dbz):
    """Reflektivität -> Regenrate in mm/h (Marshall-Palmer: Z = 200 * R^1,6)"""
    with np.errstate(all="ignore"):
        rate = (10.0 ** (dbz / 10.0) / 200.0) ** (1.0 / 1.6)
    return np.where(np.isfinite(dbz) & (dbz >= DBZ_MIN), rate, np.nan).astype(np.float32)


def list_hd5(sub, pattern):
    html = get(ROOT + sub + "/").text
    return sorted(set(re.findall(pattern, html)))


def product_250(out, manifest):
    stamps = list_hd5("composite/dmax", r"composite_dmax_(\d{8}_\d{4})-hd5")
    if not stamps:
        raise RuntimeError("keine dmax-Dateien gefunden")
    frames, latest_dbz, remap = [], None, None
    for s in stamps[-N250:]:
        try:
            dbz, geo, when, qty = read_composite(get(f"{ROOT}composite/dmax/composite_dmax_{s}-hd5").content)
        except Exception as e:
            log("  dmax", s, "übersprungen:", repr(e))
            continue
        if remap is None:
            log(f"dmax: Größe {dbz.shape}, Größe der Pixel {geo.get('xscale')} m, Größe: {qty}")
            remap = build_remapper(geo, dbz.shape, BOUNDS250, WIDTH250)
        t = dt.datetime.strptime(s, "%Y%m%d_%H%M").replace(tzinfo=UTC)
        name = f"r250_{s}.png"
        save_png(remap(dbz_to_rate(dbz)), PARAMS["radar250"], os.path.join(out, name))
        frames.append({"valid": iso(t), "file": name, "kind": "obs"})
        latest_dbz, latest_t = dbz, t
    if not frames:
        raise RuntimeError("keine dmax-Datei lesbar")
    manifest["params"]["radar250"] = {**PARAMS["radar250"], "bounds": [[BOUNDS250[0], BOUNDS250[1]], [BOUNDS250[2], BOUNDS250[3]]], "frames": frames}
    save_png(remap(latest_dbz), PARAMS["hail250"], os.path.join(out, "hail250.png"))
    manifest["params"]["hail250"] = {**PARAMS["hail250"], "bounds": [[BOUNDS250[0], BOUNDS250[1]], [BOUNDS250[2], BOUNDS250[3]]],
                                      "frames": [{"valid": iso(latest_t), "file": "hail250.png"}]}


def product_vil(out, manifest):
    vil, geo, when, qty = read_composite(get(ROOT + "composite/vii/composite_VII_latest-hd5").content)
    log(f"vii: Größe {vil.shape}, Größe: {qty}")
    remap = build_remapper(geo, vil.shape)
    save_png(remap(vil), PARAMS["vil"], os.path.join(out, "vil.png"))
    manifest["params"]["vil"] = {**PARAMS["vil"], "frames": [{"valid": iso(when or dt.datetime.now(UTC)), "file": "vil.png"}]}


def read_sweep(raw):
    """Einzelner Radar-Scan in Polarkoordinaten -> Standort, Strahlen, Entfernungsbins, Geschwindigkeit (m/s)"""
    with h5py.File(io.BytesIO(raw), "r") as h:
        w = h["where"].attrs
        site = {"lat": float(attr(w, "lat")), "lon": float(attr(w, "lon")), "height": float(attr(w, "height", 0.0))}
        ds = h["dataset1"]
        dw = ds["where"].attrs
        nb, nr = int(attr(dw, "nbins")), int(attr(dw, "nrays"))
        rstart, rscale = float(attr(dw, "rstart", 0.0)) * 1000.0, float(attr(dw, "rscale"))
        elev = float(attr(dw, "elangle", 0.5))
        g = ds["data1"]
        gw = g["what"].attrs
        raw_arr = g["data"][()]
        gain, offset = float(attr(gw, "gain", 1.0)), float(attr(gw, "offset", 0.0))
        v = raw_arr.astype(np.float32) * np.float32(gain) + np.float32(offset)
        for key in ("nodata", "undetect"):
            q = attr(gw, key)
            if q is not None:
                v[raw_arr == q] = np.nan
        az, method = None, "gleichmäßig"
        if "how" in ds:
            hw = ds["how"].attrs
            if "startazA" in hw and "stopazA" in hw:
                sa, ea = np.asarray(hw["startazA"], float), np.asarray(hw["stopazA"], float)
                if sa.size == nr:
                    az, method = (sa + ((ea - sa) % 360.0) / 2.0) % 360.0, "startazA/stopazA"
        if az is None:
            az = (np.arange(nr) + 0.5) * 360.0 / nr
        when = None
        try:
            ww = h["what"].attrs
            when = dt.datetime.strptime(str(attr(ww, "date")) + str(attr(ww, "time"))[:4], "%Y%m%d%H%M").replace(tzinfo=UTC)
        except Exception:
            pass
    return {"site": site, "v": v, "az": az, "rstart": rstart, "rscale": rscale, "elev": elev, "nb": nb, "nr": nr, "when": when, "method": method}


def product_doppler(out, manifest):
    S, W, N, E = BOUNDS
    my = lambda la: np.log(np.tan(np.pi / 4 + np.radians(la) / 2))
    height = int(round(WIDTH_DOP * (my(N) - my(S)) / np.radians(E - W)))
    lons = np.linspace(W, E, WIDTH_DOP)
    lats = np.degrees(2 * np.arctan(np.exp(np.linspace(my(N), my(S), height))) - np.pi / 2)
    LON, LAT = np.meshgrid(lons, lats)
    sites = [x.strip("/") for x in re.findall(r'href="([a-z0-9]+/)"', get(ROOT + "sites/sweep_pcp_v/").text)]
    log("Doppler: Standorte:", sites)
    best = np.full(LON.shape, np.inf, np.float32)
    val = np.full(LON.shape, np.nan, np.float32)
    newest, used = None, 0
    for s in sites:
        try:
            base = f"{ROOT}sites/sweep_pcp_v/{s}/hdf5/filter_simple/"
            names = [x for x in re.findall(r'href="([^"]+)"', get(base).text) if not x.startswith("..") and not x.startswith("?")]
            sw = read_sweep(get(base + names[-1]).content)
            tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(
                f"+proj=aeqd +lat_0={sw['site']['lat']} +lon_0={sw['site']['lon']} +x_0=0 +y_0=0 +datum=WGS84 +units=m"), always_xy=True)
            X, Y = tr.transform(LON, LAT)
            rng = np.hypot(X, Y) / max(np.cos(np.radians(sw["elev"])), 0.5)
            azp = np.degrees(np.arctan2(X, Y)) % 360.0
            order = np.argsort(sw["az"]); azs = sw["az"][order]; nr = sw["nr"]
            pos = np.searchsorted(azs, azp)
            left, right = (pos - 1) % nr, pos % nr
            dl = np.abs((azp - azs[left] + 180.0) % 360.0 - 180.0)
            dr = np.abs((azp - azs[right] + 180.0) % 360.0 - 180.0)
            ray = order[np.where(dl <= dr, left, right)]
            b = np.floor((rng - sw["rstart"]) / sw["rscale"]).astype(np.int64)
            ok = (b >= 0) & (b < sw["nb"]) & (rng <= SITE_RANGE_M)
            vv = np.full(LON.shape, np.nan, np.float32)
            vv[ok] = sw["v"][ray[ok], b[ok]]
            take = np.isfinite(vv) & (rng < best)
            val[take] = vv[take]; best[take] = rng[take]
            used += 1
            if sw["when"] and (newest is None or sw["when"] > newest):
                newest = sw["when"]
            if used == 1:
                log(f"Doppler: {s}: {sw['nr']} Strahlen x {sw['nb']} Bins, Bin {sw['rscale']} m, Strahlwinkel aus {sw['method']}")
        except Exception as e:
            log("  Doppler", s, "übersprungen:", repr(e))
    if not used:
        raise RuntimeError("kein Radarstandort lesbar")
    log(f"Doppler: {used} Standorte, {np.isfinite(val).mean() * 100:.0f} % der Fläche mit Werten")
    save_png(val, PARAMS["doppler"], os.path.join(out, "doppler.png"))
    manifest["params"]["doppler"] = {**PARAMS["doppler"], "frames": [{"valid": iso(newest or dt.datetime.now(UTC)), "file": "doppler.png"}]}


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

    import traceback
    for name, fn in (("Radar 250 m + Hagel", product_250), ("Gewitterkern (VIL)", product_vil), ("Doppler", product_doppler)):
        try:
            fn(args.out, manifest)
        except Exception:
            log(f"FEHLER bei {name}:")
            log(traceback.format_exc())
    order = ["radar", "radar250", "hail250", "doppler", "vil", "sum_past1h", "sum_next2h", "max_next2h"]
    manifest["params"] = {k: manifest["params"][k] for k in order if k in manifest["params"]}

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    log("Fertig:", args.out, "->", ", ".join(manifest["params"]))


if __name__ == "__main__":
    main()
