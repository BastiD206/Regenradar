# Regenradar

Eigener Regenradar für die Wetter-App, erzeugt aus den offenen Daten des Deutschen Wetterdienstes (DWD):
Messung der letzten Stunde plus 2-Stunden-Prognose, alle 10 Minuten neu.

## Einrichten (einmalig)
1. Auf GitHub ein **neues öffentliches Repository** mit dem Namen `Regenradar` anlegen.
2. Die Ordner `radar` und `.github` aus diesem Paket hochladen (Add file → Upload files).
   Falls `.github` beim Hochladen fehlt: Add file → Create new file, als Namen `.github/workflows/radar.yml` eingeben und den Inhalt der Datei einfügen.
3. Settings → Pages → **Source: GitHub Actions**.
4. Reiter Actions → „Regenradar“ → **Run workflow**.
5. Nach dem grünen Haken muss sich `https://DEIN-NAME.github.io/Regenradar/radar/manifest.json` öffnen lassen.

Heißt das Repository anders oder ist dein Name ein anderer, in der App-Datei die Zeile `RADAR_MAPS_BASE` anpassen.

## Hinweise
- GitHub schaltet geplante Workflows in öffentlichen Repositories nach 60 Tagen ohne Aktivität ab. Dann einfach unter Actions wieder aktivieren.
- Datenbasis: Deutscher Wetterdienst (DWD), RADVOR-RV, Open Data.
