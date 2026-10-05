"""Descarga ganadores y premio individual por categoría desde melaterevancha.com.

La página se consulta por fecha (melate-resultados.php?del-dia=AAAA-MM-DD) y está
disponible desde el concurso 2764 (01/06/2014). Las fechas salen de
data/processed/melate_historico.csv (ejecutar antes src/scraper.py). Cada tabla se
identifica por el número de sorteo de su encabezado, no por la fecha consultada.

De cada página solo se guarda en caché el fragmento con las tablas de premios
(data/raw/premios/<fecha>.html), para no almacenar ~170 KB por página.

El sitio está detrás de CloudFront y bloquea (HTTP 403) las ráfagas de peticiones,
así que se descarga de una en una con pausa. Ante un 403/429 el script se detiene;
basta con volver a ejecutarlo más tarde, porque la caché conserva lo avanzado.

Uso:
    python src/scraper_premios.py              # descarga lo que falte y regenera el CSV
    python src/scraper_premios.py --pausa 6    # más lento si vuelve a bloquear
    python src/scraper_premios.py --solo-parse
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://www.melaterevancha.com/melate-resultados.php?del-dia={fecha}"
ROOT = Path(__file__).resolve().parents[1]
HIST_CSV = ROOT / "data" / "processed" / "melate_historico.csv"
RAW_DIR = ROOT / "data" / "raw" / "premios"
OUT_CSV = ROOT / "data" / "processed" / "premios_por_categoria.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (proyecto personal de análisis estadístico)"}
PRIMER_CONCURSO = 2764
SIN_DATOS = "<!-- sin premios -->"
ENCABEZADO = re.compile(r"Premios\s+Melate\s*(Revancha|Revanchita)?\s+del\s+(\d{2}/\d{2}/\d{4})\s+Sorteo\s+(\d+)")


def fragmento_premios(html: str) -> str:
    """Recorta el HTML desde el primer encabezado de premios hasta el cierre de la última tabla."""
    ini = html.find("<h4>Premios")
    if ini < 0:
        return SIN_DATOS
    fin = html.rfind("</table>") + len("</table>")
    return html[ini:fin] if fin > ini else SIN_DATOS


class Bloqueado(RuntimeError):
    """El sitio respondió 403/429: hay que parar y reintentar más tarde, no insistir."""


def descargar(session: requests.Session, fecha: str, pausa: float, reintentos: int = 3) -> bool:
    destino = RAW_DIR / f"{fecha}.html"
    if destino.exists():
        return False
    for intento in range(reintentos):
        try:
            r = session.get(URL.format(fecha=fecha), headers=HEADERS, timeout=30)
            if r.status_code in (403, 429):
                raise Bloqueado(f"HTTP {r.status_code} en {fecha}")
            r.raise_for_status()
            destino.write_text(fragmento_premios(r.content.decode("utf-8", errors="replace")), encoding="utf-8")
            time.sleep(pausa)  # ser amables con el servidor
            return True
        except requests.RequestException:
            time.sleep(5 * (intento + 1))
    print(f"  ! no se pudo descargar {fecha}")
    return False


def parsear(fecha: str) -> list[dict]:
    soup = BeautifulSoup((RAW_DIR / f"{fecha}.html").read_text(encoding="utf-8"), "lxml")
    filas = []
    for h4 in soup.find_all("h4"):
        m = ENCABEZADO.search(h4.get_text(" ", strip=True))
        if not m:
            continue
        juego = m.group(1) or "Melate"
        tabla = h4.find_next("table")
        for pos, tr in enumerate(tabla.select("tbody tr"), start=1):
            celdas = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(celdas) != 4:
                continue
            aciertos = celdas[1]
            filas.append(
                {
                    "concurso": int(m.group(3)),
                    "fecha": pd.to_datetime(m.group(2), dayfirst=True),
                    "juego": juego,
                    "categoria": pos,
                    "aciertos": aciertos,
                    "naturales": int(re.match(r"\d+", aciertos).group()),
                    "con_adicional": "adicional" in aciertos,
                    "ganadores": int(re.sub(r"\D", "", celdas[2]) or 0),
                    "premio_individual": float(re.sub(r"[^\d.]", "", celdas[3]) or 0),
                }
            )
    return filas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-parse", action="store_true")
    ap.add_argument("--pausa", type=float, default=4.0, help="segundos entre peticiones")
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    hist = pd.read_csv(HIST_CSV, parse_dates=["fecha"])
    mel = hist[(hist.juego == "Melate") & (hist.concurso >= PRIMER_CONCURSO)].set_index("concurso")["fecha"]
    fechas = sorted(mel.dt.strftime("%Y-%m-%d").unique())

    # Una sola petición a la vez: el sitio está detrás de CloudFront y bloquea ráfagas.
    with requests.Session() as session:
        try:
            if not args.solo_parse:
                pendientes = [f for f in fechas if not (RAW_DIR / f"{f}.html").exists()]
                print(f"Fechas: {len(fechas)} | pendientes: {len(pendientes)} | tiempo estimado: {len(pendientes) * args.pausa / 60:.0f} min")
                for i, f in enumerate(pendientes, start=1):
                    descargar(session, f, args.pausa)
                    if i % 50 == 0:
                        print(f"  {i}/{len(pendientes)} descargadas")

                # Concursos que no aparecieron (fecha mal capturada en la fuente): probar días vecinos
                df = pd.DataFrame([f for fecha in fechas if (RAW_DIR / f"{fecha}.html").exists() for f in parsear(fecha)])
                faltan = sorted(set(mel.index) - set(df.concurso))
                if faltan:
                    print(f"Concursos sin tabla en su fecha: {len(faltan)}; probando días vecinos…")
                    extra = {(mel[c] + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for c in faltan for d in (-3, -2, -1, 1, 2, 3)}
                    for f in sorted(extra - set(fechas)):
                        descargar(session, f, args.pausa)
                    fechas = sorted(set(fechas) | extra)
        except Bloqueado as e:
            print(f"BLOQUEADO ({e}). Se detiene la descarga; vuelve a ejecutar más tarde (la caché conserva lo avanzado).")
        fechas = sorted({p.stem for p in RAW_DIR.glob("*.html")} | set(fechas))

    df = pd.DataFrame([f for fecha in fechas if (RAW_DIR / f"{fecha}.html").exists() for f in parsear(fecha)])
    df = df.drop_duplicates(["concurso", "juego", "categoria"]).sort_values(["concurso", "juego", "categoria"])
    df.to_csv(OUT_CSV, index=False)
    faltan = sorted(set(mel.index) - set(df.concurso))
    print(f"Guardado {OUT_CSV} -> {len(df)} filas, {df.concurso.nunique()} concursos")
    print(f"Concursos sin datos de premios ({len(faltan)}): {faltan[:30]}{' …' if len(faltan) > 30 else ''}")


if __name__ == "__main__":
    main()
