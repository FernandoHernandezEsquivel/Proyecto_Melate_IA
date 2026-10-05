"""Descarga el histórico de Melate, Revancha y Revanchita desde resultadosmelate.mx.

Cada concurso tiene su propia página (https://resultadosmelate.mx/melate-<N>).
El HTML crudo se guarda en data/raw/html/ como caché, de modo que volver a
ejecutar el script solo descarga los concursos nuevos.

Uso:
    python src/scraper.py              # descarga todo lo que falte y regenera el CSV
    python src/scraper.py --solo-parse # solo regenera el CSV desde la caché
"""
from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://resultadosmelate.mx"
HISTORICO_URL = f"{BASE_URL}/melate-revancha-revanchita-historico"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "html"
OUT_CSV = ROOT / "data" / "processed" / "melate_historico.csv"
PARCHES_CSV = ROOT / "data" / "raw" / "parches_manuales.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (proyecto personal de análisis estadístico)"}
JUEGOS = ("Melate", "Revancha", "Revanchita")


def decodificar(contenido: bytes) -> str:
    try:
        return contenido.decode("utf-8")
    except UnicodeDecodeError:
        return contenido.decode("cp1252", errors="replace")


def ultimo_concurso(session: requests.Session) -> int:
    html = decodificar(session.get(HISTORICO_URL, headers=HEADERS, timeout=30).content)
    return max(int(n) for n in re.findall(r'href="/melate-(\d+)"', html))


def descargar(session: requests.Session, concurso: int, reintentos: int = 3) -> bool:
    destino = RAW_DIR / f"melate-{concurso}.html"
    if destino.exists() and destino.stat().st_size > 5000:
        return False
    for intento in range(reintentos):
        try:
            r = session.get(f"{BASE_URL}/melate-{concurso}", headers=HEADERS, timeout=30)
            r.raise_for_status()
            destino.write_bytes(r.content)
            time.sleep(0.25)  # ser amables con el servidor
            return True
        except requests.RequestException:
            time.sleep(2 * (intento + 1))
    raise RuntimeError(f"No se pudo descargar el concurso {concurso}")


def parsear(concurso: int) -> list[dict]:
    html = decodificar((RAW_DIR / f"melate-{concurso}.html").read_bytes())
    soup = BeautifulSoup(html, "lxml")

    titulo = soup.select_one(".resultadotitulo").get_text(strip=True)
    fecha = pd.to_datetime(re.search(r"(\d{2}/\d{2}/\d{4})", titulo).group(1), dayfirst=True)

    filas = []
    for bloque in soup.select(".cuerpo2 > .cuerpo1"):
        juego = bloque.select_one("[class^=filabolacab] > div").get_text(strip=True)
        if juego not in JUEGOS:
            continue
        principales = [int(t) for b in bloque.select(".bola") if (t := b.get_text(strip=True))]
        if not principales:  # el juego aún no existía en ese concurso
            continue
        if len(principales) != 6:
            raise ValueError(f"Concurso {concurso} {juego}: {len(principales)} números")
        adicional = bloque.select_one(".bola2")
        adicional = adicional.get_text(strip=True) if adicional else ""
        bolsa_txt = bloque.find(string=re.compile("Bolsa acumulada"))
        bolsa = int(re.sub(r"\D", "", bolsa_txt)) if bolsa_txt else None
        filas.append(
            {
                "concurso": concurso,
                "fecha": fecha,
                "juego": juego,
                **{f"r{i}": n for i, n in enumerate(principales, start=1)},  # orden de extracción
                "adicional": int(adicional) if adicional else None,
                "bolsa": bolsa or None,  # el sitio publica $0 cuando no hay dato
            }
        )
    return filas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-parse", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        ultimo = ultimo_concurso(session)
        print(f"Último concurso publicado: {ultimo}")
        if not args.solo_parse:
            nuevos = 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futuros = [pool.submit(descargar, session, c) for c in range(1, ultimo + 1)]
                for i, f in enumerate(as_completed(futuros), start=1):
                    nuevos += f.result()
                    if i % 250 == 0:
                        print(f"  {i}/{ultimo} revisados ({nuevos} descargados)")
            print(f"Descargas nuevas: {nuevos}")

    filas = [fila for c in range(1, ultimo + 1) for fila in parsear(c)]
    df = pd.DataFrame(filas)
    # Concursos que el sitio publica sin números, completados a mano desde otra fuente
    parches = pd.read_csv(PARCHES_CSV, parse_dates=["fecha"]).drop(columns="fuente")
    parches = parches[~parches.set_index(["concurso", "juego"]).index.isin(df.set_index(["concurso", "juego"]).index)]
    df = pd.concat([df, parches], ignore_index=True)
    print(f"Filas completadas con parches manuales: {len(parches)}")
    cols_r = [f"r{i}" for i in range(1, 7)]
    ordenados = pd.DataFrame(df[cols_r].apply(sorted, axis=1).tolist(), columns=[f"n{i}" for i in range(1, 7)])
    df = pd.concat([df, ordenados], axis=1)
    df[["adicional", "bolsa"]] = df[["adicional", "bolsa"]].astype("Int64")
    df = df[["concurso", "fecha", "juego", *ordenados.columns, "adicional", *cols_r, "bolsa"]]
    df.sort_values(["concurso", "juego"]).to_csv(OUT_CSV, index=False)
    print(f"Guardado {OUT_CSV} -> {len(df)} filas")
    print(df.groupby("juego")["concurso"].agg(["count", "min", "max"]))


if __name__ == "__main__":
    main()
