"""Carga y limpieza de los datos procesados, compartida por los notebooks."""
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESADOS = ROOT / "data" / "processed"
N, K = 56, 6
N_COLS = [f"n{i}" for i in range(1, K + 1)]

# P(acertar exactamente j números naturales) con una combinación de 6 de 56
P_NAT = {j: comb(K, j) * comb(N - K, K - j) / comb(N, K) for j in range(K + 1)}


def cargar_historico() -> pd.DataFrame:
    return pd.read_csv(PROCESADOS / "melate_historico.csv", parse_dates=["fecha"])


def cargar_premios(limpio: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (premios, descartados).

    Con limpio=True se quitan Revanchita (solo publica el 1er lugar) y los concursos
    cuya tabla de premios es inválida en la fuente: tablas idénticas a las de otro
    concurso (copiadas) o con 0 ganadores de 2 aciertos.
    """
    pr = pd.read_csv(PROCESADOS / "premios_por_categoria.csv", parse_dates=["fecha"])
    if not limpio:
        return pr, pr.iloc[0:0]
    pr = pr[pr.juego.isin(["Melate", "Revancha"])]
    motivos = {}
    for juego, x in pr.groupby("juego"):
        tabla = x[x.categoria > 1].pivot(index="concurso", columns="categoria", values="ganadores")
        for c in tabla.index[tabla.duplicated(keep=False)]:
            motivos.setdefault(c, f"tabla de {juego} duplicada en otro concurso")
        dos = x[x.naturales == 2].groupby("concurso").ganadores.sum()
        for c in dos.index[dos == 0]:
            motivos.setdefault(c, f"{juego} con 0 ganadores de 2 aciertos")
    descartados = pd.Series(motivos, name="motivo").rename_axis("concurso").sort_index().reset_index()
    return pr[~pr.concurso.isin(motivos)].copy(), descartados


def popularidad(pr: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (juego, concurso) con ganadores por número de aciertos, ventas
    implícitas V y el índice de popularidad R_k = ganadores(k) / (V·P(k))."""
    g = (pr.groupby(["juego", "concurso", "fecha", "naturales"]).ganadores.sum()
           .unstack("naturales").reset_index())
    g.columns.name = None
    g["V"] = g[2] / P_NAT[2]
    for k in (3, 4, 5):
        g[f"E{k}"] = g["V"] * P_NAT[k]
        g[f"R{k}"] = g[k] / g[f"E{k}"]
    return g
