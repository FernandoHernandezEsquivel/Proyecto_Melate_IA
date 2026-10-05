"""Sugiere combinaciones de Melate poco jugadas según el modelo de popularidad.

No cambia la probabilidad de ganar (todas las combinaciones tienen 1 en 32,468,436).
Lo que cambia es con cuántas personas compartirías el premio de las categorías que se
reparten (2º a 5º lugar) si aciertas.

Uso:
    python src/generar.py                 # 5 combinaciones del 10% menos jugado
    python src/generar.py -n 10 --percentil 5
    python src/generar.py --evaluar 7 13 21 28 30 31   # qué tan jugada es una combinación
"""
from __future__ import annotations

import argparse

import numpy as np

import datos
from popularidad import ModeloPopularidad, combinaciones_al_azar

MODELO = datos.ROOT / "models" / "popularidad_melate.npz"


def ganadoras_previas() -> set[tuple]:
    h = datos.cargar_historico()
    return set(map(tuple, np.sort(h[datos.N_COLS].to_numpy(), axis=1)))


def es_patron(c: np.ndarray, previas: set[tuple]) -> bool:
    """Reglas de prudencia (no aprendidas de los datos): ganadoras anteriores, ≥3 consecutivos, progresiones."""
    dif = np.diff(c)
    racha = max(len(s) for s in "".join("1" if x == 1 else "0" for x in dif).split("0")) + 1
    return tuple(c) in previas or racha >= 3 or len(set(dif)) == 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=5, help="cuántas combinaciones sugerir")
    ap.add_argument("--percentil", type=float, default=10, help="buscar dentro del X%% menos jugado")
    ap.add_argument("--evaluar", type=int, nargs=6, metavar="N", help="evaluar una combinación")
    ap.add_argument("--semilla", type=int, default=None)
    args = ap.parse_args()

    modelo = ModeloPopularidad.cargar(MODELO)
    rng = np.random.default_rng(args.semilla)
    referencia = modelo.popularidad(combinaciones_al_azar(300_000, rng))

    if args.evaluar:
        c = np.sort(np.array(args.evaluar))
        if len(set(c)) != 6 or c.min() < 1 or c.max() > 56:
            ap.error("la combinación debe tener 6 números distintos entre 1 y 56")
        p = modelo.popularidad(c[None, :])[0]
        print(f"{'-'.join(f'{x:02d}' for x in c)}: popularidad {p:.2f} (1 = promedio), "
              f"más jugada que el {(referencia < p).mean():.0%} de las combinaciones")
        return

    umbral = np.percentile(referencia, args.percentil)
    previas = ganadoras_previas()
    salida = []
    while len(salida) < args.n:
        cand = combinaciones_al_azar(20_000, rng)
        cand = cand[modelo.popularidad(cand) <= umbral]
        salida += [c for c in cand if not es_patron(c, previas)][: args.n - len(salida)]
    print(f"Combinaciones del {args.percentil:g}% menos jugado (popularidad ≤ {umbral:.2f}; 1 = promedio):")
    for c in salida:
        print(f"  {'-'.join(f'{x:02d}' for x in c)}   popularidad {modelo.popularidad(c[None, :])[0]:.2f}")


if __name__ == "__main__":
    main()
