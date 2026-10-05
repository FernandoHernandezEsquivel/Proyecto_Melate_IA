"""Genera la app: inserta el modelo entrenado y sus tablas en app/plantilla.html.

Salidas:
    app/boleto_poco_jugado.html   versión para publicar como Artifact de Claude
    docs/index.html               versión autónoma para GitHub Pages

La página calcula la popularidad en el navegador con la misma fórmula que ModeloPopularidad.popularidad:
    pop(c) = q + (1−q) · exp(Σθ_c + log γ·consecutivos + log δ·penalización_decena + log(C_TOTAL/Z))

Uso (después de entrenar el modelo en notebooks/03):
    python src/exportar_app.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

import datos
from popularidad import C_TOTAL, ModeloPopularidad, combinaciones_al_azar

APP = datos.ROOT / "app"
MODELO = datos.ROOT / "models" / "popularidad_melate.npz"


def tabla_beneficio(modelo: ModeloPopularidad, V: float, rng, combos: int = 2500, muestras: int = 150) -> dict:
    """Premio esperado en 4 y 5 aciertos (relativo a una combinación al azar) según la popularidad de la combinación.
    Se simulan sorteos en los que la combinación acierta k números y se cuentan los demás ganadores esperados."""
    base = combinaciones_al_azar(200_000, rng)
    pop = modelo.popularidad(base)
    # cubrir todo el rango de popularidad: mitad al azar, mitad repartida por cuantiles
    orden = np.argsort(pop)
    elegidas = np.r_[rng.choice(len(base), combos // 2, replace=False), orden[np.linspace(0, len(base) - 1, combos // 2).astype(int)]]
    C = base[elegidas]
    p = pop[elegidas]
    salida = {}
    for k in (4, 5):
        rep = np.repeat(C, muestras, axis=0)
        resto = np.array([np.setdiff1d(np.arange(1, 57), c) for c in C]).repeat(muestras, axis=0)
        elige = lambda A, m: np.take_along_axis(A, rng.random(A.shape).argsort(axis=1)[:, :m], axis=1)
        Wk = np.sort(np.c_[elige(rep, k), elige(resto, 6 - k)], axis=1)
        lam = V * modelo.prob(Wk)[:, k]
        salida[k] = ((1 - np.exp(-lam)) / lam).reshape(len(C), muestras).mean(axis=1)
    # referencia: combinaciones al azar (la primera mitad)
    ref = {k: salida[k][: combos // 2].mean() for k in (4, 5)}
    # curva suave: mediana por intervalos de log-popularidad
    bordes = np.quantile(np.log(p), np.linspace(0, 1, 26))
    filas = []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        s = (np.log(p) >= lo) & (np.log(p) <= hi)
        filas.append([float(np.exp(np.median(np.log(p[s])))), *(float(np.median(salida[k][s]) / ref[k]) for k in (4, 5))])
    return {"puntos": filas, "nota": "popularidad, premio relativo 4 aciertos, premio relativo 5 aciertos"}


def tabla_premios(desde: str = "2024-01-01") -> dict:
    """Premio individual promedio por juego y categoría (solo sorteos con ganadores en esa categoría),
    y mediana del premio mayor pagado. Son cifras antes de impuestos."""
    pr, _ = datos.cargar_premios(limpio=False)
    r = pr[(pr.fecha >= desde) & (pr.ganadores > 0)]
    medias = r.groupby(["juego", "categoria"]).premio_individual.mean()
    salida = {f"{j[0]}{c}": round(float(v), 2) for (j, c), v in medias.items() if c > 1}
    mayores = pr[(pr.categoria == 1) & (pr.ganadores > 0) & (pr.fecha >= "2020-01-01")].premio_individual
    salida["mayor_mediana"] = round(float(mayores.median()), -5)
    salida["desde"] = desde[:4]
    return salida


def main() -> None:
    rng = np.random.default_rng(7)
    modelo = ModeloPopularidad.cargar(MODELO)
    meta = np.load(MODELO)
    with torch.no_grad():
        Z = modelo._dp(torch, torch.as_tensor(modelo.theta, dtype=torch.float64),
                       torch.tensor(modelo.log_gamma, dtype=torch.float64),
                       torch.tensor(modelo.log_delta, dtype=torch.float64), None).item()

    referencia = modelo.popularidad(combinaciones_al_azar(500_000, rng))
    pr, _ = datos.cargar_premios()
    g = datos.popularidad(pr)
    V = float(g.loc[g.juego == "Melate", "V"].median())
    hist = datos.cargar_historico()
    previas = sorted({"-".join(map(str, r)) for r in np.sort(hist[datos.N_COLS].to_numpy(), axis=1)})

    payload = {
        "theta": np.round(modelo.theta, 6).tolist(),
        "logGamma": modelo.log_gamma,
        "logDelta": modelo.log_delta,
        "q": modelo.q,
        "logK": float(np.log(C_TOTAL / Z)),
        "pesos": np.round(modelo.pesos, 4).tolist(),
        "cuantiles": np.round(np.quantile(referencia, np.linspace(0, 1, 201)), 5).tolist(),
        "beneficio": tabla_beneficio(modelo, V, rng),
        "premios": tabla_premios(),
        "previas": previas,
        "meta": {
            "desde": str(meta["desde"]), "hasta": str(meta["hasta"]),
            "concursos": int(g.concurso.nunique()), "combinaciones": int(len(g)),
            "ventas_mediana": round(V), "ultimo_concurso": int(hist.concurso.max()),
            "fecha_ultimo": str(hist.fecha.max().date()),
        },
    }
    plantilla = (APP / "plantilla.html").read_text(encoding="utf-8")
    html = plantilla.replace("/*__DATOS__*/null", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    destino = APP / "boleto_poco_jugado.html"
    destino.write_text(html, encoding="utf-8")
    print(f"Escrito {destino} ({destino.stat().st_size / 1024:.0f} KB)")

    # Versión autónoma para GitHub Pages (docs/index.html): documento HTML completo
    cabeza, cuerpo = html.split('<div class="envoltura">', 1)
    pagina = ('<!doctype html>\n<html lang="es-MX">\n<head>\n<meta charset="utf-8">\n'
              '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
              f'{cabeza.strip()}\n</head>\n<body>\n<div class="envoltura">{cuerpo}\n</body>\n</html>\n')
    docs = datos.ROOT / "docs" / "index.html"
    docs.parent.mkdir(exist_ok=True)
    docs.write_text(pagina, encoding="utf-8")
    print(f"Escrito {docs} (GitHub Pages)")
    print("Beneficio (popularidad → premio relativo 4 y 5 aciertos):")
    for fila in payload["beneficio"]["puntos"][::5]:
        print("  ", [round(v, 3) for v in fila])


if __name__ == "__main__":
    main()
