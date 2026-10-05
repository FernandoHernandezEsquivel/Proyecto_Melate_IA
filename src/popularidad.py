"""Modelos estructurales de popularidad de combinaciones de Melate.

Supuesto común: una combinación jugada viene de una mezcla
  - con probabilidad q, "al azar" (la máquina elige; uniforme sobre las C(56,6));
  - con probabilidad 1−q, elegida por la persona, con probabilidad proporcional a un peso w(c).

ModeloPesos:        w(c) = ∏_{i∈c} a_i                      (un peso por número)
ModeloPopularidad:  w(c) = ∏_{i∈c} a_i · γ^{consecutivos(c)} · δ^{pares en la misma decena(c)}

Para un sorteo con combinación ganadora s, ambos calculan de forma exacta la probabilidad de
que una combinación jugada acierte k números:
  - ModeloPesos con polinomios simétricos elementales: e_k(a_s)·e_{6−k}(a_resto)/e_6(a);
  - ModeloPopularidad con programación dinámica sobre los números 1..56 (estado: cuántos
    elegidos, cuántos de ellos están en s, si el anterior fue elegido y cuántos van en la decena),
    con gradientes automáticos de PyTorch.

Ajuste: se condiciona en los ganadores de 2 aciertos (que fijan las ventas del sorteo) y se
comparan ganadores esperados y observados de 3, 4 y 5 aciertos en escala log, con pérdida de
Huber (robusta a atípicos) y una varianza extra por categoría (τ_k) estimada de los datos.
"""
from __future__ import annotations

from math import comb

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

N, K = 56, 6
C_TOTAL = comb(N, K)
P_UNIF = np.array([comb(K, k) * comb(N - K, K - k) / C_TOTAL for k in range(K + 1)])
DECENA = np.arange(1, N + 1) // 10        # 1–9, 10–19, …, 50–56
T_MAX = 4                                  # tope del contador por decena (≥5 en una decena es rarísimo)


def huber(r, delta):
    ar = np.abs(r)
    return np.where(ar <= delta, 0.5 * r**2, delta * (ar - 0.5 * delta))


def combinaciones_al_azar(m: int, rng: np.random.Generator) -> np.ndarray:
    """m combinaciones uniformes de 6 números (1..56), ordenadas."""
    return np.sort(rng.random((m, N)).argpartition(K, axis=1)[:, :K] + 1, axis=1)


def rasgos_patron(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(parejas consecutivas, penalización por decena) de cada combinación, igual que en la programación dinámica."""
    C = np.sort(C, axis=1)
    consec = (np.diff(C, axis=1) == 1).sum(axis=1)
    cuenta = np.zeros((len(C), DECENA.max() + 1), dtype=int)
    np.add.at(cuenta, (np.arange(len(C))[:, None], DECENA[C - 1]), 1)
    # el t-ésimo número de una decena (t = 0, 1, …) aporta δ^min(t, T_MAX)
    pen = np.array([sum(min(t, T_MAX) for t in range(n)) for n in range(K + 1)])
    return consec, pen[cuenta].sum(axis=1)


class _Base:
    ks = (3, 4, 5)

    def __init__(self, lam: float = 3.0, delta: float = 1.5):
        self.lam, self.delta = lam, delta
        self.theta = self.q = self.tau = None

    # --- interfaz común ---
    def prob(self, W: np.ndarray) -> np.ndarray:
        """P(una combinación jugada acierta k números), k = 0..5, para cada combinación ganadora W (D, 6)."""
        raise NotImplementedError

    def R(self, W: np.ndarray) -> np.ndarray:
        """Índice de popularidad predicho R_k (D, 3) para k = 3, 4, 5, en la escala del observado."""
        P, ks = self.prob(W), list(self.ks)
        return (P[:, ks] / P[:, [2]]) / (P_UNIF[ks] / P_UNIF[2])

    @property
    def pesos(self) -> np.ndarray:
        a = np.exp(self.theta)
        return a / a.mean()

    def _tau(self, W, G):
        P = self.prob(W)
        E = G[:, [0]] * P[:, list(self.ks)] / P[:, [2]]
        crudo = np.log(G[:, 1:] + 0.5) - np.log(E + 0.5)
        mad = 1.4826 * np.median(np.abs(crudo - np.median(crudo, axis=0)), axis=0)
        return np.sqrt(np.maximum(mad**2 - np.median(1 / (E + 0.5), axis=0), 1e-4))


class ModeloPesos(_Base):
    """Solo un peso por número más la fracción al azar."""

    @staticmethod
    def _simetricos(A):
        E = np.zeros((A.shape[0], K + 1))
        E[:, 0] = 1.0
        for j in range(A.shape[1]):
            E[:, 1:] = E[:, 1:] + A[:, [j]] * E[:, :-1]
        return E

    def _prob(self, theta, q, W):
        a = np.exp(theta)
        fuera = np.broadcast_to(a, (len(W), N)).copy()
        fuera[np.arange(len(W))[:, None], W - 1] = 0.0
        e_s, e_f = self._simetricos(a[W - 1]), self._simetricos(fuera)
        e6 = self._simetricos(a[None, :])[0, K]
        k = np.arange(K)
        return q * P_UNIF[:K] + (1 - q) * e_s[:, k] * e_f[:, K - k] / e6

    def prob(self, W):
        return self._prob(self.theta, self.q, W)

    def fit(self, W, G, iteraciones=3):
        x = np.r_[np.zeros(N - 1), logit(0.3)]
        self.tau = np.full(len(self.ks), 0.1)
        for _ in range(iteraciones):
            def objetivo(x_):
                theta, q = np.r_[x_[:-1], -x_[:-1].sum()], expit(x_[-1])
                P = self._prob(theta, q, W)
                E = G[:, [0]] * P[:, list(self.ks)] / P[:, [2]]
                r = (np.log(G[:, 1:] + 0.5) - np.log(E + 0.5)) / np.sqrt(1 / (E + 0.5) + self.tau**2)
                return huber(r, self.delta).sum() + self.lam * (theta**2).sum()

            x = minimize(objetivo, x, method="L-BFGS-B", options={"maxiter": 500}).x
            self.theta, self.q = np.r_[x[:-1], -x[:-1].sum()], float(expit(x[-1]))
            self.tau = self._tau(W, G)
        return self

    def popularidad(self, C):
        a = np.exp(self.theta)
        e6 = self._simetricos(a[None, :])[0, K]
        return self.q + (1 - self.q) * np.prod(a[C - 1], axis=1) * C_TOTAL / e6


class ModeloPopularidad(_Base):
    """Peso por número + factor por pareja consecutiva (γ) + factor por par en la misma decena (δ)."""

    LOTE = 300  # sorteos por lote en la programación dinámica (limita la memoria de autograd)

    def __init__(self, lam: float = 3.0, delta: float = 1.5):
        super().__init__(lam, delta)
        self.log_gamma = self.log_delta = 0.0

    @property
    def gamma(self):
        return float(np.exp(self.log_gamma))

    @property
    def delta_decena(self):
        return float(np.exp(self.log_delta))

    @staticmethod
    def _dp(torch, theta, lg, ld, W):
        """Suma de pesos de las combinaciones que aciertan m = 0..5 números de cada fila de W.
        W=None calcula la suma total Z (sin distinguir aciertos). Devuelve tensor (D, 6) o escalar."""
        D = 1 if W is None else len(W)
        dt = theta.dtype
        miembro = torch.zeros((D, N), dtype=torch.bool)
        if W is not None:
            miembro[torch.arange(D)[:, None], torch.as_tensor(W - 1)] = True
        a = torch.exp(theta)
        t_idx = torch.arange(T_MAX + 1, dtype=dt)
        # factor por (anterior elegido p, contador en la decena t)
        F = torch.exp(torch.stack([ld * t_idx, lg + ld * t_idx]))            # (2, T+1)
        S = torch.zeros((D, K + 1, K, 2, T_MAX + 1), dtype=dt)              # (D, j, m, prev, t)
        S[:, 0, 0, 0, 0] = 1.0
        z = lambda *s: torch.zeros(s, dtype=dt)
        for i in range(N):
            if i > 0 and DECENA[i] != DECENA[i - 1]:                         # empieza decena: t = 0
                S = torch.cat([S.sum(-1, keepdim=True), z(D, K + 1, K, 2, T_MAX)], -1)
            no = S.sum(3)                                                     # no elegir i
            si = (S * F).sum(3) * a[i]                                        # elegir i
            si = torch.cat([z(D, 1, K, T_MAX + 1), si[:, :-1]], 1)            # j → j+1
            si = torch.cat([z(D, K + 1, K, 1), si[..., : T_MAX - 1], si[..., T_MAX - 1:].sum(-1, keepdim=True)], -1)  # t → min(t+1, T)
            en_s = torch.cat([z(D, K + 1, 1, T_MAX + 1), si[:, :, :-1]], 2)   # m → m+1 si i está en la ganadora
            si = torch.where(miembro[:, i, None, None, None], en_s, si)
            S = torch.stack([no, si], 3)
        Zm = S[:, K].sum((-1, -2))                                            # (D, m=0..5)
        return Zm[0].sum() if W is None else Zm

    def _prob_t(self, torch, theta, lg, ld, q, W):
        Pm = self._dp(torch, theta, lg, ld, W) / self._dp(torch, theta, lg, ld, None)
        return q * torch.as_tensor(P_UNIF[:K]) + (1 - q) * Pm

    def prob(self, W):
        import torch
        with torch.no_grad():
            th = torch.as_tensor(self.theta, dtype=torch.float64)
            lg, ld = torch.tensor(self.log_gamma, dtype=torch.float64), torch.tensor(self.log_delta, dtype=torch.float64)
            return torch.cat([self._prob_t(torch, th, lg, ld, self.q, W[i:i + self.LOTE])
                              for i in range(0, len(W), self.LOTE)]).numpy()

    def fit(self, W, G, iteraciones=3, x0=None):
        import torch
        torch.set_num_threads(max(1, torch.get_num_threads()))
        # arranque: pesos del modelo simple (rápido) y γ = δ = 1
        if x0 is None:
            base = ModeloPesos(self.lam, self.delta).fit(W, G, iteraciones=1)
            x0 = np.r_[base.theta[:-1], 0.0, 0.0, logit(np.clip(base.q, 0.05, 0.95))]
        x = np.asarray(x0, float)
        self.tau = np.full(len(self.ks), 0.1)
        Gt = torch.tensor(np.asarray(G), dtype=torch.float64)  # copia: G puede venir de solo lectura
        ks = list(self.ks)

        for _ in range(iteraciones):
            tau = torch.as_tensor(self.tau)

            def objetivo(x_):
                xt = torch.tensor(x_, dtype=torch.float64, requires_grad=True)

                def params():  # se reconstruye en cada lote: cada backward libera su grafo
                    theta = torch.cat([xt[: N - 1], -xt[: N - 1].sum(0, keepdim=True)])
                    return theta, xt[N - 1], xt[N], torch.sigmoid(xt[N + 1])

                theta = params()[0]
                total = self.lam * (theta**2).sum()
                total.backward()
                valor = total.item()
                for i in range(0, len(W), self.LOTE):
                    P = self._prob_t(torch, *params(), W[i:i + self.LOTE])
                    g = Gt[i:i + self.LOTE]
                    E = g[:, [0]] * P[:, ks] / P[:, [2]]
                    r = (torch.log(g[:, 1:] + 0.5) - torch.log(E + 0.5)) / torch.sqrt(1 / (E + 0.5) + tau**2)
                    ar = r.abs()
                    perdida = torch.where(ar <= self.delta, 0.5 * r**2, self.delta * (ar - 0.5 * self.delta)).sum()
                    perdida.backward()
                    valor += perdida.item()
                return valor, xt.grad.numpy().copy()

            x = minimize(objetivo, x, jac=True, method="L-BFGS-B", options={"maxiter": 400}).x
            self._fijar(x)
            self.tau = self._tau(W, G)
        return self

    def _fijar(self, x):
        self.theta = np.r_[x[: N - 1], -x[: N - 1].sum()]
        self.log_gamma, self.log_delta, self.q = float(x[N - 1]), float(x[N]), float(expit(x[N + 1]))

    def popularidad(self, C):
        import torch
        with torch.no_grad():
            Z = self._dp(torch, torch.as_tensor(self.theta, dtype=torch.float64), torch.tensor(self.log_gamma, dtype=torch.float64),
                         torch.tensor(self.log_delta, dtype=torch.float64), None).item()
        consec, pen = rasgos_patron(C)
        w = np.exp(self.theta[C - 1].sum(axis=1) + self.log_gamma * consec + self.log_delta * pen)
        return self.q + (1 - self.q) * w * C_TOTAL / Z

    # --- persistencia ---
    def guardar(self, ruta, **extra):
        np.savez(ruta, theta=self.theta, log_gamma=self.log_gamma, log_delta=self.log_delta, q=self.q,
                 tau=self.tau, lam=self.lam, **extra)

    @classmethod
    def cargar(cls, ruta):
        f = np.load(ruta)
        m = cls(lam=float(f["lam"]))
        m.theta, m.tau = f["theta"], f["tau"]
        m.log_gamma, m.log_delta, m.q = float(f["log_gamma"]), float(f["log_delta"]), float(f["q"])
        return m
