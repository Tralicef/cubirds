"""
Métodos con aproximación de funciones de Sutton & Barto (Parte II) y de
gradiente de política (cap. 13), sobre afterstates.

- Métodos de valor: v̂(x) = w·x + b estima la recompensa hasta el final de la
  partida del afterstate de cada jugada (x = las 65 características por jugada
  de cubirds/rl/features.py, o tile coding). Se juega la de mayor v̂.
- Gradiente de política: preferencia lineal h(x) = θ·x por jugada y softmax.
"""

import numpy as np

from cubirds.rl.features import DIM, G_DIM, decision
from cubirds.rl.sb.common import key


class LinearBase:
    chapter = ""
    label = ""
    desc = ""
    python_only = False

    def __init__(self, rng, **hp):
        self.rng = rng
        self.hp = hp
        self.dim = DIM
        self.w = np.zeros(self.dim)
        self.b = 0.0
        self.t = 0

    # características del afterstate de cada jugada legal
    def feats(self, ep):
        return decision(ep.g)[1].astype(np.float64)

    def v(self, x):
        return x @ self.w + self.b

    def greedy(self, X):
        return int(np.argmax(self.v(X)))

    def explore(self, X):
        if self.rng.random() < self.hp.get("epsilon", 0.1):
            return self.rng.randrange(len(X))
        return self.greedy(X)

    def play(self, ep):
        traj = []
        while not ep.done:
            X = self.feats(ep)
            j = self.explore(X)
            r = ep.step(j)
            self.t += 1
            traj.append((X, j, r))
            self.online(traj)
        if traj:
            self.end(traj)

    def online(self, traj):
        pass

    def end(self, traj):
        pass

    def choose(self, ep):
        return self.greedy(self.feats(ep))

    def export_data(self):
        return {"kind": "lineal", "w": np.round(self.w, 6).tolist(), "b": round(float(self.b), 6)}

    def _sgd(self, x, target, alpha):
        err = target - self.v(x)
        self.w += alpha * err * x
        self.b += alpha * err


# ----------------------- Capítulo 9: predicción -----------------------

class GradientMC(LinearBase):
    chapter = "9.3"
    label = "Monte Carlo por gradiente (lineal)"
    desc = "Ajusta v̂ al retorno real hasta el final de la partida con descenso de gradiente estocástico."

    def end(self, traj):
        g = 0.0
        for X, j, r in reversed(traj):
            g += r
            self._sgd(X[j], g, self.hp.get("alpha", 0.01))


class LSTD(LinearBase):
    chapter = "9.8"
    label = "LSTD"
    desc = "TD por mínimos cuadrados: junta todas las transiciones y resuelve los pesos de una vez (sin paso α)."

    def __init__(self, rng, **hp):
        super().__init__(rng, **hp)
        d = self.dim + 1
        self.A = np.eye(d) * hp.get("reg", 1.0)
        self.bv = np.zeros(d)
        self.episodes = 0

    def _solve(self):
        sol = np.linalg.solve(self.A, self.bv)
        self.w, self.b = sol[:-1], sol[-1]

    def end(self, traj):
        for t, (X, j, r) in enumerate(traj):
            x = np.append(X[j], 1.0)
            x1 = np.append(traj[t + 1][0][traj[t + 1][1]], 1.0) if t + 1 < len(traj) else np.zeros(self.dim + 1)
            self.A += np.outer(x, x - x1)
            self.bv += r * x
        self.episodes += 1
        if self.episodes % self.hp.get("solve_every", 200) == 0:
            self._solve()

    def export_data(self):
        self._solve()
        return super().export_data()


# ----------------------- Capítulo 10: control -----------------------

class SemiGradientSarsa(LinearBase):
    chapter = "10.1"
    label = "SARSA semi-gradiente (lineal)"
    desc = "SARSA con v̂ lineal: target = r + v̂(afterstate siguiente elegido)."

    def online(self, traj):
        if len(traj) >= 2:
            X, j, r = traj[-2]
            X1, j1, _ = traj[-1]
            self._sgd(X[j], r + self.v(X1[j1]), self.hp.get("alpha", 0.01))

    def end(self, traj):
        X, j, r = traj[-1]
        self._sgd(X[j], r, self.hp.get("alpha", 0.01))


class NStepSemiGradientSarsa(LinearBase):
    chapter = "10.2"
    label = "SARSA semi-gradiente de n pasos"
    desc = "Recompensas de n turnos reales más v̂ del afterstate n turnos después (n = 3)."

    def end(self, traj):
        n, alpha, T = self.hp.get("n", 3), self.hp.get("alpha", 0.01), len(traj)
        for t in range(T):
            g = sum(traj[i][2] for i in range(t, min(t + n, T)))
            if t + n < T:
                X, j, _ = traj[t + n]
                g += self.v(X[j])
            X, j, _ = traj[t]
            self._sgd(X[j], g, alpha)


class TileCodingSarsa(SemiGradientSarsa):
    chapter = "9.5.4"
    label = "SARSA con tile coding"
    desc = "SARSA semi-gradiente sobre tile coding: 8 grillas desplazadas sobre 5 variables del afterstate."
    python_only = True
    TILINGS, TILES = 8, 4

    def __init__(self, rng, **hp):
        super().__init__(rng, **hp)
        self.dim = self.TILINGS * self.TILES ** 5
        self.w = np.zeros(self.dim)

    def feats(self, ep):
        """Índices de las 8 celdas activas (una por grilla) de cada jugada: x es binario y disperso."""
        g = ep.g
        legal = g.legal_actions()
        X = decision(g)[1]
        out = np.zeros((len(legal), self.TILINGS), dtype=np.int64)
        for i, a in enumerate(legal):
            k = key(g, a)
            # pájaros que se lleva, mejor bandada, cuánto acerca, lo que le falta, ganancia de mano (Afinado)
            dims = [k[0] / 4, k[1] / 2, k[2] / 2, k[3] / 3, min(max(X[i, G_DIM + 22] + 0.5, 0), 1)]
            for tl in range(self.TILINGS):
                off = tl / self.TILINGS
                idx = 0
                for d in dims:
                    cell = min(self.TILES - 1, int(min(max(d, 0), 1) * (self.TILES - 1) + off))
                    idx = idx * self.TILES + cell
                out[i, tl] = tl * self.TILES ** 5 + idx
        return out

    def v(self, x):
        return self.w[x].sum(-1) + self.b

    def _sgd(self, x, target, alpha):
        err = target - self.v(x)
        np.add.at(self.w, x, alpha * err)
        self.b += alpha * err

    def export_data(self):
        return None  # sólo Python


class TrueOnlineSarsaLambda(LinearBase):
    chapter = "12.8"
    label = "True online SARSA(λ)"
    desc = "SARSA(λ) lineal con trazas holandesas: equivale exactamente a la versión 'hacia adelante' de λ-retornos."

    def end(self, traj):
        lam, alpha = self.hp.get("lam", 0.8), self.hp.get("alpha", 0.01)
        z = np.zeros(self.dim)
        q_old = 0.0
        for t, (X, j, r) in enumerate(traj):
            x = X[j]
            q = self.v(x)
            q1 = self.v(traj[t + 1][0][traj[t + 1][1]]) if t + 1 < len(traj) else 0.0
            delta = r + q1 - q
            z = lam * z + (1 - alpha * lam * z @ x) * x
            self.w += alpha * (delta + q - q_old) * z - alpha * (q - q_old) * x
            self.b += alpha * delta
            q_old = q1


class SemiGradientQ(LinearBase):
    chapter = "11.1"
    label = "Q-learning semi-gradiente (lineal)"
    desc = "Off-policy con aproximación: target = r + máx v̂ del turno siguiente (la 'tríada mortal' del cap. 11)."

    def online(self, traj):
        if len(traj) >= 2:
            X, j, r = traj[-2]
            X1, _, _ = traj[-1]
            self._sgd(X[j], r + self.v(X1).max(), self.hp.get("alpha", 0.01))

    def end(self, traj):
        X, j, r = traj[-1]
        self._sgd(X[j], r, self.hp.get("alpha", 0.01))


# ----------------------- Capítulo 13: gradiente de política -----------------------

class PolicyGradientBase(LinearBase):
    """Política softmax con preferencias lineales h(x) = θ·x por jugada."""

    def __init__(self, rng, **hp):
        super().__init__(rng, **hp)
        self.theta = np.zeros(self.dim)
        self.u = np.zeros(G_DIM + 1)  # crítico / línea de base sobre el estado global (+ sesgo)

    def probs(self, X):
        h = X @ self.theta
        e = np.exp(h - h.max())
        return e / e.sum()

    def explore(self, X):
        p = self.probs(X)
        return int(self.rng.choices(range(len(X)), p.tolist())[0])

    def greedy(self, X):
        return int(np.argmax(X @ self.theta))

    def grad_log(self, X, j):
        return X[j] - self.probs(X) @ X

    @staticmethod
    def g(X):
        return np.append(X[0][:G_DIM], 1.0)  # estado global (igual para todas las jugadas)

    def export_data(self):
        return {"kind": "lineal", "w": np.round(self.theta, 6).tolist(), "b": 0.0}


class Reinforce(PolicyGradientBase):
    chapter = "13.3"
    label = "REINFORCE"
    desc = "Gradiente de política Monte Carlo: sube la probabilidad de las jugadas según el retorno de la partida."

    def end(self, traj):
        alpha = self.hp.get("alpha", 0.05)
        G = 0.0
        for X, j, r in reversed(traj):
            G += r
            self.theta += alpha * G * self.grad_log(X, j)


class ReinforceBaseline(PolicyGradientBase):
    chapter = "13.4"
    label = "REINFORCE con línea de base"
    desc = "REINFORCE restando una estimación del valor del estado, lo que reduce mucho la varianza."

    def end(self, traj):
        a_th, a_w = self.hp.get("alpha", 0.05), self.hp.get("alpha_w", 0.02)
        G = 0.0
        for X, j, r in reversed(traj):
            G += r
            gx = self.g(X)
            delta = G - self.u @ gx
            self.u += a_w * delta * gx
            self.theta += a_th * delta * self.grad_log(X, j)


class ActorCritic(PolicyGradientBase):
    chapter = "13.5"
    label = "Actor-crítico de un paso"
    desc = "El crítico (TD) evalúa cada jugada en el momento y el actor ajusta la política con ese error."

    def online(self, traj):
        if len(traj) >= 2:
            self._step(traj[-2], traj[-1][0])

    def end(self, traj):
        self._step(traj[-1], None)

    def _step(self, step, X1):
        a_th, a_w = self.hp.get("alpha", 0.05), self.hp.get("alpha_w", 0.02)
        X, j, r = step
        gx = self.g(X)
        v1 = self.u @ self.g(X1) if X1 is not None else 0.0
        delta = r + v1 - self.u @ gx
        self.u += a_w * delta * gx
        self.theta += a_th * delta * self.grad_log(X, j)


class ActorCriticTraces(PolicyGradientBase):
    chapter = "13.5b"
    label = "Actor-crítico con trazas"
    desc = "Actor-crítico con trazas de elegibilidad en el actor y en el crítico (λ = 0,8)."

    def end(self, traj):
        a_th, a_w, lam = self.hp.get("alpha", 0.05), self.hp.get("alpha_w", 0.02), self.hp.get("lam", 0.8)
        zt = np.zeros(self.dim)
        zw = np.zeros(G_DIM + 1)
        for t, (X, j, r) in enumerate(traj):
            gx = self.g(X)
            v1 = self.u @ self.g(traj[t + 1][0]) if t + 1 < len(traj) else 0.0
            delta = r + v1 - self.u @ gx
            zw = lam * zw + gx
            zt = lam * zt + self.grad_log(X, j)
            self.u += a_w * delta * zw
            self.theta += a_th * delta * zt


ALGORITHMS = {
    "mc_gradiente": (GradientMC, dict(epsilon=0.1, alpha=0.01)),
    "lstd": (LSTD, dict(epsilon=0.1, reg=1.0, solve_every=200)),
    "sarsa_semigradiente": (SemiGradientSarsa, dict(epsilon=0.1, alpha=0.01)),
    "sarsa_semigradiente_n": (NStepSemiGradientSarsa, dict(epsilon=0.1, alpha=0.01, n=3)),
    "sarsa_tiles": (TileCodingSarsa, dict(epsilon=0.1, alpha=0.1 / 8)),
    "true_online_sarsa_lambda": (TrueOnlineSarsaLambda, dict(epsilon=0.1, alpha=0.01, lam=0.8)),
    "q_semigradiente": (SemiGradientQ, dict(epsilon=0.1, alpha=0.01)),
    "reinforce": (Reinforce, dict(alpha=0.05)),
    "reinforce_baseline": (ReinforceBaseline, dict(alpha=0.05, alpha_w=0.02)),
    "actor_critico": (ActorCritic, dict(alpha=0.05, alpha_w=0.02)),
    "actor_critico_trazas": (ActorCriticTraces, dict(alpha=0.05, alpha_w=0.02, lam=0.8)),
}
