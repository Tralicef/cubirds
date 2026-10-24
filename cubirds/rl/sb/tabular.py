"""
Algoritmos tabulares de Sutton & Barto sobre afterstates discretos.

Todos aprenden un valor por clave de afterstate (recompensa esperada) y, al
jugar sin explorar, eligen la jugada de mayor valor. Cada clase indica el
capítulo y la sección del libro.
"""

import heapq
import math
from collections import defaultdict


class Tabular:
    chapter = ""
    label = ""
    desc = ""

    def __init__(self, rng, **hp):
        self.rng = rng
        self.hp = hp
        self.Q = defaultdict(float)  # valor estimado por clave
        self.N = defaultdict(int)    # visitas por clave
        self.t = 0                   # decisiones tomadas

    # --- política ---
    def score(self, k):
        """Valor usado para elegir sin explorar (mayor = mejor)."""
        return self.Q[k] if k in self.Q else self.default()

    def default(self):
        return sum(self.Q.values()) / len(self.Q) if self.Q else 0.0

    def greedy(self, ks):
        scores = [self.score(k) for k in ks]
        return scores.index(max(scores))

    def explore(self, ks):
        if self.rng.random() < self.hp.get("epsilon", 0.1):
            return self.rng.randrange(len(ks))
        return self.greedy(ks)

    # --- aprendizaje: una partida completa ---
    def play(self, ep):
        while not ep.done:
            ks = ep.keys()
            j = self.explore(ks)
            r = ep.step(j)
            self.t += 1
            self.learn_step(ks, j, r)

    def learn_step(self, ks, j, r):
        raise NotImplementedError

    def table(self):
        """Clave -> valor, para exportar (el agente elige el máximo)."""
        return {k: self.score(k) for k in self.Q}


# ======================= Capítulo 2: bandidos =======================
# Como bandidos, sólo miran la recompensa inmediata de la jugada (cuánto subió
# el potencial hasta su próximo turno). La clave del afterstate hace de contexto.

class EpsilonGreedy(Tabular):
    chapter = "2.2–2.4"
    label = "ε-greedy (promedios)"
    desc = "Bandido ε-greedy con promedios muestrales incrementales (2.4): aprende la ganancia inmediata de cada afterstate."

    def learn_step(self, ks, j, r):
        k = ks[j]
        self.N[k] += 1
        self.Q[k] += (r - self.Q[k]) / self.N[k]


class Optimistic(Tabular):
    chapter = "2.6"
    label = "Valores iniciales optimistas"
    desc = "Greedy puro con valor inicial alto (0,5, optimista) y paso constante: explora porque lo desconocido parece bueno."

    def score(self, k):
        return self.Q.get(k, 0.5)

    def explore(self, ks):
        return self.greedy(ks)

    def learn_step(self, ks, j, r):
        k = ks[j]
        q = self.Q.get(k, 0.5)
        self.Q[k] = q + self.hp.get("alpha", 0.1) * (r - q)
        self.N[k] += 1


class UCB(Tabular):
    chapter = "2.7"
    label = "UCB"
    desc = "Upper Confidence Bound: prefiere afterstates poco probados (bono c·√(ln t / N))."

    def explore(self, ks):
        c = self.hp.get("c", 0.2)
        for i, k in enumerate(ks):
            if self.N[k] == 0:
                return i
        lt = math.log(self.t + 1)
        vals = [self.Q[k] + c * math.sqrt(lt / self.N[k]) for k in ks]
        return vals.index(max(vals))

    def learn_step(self, ks, j, r):
        k = ks[j]
        self.N[k] += 1
        self.Q[k] += (r - self.Q[k]) / self.N[k]


class GradientBandit(Tabular):
    chapter = "2.8"
    label = "Bandido por gradiente"
    desc = "Preferencias H con softmax y línea de base (promedio de recompensas); no estima valores."

    def __init__(self, rng, **hp):
        super().__init__(rng, **hp)
        self.H = defaultdict(float)
        self.baseline = 0.0

    def _probs(self, ks):
        m = max(self.H[k] for k in ks)
        e = [math.exp(self.H[k] - m) for k in ks]
        s = sum(e)
        return [x / s for x in e]

    def explore(self, ks):
        return self.rng.choices(range(len(ks)), self._probs(ks))[0]

    def score(self, k):
        return self.H.get(k, 0.0)

    def learn_step(self, ks, j, r):
        alpha = self.hp.get("alpha", 0.1)
        p = self._probs(ks)
        self.baseline += (r - self.baseline) / self.t
        delta = r - self.baseline
        for i, k in enumerate(ks):
            self.H[k] += alpha * delta * ((1 if i == j else 0) - p[i])
        self.Q[ks[j]] = 0.0  # para que table() incluya la clave

    def table(self):
        return dict(self.H)


# ============ Capítulos 5-8 y 12: retorno hasta el final de la partida ============
# V(k) = recompensa esperada desde el afterstate k hasta que termina la partida
# (incluida la de este turno). La política greedy elige el mayor V.

class Episodic(Tabular):
    """Juega la partida y guarda la trayectoria: opciones, índice elegido y recompensa."""

    def play(self, ep):
        traj = []
        while not ep.done:
            ks = ep.keys()
            j = self.explore(ks)
            r = ep.step(j)
            self.t += 1
            traj.append((ks, j, r))
            self.online(traj)
        if traj:
            self.end(traj)

    def online(self, traj):
        pass

    def end(self, traj):
        pass

    def eps_probs(self, ks):
        """Probabilidades ε-greedy sobre las opciones."""
        eps = self.hp.get("epsilon", 0.1)
        n = len(ks)
        p = [eps / n] * n
        p[self.greedy(ks)] += 1 - eps
        return p


class MonteCarloOnPolicy(Episodic):
    chapter = "5.4"
    label = "Monte Carlo on-policy (ε-soft)"
    desc = "Promedia el retorno real hasta el final de la partida de cada afterstate visitado (every-visit)."

    def end(self, traj):
        g = 0.0
        for ks, j, r in reversed(traj):
            g += r
            k = ks[j]
            self.N[k] += 1
            self.Q[k] += (g - self.Q[k]) / self.N[k]


class MonteCarloOffPolicy(Episodic):
    chapter = "5.7"
    label = "Monte Carlo off-policy (IS ponderado)"
    desc = "Aprende la política greedy a partir de jugadas ε-greedy, corrigiendo con importance sampling ponderado."

    def __init__(self, rng, **hp):
        super().__init__(rng, **hp)
        self.C = defaultdict(float)

    def end(self, traj):
        g, w = 0.0, 1.0
        eps = self.hp.get("epsilon", 0.3)
        for ks, j, r in reversed(traj):
            g += r
            k = ks[j]
            self.C[k] += w
            self.Q[k] += w / self.C[k] * (g - self.Q[k])
            self.N[k] += 1
            if j != self.greedy(ks):
                break
            w *= 1.0 / (1 - eps + eps / len(ks))


class _TD(Episodic):
    """Actualización TD de un paso en línea: target(paso, siguiente) define el algoritmo."""

    def online(self, traj):
        if len(traj) >= 2:
            self._update(traj[-2], self.target(traj[-2], traj[-1]))

    def end(self, traj):
        self._update(traj[-1], traj[-1][2])  # último turno: no hay después

    def _update(self, step, target):
        ks, j, _ = step
        k = ks[j]
        self.N[k] += 1
        self.Q[k] += self.hp.get("alpha", 0.05) * (target - self.Q[k])


class Sarsa(_TD):
    chapter = "6.4"
    label = "SARSA"
    desc = "TD on-policy: V(k) ← V(k) + α[r + V(k siguiente elegido) − V(k)]."

    def target(self, step, nxt):
        ks1, j1, _ = nxt
        return step[2] + self.Q[ks1[j1]]


class QLearning(_TD):
    chapter = "6.5"
    label = "Q-learning"
    desc = "TD off-policy: usa el mejor afterstate del turno siguiente, juegue el que juegue."

    def target(self, step, nxt):
        return step[2] + max(self.Q[k] for k in nxt[0])


class ExpectedSarsa(_TD):
    chapter = "6.6"
    label = "Expected SARSA"
    desc = "TD con el valor esperado del turno siguiente según la política ε-greedy."

    def target(self, step, nxt):
        ks1 = nxt[0]
        return step[2] + sum(pi * self.Q[k] for pi, k in zip(self.eps_probs(ks1), ks1))


class DoubleQ(Episodic):
    chapter = "6.7"
    label = "Double Q-learning"
    desc = "Dos tablas: una elige el mejor afterstate y la otra lo valúa, para evitar el sesgo de maximización."

    def __init__(self, rng, **hp):
        super().__init__(rng, **hp)
        self.A = defaultdict(float)
        self.B = defaultdict(float)

    def score(self, k):
        if k in self.A or k in self.B:
            return (self.A[k] + self.B[k]) / 2
        return 0.0

    def _upd(self, step, nxt):
        ks, j, r = step
        k = ks[j]
        X, Y = (self.A, self.B) if self.rng.random() < 0.5 else (self.B, self.A)
        if nxt is None:
            target = r
        else:
            ks1 = nxt[0]
            vals = [X[x] for x in ks1]
            target = r + Y[ks1[vals.index(max(vals))]]
        X[k] += self.hp.get("alpha", 0.05) * (target - X[k])
        self.Q[k] = 0.0
        self.N[k] += 1

    def online(self, traj):
        if len(traj) >= 2:
            self._upd(traj[-2], traj[-1])

    def end(self, traj):
        self._upd(traj[-1], None)

    def table(self):
        return {k: (self.A[k] + self.B[k]) / 2 for k in self.Q}


class NStepSarsa(Episodic):
    chapter = "7.2"
    label = "SARSA de n pasos"
    desc = "Suma las recompensas de n turnos reales y completa con V del afterstate n turnos después (n = 3)."

    def end(self, traj):
        n, alpha = self.hp.get("n", 3), self.hp.get("alpha", 0.05)
        T = len(traj)
        for t in range(T):
            g = sum(traj[i][2] for i in range(t, min(t + n, T)))
            if t + n < T:
                ks, j, _ = traj[t + n]
                g += self.Q[ks[j]]
            ks, j, _ = traj[t]
            k = ks[j]
            self.N[k] += 1
            self.Q[k] += alpha * (g - self.Q[k])


class TreeBackup(Episodic):
    chapter = "7.5"
    label = "Tree backup de n pasos"
    desc = "n pasos off-policy sin importance sampling: en cada paso mezcla lo jugado con el valor de la jugada greedy."

    def end(self, traj):
        n, alpha = self.hp.get("n", 3), self.hp.get("alpha", 0.05)
        T = len(traj)
        for t in range(T):
            end = min(t + n, T)
            if end == T:
                g = traj[T - 1][2]
            else:
                ks = traj[end][0]
                g = traj[end - 1][2] + self.Q[ks[self.greedy(ks)]]
            for i in range(end - 1, t, -1):
                ks, j, _ = traj[i]
                gi = self.greedy(ks)
                # política objetivo greedy: si lo jugado no es la greedy, el valor viene de la greedy
                g = traj[i - 1][2] + (g if j == gi else self.Q[ks[gi]])
            ks, j, _ = traj[t]
            k = ks[j]
            self.N[k] += 1
            self.Q[k] += alpha * (g - self.Q[k])


class DynaQ(QLearning):
    chapter = "8.2"
    label = "Dyna-Q"
    desc = "Q-learning más planificación: repasa transiciones guardadas en un modelo aprendido (n repasos por paso)."

    def __init__(self, rng, **hp):
        super().__init__(rng, **hp)
        self.model = defaultdict(list)  # k -> [(r, opciones siguientes o None)]
        self.visited = []

    def _store(self, k, r, ks1):
        bucket = self.model[k]
        if not bucket:
            self.visited.append(k)
        if len(bucket) < 50:
            bucket.append((r, ks1))
        else:
            bucket[self.rng.randrange(50)] = (r, ks1)

    def _plan(self):
        alpha = self.hp.get("alpha", 0.05)
        for _ in range(self.hp.get("planning", 5)):
            k = self.rng.choice(self.visited)
            r, ks1 = self.rng.choice(self.model[k])
            target = r if ks1 is None else r + max(self.Q[x] for x in ks1)
            self.Q[k] += alpha * (target - self.Q[k])

    def online(self, traj):
        if len(traj) >= 2:
            ks, j, r = traj[-2]
            self._store(ks[j], r, traj[-1][0])
            super().online(traj)
            self._plan()

    def end(self, traj):
        ks, j, r = traj[-1]
        self._store(ks[j], r, None)
        super().end(traj)
        self._plan()


class PrioritizedSweeping(DynaQ):
    chapter = "8.4"
    label = "Prioritized sweeping"
    desc = "Dyna que repasa primero los afterstates cuyo valor cambió más y propaga hacia sus predecesores."

    def __init__(self, rng, **hp):
        super().__init__(rng, **hp)
        self.pred = defaultdict(set)  # afterstate siguiente -> afterstates previos
        self.queue = []

    def _priority(self, k):
        samples = self.model.get(k)
        if not samples:
            return 0.0
        r, ks1 = self.rng.choice(samples)
        target = r if ks1 is None else r + max(self.Q[x] for x in ks1)
        return abs(target - self.Q[k])

    def _store(self, k, r, ks1):
        super()._store(k, r, ks1)
        if ks1 is not None:
            for x in ks1:
                self.pred[x].add(k)
        p = self._priority(k)
        if p > self.hp.get("theta", 0.01):
            heapq.heappush(self.queue, (-p, self.rng.random(), k))

    def _plan(self):
        alpha = self.hp.get("alpha", 0.05)
        for _ in range(self.hp.get("planning", 5)):
            if not self.queue:
                return
            _, _, k = heapq.heappop(self.queue)
            r, ks1 = self.rng.choice(self.model[k])
            target = r if ks1 is None else r + max(self.Q[x] for x in ks1)
            self.Q[k] += alpha * (target - self.Q[k])
            for prev in list(self.pred[k])[:10]:
                p = self._priority(prev)
                if p > self.hp.get("theta", 0.01):
                    heapq.heappush(self.queue, (-p, self.rng.random(), prev))
        if len(self.queue) > 5000:
            self.queue = heapq.nsmallest(2000, self.queue)
            heapq.heapify(self.queue)


class SarsaLambda(Episodic):
    chapter = "12.7"
    label = "SARSA(λ) tabular"
    desc = "SARSA con trazas de elegibilidad (reemplazo): el error de cada turno corrige también a los anteriores."

    def end(self, traj):
        lam, alpha = self.hp.get("lam", 0.8), self.hp.get("alpha", 0.05)
        z = defaultdict(float)
        for t, (ks, j, r) in enumerate(traj):
            k = ks[j]
            nxt = self.Q[traj[t + 1][0][traj[t + 1][1]]] if t + 1 < len(traj) else 0.0
            delta = r + nxt - self.Q[k]
            z[k] = 1.0  # traza de reemplazo
            for x in list(z):
                self.Q[x] += alpha * delta * z[x]
                z[x] *= lam
            self.N[k] += 1


class WatkinsQLambda(Episodic):
    chapter = "12.10"
    label = "Q(λ) de Watkins"
    desc = "Q-learning con trazas: las trazas se cortan cuando la jugada no fue la greedy."

    def end(self, traj):
        lam, alpha = self.hp.get("lam", 0.8), self.hp.get("alpha", 0.05)
        z = defaultdict(float)
        for t, (ks, j, r) in enumerate(traj):
            k = ks[j]
            nxt = max(self.Q[x] for x in traj[t + 1][0]) if t + 1 < len(traj) else 0.0
            delta = r + nxt - self.Q[k]
            z[k] = 1.0
            for x in list(z):
                self.Q[x] += alpha * delta * z[x]
                z[x] *= lam
            self.N[k] += 1
            if t + 1 < len(traj) and traj[t + 1][1] != self.greedy(traj[t + 1][0]):
                z.clear()


ALGORITHMS = {
    "epsilon_greedy": (EpsilonGreedy, dict(epsilon=0.1)),
    "optimista": (Optimistic, dict(alpha=0.1)),
    "ucb": (UCB, dict(c=0.2)),
    "gradiente_bandido": (GradientBandit, dict(alpha=0.1)),
    "mc_on_policy": (MonteCarloOnPolicy, dict(epsilon=0.1)),
    "mc_off_policy": (MonteCarloOffPolicy, dict(epsilon=0.3)),
    "sarsa": (Sarsa, dict(epsilon=0.1, alpha=0.05)),
    "q_learning": (QLearning, dict(epsilon=0.1, alpha=0.05)),
    "expected_sarsa": (ExpectedSarsa, dict(epsilon=0.1, alpha=0.05)),
    "double_q": (DoubleQ, dict(epsilon=0.1, alpha=0.05)),
    "sarsa_n": (NStepSarsa, dict(epsilon=0.1, alpha=0.05, n=3)),
    "tree_backup": (TreeBackup, dict(epsilon=0.3, alpha=0.05, n=3)),
    "dyna_q": (DynaQ, dict(epsilon=0.1, alpha=0.05, planning=5)),
    "prioritized_sweeping": (PrioritizedSweeping, dict(epsilon=0.1, alpha=0.05, planning=5, theta=0.01)),
    "sarsa_lambda": (SarsaLambda, dict(epsilon=0.1, alpha=0.05, lam=0.8)),
    "q_lambda": (WatkinsQLambda, dict(epsilon=0.1, alpha=0.05, lam=0.8)),
}
