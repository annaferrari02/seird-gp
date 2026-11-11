"""Genetic programming per β(t) (DEAP + NSGA-II).

Scelte (11/11/2026, vedi notebook 05):
- β(t) = β_cost · exp(g(x(t))), con g l'albero evoluto e β_cost il β della baseline costante
  nella stessa finestra: sempre positiva, effetti moltiplicativi ("10 punti di mobilità in meno
  moltiplicano β per e^(−0.1c)"), e g = 0 riproduce la baseline costante. Senza β_cost il GP
  doveva prima indovinare il livello di β (≈ 0.1): le formule con la struttura giusta ma costanti
  casuali facevano esplodere l'epidemia e venivano eliminate prima di essere ottimizzate.
  Versione protetta: β fra BETA_MIN e BETA_MAX;
- tre configurazioni di terminali: solo t, solo mobilità, t + mobilità (domanda D6);
- σ fisso, γ(t) e μ(t) dai dati (come le baseline), β_cost dalla baseline costante nella stessa
  finestra; E0 stimato per ogni formula (ricerca in una dimensione), come nelle baseline: con E0
  fisso la baseline lineare in work passava da errore 0.14 a 1.54;
- due obiettivi da minimizzare con NSGA-II: fitness normalizzata sul training e numero di
  nodi (parsimony pressure contro il bloat), profondità massima 6;
- (μ + λ) con variazione "crossover OPPURE mutazione" (varOr), come nella lezione sul GP;
  duplicati eliminati prima della selezione e rimpiazzati da individui casuali (diversità);
- ogni 5 generazioni le costanti delle formule non dominate si ottimizzano con Nelder-Mead;
- integratore RK4 a passo fisso (`model.simulate_fixed`): circa 2 ms per valutazione.

Regola sul test, come per le baseline: nessun dato futuro degli esiti; la mobilità osservata
si usa anche nel test, γ e μ restano all'ultimo valore del training.
"""

import copy
import functools
import json
import operator
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools
from scipy.optimize import minimize, minimize_scalar

from .data import RESULTS_DIR, TRENTO_POPULATION
from .fitting import (
    data_scale, empirical_rates, freeze_after, initial_state, normalized_fitness, rmsd,
)
from .model import COMPARTMENTS, SEIRDParams, simulate_fixed

BETA_MIN, BETA_MAX = 1e-4, 2.0
LOG_BETA_MIN, LOG_BETA_MAX = np.log(BETA_MIN), np.log(BETA_MAX)
MAX_DEPTH = 6
BAD = 1e6  # fitness di una formula che non si può valutare

MOBILITY_TERMINALS = [f"{name}_lag{k}" for name in ("retail", "work") for k in (0, 3, 7, 10, 14)]
CONFIGS = {
    "tempo": ["t"],
    "mobilita": MOBILITY_TERMINALS,
    "tempo+mobilita": ["t"] + MOBILITY_TERMINALS,
}
T_SCALE = 30.0  # t in mesi (giorni / 30): costanti di ordine 1 bastano


# --- Primitive protette (closure) -----------------------------------------------------------

def protected_div(a, b):
    """a / b, con 1 dove |b| < 1e-6 (convenzione classica del GP)."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    out = np.ones_like(a)
    ok = np.abs(b) > 1e-6
    out[ok] = a[ok] / b[ok]
    return out


def _random_constant(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 3)


def build_pset(terminals) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("BETA", len(terminals))
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(terminals)})
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(protected_div, 2, name="div")
    pset.addPrimitive(np.negative, 1, name="neg")
    # costanti casuali (ephemeral random constants); g è uno scarto da β_cost, quindi piccolo
    pset.addEphemeralConstant("c", functools.partial(_random_constant, -2.0, 2.0))
    return pset


if not hasattr(creator, "FitnessGP"):
    creator.create("FitnessGP", base.Fitness, weights=(-1.0, -1.0))  # errore, nodi
    creator.create("IndividualGP", gp.PrimitiveTree, fitness=creator.FitnessGP)


# --- Problema: dati della finestra già preparati -------------------------------------------

@dataclass
class Problem:
    """Tutto ciò che serve per valutare una formula su una finestra."""
    obs: pd.DataFrame            # dati della finestra (train + test)
    n_train: int
    X: pd.DataFrame              # terminali, stesso indice di obs
    sigma: float
    E0: float | None             # None: E0 stimato per ogni formula (come nelle baseline)
    rates: pd.DataFrame          # γ, μ dai dati, congelati dopo il training
    beta_ref: float = 1.0        # β della baseline costante: β(t) = beta_ref · exp(g)
    max_abs_g: float | None = None  # vincolo di plausibilità (None = nessuno), vedi `plausible`
    N: float = TRENTO_POPULATION
    scale: dict = field(init=False)

    def __post_init__(self):
        self.scale = data_scale(self.obs.iloc[:self.n_train])
        self.cols = {c: self.X[c].to_numpy(float) for c in self.X}
        # stesso intervallo per E0 delle baseline (fit_baseline, fit_beta)
        self.E0_max = 10 * max(float(self.obs["totale_positivi"].iloc[0]), 50.0)
        r = self.rates
        self._params = {n: SEIRDParams(self.sigma, r["gamma"].to_numpy()[:n], r["mu"].to_numpy()[:n])
                        for n in (self.n_train, len(self.obs))}

    def beta(self, func, terminals, n=None) -> np.ndarray:
        """β giornaliera della formula sui primi n giorni (tutti se n è None)."""
        n = n or len(self.obs)
        args = [self.cols[c][:n] for c in terminals]
        with np.errstate(all="ignore"):
            g = np.broadcast_to(np.asarray(func(*args), float), (n,))
        if not np.all(np.isfinite(g)):
            raise FloatingPointError("formula non finita")
        return np.exp(np.clip(np.log(self.beta_ref) + g, LOG_BETA_MIN, LOG_BETA_MAX))

    def simulate(self, beta: np.ndarray, E0: float, n=None) -> pd.DataFrame:
        n = n or len(self.obs)
        y = simulate_fixed(beta, self._params[n], initial_state(self.obs, E0, self.N), n)
        return pd.DataFrame(y, index=self.obs.index[:n], columns=COMPARTMENTS)

    def _error_given_E0(self, beta_train: np.ndarray, E0: float) -> float:
        sim = self.simulate(beta_train, E0, self.n_train)
        err = normalized_fitness(sim, self.obs.iloc[:self.n_train], scale=self.scale)
        return err if np.isfinite(err) else BAD

    def fit(self, func, terminals) -> tuple:
        """(errore sul training, E0) della formula. Se E0 non è fissato, si cerca il migliore
        in [0, E0_max] con una ricerca in una dimensione (circa 15–20 simulazioni): E0 e la
        forma di β sono legati, e con E0 fisso le formule buone risultano pessime."""
        try:
            if not self.plausible(func, terminals):
                return BAD, np.nan
            b = self.beta(func, terminals, self.n_train)
            if self.E0 is not None:
                return self._error_given_E0(b, self.E0), self.E0
            res = minimize_scalar(lambda e: self._error_given_E0(b, e), bounds=(0.0, self.E0_max),
                                  method="bounded", options={"xatol": 0.5, "maxiter": 25})
            return float(res.fun), float(res.x)
        except (FloatingPointError, RuntimeError, ValueError, OverflowError):
            return BAD, np.nan

    def plausible(self, func, terminals) -> bool:
        """Vincolo di plausibilità (death penalty, lezione sui vincoli): |g| ≤ max_abs_g, cioè
        β fra β_cost·e^(−max_abs_g) e β_cost·e^(max_abs_g), su TUTTA la finestra. Usa solo gli
        input (t e mobilità, noti anche nel test), mai gli esiti: non viola la regola sul test.
        Scarta le formule con singolarità, es. c / (2.2 − t), che piegano β sul training e
        divergono fuori (nella prova con σ = 1/3: errori di test fino a 150)."""
        if self.max_abs_g is None:
            return True
        args = [self.cols[c] for c in terminals]
        with np.errstate(all="ignore"):
            g = np.broadcast_to(np.asarray(func(*args), float), (len(self.obs),))
        return bool(np.all(np.isfinite(g)) and np.abs(g).max() <= self.max_abs_g)

    def train_error(self, func, terminals) -> float:
        return self.fit(func, terminals)[0]

    def evaluate_full(self, func, terminals) -> dict:
        """Errori su train e test e serie β(t), con le stesse misure delle baseline."""
        _, E0 = self.fit(func, terminals)
        b = self.beta(func, terminals)
        sim = self.simulate(b, E0)
        out = {"beta": pd.Series(b, index=self.obs.index), "simulation": sim, "E0": E0}
        for period, sl in {"train": slice(None, self.n_train), "test": slice(self.n_train, None)}.items():
            o, s = self.obs.iloc[sl], sim.iloc[sl]
            out[f"fitness_norm {period}"] = normalized_fitness(s, o, scale=self.scale)
            out[f"RMSD D {period}"] = rmsd(s["D"], o["deceduti"])
            out[f"RMSD I {period}"] = rmsd(s["I"], o["totale_positivi"])
        return out


def make_problem(df: pd.DataFrame, inputs: pd.DataFrame, start: str, end: str, sigma: float,
                 beta_ref: float, E0: float | None = None, train_frac: float = 2 / 3,
                 max_abs_g: float | None = None) -> Problem:
    """Prepara la finestra: terminali (t in mesi + mobilità), γ(t), μ(t) congelati nel test."""
    obs = df.loc[start:end]
    n_train = max(2, int(round(len(obs) * train_frac)))
    X = inputs.reindex(obs.index)[MOBILITY_TERMINALS].copy()
    if X.isna().any().any():
        raise ValueError("Input di mobilità mancanti nella finestra")
    X["t"] = np.arange(len(obs)) / T_SCALE
    rates = freeze_after(empirical_rates(df).loc[obs.index, ["gamma", "mu"]], obs.index[n_train - 1])
    return Problem(obs, n_train, X, sigma, E0, rates, beta_ref, max_abs_g)


# --- Algoritmo -------------------------------------------------------------------------------

@dataclass
class GPConfig:
    pop_size: int = 200       # μ
    n_offspring: int = 200    # λ
    n_gen: int = 50
    cxpb: float = 0.8         # crossover OPPURE mutazione (varOr): cxpb + mutpb <= 1
    mutpb: float = 0.2
    max_depth: int = MAX_DEPTH
    init_depth: tuple = (1, 3)
    tune_every: int = 5       # ogni quante generazioni si ottimizzano le costanti del front
    tune_evals: int = 60      # valutazioni di Nelder-Mead per formula (0 = niente ottimizzazione)


def make_toolbox(problem: Problem, terminals, cfg: GPConfig):
    pset = build_pset(terminals)
    tb = base.Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=cfg.init_depth[0], max_=cfg.init_depth[1])
    tb.register("individual", tools.initIterate, creator.IndividualGP, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("compile", gp.compile, pset=pset)

    def evaluate(ind):
        return problem.train_error(tb.compile(expr=ind), terminals), float(len(ind))

    def mutate(ind):
        """Una delle mutazioni della lezione sul GP, scelta a caso."""
        r = random.random()
        if r < 0.4:   # sottoalbero nuovo (growth)
            return gp.mutUniform(ind, expr=tb.expr_mut, pset=pset)
        if r < 0.6:   # cambio di un nodo (function/terminal node mutation)
            return gp.mutNodeReplacement(ind, pset=pset)
        if r < 0.8:   # potatura (truncation)
            return gp.mutShrink(ind)
        return mutate_constants(ind)  # gaussiana sulle costanti

    tb.register("evaluate", evaluate)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genFull, min_=0, max_=2)
    tb.register("mutate", mutate)
    tb.register("select", tools.selNSGA2)
    limit = gp.staticLimit(key=operator.attrgetter("height"), max_value=cfg.max_depth)
    tb.decorate("mate", limit)
    tb.decorate("mutate", limit)
    return tb, pset


def mutate_constants(ind, sd: float = 0.3):
    """Mutazione gaussiana delle costanti: sposta tutte le costanti di un rumore N(0, sd)."""
    for i, node in enumerate(ind):
        if isinstance(type(node), gp.MetaEphemeral):
            new = copy.copy(node)
            new.value = round(node.value + random.gauss(0, sd), 3)
            ind[i] = new
    return ind,


def tune_constants(ind, tb, max_evals: int) -> int:
    """Ottimizzazione locale delle costanti (Nelder-Mead) a struttura fissa.

    Il GP trova bene la *forma* della formula ma lentamente il valore esatto delle costanti:
    è il classico passo "ibrido" (GP + ricerca locale). Aggiorna l'individuo e la fitness solo
    se l'errore migliora. Restituisce il numero di valutazioni usate.
    """
    idx = [i for i, node in enumerate(ind) if isinstance(type(node), gp.MetaEphemeral)]
    if not idx or max_evals <= 0:
        return 0
    x0 = np.array([ind[i].value for i in idx], float)
    # Attenzione: copiando un albero DEAP i nodi restano condivisi con gli altri individui.
    # Le costanti si modificano solo su copie dei nodi, mai in place.
    work = copy.deepcopy(ind)
    for i in idx:
        work[i] = copy.copy(work[i])
    n_evals = 0

    def f(x):
        nonlocal n_evals
        n_evals += 1
        for i, v in zip(idx, x):
            work[i].value = float(v)
        return tb.evaluate(work)[0]

    res = minimize(f, x0, method="Nelder-Mead",
                   options={"maxfev": max_evals, "xatol": 1e-4, "fatol": 1e-6})
    if res.fun < ind.fitness.values[0]:
        for i, v in zip(idx, res.x):
            node = copy.copy(ind[i])
            node.value = round(float(v), 4)
            ind[i] = node
        ind.fitness.values = tb.evaluate(ind)
    return n_evals


def run_gp(problem: Problem, terminals, cfg: GPConfig = GPConfig(), seed: int = 0, verbose=False):
    """Una run di GP: (μ + λ) con NSGA-II, variazione varOr (crossover OPPURE mutazione) e,
    ogni `tune_every` generazioni, ottimizzazione delle costanti delle formule non dominate.
    Restituisce il Pareto front (ordinato per numero di nodi) e il logbook."""
    random.seed(seed)
    np.random.seed(seed)
    tb, pset = make_toolbox(problem, terminals, cfg)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("median", np.median, axis=0)
    log = tools.Logbook()
    log.header = ["gen", "nevals", "min", "median"]
    front = tools.ParetoFront()

    def evaluate_invalid(inds):
        todo = [ind for ind in inds if not ind.fitness.valid]
        for ind in todo:
            ind.fitness.values = tb.evaluate(ind)
        return len(todo)

    def select_unique(candidates):
        """NSGA-II su individui distinti. Senza questo passo la popolazione si riempie di copie
        delle formule più piccole (una costante) e l'evoluzione si ferma; i posti liberi vanno a
        individui nuovi casuali ("immigrati"). Restituisce anche le valutazioni fatte."""
        unique = list({str(ind): ind for ind in candidates}.values())
        extra = [] if len(unique) >= cfg.pop_size else tb.population(n=cfg.pop_size - len(unique))
        n_new = evaluate_invalid(extra)
        return tb.select(unique + extra, cfg.pop_size), n_new

    pop = tb.population(n=cfg.pop_size)
    n = evaluate_invalid(pop)
    pop, k = select_unique(pop)
    n += k
    front.update(pop)
    log.record(gen=0, nevals=n, **stats.compile(pop))
    for gen in range(1, cfg.n_gen + 1):
        offspring = algorithms.varOr(pop, tb, cfg.n_offspring, cfg.cxpb, cfg.mutpb)
        n = evaluate_invalid(offspring)
        pop, k = select_unique(pop + offspring)
        n += k
        if cfg.tune_evals and gen % cfg.tune_every == 0:
            best = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
            for ind in best:
                n += tune_constants(ind, tb, cfg.tune_evals)
            pop = tb.select(pop, cfg.pop_size)  # ricalcola ranghi e crowding
        front.update(pop)
        log.record(gen=gen, nevals=n, **stats.compile(pop))
        if verbose:
            print(log.stream)
    rows = []
    for ind in sorted(front, key=len):
        err, size = ind.fitness.values
        if err >= BAD:
            continue
        full = problem.evaluate_full(tb.compile(expr=ind), terminals)
        rows.append({"formula": str(ind), "nodi": int(size), "profondità": ind.height,
                     "terminali usati": sorted({n.value for n in ind
                                                if isinstance(n, gp.Terminal) and isinstance(n.value, str)}),
                     "E0": full["E0"],
                     **{k: v for k, v in full.items() if k.startswith(("fitness", "RMSD"))},
                     "beta": full["beta"]})
    return pd.DataFrame(rows), log


def knee(front: pd.DataFrame, col: str = "fitness_norm train") -> int:
    """Indice della formula al "gomito" del Pareto front: massima distanza dalla retta
    che unisce la formula più semplice e quella più precisa (assi normalizzati)."""
    if len(front) <= 2:
        return int(front[col].idxmin())
    x = (front["nodi"] - front["nodi"].min()) / max(np.ptp(front["nodi"]), 1)
    y = (front[col] - front[col].min()) / max(np.ptp(front[col]), 1e-12)
    x0, y0, x1, y1 = x.iloc[0], y.iloc[0], x.iloc[-1], y.iloc[-1]
    d = np.abs((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) / max(np.hypot(x1 - x0, y1 - y0), 1e-12)
    return int(front.index[int(np.argmax(d.to_numpy()))])


def save_front(front: pd.DataFrame, path) -> None:
    out = front.drop(columns="beta").copy()
    out["beta"] = [b.round(6).tolist() for b in front["beta"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out.to_dict(orient="records"), ensure_ascii=False, default=str))


GP_RESULTS_DIR = RESULTS_DIR / "gp"
