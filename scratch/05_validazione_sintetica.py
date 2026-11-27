#!/usr/bin/env python
# coding: utf-8

# # 05 — Validazione del GP su epidemie sintetiche
# 
# **Domanda.** Prima di interpretare il GP sui dati veri (notebook 06), che cosa sa ritrovare quando la risposta è nota? In particolare:
# 1. se β dipende dalla mobilità con un certo ritardo, il GP trova **la variabile** e **il ritardo** giusti, anche se i ritardi della mobilità sono fortemente correlati fra loro?
# 2. se β **non** dipende dalla mobilità (solo dal tempo, o dalle restrizioni), il GP con la mobilità **inventa un effetto** che non c'è? Cioè: il risultato del notebook 06 "la mobilità prevede meglio del tempo" potrebbe uscire anche senza un vero effetto della mobilità?
# 3. se β è costante, il GP resta su formule semplici?
# 
# **Come.** Si simula il SEIRD con una β(t) scelta da noi, si aggiunge rumore, e si passa tutto alla **stessa pipeline** del notebook 06: γ(t) e μ(t) stimati dai dati (sintetici), congelati nel test; E₀ stimato per ogni formula; β_cost dalla baseline costante; stessi terminali, stesse impostazioni del GP, stessa regola di scelta (la più semplice entro il 5% del miglior errore di training).
# 
# Per restare vicini al caso reale, dai dati veri della seconda ondata (1/9/2020 – 31/1/2021) vengono: la **mobilità** (con i suoi ritardi correlati), lo **stato iniziale** (I₀, R₀, D₀), **γ(t) e μ(t)** usati per generare i dati, e il **livello** di β (baseline costante, σ = 1/4). σ è noto e uguale a quello usato nel fit (1/4): qui non si studia l'errore su σ.
# 
# **Scenari** (fissati il 12/11/2026, prima di guardare i risultati sintetici; codice in `src/seird_gp/synthetic.py`):
# 
# | | Nome | β vera | A cosa serve |
# |---|---|---|---|
# | A | mobilità | c · exp(1.5 · work_lag7) | ritrova variabile e ritardo? Ritardo vero **7**, diverso dal 3 trovato sui dati veri, per non costruire uno scenario che dia ragione al notebook 06 |
# | B | tempo | c · exp(−0.2 · t), t in mesi | controllo negativo: la mobilità nella seconda ondata è in parte correlata con t |
# | C | restrizioni | c · fattore per livello (gialla 0.75, rossa/arancione 0.55, …), a gradini nelle date dei provvedimenti | controllo negativo: restrizioni e mobilità cambiano insieme |
# | D | costante | c | il GP deve restare su formule quasi costanti |
# 
# c è scelta in modo che la media geometrica di β sul training sia uguale al livello dei dati veri. Rumore: binomiale negativa (k = 20) sui guariti e decessi giornalieri, log-normale (5%) sui positivi attuali.
# 
# Run con `scripts/run_synthetic.py` (4 scenari × 3 configurazioni × 5 seed). Qui si leggono i risultati da `results/synthetic/` e `results/gp_synth/`.

# **Criteri, fissati prima dei risultati.** Le soglie sono convenzionali ma decise ora, così non si adattano ai risultati.
# 
# | Scenario | Criterio | Se non è rispettato |
# |---|---|---|
# | A | A1: nelle configurazioni con la mobilità, almeno il 60% delle formule scelte usa `work` | il GP non sa riconoscere quale indice conta: nel notebook 06 non si può dire "conta work" |
# | A | A2: fra le formule che usano `work`, almeno il 50% usa il ritardo 7 → ritardo **identificato**; se invece la maggioranza usa ritardi fra 3 e 10 → identificato **solo a ±4 giorni** | nel notebook 06 il "ritardo di 3 giorni" non si può interpretare (o solo come "fra 0 e 7 circa") |
# | A | A3: con tempo + mobilità, al massimo il 30% delle formule usa t | il GP preferisce t anche quando la causa è la mobilità |
# | B, C | B/C: rapporto fra l'errore di test mediano della configurazione "mobilità" e quello di "tempo". Sui dati veri (notebook 06) è circa 0.46. Se negli scenari senza effetto della mobilità il rapporto è anch'esso ≤ 0.6 | il risultato "la mobilità prevede meglio del tempo" **non è una prova** di un effetto della mobilità: esce anche senza |
# | D | D1: almeno il 60% delle formule scelte è quasi costante (β entro ±20% del suo valore mediano su tutta la finestra) | il GP inventa variazioni dal rumore |
# | tutti | errore di test mediano della configurazione "giusta" (A: mobilità, B: tempo, D: qualunque) entro 1.5 volte l'oracolo | la ricerca non arriva alla formula giusta anche quando può esprimerla |
# 
# Lo scenario C non ha una configurazione "giusta" (i gradini non si scrivono con t o con la mobilità): serve a vedere quanto la mobilità **imita** le restrizioni.

# In[1]:


import sys
print(sys.executable)


# In[2]:


import glob, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seird_gp.data import PROCESSED_DIR, RESULTS_DIR
from seird_gp.beta_models import load_restrictions
from seird_gp.synthetic import GP_SYNTH_DIR, SCENARIOS, SYNTH_DIR

SIGMA = "4"           # σ = 1/4, come nel notebook 06
TOL = 0.05            # stessa regola di scelta del notebook 06
CONFIG_ORDER = ["tempo", "mobilita", "tempo+mobilita"]
RIGHT_CONFIG = {"A": "mobilita", "B": "tempo", "C": None, "D": None}
CRIT = {"A1_work": 0.6, "A2_lag7": 0.5, "A3_t_max": 0.3, "BC_ratio": 0.6,
        "D1_flat": 0.6, "D1_band": np.log(1.2), "oracle_x": 1.5}
REAL_RATIO = 0.46     # mobilità / tempo, errore di test mediano sui dati veri (notebook 06, σ = 1/4)

def choose(front, tol=TOL):
    ok = front[front["fitness_norm train"] <= (1 + tol) * front["fitness_norm train"].min()]
    return ok.sort_values(["nodi", "fitness_norm train"]).iloc[0]

def lags(terms):
    return sorted({int(t.split("lag")[1]) for t in terms if "lag" in t})

meta = {k: json.load(open(SYNTH_DIR / f"{k}.json")) for k in SCENARIOS if (SYNTH_DIR / f"{k}.json").exists()}
data = {k: pd.read_csv(SYNTH_DIR / f"{k}.csv", index_col=0, parse_dates=True) for k in meta}

rows = []
for f in sorted(glob.glob(str(GP_SYNTH_DIR / f"*_s{SIGMA}_*.json"))):
    m = re.search(r"([A-Z])_s([\d.]+)_(.+)_seed(\d+)\.json$", f)
    front = pd.DataFrame(json.load(open(f)))
    front["beta"] = front["beta"].apply(np.array)
    ch = choose(front)
    terms = list(ch["terminali usati"])
    logb = np.log(ch["beta"])
    rows.append({"scenario": m.group(1), "config": m.group(3), "seed": int(m.group(4)),
                 "formula": ch["formula"], "nodi": int(ch["nodi"]),
                 "train": ch["fitness_norm train"], "test": ch["fitness_norm test"],
                 "terminali": terms, "ritardi": lags(terms),
                 "usa t": "t" in terms, "usa work": any(t.startswith("work") for t in terms),
                 "usa retail": any(t.startswith("retail") for t in terms),
                 "usa mobilità": any("lag" in t for t in terms),
                 "lag work": sorted({int(t.split("lag")[1]) for t in terms if t.startswith("work")}),
                 "quasi costante": bool(np.abs(logb - np.median(logb)).max() <= CRIT["D1_band"]),
                 "beta": ch["beta"]})
chosen = pd.DataFrame(rows)
print(len(chosen), "run;", chosen.groupby(["scenario", "config"]).size().to_dict())


# In[ ]:


for k in SCENARIOS:
    p = SYNTH_DIR / f"{k}.json"
    if p.exists():
        txt = p.read_text(encoding="utf-8")
        print(k, p.stat().st_size, repr(txt[:80]))


# ## 1. Gli scenari
# 
# A sinistra la β vera; a destra i positivi attuali sintetici (con rumore) accanto a quelli veri, per controllare che le epidemie simulate abbiano dimensioni realistiche. In grigio il periodo di test.

# In[1]:


import numpy as np, pandas as pd
print("ok")


# In[2]:


print("ancora vivo")


# In[3]:


import matplotlib
print(matplotlib.__version__, matplotlib.get_backend())
import matplotlib.pyplot as plt


# In[4]:


print("vivo dopo matplotlib?")


# In[ ]:


real = pd.read_csv(PROCESSED_DIR / "trento_clean.csv", index_col=0, parse_dates=True)
restr = load_restrictions()
fig, ax = plt.subplots(len(meta), 2, figsize=(12, 2.6 * len(meta)), sharex=True, squeeze=False)
for i, (k, d) in enumerate(data.items()):
    n_train = int(round(len(d) * 2 / 3))
    ax[i, 0].plot(d.index, d["beta vero"], color="#a33")
    ax[i, 0].set_ylabel("β vera")
    ax[i, 0].set_title(f"{k} — {meta[k]['nome']}: {meta[k]['formula vera']}", fontsize=10, loc="left")
    ax[i, 1].plot(d.index, d["totale_positivi"], color="#1f4e79", lw=1, label="sintetici")
    ax[i, 1].plot(d.index, real.loc[d.index, "totale_positivi"], color="#7f7f7f", lw=1, label="veri")
    ax[i, 1].set_ylabel("positivi attuali")
    for a in ax[i]:
        a.axvspan(d.index[n_train], d.index[-1], color="#eee")
        for s in restr.inizio[(restr.inizio > d.index[0]) & (restr.inizio <= d.index[-1])]:
            a.axvline(s, color="#a33", lw=0.5, ls="--")
ax[0, 1].legend(fontsize=8)
plt.tight_layout()

pd.DataFrame({k: {"β_cost": m["β_cost (baseline costante sui dati sintetici)"],
                  "livello vero": m["livello β (dati veri)"], "E0 vero": m["E0 vero"],
                  "E0 oracolo": m["oracolo E0"]} for k, m in meta.items()}).T.round(4)


# ## 2. Quanto errore resta anche con la β vera
# 
# L'**oracolo** è la formula vera valutata dentro la pipeline: è il meglio che il GP possa ottenere, e il termine di confronto per tutto il resto. Le altre colonne scompongono il suo errore:
# - **solo rumore**: traiettoria vera senza rumore contro i dati rumorosi;
# - **γ μ veri**: oracolo con i tassi usati per generare i dati, anche nel test;
# - **γ μ veri congelati**: come sopra, ma con i tassi congelati nel test come nella pipeline. La differenza con la colonna precedente è l'effetto del **congelamento** nel test;
# - **oracolo**: con γ e μ **stimati** dai dati (media mobile all'indietro di 7 giorni), come nella pipeline. La differenza con la colonna precedente è l'effetto della **stima** dei tassi, che è in ritardo rispetto a quelli veri.
# 
# Questa scomposizione vale anche per i dati veri: una parte dell'errore del notebook 06 non dipende da β, ma da come si stimano e si congelano γ e μ.

# In[ ]:


cols = ["solo rumore", "oracolo, γ μ veri", "oracolo, γ μ veri congelati", "oracolo"]
oracle_tab = pd.DataFrame({(k, p): {c: m[f"{c} {p}"] for c in cols}
                           for k, m in meta.items() for p in ("train", "test")}).T
oracle_tab.index.names = ["scenario", "periodo"]
oracle_tab.round(3)


# ## 3. Errore del GP rispetto all'oracolo
# 
# Mediana e intervallo interquartile sui seed della formula scelta. `/ oracolo` = errore di test mediano diviso per l'errore di test dell'oracolo (1 = come la formula vera; < 1 è possibile per caso nel test, o perché una formula sbagliata compensa gli errori su γ e μ).

# In[ ]:


def iqr(x):
    q = x.quantile([.25, .75])
    return f"{q.iloc[0]:.2f}–{q.iloc[1]:.2f}"

summ = (chosen.groupby(["scenario", "config"])
        .agg(train=("train", "median"), test=("test", "median"), test_IQR=("test", iqr),
             nodi=("nodi", "median"), seed=("seed", "count"))
        .reindex(CONFIG_ORDER, level="config"))
summ["oracolo test"] = [meta[k]["oracolo test"] for k in summ.index.get_level_values("scenario")]
summ["/ oracolo"] = summ["test"] / summ["oracolo test"]
summ.to_csv(RESULTS_DIR / "05_sintetici_gp.csv")
display(summ.round(3))

ratio = (summ["test"].xs("mobilita", level="config") / summ["test"].xs("tempo", level="config")).rename("mobilità / tempo")
print(f"Rapporto fra gli errori di test mediani (sui dati veri: {REAL_RATIO})")
ratio.round(2)


# ## 4. Che cosa usano le formule scelte
# 
# Quota di formule (sui seed) che usano ciascun ingrediente, e ritardi di `work` usati. Nello scenario A la formula vera usa solo `work_lag7`; negli scenari B, C e D nessuna formula dovrebbe aver bisogno della mobilità.

# In[ ]:


use = (chosen.groupby(["scenario", "config"])[["usa mobilità", "usa work", "usa retail", "usa t", "quasi costante"]]
       .mean().reindex(CONFIG_ORDER, level="config"))
display(use.round(2))

mob = chosen[chosen["config"] != "tempo"]
lag_counts = (mob.explode("lag work").dropna(subset=["lag work"])
              .groupby(["scenario", "config", "lag work"]).size().unstack(fill_value=0))
print("Ritardi di work usati (numero di formule):")
lag_counts


# ## 5. Verifica dei criteri

# In[ ]:


def share(df, col):
    return df[col].mean() if len(df) else np.nan

checks = []
A = chosen[chosen["scenario"] == "A"]
if len(A):
    Am = A[A["config"] != "tempo"]
    w = Am[Am["usa work"]]
    lag7 = w["lag work"].apply(lambda l: 7 in l).mean() if len(w) else np.nan
    near = w["lag work"].apply(lambda l: len(l) > 0 and all(3 <= x <= 10 for x in l)).mean() if len(w) else np.nan
    checks += [
        ("A1", "formule con la mobilità che usano work", share(Am, "usa work"), f"≥ {CRIT['A1_work']}",
         share(Am, "usa work") >= CRIT["A1_work"]),
        ("A2", "di queste, con il ritardo 7", lag7, f"≥ {CRIT['A2_lag7']}", lag7 >= CRIT["A2_lag7"]),
        ("A2b", "di queste, solo ritardi fra 3 e 10", near, "(se A2 fallisce)", near >= CRIT["A2_lag7"]),
        ("A3", "tempo+mobilità: formule che usano t", share(A[A["config"] == "tempo+mobilita"], "usa t"),
         f"≤ {CRIT['A3_t_max']}", share(A[A["config"] == "tempo+mobilita"], "usa t") <= CRIT["A3_t_max"]),
    ]
for k in ["B", "C"]:
    if k in ratio.index:
        checks.append((k, "errore test mobilità / tempo (nessun effetto vero della mobilità)", ratio[k],
                       f"> {CRIT['BC_ratio']} (dati veri: {REAL_RATIO})", ratio[k] > CRIT["BC_ratio"]))
D = chosen[chosen["scenario"] == "D"]
if len(D):
    checks.append(("D1", "formule quasi costanti (β entro ±20%)", share(D, "quasi costante"),
                   f"≥ {CRIT['D1_flat']}", share(D, "quasi costante") >= CRIT["D1_flat"]))
for k, c in RIGHT_CONFIG.items():
    sel = summ.loc[k] if k in summ.index.get_level_values("scenario") else None
    if sel is None:
        continue
    x = sel.loc[c, "/ oracolo"] if c else sel["/ oracolo"].min()
    checks.append((f"{k}-oracolo", f"errore di test / oracolo ({c or 'migliore configurazione'})", x,
                   f"≤ {CRIT['oracle_x']}", x <= CRIT["oracle_x"]))
crit = pd.DataFrame(checks, columns=["criterio", "cosa", "valore", "soglia", "rispettato"]).set_index("criterio")
crit["valore"] = crit["valore"].astype(float).round(2)
crit


# ## 6. β(t) trovata contro β vera
# 
# Una curva per seed (blu), β vera in rosso. Il periodo di test è in grigio.

# In[ ]:


keys = [k for k in SCENARIOS if k in data]
fig, ax = plt.subplots(len(keys), len(CONFIG_ORDER), figsize=(13, 2.6 * len(keys)), sharex=True, squeeze=False)
for i, k in enumerate(keys):
    d = data[k]
    n_train = int(round(len(d) * 2 / 3))
    for j, c in enumerate(CONFIG_ORDER):
        a = ax[i, j]
        for _, r in chosen[(chosen["scenario"] == k) & (chosen["config"] == c)].iterrows():
            a.plot(d.index, r["beta"], color="#1f4e79", alpha=0.4, lw=1)
        a.plot(d.index, d["beta vero"], color="#a33", lw=2)
        a.axvspan(d.index[n_train], d.index[-1], color="#eee")
        a.set_title(f"{k} ({meta[k]['nome']}) — GP {c}", fontsize=9)
        a.set_ylim(0, 2.5 * d["beta vero"].max())
plt.tight_layout()


# ## 7. Lettura
# 
# *(da scrivere dopo i risultati)* Domande guida:
# - **Variabile e ritardo (A1, A2).** Il GP trova `work` e il ritardo 7? Se trova solo un ritardo "vicino", con quale precisione si può leggere il ritardo di 3 giorni del notebook 06?
# - **Falsi positivi (B, C).** Senza un effetto vero della mobilità, la configurazione "mobilità" fa comunque molto meglio di "tempo"? Se sì, il confronto del notebook 06 non basta: serve il confronto con la baseline a tratti (o un terminale per le restrizioni).
# - **Formule semplici (D1).** Con β costante, il GP resta su formule quasi costanti?
# - **Oracolo (sezione 2).** Quanto dell'errore viene dalla stima e dal congelamento di γ e μ, e non da β? Che cosa implica per i numeri del notebook 06?
# 
# **Limiti.** Una sola realizzazione del rumore per scenario; σ noto e uguale nel fit; stessa finestra; la β vera ha la stessa forma esponenziale usata dal GP nello scenario A (caso favorevole).
