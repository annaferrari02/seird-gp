"""
Generatore di dati sintetici SIR per il framework PINN + Symbolic Regression.

Genera 4 scenari con beta_true(t) analitica nota, integrazione numerica del
modello SIR, e rumore Poissoniano sui dati osservati — stesso formato di
Millevoi et al. (Case2.txt / Case3.txt).

Scenari:
  S1 — beta costante              (sanity check)
  S2 — decadimento esponenziale   (lockdown progressivo)
  S3 — esponenziale * coseno      (decadimento + stagionalità)
  S4 — step function              (lockdown improvviso al giorno 30)

Output: una cartella data/ con un file .txt per scenario, più un file
        beta_true_values.csv con i valori analitici di beta_true(t)
        per ogni scenario (usato nella validazione SR).
"""

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pathlib import Path

# Parametri globali del modello SIR 

N     = 56e6   # popolazione Italia
delta = 1/5    # tasso di rimozione (periodo medio infettivo = 5 giorni)
I0    = 1      # individui infetti iniziali
S0    = N - I0
R0_ic = 0      # R iniziale (R0_ic per non confondersi con R0 epidemiologico)

T0 = 0         # giorno iniziale
TF = 120        # giorno finale (da 90 a 120)
N_DAYS = int(TF - T0)   # 120 punti giornalieri

# Data di riferimento (stessa di Millevoi et al.)
START_DATE = np.datetime64('2020-02-01')

# Cartella output
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Definizione degli scenari 

def beta_S1(t):
    """S1: beta costante. Caso base per sanity check del PINN."""
    return 0.6  #controllare questo valore... (prima era 0.3, faceva ca. troppo vicino a R0)

def beta_S2(t):
    """S2: decadimento esponenziale. Simula un lockdown progressivo."""
    return 0.5 * np.exp(-0.015 * float(t))

def beta_S3(t):
    """S3: esponenziale * oscillazione coseno.
    Combina lockdown progressivo con variazione stagionale/comportamentale.
    Due meccanismi sovrapposti — lo scenario più realistico e sfidante per SR."""
    t = float(t)
    return 0.5 * np.exp(-0.02 * t) * (1 + 0.4 * np.cos(2 * np.pi * t / 30))

def beta_S4(t):
    """S4: step function. Lockdown improvviso al giorno 30.
    Testa la capacità del PINN di catturare discontinuità."""
    return 0.5 if float(t) < 30 else 0.15

SCENARIOS = {
    "S1": {"func": beta_S1, "label": r"$\beta_0 = 0.3$",
           "description": "Costante"},
    "S2": {"func": beta_S2, "label": r"$0.5 \cdot e^{-0.03t}$",
           "description": "Decadimento esponenziale"},
    "S3": {"func": beta_S3, "label": r"$0.5 e^{-0.02t}(1 + 0.4\cos(2\pi t/30))$",
           "description": "Esponenziale x coseno"},
    "S4": {"func": beta_S4, "label": r"step: 0.5 → 0.15 al giorno 30",
           "description": "Step function"},
}

#  Modello SIR con beta(t) tempo-variante 

def sir_ode(x, t, delta, beta_func, N):
    """ODE del modello SIR con tasso di trasmissione tempo-variante."""
    S, I, R = x
    b = float(beta_func(t))
    force_of_infection = b * I / N
    dSdt = -force_of_infection * S
    dIdt =  force_of_infection * S - delta * I
    dRdt =  delta * I
    return [dSdt, dIdt, dRdt]

#  Generatore principale 

def generate_scenario(name, beta_func, seed=42):
    """
    Integra il SIR, aggiunge rumore Poissoniano, salva il file .txt.

    Returns:
        df       : DataFrame con i dati (formato Millevoi et al.)
        beta_true: array numpy con i valori analitici di beta_true(t)
    """
    rng = np.random.default_rng(seed)

    # Griglia temporale giornaliera
    t_span = np.arange(T0, TF, dtype=float)   # [0, 1, ..., 89]

    # Integrazione numerica del SIR
    x0 = [S0, I0, R0_ic]
    sol = odeint(sir_ode, x0, t_span, args=(delta, beta_func, N))
    S_sol = sol[:, 0]
    I_sol = sol[:, 1]
    R_sol = sol[:, 2]

    # Rumore Poissoniano sui dati osservati di I(t)
    # Clip a 1 per evitare lambda=0 nella distribuzione di Poisson
    I_obs = rng.poisson(lam=np.maximum(I_sol, 1)).astype(float)

    # Valori analitici di beta_true(t) — ground truth per validazione SR
    beta_true = np.array([beta_func(ti) for ti in t_span])

    # Reproduction number R_t = beta(t) / delta
    Rt = beta_true / delta

    # Date calendario
    dates = (START_DATE + t_span.astype('timedelta64[D]')).astype(str)

    # DataFrame nel formato Millevoi et al.
    df = pd.DataFrame({
        "Date":        dates,
        "Susceptible": S_sol,
        "Infectious":  I_sol,
        "Recovered":   R_sol,
        "I data":      I_obs,
        "beta":        beta_true,
        "Rt":          Rt,
    })

    # Salvataggio
    out_path = OUTPUT_DIR / f"{name}.txt"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  [{name}] salvato → {out_path}  "
          f"(I_max={I_sol.max():.0f}, beta range=[{beta_true.min():.3f}, {beta_true.max():.3f}])")

    return df, beta_true, t_span

#  Visualizzazione 

def plot_all_scenarios(results):
    """Genera un plot 4x2 con le traiettorie I(t) e beta_true(t) per ogni scenario."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("Dati sintetici SIR — 4 scenari\n"
                 "Sinistra: I(t) vera vs osservata | Destra: β_true(t)",
                 fontsize=13, fontweight='bold')

    dates_base = np.arange(T0, TF)

    for row, (name, info) in enumerate(SCENARIOS.items()):
        df, beta_true, t_span = results[name]

        # --- Pannello sinistro: I(t) ---
        ax_I = axes[row, 0]
        ax_I.plot(t_span, df["Infectious"].values, 'r-', lw=2, label='I(t) vera')
        ax_I.plot(t_span, df["I data"].values, 'xm', ms=4, alpha=0.6,
                  label='osservazioni (Poisson)')
        ax_I.set_ylabel("individui")
        ax_I.set_title(f"{name} — {info['description']}: I(t)")
        ax_I.legend(fontsize=8)
        ax_I.grid(alpha=0.3)

        # --- Pannello destro: beta_true(t) ---
        ax_b = axes[row, 1]
        ax_b.plot(t_span, beta_true, 'b-', lw=2)
        ax_b.set_ylabel(r"$\beta(t)$")
        ax_b.set_title(f"{name} — β_true(t): {info['label']}")
        ax_b.set_ylim(bottom=0)
        ax_b.grid(alpha=0.3)

        if row == 3:
            ax_I.set_xlabel("giorni")
            ax_b.set_xlabel("giorni")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "scenari_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot salvato → {OUTPUT_DIR}/scenari_overview.png")

#  Salvataggio beta_true per validazione SR 

def save_beta_true_csv(results):
    """
    Salva un CSV con i valori analitici di beta_true(t) per tutti gli scenari.
    Questo file viene usato nella Fase 3 (SR) per il confronto Livello B:
    MSE(beta_SR, beta_true).
    """
    t_span = np.arange(T0, TF, dtype=float)
    df_beta = pd.DataFrame({"t": t_span})
    for name in SCENARIOS:
        _, beta_true, _ = results[name]
        df_beta[f"beta_true_{name}"] = beta_true

    out_path = OUTPUT_DIR / "beta_true_all_scenarios.csv"
    df_beta.to_csv(out_path, index=False)
    print(f"  beta_true analitico salvato → {out_path}")

#  Summary statistiche 

def print_summary(results):
    print("\n" + "="*60)
    print("SUMMARY — Statistiche descrittive per scenario")
    print("="*60)
    for name, (df, beta_true, t_span) in results.items():
        I = df["Infectious"].values
        I_obs = df["I data"].values
        noise_ratio = np.std(I_obs - I) / np.mean(I)
        print(f"\n{name} ({SCENARIOS[name]['description']}):")
        print(f"  I_max          = {I.max():>12.0f} individui al giorno {I.argmax()}")
        print(f"  I_min          = {I.min():>12.2f}")
        print(f"  beta range     = [{beta_true.min():.4f}, {beta_true.max():.4f}]")
        print(f"  R0 iniziale    = {beta_true[0]/delta:.2f}")
        print(f"  rumore rel.    ≈ {noise_ratio:.3f}  (std(I_obs-I)/mean(I))")
        print(f"  N punti        = {len(I)}")

#  Main 

if __name__ == "__main__":
    print("Generazione dati sintetici SIR\n" + "─"*40)

    results = {}
    for name, info in SCENARIOS.items():
        df, beta_true, t_span = generate_scenario(name, info["func"], seed=42)
        results[name] = (df, beta_true, t_span)

    plot_all_scenarios(results)
    save_beta_true_csv(results)
    print_summary(results)

    print("\n✓ Generazione completata.")
    print(f"  File nella cartella: {OUTPUT_DIR.resolve()}")