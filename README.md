# seird-gp

Genetic programming per far evolvere β(t) di un modello SEIRD sui dati COVID-19 della Provincia Autonoma di Trento, con la mobilità come variabile di input.

## Struttura

```
seird-gp/
├── src/seird_gp/      # codice riutilizzabile (modello, dati, GP)
│   └── model.py       # SEIRD con β(t) arbitraria
├── notebooks/         # analisi esplorative, numerate: 01_..., 02_...
├── data/
│   ├── raw/           # dati scaricati, mai modificati (non versionati)
│   └── processed/     # dati puliti generati dal codice (non versionati)
├── results/           # figure e output (non versionati)
└── tests/
```

Regola: i notebook importano da `seird_gp`; la logica che serve in più di un notebook va in `src/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Download dati 
```bash
python -m src.seird_gp.data 
```

## Modello

SEIRD con popolazione costante N:

- dS/dt = −β(t) S I / N
- dE/dt = β(t) S I / N − σ E
- dI/dt = σ E − (γ + μ) I
- dR/dt = γ I
- dD/dt = μ I

β(t) è una funzione qualsiasi; nel progetto sarà l'espressione evoluta dal GP a partire dalla mobilità.
