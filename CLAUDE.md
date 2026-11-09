# Contesto per Claude

Progetto: genetic programming per far evolvere β(t) di un SEIRD sui dati COVID della P.A. Trento, con la mobilità come variabile. Base: lavoro di Pianesi e Pojer.

Convenzioni:
- Codice riutilizzabile in `src/seird_gp/`, analisi in `notebooks/` (numerati `01_`, `02_`, ...).
- `data/` e `results/` non sono versionati: tutto deve essere rigenerabile dal codice.
- Ogni funzione nuova in `src/` ha almeno un test in `tests/`; `pytest` deve passare prima del commit.
- Lavorare su branch e aprire PR; niente push diretti su `main`.
- Commenti e messaggi di commit in italiano.
