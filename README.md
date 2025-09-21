# football-playstyle-map

[![CI](https://github.com/maximotk/football-playstyle-map/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/maximotk/football-playstyle-map/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/maximotk/football-playstyle-map/branch/main/graph/badge.svg)](https://codecov.io/gh/maximotk/football-playstyle-map)

This repository is the implementation of **Football Playstyle Map**, a work-in-progress project that applies **unsupervised learning techniques** to football (soccer) data.

Currently, the dashboard focuses on:
- **Tab 1**: Football playstyle maps for major tournaments (World Cups, Euros, Copa América, AFCON), categorizing teams and matches by playstyle.  
- Use of **Non-negative Matrix Factorization (NMF)** to extract latent factors (dimensions) of playstyle.  
- Interactive visualization of team clusters and dominant tactical patterns.  

⚠️ **Note**: This project is still under development. Upcoming extensions (future dashboard tabs) will include:
- **Game-phase clustering** (separating offensive/defensive phases).  
- **Generative models** for simulating alternative tactical scenarios.


---

## 📦 Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

Alternatively, build a reproducible Docker container:

```bash
docker build -t football-playstyle-map .
docker run -p 8501:8501 football-playstyle-map
```

---

## 🏋️ Training / Preprocessing

This repository does not train deep models; instead, it performs:
- **Preprocessing** (missing values, constants, scaling)  
- **Feature factorization** with NMF  
- **Rule-based clustering** of playstyles  
- **Dimensionality reduction** (t-SNE/UMAP) for visualization

The preprocessing and NMF factorization are applied across **all tournaments combined** in order to learn global playstyle dimensions.  
However, the visualizations and clustering results are displayed only for the **competition selected in the dashboard sidebar**.  

You can also control whether:
- Only **team-level features** are included (`_self` / unilateral mode).  
- Or both team and **opponent features** are used (`_opp` mode).


The preprocessing and factorization are fully automated in the dashboard.  

If you want to re-run preprocessing manually:

```bash
python -m code.src.tab1.preprocessing
```

---

## 📊 Evaluation

To run the full **unit tests** and check reproducibility:

```bash
PYTHONPATH=. pytest --cov=code/src --cov-report=term-missing
```

This will execute both component-level and integration tests.

---

## 🎮 Running the Dashboard

To launch the interactive Streamlit dashboard locally:

```bash
streamlit run dashboard.py
```

By default the app loads **engineered StatsBomb match features** from `data/processed/`.

---

## 📥 Data

- The dataset is **match-level**, with approximately **500 engineered features per team per match**.  
- Original raw data comes from [StatsBomb](https://github.com/statsbomb) via [StatsBombPy](https://github.com/statsbomb/statsbombpy).  
- Event-level StatsBomb data was processed and aggregated by me into the match-level dataset used here.  
- Engineered features are stored in `data/processed/`.  

⚠️ Data generation and cleaning are still in progress. Future versions may include updated or extended features.


---

## 📜 License

This repository is part of a university project and is shared for **academic evaluation purposes only**.  
It is **not open for external contributions** at this time.  

---

## 📚 References

- StatsBombPy: https://github.com/statsbomb/statsbombpy  
