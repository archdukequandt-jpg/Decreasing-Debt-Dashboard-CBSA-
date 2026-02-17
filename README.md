# CBSA Household Debt Dashboard (Streamlit)

This dashboard is a lightweight Streamlit app for exploring **household debt ranges** (`low` / `high`)
by **CBSA** across **year/quarter**, optionally normalized by **2022 population**.

## Data
Place the CSV here:

- `cbsa_population_debt_merged.csv`

Expected columns (minimum):
- `year`, `qtr`, `cbsa`, `low`, `high`, `NAME`
- population columns like `POPESTIMATE2022` (used for optional per-capita normalization)

## Run
### Option A: Python
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option B: Included launchers
- macOS: `run_app.command`
- Windows: `run_app.bat`
- Cross-platform helper: `run_app.py`

## Notes
- The app splits `NAME` into `COUNTY` and `STATE` using the first comma.
- `mid` is computed as `(low + high) / 2` (when both exist) and used for ranking/trends.
