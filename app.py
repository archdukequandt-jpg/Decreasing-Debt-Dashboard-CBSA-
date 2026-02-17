import streamlit as st
from pathlib import Path

st.set_page_config(page_title="CBSA Household Debt Dashboard", layout="wide")

st.sidebar.title("CBSA Debt Dashboard")
page = st.sidebar.radio("Page", ["Home", "Debt & Population"], index=0)

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "cbsa_population_debt_merged.csv"

def render_home():
    st.title("CBSA Household Debt Dashboard")
    st.markdown(
        """
This app visualizes **household debt ranges** (`low` / `high`) by **CBSA** (Core Based Statistical Area)
across **year/quarter**, and includes Census **population estimates** for comparison.

Use **Debt & Population** to:
- view **top CBSAs** by debt (low/high/midpoint)
- explore **debt trends** over time for selected CBSAs
- compare debt & population growth (CAGR) and divergence
- download filtered datasets
"""
    )
    st.info("Data source expected in the same folder as app.py: `cbsa_population_debt_merged.csv`")
    if DATA_FILE.exists():
        st.success("Data file found.")
    else:
        st.error("Missing data file. Put `cbsa_population_debt_merged.csv` in this folder (same folder as app.py).")

def render_debt_population():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # ---------------------------------------------
    # Debt & Population Dashboard (CBSA)
    # ---------------------------------------------
    
    DATA_PATH = Path(__file__).resolve().parent / "cbsa_population_debt_merged.csv"
    
    
    @st.cache_data(show_spinner=False)
    def load_master(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
    
        # Standardize cbsa to 5-digit string
        df["cbsa"] = pd.to_numeric(df["cbsa"], errors="coerce").astype("Int64")
        df["cbsa"] = df["cbsa"].astype("string").str.zfill(5)
    
        # Standardize year/qtr
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["qtr"] = pd.to_numeric(df["qtr"], errors="coerce").astype("Int64")
    
        # Debt numeric
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
    
        # Split NAME -> COUNTY/STATE (supports multi-state codes like NY-NJ-PA)
        df["NAME"] = df["NAME"].astype("string")
        parts = df["NAME"].str.split(",", n=1, expand=True)
        df["COUNTY"] = parts[0].astype("string").str.strip()
        df["STATE"] = np.where(parts.shape[1] > 1, parts[1].astype("string").str.strip(), pd.NA)
    
        # A convenient midpoint series for ranking / charts
        df["mid"] = np.nanmean(np.vstack([df["low"].to_numpy(), df["high"].to_numpy()]), axis=0)
    
        # Period label for plotting
        df["period"] = df["year"].astype("string") + "-Q" + df["qtr"].astype("string")
    
        return df
    
    
    def make_barh(frame: pd.DataFrame, label_col: str, value_col: str, title: str, xlabel: str):
        fig, ax = plt.subplots()
        plot_df = frame.sort_values(value_col, ascending=True)
        ax.barh(plot_df[label_col].astype(str), plot_df[value_col].astype(float))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        fig.tight_layout()
        return fig
    
    
    def make_line(frame: pd.DataFrame, x_col: str, y_col: str, hue_col: str, title: str, ylabel: str):
        fig, ax = plt.subplots()
        for key, grp in frame.groupby(hue_col):
            ax.plot(grp[x_col], grp[y_col], marker="o", linewidth=1.5, label=str(key))
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig
    
    
    
    # -----------------------------
    # Growth helpers
    # -----------------------------
    def _time_float(year: int, qtr: int) -> float:
        """Convert (year, qtr) -> fractional year for CAGR calculations."""
        return float(year) + (float(qtr) - 1.0) / 4.0
    
    
    def cagr(start: float, end: float, years: float) -> float:
        """Compound annual growth rate. Returns np.nan for invalid inputs."""
        try:
            start = float(start)
            end = float(end)
            years = float(years)
        except Exception:
            return np.nan
        if not np.isfinite(start) or not np.isfinite(end) or not np.isfinite(years):
            return np.nan
        if start <= 0 or end <= 0 or years <= 0:
            return np.nan
        return (end / start) ** (1.0 / years) - 1.0
    
    
    def pop_series_for_years(frame: pd.DataFrame) -> pd.Series:
        """Map each row's year to available population estimate columns (2020-2022)."""
        y = frame["year"].astype("Int64")
        pop = pd.Series(np.nan, index=frame.index, dtype="float64")
        if "POPESTIMATE2020" in frame.columns:
            pop = np.where(y == 2020, pd.to_numeric(frame["POPESTIMATE2020"], errors="coerce"), pop)
        if "POPESTIMATE2021" in frame.columns:
            pop = np.where(y == 2021, pd.to_numeric(frame["POPESTIMATE2021"], errors="coerce"), pop)
        if "POPESTIMATE2022" in frame.columns:
            pop = np.where(y == 2022, pd.to_numeric(frame["POPESTIMATE2022"], errors="coerce"), pop)
        return pd.Series(pop, index=frame.index, dtype="float64")
    
    
    def make_dual_axis(frame: pd.DataFrame, x_col: str, debt_col: str, pop_col: str, title: str):
        """Debt on left axis, population on right axis."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
    
        ax1.plot(frame[x_col], frame[debt_col], marker="o", linewidth=1.6, label="Debt")
        ax2.plot(frame[x_col], frame[pop_col], marker="o", linewidth=1.6, label="Population")
    
        ax1.set_title(title)
        ax1.set_xlabel("")
        ax1.set_ylabel("Debt")
        ax2.set_ylabel("Population")
    
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    
        # Combine legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")
    
        fig.tight_layout()
        return fig
    
    
    def make_growth_scatter(growth_df: pd.DataFrame, title: str, label_top: int = 12):
        fig, ax = plt.subplots()
        x = growth_df["pop_cagr"].to_numpy()
        y = growth_df["debt_cagr"].to_numpy()
        ax.scatter(x, y)
    
        ax.axhline(0, linewidth=1.0)
        ax.axvline(0, linewidth=1.0)
    
        ax.set_title(title)
        ax.set_xlabel("Population CAGR")
        ax.set_ylabel("Debt CAGR")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    
        # Label the most divergent points (by absolute gap)
        lab = growth_df.copy()
        lab["gap"] = (lab["debt_cagr"] - lab["pop_cagr"]).abs()
        lab = lab.dropna(subset=["debt_cagr", "pop_cagr"]).sort_values("gap", ascending=False).head(label_top)
        for _, r in lab.iterrows():
            ax.annotate(str(r["CBSA_NAME"]), (r["pop_cagr"], r["debt_cagr"]), fontsize=8)
    
        fig.tight_layout()
        return fig
    
    st.title("Debt & Population")
    
    if not DATA_PATH.exists():
        st.error(f"Missing data file: {DATA_PATH}")
        st.stop()
    
    df = load_master(DATA_PATH)
    
    # Basic validation
    need_cols = ["cbsa", "year", "qtr", "low", "high", "COUNTY", "STATE", "POPESTIMATE2022"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        st.error(f"Data is missing required columns: {missing}")
        st.stop()
    
    # Drop rows with no debt info (should already be done upstream, but keep dashboard safe)
    df = df.dropna(subset=["low", "high"], how="all").copy()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    section = st.sidebar.radio(
        "Section",
        ["Overview", "Top CBSAs", "Debt trends", "Growth & divergence", "Data explorer"],
        index=0,
    )
    
    top_n = st.sidebar.slider("Top N", min_value=5, max_value=50, value=15, step=1)
    
    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    qtrs = sorted(df["qtr"].dropna().astype(int).unique().tolist())
    
    year_min, year_max = min(years), max(years)
    year_range = st.sidebar.slider("Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max))
    
    qtr_pick = st.sidebar.multiselect("Quarter(s)", options=qtrs, default=qtrs)
    
    state_list = sorted(df["STATE"].dropna().astype(str).unique().tolist())
    state_pick = st.sidebar.selectbox("State (optional)", ["(All states)"] + state_list, index=0)
    
    # Optional CBSA multi-select (shown after state filter so list is shorter)
    filtered = df[(df["year"].between(year_range[0], year_range[1])) & (df["qtr"].isin(qtr_pick))].copy()
    if state_pick != "(All states)":
        filtered = filtered[filtered["STATE"].astype(str) == state_pick].copy()
    
    cbsa_label = filtered["COUNTY"].astype(str) + " (" + filtered["STATE"].astype(str) + ") — " + filtered["cbsa"].astype(str)
    filtered = filtered.assign(cbsa_label=cbsa_label)
    
    cbsa_options = sorted(filtered["cbsa_label"].dropna().unique().tolist())
    cbsa_selection = st.sidebar.multiselect(
        "Select CBSAs (optional, affects Trends/Data Explorer)",
        options=cbsa_options,
        default=[],
    )
    
    if cbsa_selection:
        filtered = filtered[filtered["cbsa_label"].isin(cbsa_selection)].copy()
    
    # ---------------------------------------------
    # Overview
    # ---------------------------------------------
    if section == "Overview":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows (filtered)", f"{len(filtered):,}")
        c2.metric("Unique CBSAs", f"{filtered['cbsa'].nunique():,}")
        c3.metric("Year range", f"{year_range[0]}–{year_range[1]}")
        c4.metric("Quarters", ", ".join(map(str, qtr_pick)) if qtr_pick else "None")
    
        st.markdown("### Notes on fields")
        st.write(
            "- `low` / `high`: debt range values from the source dataset.\n"
            "- `mid`: simple midpoint = (low + high) / 2 (used for ranking and charts).\n"
            "- `POPESTIMATE2022`: used for optional per-capita normalization (Top CBSAs tab)."
        )
    
        st.markdown("### Preview")
        st.dataframe(
            filtered[["year", "qtr", "cbsa", "COUNTY", "STATE", "low", "high", "mid", "POPESTIMATE2022"]].head(50),
            use_container_width=True,
        )
    
    # ---------------------------------------------
    # Top CBSAs
    # ---------------------------------------------
    elif section == "Top CBSAs":
        st.subheader("Top CBSAs by debt (midpoint)")
    
        # Choose a specific period to rank
        rank_year = st.sidebar.selectbox("Rank year", options=years, index=len(years) - 1)
        rank_qtr = st.sidebar.selectbox("Rank quarter", options=qtrs, index=len(qtrs) - 1)
    
        metric_mode = st.sidebar.selectbox("Metric", ["Midpoint (mid)", "Low", "High"], index=0)
        per_capita = st.sidebar.checkbox("Normalize by population (per 1,000 residents)", value=False)
    
        snap = df[(df["year"] == rank_year) & (df["qtr"] == rank_qtr)].copy()
        if state_pick != "(All states)":
            snap = snap[snap["STATE"].astype(str) == state_pick].copy()
    
        # If CBSA selection made, apply it
        if cbsa_selection:
            snap = snap.assign(cbsa_label=snap["COUNTY"].astype(str) + " (" + snap["STATE"].astype(str) + ") — " + snap["cbsa"].astype(str))
            snap = snap[snap["cbsa_label"].isin(cbsa_selection)].copy()
    
        value_col = {"Midpoint (mid)": "mid", "Low": "low", "High": "high"}[metric_mode]
        snap = snap.dropna(subset=[value_col])
    
        snap["POP_2022"] = pd.to_numeric(snap["POPESTIMATE2022"], errors="coerce")
    
        if per_capita:
            snap["VALUE"] = np.where(snap["POP_2022"] > 0, (snap[value_col] / snap["POP_2022"]) * 1000.0, np.nan)
            xlabel = f"{metric_mode} per 1,000 residents (using 2022 population)"
        else:
            snap["VALUE"] = snap[value_col]
            xlabel = metric_mode
    
        snap["CBSA_NAME"] = snap["COUNTY"].astype(str) + ", " + snap["STATE"].astype(str)
    
        show = snap.sort_values("VALUE", ascending=False)[
            ["CBSA_NAME", "cbsa", "VALUE", "low", "high", "POP_2022"]
        ].head(top_n).copy()
    
        st.dataframe(show, use_container_width=True)
    
        fig = make_barh(
            show,
            label_col="CBSA_NAME",
            value_col="VALUE",
            title=f"Top {min(top_n, len(show))} CBSAs — {metric_mode} ({rank_year} Q{rank_qtr})" + ("" if state_pick == "(All states)" else f" — {state_pick}"),
            xlabel=xlabel,
        )
        st.pyplot(fig)
    
        st.download_button(
            "Download top-N CSV",
            show.to_csv(index=False).encode("utf-8"),
            file_name=f"top_cbsa_debt_{rank_year}_Q{rank_qtr}.csv",
            mime="text/csv",
        )
    
    # ---------------------------------------------
    # Debt trends
    # ---------------------------------------------
    elif section == "Debt trends":
        st.subheader("Debt trends over time")
    
        if not cbsa_selection:
            st.info("Pick one or more CBSAs in the sidebar to view trend lines.")
            st.stop()
    
        trend_metric = st.sidebar.selectbox("Trend metric", ["Midpoint (mid)", "Low", "High"], index=0)
        ycol = {"Midpoint (mid)": "mid", "Low": "low", "High": "high"}[trend_metric]
    
        show_population = st.sidebar.checkbox("Overlay population (2nd axis)", value=True)
        plot_as_index = st.sidebar.checkbox("Plot as index (start=100)", value=False)
        max_cbsa_for_dual = 4
    
        # Build an ordered time axis
        plot_df = filtered.dropna(subset=[ycol]).copy()
        plot_df["time_order"] = plot_df["year"].astype(int) * 10 + plot_df["qtr"].astype(int)
        plot_df = plot_df.sort_values(["cbsa_label", "time_order"])
    
        # Optional index scaling for easier comparison across CBSAs
        if plot_as_index:
            plot_df = plot_df.copy()
            for key, grp in plot_df.groupby("cbsa_label"):
                idx = grp.index
                base_val = grp.iloc[0][ycol]
                if pd.notna(base_val) and float(base_val) != 0:
                    plot_df.loc[idx, ycol] = (grp[ycol] / base_val) * 100.0
    
        if show_population:
            max_cbsa_for_dual = 4
            if len(cbsa_selection) > max_cbsa_for_dual:
                st.warning(
                    f"Population overlay is clearest with <= {max_cbsa_for_dual} CBSAs selected. "
                    f"You selected {len(cbsa_selection)}; showing debt-only line chart."
                )
                fig = make_line(
                    plot_df,
                    x_col="period",
                    y_col=ycol,
                    hue_col="cbsa_label",
                    title=f"{trend_metric} over time" + (" (index=100)" if plot_as_index else ""),
                    ylabel=trend_metric + (" (index=100)" if plot_as_index else ""),
                )
                st.pyplot(fig)
            else:
                # For clarity, render one dual-axis chart per CBSA
                for cbsa_name, grp in plot_df.groupby("cbsa_label"):
                    grp = grp.copy()
                    grp["population"] = pop_series_for_years(grp)
    
                    # Optionally index population as well
                    if plot_as_index:
                        base_p = grp["population"].dropna().iloc[0] if grp["population"].notna().any() else np.nan
                        if pd.notna(base_p) and float(base_p) != 0:
                            grp["population"] = (grp["population"] / base_p) * 100.0
    
                    fig = make_dual_axis(
                        grp,
                        x_col="period",
                        debt_col=ycol,
                        pop_col="population",
                        title=f"{cbsa_name} — {trend_metric} vs Population" + (" (index=100)" if plot_as_index else ""),
                    )
                    st.pyplot(fig)
        else:
            fig = make_line(
                plot_df,
                x_col="period",
                y_col=ycol,
                hue_col="cbsa_label",
                title=f"{trend_metric} over time" + (" (index=100)" if plot_as_index else ""),
                ylabel=trend_metric + (" (index=100)" if plot_as_index else ""),
            )
            st.pyplot(fig)
    
        st.markdown("### Trend data")
        st.dataframe(
            plot_df[["year", "qtr", "cbsa", "COUNTY", "STATE", "low", "high", "mid", "POPESTIMATE2022"]]
            .sort_values(["cbsa", "year", "qtr"])
            .reset_index(drop=True),
            use_container_width=True,
        )
    
        st.download_button(
            "Download selected trend data CSV",
            plot_df.to_csv(index=False).encode("utf-8"),
            file_name="selected_cbsa_debt_trends.csv",
            mime="text/csv",
        )
    
    
    # ---------------------------------------------
    # Growth & divergence
    # ---------------------------------------------
    elif section == "Growth & divergence":
        st.subheader("Growth & divergence: Debt vs Population")
    
        st.write(
            "This section computes **CAGR** for debt and population and highlights places where they diverge.\n\n"
            "- Debt CAGR is computed over your filtered time window (Year range + Quarter selection) using the chosen debt metric.\n"
            "- Population CAGR is computed from **2020 → 2022** (the population data available in this merged dataset)."
        )
    
        growth_metric = st.sidebar.selectbox("Growth metric (debt)", ["Midpoint (mid)", "Low", "High"], index=0)
        debt_col = {"Midpoint (mid)": "mid", "Low": "low", "High": "high"}[growth_metric]
    
        # Prepare a CBSA-level table
        base = filtered.dropna(subset=[debt_col]).copy()
        base["time_order"] = base["year"].astype(int) * 10 + base["qtr"].astype(int)
        base = base.sort_values(["cbsa", "time_order"])
    
        rows = []
        for cbsa, grp in base.groupby("cbsa"):
            grp = grp.dropna(subset=[debt_col]).copy()
            if grp.empty:
                continue
    
            # Debt CAGR over selected range
            s = grp.iloc[0]
            e = grp.iloc[-1]
            debt_start = float(s[debt_col]) if pd.notna(s[debt_col]) else np.nan
            debt_end = float(e[debt_col]) if pd.notna(e[debt_col]) else np.nan
            years_elapsed = _time_float(int(e["year"]), int(e["qtr"])) - _time_float(int(s["year"]), int(s["qtr"]))
            debt_c = cagr(debt_start, debt_end, years_elapsed)
    
            # Population CAGR 2020->2022 (available cols)
            pop_2020 = pd.to_numeric(grp["POPESTIMATE2020"].iloc[0] if "POPESTIMATE2020" in grp.columns else np.nan, errors="coerce")
            pop_2022 = pd.to_numeric(grp["POPESTIMATE2022"].iloc[0] if "POPESTIMATE2022" in grp.columns else np.nan, errors="coerce")
            pop_c = cagr(pop_2020, pop_2022, 2.0)
    
            cbsa_name = (grp["COUNTY"].astype(str).iloc[0] + ", " + grp["STATE"].astype(str).iloc[0]) if ("COUNTY" in grp.columns and "STATE" in grp.columns) else str(cbsa)
    
            # Quadrant classification
            debt_up = pd.notna(debt_c) and debt_c > 0
            pop_up = pd.notna(pop_c) and pop_c > 0
            if debt_up and pop_up:
                bucket = "Debt ↑ | Pop ↑"
            elif debt_up and (not pop_up):
                bucket = "Debt ↑ | Pop ↓/flat"
            elif (not debt_up) and pop_up:
                bucket = "Debt ↓/flat | Pop ↑"
            else:
                bucket = "Debt ↓/flat | Pop ↓/flat"
    
            rows.append({
                "CBSA_NAME": cbsa_name,
                "cbsa": cbsa,
                "debt_start": debt_start,
                "debt_end": debt_end,
                "debt_years": years_elapsed,
                "debt_cagr": debt_c,
                "pop_2020": pop_2020,
                "pop_2022": pop_2022,
                "pop_cagr": pop_c,
                "bucket": bucket,
                "gap_debt_minus_pop": (debt_c - pop_c) if (pd.notna(debt_c) and pd.notna(pop_c)) else np.nan,
            })
    
        growth_df = pd.DataFrame(rows)
        if growth_df.empty:
            st.warning("No data available for growth calculations under the current filters.")
            st.stop()
    
        # Summary counts
        st.markdown("### Where debt and population are (and are not) growing")
        counts = growth_df["bucket"].value_counts(dropna=False).rename_axis("bucket").reset_index(name="count")
        st.dataframe(counts, use_container_width=True)
    
        # Filter by quadrant
        bucket_pick = st.multiselect(
            "Show buckets",
            options=counts["bucket"].tolist(),
            default=counts["bucket"].tolist(),
        )
        show = growth_df[growth_df["bucket"].isin(bucket_pick)].copy()
    
        # Rank by divergence
        rank_mode = st.selectbox("Rank by", ["Largest debt-pop gap", "Fastest debt CAGR", "Fastest population CAGR"], index=0)
        if rank_mode == "Largest debt-pop gap":
            show = show.sort_values("gap_debt_minus_pop", ascending=False)
        elif rank_mode == "Fastest debt CAGR":
            show = show.sort_values("debt_cagr", ascending=False)
        else:
            show = show.sort_values("pop_cagr", ascending=False)
    
        st.markdown("### CBSA growth table (CAGR)")
        fmt = show.copy()
        for c in ["debt_cagr", "pop_cagr", "gap_debt_minus_pop"]:
            if c in fmt.columns:
                fmt[c] = (fmt[c] * 100.0).round(2)
        st.dataframe(
            fmt[["CBSA_NAME", "cbsa", "bucket", "debt_cagr", "pop_cagr", "gap_debt_minus_pop", "debt_years", "pop_2020", "pop_2022"]]
            .rename(columns={"debt_cagr": "Debt CAGR (%)", "pop_cagr": "Pop CAGR (%)", "gap_debt_minus_pop": "Debt - Pop (pp)"}),
            use_container_width=True,
        )
    
        # Scatter plot
        st.markdown("### Scatter: Population CAGR vs Debt CAGR")
        fig = make_growth_scatter(growth_df, title="Population CAGR (2020–2022) vs Debt CAGR (selected window)")
        st.pyplot(fig)
    
        st.download_button(
            "Download growth table CSV",
            growth_df.to_csv(index=False).encode("utf-8"),
            file_name="cbsa_growth_debt_vs_population.csv",
            mime="text/csv",
        )
    
    
    # ---------------------------------------------
    # Data explorer
    # ---------------------------------------------
    else:
        st.subheader("Data explorer")
    
        cols = ["year", "qtr", "cbsa", "COUNTY", "STATE", "low", "high", "mid", "LSAD", "POPESTIMATE2020", "POPESTIMATE2021", "POPESTIMATE2022"]
        cols = [c for c in cols if c in filtered.columns]
    
        st.dataframe(filtered[cols].sort_values(["year", "qtr", "cbsa"]), use_container_width=True)
    
        st.download_button(
            "Download filtered CSV",
            filtered[cols].to_csv(index=False).encode("utf-8"),
            file_name="filtered_cbsa_population_debt.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    render_debt_population()

if page == "Home":
    render_home()
else:
    render_debt_population()
