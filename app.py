"""
AgriStat Studio – Professional Experimental Design & Biometrics Engine
Designs : CRD | RCBD | Factorial (RCBD) | Split-Plot (RCBD)
Methods : Tukey HSD, LSD, Bonferroni, Duncan (DMRT), SNK
Engine  : Pure NumPy / SciPy / Pandas (Type I / Type III SS, EMS Error Architecture)
"""
from __future__ import annotations

import io
import itertools
import string
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

warnings.filterwarnings("ignore")

# ───────────────────────── Page Configuration ─────────────────────────
st.set_page_config(
    page_title="AgriStat Studio | Biometrics Engine",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────── Elite UI Styling ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.maateen.me/kalpurush/font.css');

/* Hide Streamlit Top Header, Fork, Hamburger & Deploy Button */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
.stDeployButton { display: none !important; }

/* Global Typography: Times New Roman & Kalpurush */
html, body, .stApp, p, label, li, td, th, h1, h2, h3, h4, input, button {
    font-family: 'Times New Roman', 'Kalpurush', 'Liberation Serif', serif !important;
}

/* Metric / KPI Cards (JMP Inspired) */
[data-testid="stMetric"] {
    background: #0f172a08;
    border: 1px solid #0f172a18;
    border-radius: 10px;
    padding: 10px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}

/* Tab Headers */
button[data-baseweb="tab"] {
    font-size: 0.98rem !important;
    font-weight: 600 !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    border-right: 1px solid #e2e8f0;
}
[data-testid="collapsedControl"] {
    top: 15px;
    left: 15px;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Kalpurush", "Liberation Serif", "DejaVu Serif"]
})

# ───────────────────────── Statistical Core Constants ─────────────────────────
TRANSFORMS = {
    "None": lambda y: y,
    "log10": lambda y: np.log10(y),
    "sqrt": lambda y: np.sqrt(y),
    "arcsine (%)": lambda y: np.degrees(np.arcsin(np.sqrt(y / 100))),
}
METHODS = ["Tukey HSD", "LSD", "Bonferroni", "Duncan (DMRT)", "SNK"]

DESIGNS = {
    "CRD": dict(terms=[("A",)], err={}, needs_rep=False, two_factor=False, label="Completely Randomized (CRD)"),
    "RCBD": dict(terms=[("REP",), ("A",)], err={}, needs_rep=True, two_factor=False, label="Randomized Block (RCBD)"),
    "FACT": dict(terms=[("REP",), ("A",), ("B",), ("A", "B")], err={}, needs_rep=True, two_factor=True, label="Factorial RCBD (A × B)"),
    "SPLIT": dict(
        terms=[("REP",), ("A",), ("REP", "A"), ("B",), ("A", "B")],
        err={("REP",): ("REP", "A"), ("A",): ("REP", "A")},
        needs_rep=True, two_factor=True, label="Split-Plot RCBD (Main/Sub)"
    ),
}
SPLIT_ERR_A = ("REP", "A")

class AnalysisError(ValueError):
    def __init__(self, en, bn=None):
        super().__init__(en)
        self.en, self.bn = en, bn or en

# ───────────────────────── Language Switcher ─────────────────────────
st.sidebar.markdown("### 🌐 Interface / ভাষা")
BN = st.sidebar.radio("Lang", ("English", "বাংলা"), horizontal=True, label_visibility="collapsed") == "বাংলা"

def tr(en, bn):
    return bn if BN else en

# ───────────────────────── Mathematical Utilities ─────────────────────────
def sig_stars(p, alpha=0.05):
    if p is None or pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < alpha else "ns"

def fmt_p(p):
    if p is None or pd.isna(p):
        return ""
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"

def _sum_cols(codes, n_levels):
    m = np.zeros((len(codes), n_levels - 1))
    for j in range(n_levels - 1):
        m[codes == j, j] = 1.0
    m[codes == n_levels - 1, :] = -1.0
    return m

def term_matrix(df, term, levels):
    out = None
    for f in term:
        codes = df[f].map({lv: i for i, lv in enumerate(levels[f])}).to_numpy()
        mf = _sum_cols(codes, len(levels[f]))
        out = mf if out is None else (out[:, :, None] * mf[:, None, :]).reshape(len(df), -1)
    return out

def _ls(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    resid = y - fitted
    return beta, fitted, resid, float(resid @ resid), int(np.linalg.matrix_rank(X))

def _rss_rank(X, y):
    r = _ls(X, y)
    return r[3], r[4]

def term_label(term, labels):
    return " × ".join(labels[f] for f in term)

# ───────────────────────── Data Engine ─────────────────────────
def prepare(df, design, cols, y, tf="None", avg_dup=False):
    spec = DESIGNS[design]
    sel = {"A": cols["a"]}
    if spec["needs_rep"]:
        sel["REP"] = cols["rep"]
    if spec["two_factor"]:
        sel["B"] = cols["b"]
    used = list(sel.values()) + [y]
    if len(set(used)) != len(used):
        raise AnalysisError("Treatment, block and response columns must all be different.",
                            "ট্রিটমেন্ট, ব্লক ও রেসপন্স কলাম আলাদা হতে হবে।")
    d = df[used].copy()
    d.columns = list(sel.keys()) + ["Y0"]
    d["Y0"] = pd.to_numeric(d["Y0"], errors="coerce")
    d = d.dropna()
    for f in sel:
        d[f] = d[f].astype(str).str.strip()
    if len(d) == 0:
        raise AnalysisError("No valid numeric data.", "কোনো বৈধ সংখ্যাভিত্তিক ডেটা নেই।")

    if avg_dup:
        keys = list(sel.keys())
        d = d.groupby(keys, sort=False, as_index=False)["Y0"].mean()

    if tf == "log10" and (d["Y0"] <= 0).any():
        raise AnalysisError("log10 needs all values > 0.", "log10 এর জন্য সব মান > 0 হতে হবে।")
    if tf == "sqrt" and (d["Y0"] < 0).any():
        raise AnalysisError("sqrt needs all values ≥ 0.", "sqrt এর জন্য সব মান ≥ 0 হতে হবে।")
    if tf == "arcsine (%)" and ((d["Y0"] < 0) | (d["Y0"] > 100)).any():
        raise AnalysisError("arcsine needs values between 0 and 100.", "arcsine এর জন্য মান ০–১০০ এর মধ্যে হতে হবে।")
    d["Y"] = TRANSFORMS[tf](d["Y0"])

    levels = {f: list(dict.fromkeys(d[f])) for f in sel}
    if len(levels["A"]) < 2:
        raise AnalysisError("Need at least 2 levels of the treatment factor.", "ট্রিটমেন্ট ফ্যাক্টরে কমপক্ষে ২টি লেভেল লাগবে।")
    if spec["two_factor"] and len(levels["B"]) < 2:
        raise AnalysisError("Need at least 2 levels of factor B.", "ফ্যাক্টর B তে কমপক্ষে ২টি লেভেল লাগবে।")
    if spec["needs_rep"] and len(levels["REP"]) < 2:
        raise AnalysisError("Need at least 2 replications/blocks.", "কমপক্ষে ২টি রেপ্লিকেশন/ব্লক লাগবে।")

    keys = [f for f in ("REP", "A", "B") if f in sel]
    if design == "CRD":
        cnt = d.groupby("A").size()
        balanced = cnt.nunique() == 1
        n_missing = n_dupes = 0
    else:
        cell = d.groupby(keys).size()
        expected = int(np.prod([len(levels[f]) for f in keys]))
        n_missing = expected - len(cell)
        n_dupes = int((cell > 1).sum())
        balanced = n_missing == 0 and n_dupes == 0
    info = dict(balanced=balanced, n_missing=n_missing, n_dupes=n_dupes, levels=levels, keys=keys)
    if design == "SPLIT" and not balanced:
        raise AnalysisError(
            f"Split-plot requires a balanced layout. Found {n_missing} missing, {n_dupes} duplicate cells.",
            f"স্প্লিট-প্লটের জন্য সম্পূর্ণ ব্যালান্সড ডেটা প্রয়োজন ({n_missing} মিসিং, {n_dupes} ডুপ্লিকেট সেল পাওয়া গেছে)।")
    return d, info

# ───────────────────────── General Linear Model (SAS/SPSS Equivalent) ─────────────────────────
def fit_model(d, design, labels=None, sstype="III"):
    labels = labels or {"REP": "Replication", "A": "A", "B": "B"}
    spec = DESIGNS[design]
    terms = spec["terms"]
    levels = {f: list(dict.fromkeys(d[f])) for f in {f for t in terms for f in t}}
    n = len(d)
    y = d["Y"].to_numpy(float)
    ones = np.ones((n, 1))
    mats = [term_matrix(d, t, levels) for t in terms]
    X = np.hstack([ones] + mats)
    beta, fitted, resid, rss, rank = _ls(X, y)
    dfe = n - rank
    if dfe <= 0:
        raise AnalysisError("No error degrees of freedom (DF <= 0).", "এরর ডিগ্রি অব ফ্রিডম (DF) নেই।")

    rows = {}
    for i, t in enumerate(terms):
        if sstype == "I":
            r0, k0 = _rss_rank(np.hstack([ones] + mats[:i]), y)
            r1, k1 = _rss_rank(np.hstack([ones] + mats[:i + 1]), y)
            ss, df_ = r0 - r1, k1 - k0
        else:
            r0, k0 = _rss_rank(np.hstack([ones] + mats[:i] + mats[i + 1:]), y)
            ss, df_ = r0 - rss, rank - k0
        rows[t] = dict(df=int(df_), ss=max(float(ss), 0.0))
    for t, v in rows.items():
        v["ms"] = v["ss"] / v["df"] if v["df"] > 0 else np.nan

    mse = rss / dfe
    err_terms = set(spec["err"].values())
    E = {"RES": (mse, dfe)}
    for et in err_terms:
        E[et] = (rows[et]["ms"], rows[et]["df"])

    out = []
    for t in terms:
        v = rows[t]
        is_err = t in err_terms
        ek = spec["err"].get(t, "RES")
        ms_e, df_e = E[ek]
        if is_err or v["df"] <= 0:
            F = p = eta = np.nan
        else:
            F = v["ms"] / ms_e if ms_e > 0 else np.inf
            p = float(stats.f.sf(F, v["df"], df_e)) if np.isfinite(F) else 0.0
            eta = v["ss"] / (v["ss"] + ms_e * df_e) if (v["ss"] + ms_e * df_e) > 0 else 0
        if is_err:
            name, err_name = "Error (a)", ""
        else:
            name = term_label(t, labels)
            err_name = "Error (a)" if ek != "RES" else ("Error (b)" if design == "SPLIT" else "Error")
        out.append(dict(key=t, Source=name, DF=v["df"], SS=v["ss"], MS=v["ms"], F=F, P=p, EtaP=eta, ErrorTerm=err_name))

    out.append(dict(key="RES", Source="Error (b)" if design == "SPLIT" else "Residual Error",
                    DF=int(dfe), SS=rss, MS=mse, F=np.nan, P=np.nan, EtaP=np.nan, ErrorTerm=""))
    tss = float(((y - y.mean()) ** 2).sum())
    out.append(dict(key="TOT", Source="Corrected Total", DF=n - 1, SS=tss, MS=np.nan, F=np.nan, P=np.nan, EtaP=np.nan, ErrorTerm=""))
    tbl = pd.DataFrame(out)

    hat = np.einsum("ij,ji->i", X, np.linalg.pinv(X))
    return dict(tbl=tbl, terms=terms, levels=levels, beta=beta, fitted=fitted, resid=resid, rss=rss, dfe=dfe,
                mse=mse, tss=tss, r2=1 - rss / tss if tss > 0 else np.nan, E=E, hat=hat, n=n, labels=labels,
                sstype=sstype, design=design, rank=rank)

def display_anova(tbl, alpha):
    t = tbl.copy()
    return pd.DataFrame({
        "Source of Variation": t["Source"],
        "DF": t["DF"].astype(int),
        "Sum of Squares": t["SS"].round(3),
        "Mean Square": t["MS"].round(3),
        "F-Ratio": t["F"].round(3),
        "P-Value": [fmt_p(p) for p in t["P"]],
        "Sig.": [sig_stars(p, alpha) for p in t["P"]],
        "Partial η²": t["EtaP"].round(3),
        "Error Term": t["ErrorTerm"]
    }).fillna("—")

# ───────────────────────── LS-Means & Post-Hoc Separation ─────────────────────────
def ls_means(d, fit, family):
    terms, levels, beta = fit["terms"], fit["levels"], fit["beta"]
    facs = list(levels.keys())
    grid = pd.DataFrame(list(itertools.product(*[levels[f] for f in facs])), columns=facs)
    Xg = np.column_stack([np.ones(len(grid))] + [term_matrix(grid, t, levels) for t in terms])
    grid["pred"] = Xg @ beta
    g = grid.groupby(list(family), sort=False)["pred"].mean()
    combos = list(itertools.product(*[levels[f] for f in family]))
    vals = []
    for c in combos:
        vals.append(g.loc[c[0]] if len(family) == 1 else g.loc[c])
    return combos, np.array(vals, float)

def make_crit(method, alpha, mse, dfe, r, k):
    se = np.sqrt(mse / r)
    if method == "LSD":
        c = stats.t.ppf(1 - alpha / 2, dfe) * np.sqrt(2) * se
        return lambda p: c
    if method == "Bonferroni":
        m = max(k * (k - 1) / 2, 1)
        c = stats.t.ppf(1 - alpha / (2 * m), dfe) * np.sqrt(2) * se
        return lambda p: c
    if method == "Tukey HSD":
        c = stats.studentized_range.ppf(1 - alpha, k, dfe) * se
        return lambda p: c
    if method == "SNK":
        return lambda p: stats.studentized_range.ppf(1 - alpha, max(p, 2), dfe) * se
    return lambda p: stats.studentized_range.ppf((1 - alpha) ** (max(p, 2) - 1), max(p, 2), dfe) * se  # Duncan

def letter_groups(means, crit):
    means = np.asarray(means, float)
    order = np.argsort(-means)
    m, k = means[order], len(means)
    groups = set()
    for i in range(k):
        j = i
        while j + 1 < k and m[i] - m[j + 1] < crit(j + 2 - i):
            j += 1
        groups.add((i, j))
    maximal = sorted(g for g in groups if not any(h != g and h[0] <= g[0] and h[1] >= g[1] for h in groups))
    abc = string.ascii_lowercase + string.ascii_uppercase
    lab = [""] * k
    for n, (i, j) in enumerate(maximal):
        for t in range(i, j + 1):
            lab[t] += abc[n % 52]
    out = [""] * k
    for pos, idx in enumerate(order):
        out[idx] = lab[pos]
    return out

def mean_family(d, fit, family, err_key, label, method, alpha, tf, comparable=True):
    combos, mu = ls_means(d, fit, family)
    names = [" × ".join(c) for c in combos]
    keycol = d[list(family)].agg(" × ".join, axis=1) if len(family) > 1 else d[family[0]]
    grp = d.assign(_k=keycol).groupby("_k", sort=False)
    cnt = grp.size().reindex(names)
    k = len(names)
    r = k / np.sum(1.0 / cnt.to_numpy(float))
    ms_e, df_e = fit["E"][err_key]
    se = float(np.sqrt(ms_e / r))
    sd = grp["Y"].std().reindex(names).to_numpy(float)
    mo = grp["Y0"].mean().reindex(names).to_numpy(float)
    crit = make_crit(method, alpha, ms_e, df_e, r, k)
    letters = letter_groups(mu, crit) if comparable and k >= 2 else ["—"] * k
    t = pd.DataFrame({label: names, "Rep (r)": cnt.to_numpy(int), "LS-Mean": mu.round(3), "SD": np.round(sd, 3),
                      "SE": round(se, 3), "Group": letters})
    if tf != "None":
        t.insert(2, "Orig. Mean", mo.round(3))

    pair = None
    if comparable and method in ("Tukey HSD", "LSD", "Bonferroni") and k >= 2:
        m_pairs = k * (k - 1) / 2
        rows = []
        for (iu, u), (iv, v) in itertools.combinations(enumerate(names), 2):
            diff = mu[iu] - mu[iv]
            if method == "Tukey HSD":
                p, pname = float(stats.studentized_range.sf(abs(diff) / se, k, df_e)), "P (Tukey adj.)"
            else:
                p = float(2 * stats.t.sf(abs(diff) / (np.sqrt(2) * se), df_e))
                pname = "P"
                if method == "Bonferroni":
                    p, pname = min(1.0, p * m_pairs), "P (Bonferroni adj.)"
            rows.append({"Contrast": f"{u} vs {v}", "Difference": round(diff, 3), pname: round(p, 4),
                         "Significant": "Yes" if p < alpha else "No"})
        pair = pd.DataFrame(rows)
    return dict(label=label, family=family, table=t, pair=pair, se=se, r=r, k=k, ms_e=ms_e, df_e=df_e,
                cd=crit(2) if comparable else np.nan, comparable=comparable, err_key=err_key, mu=mu, names=names)

# ───────────────────────── Diagnostics & Assumption Checks ─────────────────────────
def assumption_tests(d, fit, design):
    resid = fit["resid"]
    sh = stats.shapiro(resid) if 3 <= len(resid) <= 5000 else (np.nan, np.nan)
    grp_cols = ["A", "B"] if DESIGNS[design]["two_factor"] else ["A"]
    try:
        groups = [g["Y"].to_numpy() for _, g in d.groupby(grp_cols) if len(g) > 1]
        lev = stats.levene(*groups, center="median") if len(groups) >= 2 else (np.nan, np.nan)
    except Exception:
        lev = (np.nan, np.nan)
    return sh, lev

def studentized(fit):
    h = np.clip(fit["hat"], 0, 0.999999)
    e, dfe, rss = fit["resid"], fit["dfe"], fit["rss"]
    if dfe <= 1:
        return np.full(len(e), np.nan)
    s2_i = (rss - e ** 2 / (1 - h)) / (dfe - 1)
    return e / np.sqrt(np.clip(s2_i, 1e-300, None) * (1 - h))

# ───────────────────────── Multi-Design Demo Data Generator ─────────────────────────
def generate_sample_dataset(design):
    rng = np.random.default_rng(101)
    if design == "CRD":
        trts = ["Control", "Biochar 5 t/ha", "Biochar 10 t/ha", "Biochar 15 t/ha"]
        rows = []
        for t, base in zip(trts, [40, 48, 54, 51]):
            for _ in range(5):
                rows.append({"Treatment": t, "Grain_Yield": round(base + rng.normal(0, 2.2), 2),
                             "Biomass": round(base * 2.1 + rng.normal(0, 3.5), 2)})
        return pd.DataFrame(rows)
    elif design == "RCBD":
        trts = ["T0 (0 mM)", "T1 (50 mM)", "T2 (100 mM)", "T3 (150 mM)", "T4 (200 mM)"]
        rows = []
        for r in range(4):
            blk = rng.normal(0, 1.2)
            for t, base in zip(trts, [45, 41, 35, 27, 20]):
                rows.append({"Replication": f"R{r+1}", "Salinity": t,
                             "Grain_Yield": round(base + blk + rng.normal(0, 1.0), 2),
                             "Plant_Height": round(base * 1.8 + blk + rng.normal(0, 1.5), 2)})
        return pd.DataFrame(rows)
    elif design == "FACT":
        irr = ["I0 (Dry)", "I1 (Moderate)", "I2 (Adequate)"]
        nit = ["N0", "N60", "N120"]
        rows = []
        for r in range(3):
            blk = rng.normal(0, 1.0)
            for i, base_i in zip(irr, [30, 42, 50]):
                for n, base_n in zip(nit, [0, 8, 14]):
                    rows.append({"Block": f"B{r+1}", "Irrigation": i, "Nitrogen": n,
                                 "Yield_t_ha": round(base_i + base_n + blk + rng.normal(0, 1.1), 2)})
        return pd.DataFrame(rows)
    else:  # SPLIT
        main_plots = ["Tillage_Zero", "Tillage_Conv", "Tillage_Deep"]
        sub_plots = ["Var_A (Saline)", "Var_B (Local)", "Var_C (Hybrid)"]
        rows = []
        for r in range(3):
            for mp, bmp in zip(main_plots, [32, 40, 43]):
                err_a = rng.normal(0, 1.5)
                for sp, bsp in zip(sub_plots, [8, 2, 12]):
                    err_b = rng.normal(0, 0.9)
                    rows.append({"Rep": f"R{r+1}", "MainPlot_Tillage": mp, "SubPlot_Variety": sp,
                                 "Root_Length": round(bmp + bsp + err_a + err_b, 2),
                                 "Grain_Yield": round((bmp + bsp) * 0.8 + err_a + err_b, 2)})
        return pd.DataFrame(rows)

# ───────────────────────── Publication Plot Generator (Prism Inspired) ─────────────────────────
PALETTES = {
    "Steel Blue": ["#3b82f6", "#1d4ed8", "#60a5fa", "#93c5fd"],
    "Forest Emerald": ["#059669", "#047857", "#10b981", "#6ee7b7"],
    "Academic Slate": ["#475569", "#334155", "#64748b", "#94a3b8"],
    "Okabe-Ito (Colorblind-Safe)": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
}

def render_publication_plot(family_res, response_name, palette_name, show_letters=True):
    tbl = family_res["table"]
    names = tbl[family_res["label"]].tolist()
    means = tbl["LS-Mean"].to_numpy()
    ses = tbl["SE"].to_numpy()
    groups = tbl["Group"].tolist()

    x = np.arange(len(names))
    colors = PALETTES[palette_name]
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(max(5.5, 0.95 * len(names) + 1.8), 4.8), dpi=300)
    bars = ax.bar(x, means, yerr=ses, capsize=4.5, color=bar_colors, edgecolor="#0f172a",
                  linewidth=1.1, alpha=0.9, width=0.52, error_kw={"elinewidth": 1.1, "ecolor": "#0f172a"})

    top = (means + ses).max()
    if show_letters:
        for xi, mu, se, g in zip(x, means, ses, groups):
            if g and g != "—":
                ax.text(xi, mu + se + top * 0.025, g, ha="center", va="bottom",
                        fontsize=11, fontweight="bold", fontfamily="Times New Roman")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_ylabel(f"{response_name} (LS-Mean ± SE)", fontsize=10.5, fontweight="bold")
    ax.set_xlabel(family_res["label"], fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, top * 1.20)

    # Scientific Clean Borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#94a3b8")
    fig.tight_layout()
    return fig

# ───────────────────────── Executive Header ─────────────────────────
st.markdown(f"""
<div style="padding: 6px 0 16px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: baseline;">
        <div>
            <span style="font-size: 0.72rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; 
                         background: #0ea5e914; color: #0284c7; padding: 4px 10px; border-radius: 20px; border: 1px solid #0ea5e933;">
                {tr("ENTERPRISE BIOMETRIC SUITE", "বায়োমেট্রিক রিসার্চ ইঞ্জিন")}
            </span>
            <h1 style="font-size: 2.3rem; font-weight: 800; margin: 8px 0 2px 0; color: #0f172a; letter-spacing: -0.5px;">
                AgriStat <span style="color: #0284c7;">Studio</span>
            </h1>
            <p style="font-size: 0.95rem; color: #475569; margin: 0;">
                {tr("Design-Aware General Linear Modeling • Multi-Comparison Tests • SAS-Grade Variance Decomposition", 
                    "ডিজাইন-সচেতন লিনিয়ার মডেলিং • মাল্টিপল রেঞ্জ টেস্ট • এসএএস-গ্রেড ভ্যারিয়েন্স পার্টিশন")}
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────── Sidebar Navigation ─────────────────────────
st.sidebar.markdown(f"### ⚙️ {tr('1. Experimental Setup', '১. পরীক্ষণ নকশা')}")
design_key = st.sidebar.selectbox(
    tr("Design Architecture", "নকশা নির্বাচন"),
    list(DESIGNS.keys()),
    format_func=lambda k: DESIGNS[k]["label"]
)
spec = DESIGNS[design_key]

st.sidebar.markdown(f"### 📁 {tr('2. Data Source', '২. ডেটা সোর্স')}")
source_mode = st.sidebar.radio(
    tr("Data Input Mode", "ইনপুট মোড"),
    (tr("Interactive Grid", "ইন্টারেক্টিভ গ্রিড"), tr("File Upload", "ফাইল আপলোড"), tr("Load Standard Sample", "নমুনা ডেটাসেট"))
)

raw_df = None
if source_mode == tr("Interactive Grid", "ইন্টারেক্টিভ গ্রিড"):
    st.info(tr("💡 Directly edit cells below or append observations at the bottom.",
               "💡 গ্রিডের সেলে ক্লিক করে এডিট করুন অথবা নিচে নতুন সারি যুক্ত করুন।"))
    raw_df = st.data_editor(generate_sample_dataset(design_key), num_rows="dynamic", use_container_width=True, key=f"grid_{design_key}")
elif source_mode == tr("File Upload", "ফাইল আপলোড"):
    up = st.sidebar.file_uploader(tr("Upload Spreadsheet (.csv, .xlsx)", "স্প্রেডশীট আপলোড"), type=["csv", "xlsx"])
    if up is not None:
        try:
            raw_df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            st.dataframe(raw_df.head(10), use_container_width=True)
        except Exception as ex:
            st.error(f"File Parsing Error: {ex}")
    else:
        st.info(tr("Upload your trial spreadsheet to commence analysis.", "বিশ্লেষণ শুরু করতে ফাইল আপলোড করুন।"))
else:
    raw_df = generate_sample_dataset(design_key)
    with st.expander(tr("Inspect Active Sample", "নমুনা ডেটাসেট প্রিভিউ"), expanded=False):
        st.dataframe(raw_df, use_container_width=True)

if raw_df is None or raw_df.dropna(how="all").empty:
    st.stop()

# ───────────────────────── Factor & Column Mapping ─────────────────────────
st.sidebar.markdown(f"### 🎯 {tr('3. Variable Mapping', '৩. ফ্যাক্টর ও ভ্যারিয়েবল')}")
cols_list = list(raw_df.columns)

col_map = {}
if spec["needs_rep"]:
    rep_default = [c for c in cols_list if any(k in c.lower() for k in ["rep", "block", "blk"])]
    col_map["rep"] = st.sidebar.selectbox(tr("Block / Replication (Rep)", "ব্লক / রেপ্লিকেশন"), cols_list,
                                          index=cols_list.index(rep_default[0]) if rep_default else 0)

trt_a_default = [c for c in cols_list if any(k in c.lower() for k in ["treat", "trt", "salin", "till", "geno", "a"])]
col_map["a"] = st.sidebar.selectbox(
    tr("Treatment A (Main-Plot)" if spec["two_factor"] else "Treatment Factor", "ট্রিটমেন্ট ফ্যাক্টর (বা মেইন-প্লট)"),
    cols_list, index=cols_list.index(trt_a_default[0]) if trt_a_default else min(1, len(cols_list)-1)
)

if spec["two_factor"]:
    rem_cols = [c for c in cols_list if c not in [col_map.get("rep"), col_map["a"]]]
    col_map["b"] = st.sidebar.selectbox(tr("Factor B (Sub-Plot)", "ফ্যাক্টর B (সাব-প্লট)"), rem_cols if rem_cols else cols_list)

numeric_candidates = [c for c in raw_df.select_dtypes(include=[np.number]).columns if c not in col_map.values()]
if not numeric_candidates:
    st.error(tr("No continuous response variable detected.", "কোনো সংখ্যাভিত্তিক রেসপন্স ভ্যারিয়েবল পাওয়া যায়নি।"))
    st.stop()

response_var = st.sidebar.selectbox(tr("Response Metric (Y)", "রেসপন্স ভ্যারিয়েবল (Y)"), numeric_candidates)

# ───────────────────────── Statistical Engine Controls ─────────────────────────
st.sidebar.markdown(f"### ⚙️ {tr('4. Model Parameters', '৪. মডেল ও টেস্ট সেটিংস')}")
comp_method = st.sidebar.selectbox(tr("Mean Separation", "গড় পৃথকীকরণ পরীক্ষা"), METHODS, index=0)
alpha_level = st.sidebar.selectbox(tr("Significance Threshold (α)", "তাৎপর্য স্তর (α)"), [0.05, 0.01, 0.001], index=0)
ss_type = st.sidebar.selectbox(tr("Sum of Squares Type", "সাম অব স্কয়ার্স টাইপ"), ["III", "I"],
                               format_func=lambda s: f"Type {s} ({'Partial / Orthogonal' if s=='III' else 'Sequential'})")
data_transform = st.sidebar.selectbox(tr("Mathematical Transformation", "ট্রান্সফরমেশন"), list(TRANSFORMS.keys()))
auto_avg = st.sidebar.checkbox(tr("Average Subsampling / Duplicates", "সাব-স্যাম্পল গড় করুন"), value=False)

# Developer Credit (Corporate Bio)
st.sidebar.markdown("---")
with st.sidebar.expander(f"👨‍💻 {tr('Scientific Developer', 'ডেভেলপার পরিচিতি')}", expanded=False):
    st.markdown(f"""
    **Md. Akib Rubayiet Tahmid**  
    *{tr('Biometrics & Agricultural Research Specialist', 'বায়োমেট্রিক্স ও এগ্রিকালচারাল রিসার্চ স্পেশালিস্ট')}*  
    
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/akib-rubayiet-tahmid)
    """)

# ───────────────────────── Model Fitting ─────────────────────────
label_dict = {"REP": col_map.get("rep", "Block"), "A": col_map["a"], "B": col_map.get("b", "Factor_B")}

try:
    clean_d, d_info = prepare(raw_df, design_key, col_map, response_var, data_transform, auto_avg)
    model_fit = fit_model(clean_d, design_key, label_dict, ss_type)
except AnalysisError as ae:
    st.error(ae.bn if BN else ae.en)
    st.stop()
except Exception as e:
    st.error(f"Computation Engine Exception: {e}")
    st.stop()

# Mean Families
A_lbl, B_lbl = label_dict["A"], label_dict.get("B", "B")
if design_key in ("CRD", "RCBD"):
    fam_specs = [(("A",), "RES", A_lbl, True)]
elif design_key == "FACT":
    fam_specs = [(("A",), "RES", A_lbl, True), (("B",), "RES", B_lbl, True), (("A", "B"), "RES", f"{A_lbl} × {B_lbl}", True)]
else:  # SPLIT
    fam_specs = [(("A",), SPLIT_ERR_A, A_lbl, True), (("B",), "RES", B_lbl, True), (("A", "B"), "RES", f"{A_lbl} × {B_lbl}", False)]

families = [mean_family(clean_d, model_fit, fam, ek, lab, comp_method, alpha_level, data_transform, comp)
            for fam, ek, lab, comp in fam_specs]

sh_res, lev_res = assumption_tests(clean_d, model_fit, design_key)
stud_resid = studentized(model_fit)
grand_mean = clean_d["Y"].mean()
cv_main = (np.sqrt(model_fit["mse"]) / grand_mean) * 100

# ───────────────────────── Performance KPI Bar ─────────────────────────
kpi_cols = st.columns(5)
kpi_cols[0].metric(tr("Grand Mean (Ȳ)", "সামগ্রিক গড়"), f"{grand_mean:.2f}")
kpi_cols[1].metric(tr("CV % (Error b)", "সিভি %"), f"{cv_main:.2f}%")
kpi_cols[2].metric(tr("Model R²", "মডেল R²"), f"{model_fit['r2']:.3f}")
kpi_cols[3].metric(tr("Residual MS", "এরর মিন স্কয়ার"), f"{model_fit['mse']:.3f}")

if design_key == "SPLIT":
    cv_a = (np.sqrt(model_fit["E"][SPLIT_ERR_A][0]) / grand_mean) * 100
    kpi_cols[4].metric(tr("CV % (Error a)", "মেইন-প্লট সিভি %"), f"{cv_a:.2f}%")
else:
    kpi_cols[4].metric(tr("Design Efficiency", "ডিজাইন এফিসিয়েন্সি"), "Balanced" if d_info["balanced"] else "Unbalanced")

# ───────────────────────── Tabbed Analytical Workspace ─────────────────────────
tab_anova, tab_means, tab_plots, tab_diag, tab_batch, tab_rep = st.tabs([
    tr("📊 ANOVA Partitioning", "📊 অ্যানোভা টেবিল"),
    tr("📑 Mean Separation & Letters", "📑 গড় পৃথকীকরণ ও গ্রুপিং"),
    tr("📈 Publication Graphics", "📈 পাবলিকেশন গ্রাফ"),
    tr("🩺 Residual Diagnostics", "🩺 মডেল ডায়াগনস্টিকস"),
    tr("🧮 Multi-Trait Batch", "🧮 মাল্টি-ট্রেইট সারাংশ"),
    tr("📋 Thesis Methods & Export", "📋 রিপোর্ট ও এক্সপোর্ট")
])

# ── 1. ANOVA Tab ──
with tab_anova:
    st.markdown(f"#### {tr('Analysis of Variance Table (Two-Way/Factorial General Linear Model)', 'অ্যানালাইসিস অব ভ্যারিয়েন্স টেবিল')}")
    anova_df = display_anova(model_fit["tbl"], alpha_level)
    st.dataframe(anova_df, use_container_width=True, hide_index=True)

    if design_key == "SPLIT":
        st.info(tr("ℹ️ Note: Split-Plot models partition experimental error into Error (a) [Main-plot: Rep × A] and Error (b) [Sub-plot residual]. F-tests are automatically scaled against their respective error terms.",
                   "ℹ️ জ্ঞাতব্য: স্প্লিট-প্লট মডেলে মেইন-প্লটের F-অনুপাত Error (a) এবং সাব-প্লটের F-অনুপাত Error (b) এর প্রেক্ষিতে হিসাব করা হয়েছে।"))

    with st.expander(tr("Variance Explained & Effect Sizes (Partial η²)", "ভেরিয়েন্স ব্যাখ্যা ও পার্শিয়াল ইটা-স্কয়ার্ড")):
        sig_effects = model_fit["tbl"].dropna(subset=["F"])
        for _, r in sig_effects.iterrows():
            if r["Source"] not in ["Total", "Error (a)", "Error (b)", "Residual Error"]:
                pct = r["EtaP"] * 100 if not np.isnan(r["EtaP"]) else 0
                st.write(f"• **{r['Source']}**: accounts for **{pct:.1f}%** of the partial variance (p = {fmt_p(r['P'])})")

# ── 2. Mean Separation Tab ──
with tab_means:
    st.markdown(f"#### {tr(f'Least-Squares Means & Grouping via {comp_method} (α = {alpha_level})', f'গড় এবং {comp_method} গ্রুপিং')}")
    target_fam = st.selectbox(tr("Select Factor Family", "ফ্যাক্টর নির্বাচন"), range(len(families)),
                              format_func=lambda idx: families[idx]["label"])
    active_fam = families[target_fam]
    st.dataframe(active_fam["table"], use_container_width=True, hide_index=True)
    st.caption(tr("Letters indicate statistical groupings: Means sharing the same letter do not differ significantly.",
                  "একই অক্ষরযুক্ত গড়গুলোর মধ্যে পরিসংখ্যানগত কোনো তাৎপর্যপূর্ণ পার্থক্য নেই।"))

    if active_fam["pair"] is not None:
        with st.expander(tr("Pairwise Contrasts & Adjusted P-values", "জোড়ায় জোড়ায় তুলনা ও অ্যাডজাস্টেড পি-মান")):
            st.dataframe(active_fam["pair"], use_container_width=True, hide_index=True)

# ── 3. Publication Graphics Tab ──
with tab_plots:
    p_col1, p_col2, p_col3 = st.columns([1, 1, 1])
    plt_fam_idx = p_col1.selectbox(tr("Plot Factor", "গ্রাফের ফ্যাক্টর"), range(len(families)),
                                   format_func=lambda i: families[i]["label"], key="plt_fam")
    plt_palette = p_col2.selectbox(tr("Color Scheme", "কালার প্যালেট"), list(PALETTES.keys()))
    show_sig_letters = p_col3.checkbox(tr("Annotate Letter Groups", "অক্ষর গ্রুপিং লেবেল দেখান"), value=True)

    fig = render_publication_plot(families[plt_fam_idx], response_var, plt_palette, show_sig_letters)
    st.pyplot(fig)

    d_col1, d_col2 = st.columns(2)
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
    d_col1.download_button(tr("💾 Export 300 DPI PNG", "💾 ডাউনলোড 300 DPI PNG"), buf_png.getvalue(),
                           f"{response_var}_{families[plt_fam_idx]['label']}.png", "image/png")

    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", dpi=300, bbox_inches="tight")
    d_col2.download_button(tr("📄 Export Vector PDF", "📄 ডাউনলোড ভেক্টর PDF"), buf_pdf.getvalue(),
                           f"{response_var}_{families[plt_fam_idx]['label']}.pdf", "application/pdf")

# ── 4. Diagnostics Tab ──
with tab_diag:
    st.markdown(f"#### {tr('Gauss-Markov & Normality Assumption Diagnostics', 'মডেল এজাম্পশন ও নর্মালিটি টেস্ট')}")
    d_kpi1, d_kpi2 = st.columns(2)
    sh_p = sh_res[1]
    lev_p = lev_res[1]
    d_kpi1.metric("Shapiro-Wilk (Residual Normality)", "—" if pd.isna(sh_p) else f"{sh_p:.4f}",
                  delta="Normal" if sh_p > 0.05 else "Non-normal Violation", delta_color="normal" if sh_p > 0.05 else "inverse")
    d_kpi2.metric("Levene Test (Variance Homogeneity)", "—" if pd.isna(lev_p) else f"{lev_p:.4f}",
                  delta="Homoscedastic" if lev_p > 0.05 else "Heteroscedastic", delta_color="normal" if lev_p > 0.05 else "inverse")

    fig_diag, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(9.5, 4), dpi=300)
    ax_d1.scatter(model_fit["fitted"], model_fit["resid"], color="#0284c7", edgecolor="#0f172a", alpha=0.8)
    ax_d1.axhline(0, color="#ef4444", linestyle="--", linewidth=1.2)
    ax_d1.set_xlabel("Fitted Values", fontweight="bold")
    ax_d1.set_ylabel("Residuals", fontweight="bold")
    ax_d1.set_title("Residuals vs Fitted (Homoscedasticity)", fontweight="bold")
    ax_d1.spines["top"].set_visible(False)
    ax_d1.spines["right"].set_visible(False)

    # Q-Q plot via scipy
    sorted_res = np.sort(model_fit["resid"])
    theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_res)))
    ax_d2.scatter(theoretical_q, sorted_res, color="#0284c7", edgecolor="#0f172a", alpha=0.8)
    ax_d2.plot(theoretical_q, theoretical_q * np.std(sorted_res) + np.mean(sorted_res), color="#ef4444", linestyle="--")
    ax_d2.set_xlabel("Theoretical Quantiles", fontweight="bold")
    ax_d2.set_ylabel("Sample Quantiles", fontweight="bold")
    ax_d2.set_title("Normal Q-Q Plot", fontweight="bold")
    ax_d2.spines["top"].set_visible(False)
    ax_d2.spines["right"].set_visible(False)
    fig_diag.tight_layout()
    st.pyplot(fig_diag)

    outliers_mask = np.abs(stud_resid) > 3.0
    if np.any(outliers_mask):
        st.warning(tr(f"⚠️ {np.sum(outliers_mask)} observation(s) exhibit studentized residuals |r| > 3.0 (Potential Outliers).",
                      f"⚠️ {np.sum(outliers_mask)}টি সম্ভাব্য আউটলায়ার পাওয়া গেছে (|r| > 3.0)।"))
        st.dataframe(clean_d[outliers_mask], use_container_width=True)

# ── 5. Multi-Trait Summary Tab ──
with tab_batch:
    st.markdown(f"#### {tr('Batch Processing of All Numerical Traits', 'সব সংখ্যাভিত্তিক বৈশিষ্ট্যের এককালীন বিশ্লেষণ')}")
    batch_rows = []
    for num_v in numeric_candidates:
        try:
            d_sub, _ = prepare(raw_df, design_key, col_map, num_v, data_transform, auto_avg)
            fit_sub = fit_model(d_sub, design_key, label_dict, ss_type)
            tbl_sub = fit_sub["tbl"]
            trt_term = ("A",)
            f_val = tbl_sub.loc[tbl_sub["key"] == trt_term, "F"].values[0]
            p_val = tbl_sub.loc[tbl_sub["key"] == trt_term, "P"].values[0]
            batch_rows.append({
                "Trait": num_v,
                "Grand Mean": round(d_sub["Y"].mean(), 2),
                f"F ({label_dict['A']})": round(f_val, 2),
                "P-Value": fmt_p(p_val),
                "Sig.": sig_stars(p_val, alpha_level),
                "CV %": round((np.sqrt(fit_sub["mse"]) / d_sub["Y"].mean()) * 100, 2)
            })
        except Exception:
            continue
    if batch_rows:
        st.dataframe(pd.DataFrame(batch_rows), use_container_width=True, hide_index=True)

# ── 6. Reporting & Export Tab ──
with tab_rep:
    st.markdown(f"#### {tr('Manuscript Materials & Methods Statement', 'থিসিস বা জার্নালের জন্য মেথডলজি ড্রাফট')}")
    method_text = (
        f"Data were analysed under a {DESIGNS[design_key]['label']} layout. "
        f"Variance decomposition was evaluated via General Linear Model ({'Type III' if ss_type=='III' else 'Type I'} sum of squares) "
        f"accounting for designated block and interaction error terms. "
        f"Post-hoc pairwise mean separation was evaluated using {comp_method} at α = {alpha_level}. "
        f"Model residuals were assessed for normality (Shapiro-Wilk) and homoscedasticity (Levene's test). "
        f"Computations and matrix linear algebra were executed via AgriStat Studio."
    )
    st.code(method_text, language="markdown")

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        clean_d.to_excel(writer, sheet_name="Raw Data", index=False)
        display_anova(model_fit["tbl"], alpha_level).to_excel(writer, sheet_name="ANOVA", index=False)
        for f in families:
            f["table"].to_excel(writer, sheet_name=f["label"][:30], index=False)

    st.download_button(
        label=tr("📊 Download Complete Analysis Workbook (.xlsx)", "📊 সম্পূর্ণ এক্সেল রিপোর্ট ডাউনলোড করুন"),
        data=excel_buf.getvalue(),
        file_name=f"{response_var}_{design_key}_AgriStat_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ───────────────────────── Executive Footer ─────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #64748b; font-size: 0.88rem; padding-bottom: 24px;'>"
    f"AgriStat Studio • {tr('Biometric System Architecture by', 'ডিজাইন ও তত্ত্বাবধানে')} <b>Md. Akib Rubayiet Tahmid</b> | "
    f"<a href='https://www.linkedin.com/in/akib-rubayiet-tahmid' target='_blank' style='text-decoration: none; color: #0284c7; font-weight: 600;'>LinkedIn</a>"
    f"</div>",
    unsafe_allow_html=True
)
