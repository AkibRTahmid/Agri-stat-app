"""
AgriStat Studio – Enterprise Biometrics & Experimental Design Suite (v3.7)
Designs : CRD | RCBD | Factorial (RCBD) | Split-Plot (RCBD)
Methods : Tukey HSD, LSD, Bonferroni, Duncan (DMRT), SNK, Dunnett (vs Control)
Analytics: GLM (Type I/III SS), Trend Contrasts, Interaction Slicing, Genetic Parameters
Engine  : High-precision pure NumPy / SciPy / Pandas (No statsmodels dependency)
"""
from __future__ import annotations

import inspect
import io
import itertools
import re
import string
import warnings
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore")

# ───────────────────────── Page Configuration ─────────────────────────
st.set_page_config(
    page_title="AgriStat Studio | Biometrics Engine",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────── UI Styling (Theme-Safe & Floating Sidebar Toggle) ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.maateen.me/kalpurush/font.css');

/* Hide GitHub fork/star, Streamlit deploy button, rainbow decoration line & footer */
[data-testid="stToolbarActions"],
.stDeployButton,
[data-testid="stDecoration"],
#MainMenu,
footer { 
    display: none !important; 
}

/* Ensure header is transparent and clickable */
header[data-testid="stHeader"] { 
    background: transparent !important;
    pointer-events: auto !important;
    height: 3rem !important;
}

/* System typography */
html, body, .stApp, p, label, li, td, th, h1, h2, h3, h4, input, button, textarea {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Kalpurush",
                 "Helvetica Neue", Arial, sans-serif !important;
}

/* Translucent KPI cards for adaptive dark/light backgrounds */
[data-testid="stMetric"] {
    background: rgba(128,128,128,0.07);
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 10px;
    padding: 10px 14px;
}
button[data-baseweb="tab"] { font-size: 0.95rem !important; font-weight: 600 !important; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.20); }

/* নিশ্চিত ফ্লোটিং সাইডবার রি-ওপেন বাটন */
header [data-testid="stSidebarCollapseButton"],
header button[aria-label*="sidebar" i],
header button[aria-label*="Sidebar" i],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 99999999 !important;
    position: fixed !important;
    top: 12px !important;
    left: 12px !important;
    background: #0284c7 !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 8px !important;
    width: 38px !important;
    height: 38px !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.5) !important;
    cursor: pointer !important;
}

header [data-testid="stSidebarCollapseButton"]:hover,
[data-testid="stSidebarCollapsedControl"]:hover {
    background: #0369a1 !important;
}

header [data-testid="stSidebarCollapseButton"] svg,
header button svg,
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #ffffff !important;
    stroke: #ffffff !important;
    color: #ffffff !important;
    display: block !important;
    width: 20px !important;
    height: 20px !important;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# ───────────────────────── Core Mathematical Constants ─────────────────────────
TRANSFORMS = {
    "None": lambda y: y,
    "log10(x)": lambda y: np.log10(y),
    "log10(x + 1)": lambda y: np.log10(y + 1.0),
    "sqrt(x)": lambda y: np.sqrt(y),
    "sqrt(x + 0.5)": lambda y: np.sqrt(y + 0.5),
    "arcsine (%)": lambda y: np.degrees(np.arcsin(np.sqrt(np.clip(y / 100.0, 0, 1)))),
    "Rank Transform (Non-parametric)": lambda y: stats.rankdata(y),
}
METHODS = ["Tukey HSD", "LSD", "Bonferroni", "Duncan (DMRT)", "SNK", "Dunnett (vs Control)"]

DESIGNS = {
    "CRD": dict(terms=[("A",)], err={}, needs_rep=False, two_factor=False, label="Completely Randomized (CRD)"),
    "RCBD": dict(terms=[("REP",), ("A",)], err={}, needs_rep=True, two_factor=False, label="Randomized Block (RCBD)"),
    "FACT": dict(terms=[("REP",), ("A",), ("B",), ("A", "B")], err={}, needs_rep=True, two_factor=True, label="Factorial RCBD (A × B)"),
    "SPLIT": dict(
        terms=[("REP",), ("A",), ("REP", "A"), ("B",), ("A", "B")],
        err={("REP",): ("REP", "A"), ("A",): ("REP", "A")},
        needs_rep=True, two_factor=True, label="Split-Plot RCBD (Main/Sub)",
    ),
}
SPLIT_ERR_A = ("REP", "A")


class AnalysisError(ValueError):
    def __init__(self, en, bn=None):
        super().__init__(en)
        self.en, self.bn = en, bn or en


# ───────────────────────── Language Switcher ─────────────────────────
st.sidebar.markdown("### 🌐 Interface / ভাষা")
BN = st.sidebar.radio("Lang", ("English", "বাংলা"), horizontal=True,
                      label_visibility="collapsed", key="lang_opt") == "বাংলা"


def tr(en, bn):
    return bn if BN else en


# ───────────────────────── Streamlit Helpers ─────────────────────────
def _wide_kw(fn):
    try:
        if "stretch" in str(inspect.signature(fn).parameters["width"].annotation):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


def show_df(df, **kw):
    st.dataframe(df, **_wide_kw(st.dataframe), **kw)


def _okey(prefix, options):
    return f"{prefix}_{zlib.crc32('|'.join(map(str, options)).encode('utf-8'))}"


def _sb_caption(en, bn):
    st.sidebar.markdown(
        f"<div style='font-size:0.86rem; font-weight:600; margin:14px 0 5px 0; opacity:0.9;'>{tr(en, bn)}</div>",
        unsafe_allow_html=True
    )


def sb_select(en, bn, options, key, index=0, format_func=str, help=None):
    options = list(options)
    _sb_caption(en, bn)
    idx = index if 0 <= index < len(options) else 0
    return st.sidebar.selectbox(en, options, index=idx, key=_okey(key, options),
                                format_func=format_func, label_visibility="collapsed", help=help)


def sb_radio(en, bn, options, key, format_func=str):
    options = list(options)
    _sb_caption(en, bn)
    return st.sidebar.radio(en, options, key=key, format_func=format_func, label_visibility="collapsed")


# ───────────────────────── Mathematical Utilities ─────────────────────────
def sig_stars(p, alpha=0.05):
    if p is None or pd.isna(p):
        return ""
    if p >= alpha:
        return "ns"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*"


def fmt_p(p):
    if p is None or pd.isna(p):
        return ""
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def p_text(p):
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"


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
    return " × ".join(labels.get(f, f) for f in term)


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

    n_raw = len(df)
    d = df[used].copy()
    d.columns = list(sel.keys()) + ["Y0"]
    d["Y0"] = pd.to_numeric(d["Y0"], errors="coerce")
    d = d.dropna()

    for f in sel:
        d[f] = d[f].astype(str).str.strip()
    ok = np.ones(len(d), dtype=bool)
    for f in sel:
        ok &= (d[f] != "").to_numpy()
    d = d[ok].reset_index(drop=True)
    n_dropped = n_raw - len(d)

    if len(d) == 0:
        raise AnalysisError("No valid numeric observations remaining.", "কোনো বৈধ সংখ্যাভিত্তিক ডেটা নেই।")

    n_before_avg = len(d)
    if avg_dup and design != "CRD":
        keys = list(sel.keys())
        d = d.groupby(keys, sort=False, as_index=False)["Y0"].mean()

    # Transformation domain checks
    if tf == "log10(x)" and (d["Y0"] <= 0).any():
        raise AnalysisError("log10(x) requires all values > 0. Use log10(x + 1) for zeros.",
                            "log10(x) এর জন্য সব মান > 0 হতে হবে। শূন্য থাকলে log10(x + 1) ব্যবহার করুন।")
    if tf in ("sqrt(x)", "log10(x + 1)") and (d["Y0"] < 0).any():
        raise AnalysisError(f"{tf} requires all values ≥ 0.", f"{tf} এর জন্য সব মান ≥ 0 হতে হবে।")
    if tf == "sqrt(x + 0.5)" and (d["Y0"] < -0.5).any():
        raise AnalysisError("sqrt(x + 0.5) requires all values ≥ -0.5.", "sqrt(x + 0.5) এর জন্য মান ≥ -0.5 হতে হবে।")
    if tf == "arcsine (%)" and ((d["Y0"] < 0) | (d["Y0"] > 100)).any():
        raise AnalysisError("arcsine (%) requires percentage values between 0 and 100.", "arcsine (%) এর জন্য মান ০–১০০ এর মধ্যে হতে হবে।")

    d["Y"] = TRANSFORMS[tf](d["Y0"].to_numpy())

    levels = {f: list(dict.fromkeys(d[f])) for f in sel}
    if len(levels["A"]) < 2:
        raise AnalysisError("Factor A requires at least 2 distinct levels.", "ট্রিটমেন্ট ফ্যাক্টরে কমপক্ষে ২টি লেভেল লাগবে।")
    if spec["two_factor"] and len(levels["B"]) < 2:
        raise AnalysisError("Factor B requires at least 2 distinct levels.", "ফ্যাক্টর B তে কমপক্ষে ২টি লেভেল লাগবে।")
    if spec["needs_rep"] and len(levels["REP"]) < 2:
        raise AnalysisError("Replication requires at least 2 blocks.", "কমপক্ষে ২টি রেপ্লিকেশন/ব্লক লাগবে।")

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

    info = dict(balanced=balanced, n_missing=n_missing, n_dupes=n_dupes, levels=levels, keys=keys,
                n_raw=n_raw, n_dropped=n_dropped, n_used=len(d), averaged=(n_before_avg != len(d)))

    if design == "SPLIT" and not balanced:
        raise AnalysisError(
            f"Split-Plot GLM requires a complete, balanced layout. Found {n_missing} missing and {n_dupes} duplicate cells.",
            f"স্প্লিট-প্লটের জন্য সম্পূর্ণ ব্যালান্সড ডেটা প্রয়োজন ({n_missing} মিসিং, {n_dupes} ডুপ্লিকেট সেল)।")
    return d, info


# ───────────────────────── General Linear Model Engine ─────────────────────────
def fit_model(d, design, labels=None, sstype="III"):
    labels = labels or {"REP": "Replication", "A": "A", "B": "B"}
    spec = DESIGNS[design]
    terms = spec["terms"]
    levels = {f: list(dict.fromkeys(d[f])) for f in sorted({f for t in terms for f in t})}
    n = len(d)
    y = d["Y"].to_numpy(float)
    ones = np.ones((n, 1))
    mats = [term_matrix(d, t, levels) for t in terms]
    X = np.hstack([ones] + mats)
    beta, fitted, resid, rss, rank = _ls(X, y)
    dfe = n - rank
    if dfe <= 0:
        raise AnalysisError(
            f"Zero or negative error degrees of freedom (DF_error = {dfe} ≤ 0). "
            f"Total observations (N = {n}) are insufficient for model parameters (rank = {rank}). "
            f"ANOVA requires experimental replication (≥ 2 observations per treatment level). "
            f"Aggregated or survey summaries without field replicates cannot be analyzed using ANOVA.",
            f"এরর ডিগ্রি অব ফ্রিডম শূন্য বা ঋণাত্মক (DF_error = {dfe} ≤ 0)। "
            f"মোট পর্যবেক্ষণ সংখ্যা (N = {n}) মডেলের প্যারামিটার সংখ্যার জন্য অপর্যাপ্ত। "
            f"অ্যানোভার জন্য অবশ্যই প্রতি ট্রিটমেন্টে একাধিক রেপ্লিকেশন (কমপক্ষে ২টি) থাকতে হবে। "
            f"একক সারসংক্ষেপ বা সার্ভে ডেটায় ফিল্ড-ট্রায়াল অ্যানোভা রান করা যায় না।"
        )

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
            dfe_show = np.nan
        else:
            F = v["ms"] / ms_e if ms_e > 0 else np.inf
            p = float(stats.f.sf(F, v["df"], df_e)) if np.isfinite(F) else 0.0
            eta = v["ss"] / (v["ss"] + ms_e * df_e) if (v["ss"] + ms_e * df_e) > 0 else 0
            dfe_show = df_e
        if is_err:
            name, err_name = "Error (a)", ""
        else:
            name = term_label(t, labels)
            err_name = "Error (a)" if ek != "RES" else ("Error (b)" if design == "SPLIT" else "Error")
        out.append(dict(key=t, Source=name, DF=v["df"], SS=v["ss"], MS=v["ms"], F=F, P=p, EtaP=eta,
                        DFe=dfe_show, ErrorTerm=err_name))

    out.append(dict(key="RES", Source="Error (b)" if design == "SPLIT" else "Residual Error",
                    DF=int(dfe), SS=rss, MS=mse, F=np.nan, P=np.nan, EtaP=np.nan, DFe=np.nan, ErrorTerm=""))
    tss = float(((y - y.mean()) ** 2).sum())
    out.append(dict(key="TOT", Source="Corrected Total", DF=n - 1, SS=tss, MS=np.nan, F=np.nan, P=np.nan,
                    EtaP=np.nan, DFe=np.nan, ErrorTerm=""))
    tbl = pd.DataFrame(out)

    hat = np.einsum("ij,ji->i", X, np.linalg.pinv(X))
    return dict(tbl=tbl, terms=terms, levels=levels, beta=beta, fitted=fitted, resid=resid, rss=rss, dfe=dfe,
                mse=mse, tss=tss, r2=1 - rss / tss if tss > 0 else np.nan, E=E, hat=hat, n=n, labels=labels,
                sstype=sstype, design=design, rank=rank)


def display_anova(tbl, alpha):
    def f(x, nd=3):
        return "—" if x is None or pd.isna(x) else f"{x:.{nd}f}"
    return pd.DataFrame({
        "Source of Variation": tbl["Source"],
        "DF": tbl["DF"].astype(int),
        "Sum of Squares": [f(x) for x in tbl["SS"]],
        "Mean Square": [f(x) for x in tbl["MS"]],
        "F-Ratio": [f(x) for x in tbl["F"]],
        "P-Value": [fmt_p(p) or "—" for p in tbl["P"]],
        "Sig.": [sig_stars(p, alpha) or "—" for p in tbl["P"]],
        "Partial η²": [f(x) for x in tbl["EtaP"]],
        "Error Term": [e if e else "—" for e in tbl["ErrorTerm"]],
    })


def anova_export(tbl, alpha):
    t = tbl.drop(columns=["key"]).copy()
    t["Sig."] = [sig_stars(p, alpha) for p in tbl["P"]]
    return t.rename(columns={"Source": "Source of Variation", "SS": "Sum of Squares", "MS": "Mean Square",
                             "F": "F-Ratio", "P": "P-Value", "EtaP": "Partial eta2", "DFe": "DF (error)",
                             "ErrorTerm": "Error Term"})


# ───────────────────────── LS-Means & Multiple Comparisons ─────────────────────────
def ls_means(d, fit, family):
    terms, levels, beta = fit["terms"], fit["levels"], fit["beta"]
    facs = list(levels.keys())
    grid = pd.DataFrame(list(itertools.product(*[levels[f] for f in facs])), columns=facs)
    Xg = np.column_stack([np.ones(len(grid))] + [term_matrix(grid, t, levels) for t in terms])
    grid["pred"] = Xg @ beta
    g = grid.groupby(list(family), sort=False)["pred"].mean()
    combos = list(itertools.product(*[levels[f] for f in family]))
    vals = [g.loc[c[0]] if len(family) == 1 else g.loc[c] for c in combos]
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
    if method == "Dunnett (vs Control)":
        m = max(k - 1, 1)
        c = stats.t.ppf(1 - alpha / m, dfe) * np.sqrt(2) * se
        return lambda p: c
    return lambda p: stats.studentized_range.ppf((1 - alpha) ** (max(p, 2) - 1), max(p, 2), dfe) * se


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


def mean_family(d, fit, family, err_key, label, method, alpha, tf, ctrl_idx=0, comparable=True):
    combos, mu = ls_means(d, fit, family)
    names = [" × ".join(c) for c in combos]
    keycol = d[list(family)].agg(" × ".join, axis=1) if len(family) > 1 else d[family[0]]
    grp = d.assign(_k=keycol).groupby("_k", sort=False)
    cnt = grp.size().reindex(names).fillna(0)

    if (cnt == 0).any():
        raise AnalysisError(f"Missing treatment cell combination detected in {label}.",
                            f"{label} এ কিছু কম্বিনেশনের ডেটা অনুপস্থিত।")

    k = len(names)
    r = k / np.sum(1.0 / cnt.to_numpy(float))
    ms_e, df_e = fit["E"][err_key]
    se = float(np.sqrt(ms_e / r))
    sd = grp["Y"].std().reindex(names).fillna(0).to_numpy(float)
    mo = grp["Y0"].mean().reindex(names).to_numpy(float)
    crit = make_crit(method, alpha, ms_e, df_e, r, k)

    letters = letter_groups(mu, crit) if comparable and k >= 2 and method != "Dunnett (vs Control)" else ["—"] * k
    disp = [f"{m:.2f} ± {se:.2f}" + (f" {g}" if g not in ("", "—") else "") for m, g in zip(mu, letters)]

    t = pd.DataFrame({
        label: names,
        "Rep (r)": cnt.to_numpy(int),
        "LS-Mean": mu.round(3),
        "SEm ±": round(se, 3),
        "SD": np.round(sd, 3),
        "Group": letters,
        "Mean ± SE (Group)": disp
    })
    if tf != "None":
        t.insert(2, "Orig. Mean", mo.round(3))

    pair = None
    if comparable and k >= 2:
        rows = []
        if method == "Dunnett (vs Control)":
            ctrl_idx = min(max(0, ctrl_idx), k - 1)
            ctrl_name = names[ctrl_idx]
            for iv in range(k):
                if iv == ctrl_idx:
                    continue
                diff = mu[iv] - mu[ctrl_idx]
                se_diff = np.sqrt(2 * ms_e / r)
                t_stat = abs(diff) / se_diff
                p_raw = float(2 * stats.t.sf(t_stat, df_e))
                p_adj = min(1.0, p_raw * (k - 1))
                rows.append({
                    "Contrast (vs Control)": f"{names[iv]} vs {ctrl_name}",
                    "Difference": round(diff, 3),
                    "SE Diff": round(se_diff, 3),
                    "P (Dunnett adj.)": round(p_adj, 4),
                    "Significant": "Yes" if p_adj < alpha else "No"
                })
        else:
            m_pairs = k * (k - 1) / 2
            for (iu, u), (iv, v) in itertools.combinations(enumerate(names), 2):
                diff = mu[iu] - mu[iv]
                if method == "Tukey HSD":
                    p = float(stats.studentized_range.sf(abs(diff) / se, k, df_e))
                    pname = "P (Tukey adj.)"
                else:
                    p = float(2 * stats.t.sf(abs(diff) / (np.sqrt(2) * se), df_e))
                    pname = "P"
                    if method == "Bonferroni":
                        p = min(1.0, p * m_pairs)
                        pname = "P (Bonferroni adj.)"
                rows.append({
                    "Contrast": f"{u} vs {v}",
                    "Difference": round(diff, 3),
                    pname: round(p, 4),
                    "Significant": "Yes" if p < alpha else "No"
                })
        if rows:
            pair = pd.DataFrame(rows)

    err_name = "Error (a)" if err_key != "RES" else ("Error (b)" if fit["design"] == "SPLIT" else "Residual Error")
    return dict(
        label=label, family=family, table=t, pair=pair, se=se, r=r, k=k,
        ms_e=ms_e, df_e=df_e, cd=float(crit(2)) if comparable else np.nan,
        comparable=comparable, err_key=err_key, err_name=err_name,
        mu=mu, names=names, combos=combos, raw_df=d
    )


def build_pivot(fam, levels):
    disp = fam["table"]["Mean ± SE (Group)"].tolist()
    pv = pd.DataFrame(index=levels["A"], columns=levels["B"], dtype=object)
    for (a, b), s in zip(fam["combos"], disp):
        pv.loc[a, b] = s
    return pv


# ───────────────────────── Polynomial Orthogonal Contrasts ─────────────────────────
def compute_polynomial_contrasts(active_family):
    names = active_family["names"]
    mu = active_family["mu"]
    k = len(names)
    r = active_family["r"]
    ms_e = active_family["ms_e"]
    df_e = active_family["df_e"]

    extracted = [re.findall(r"[-+]?\d*\.\d+|\d+", str(n)) for n in names]
    if not all(len(x) > 0 for x in extracted) or k < 3:
        return None

    try:
        x_vals = np.array([float(x[0]) for x in extracted])
        s_idx = np.argsort(x_vals)
        x_s, mu_s = x_vals[s_idx], mu[s_idx]

        x_bar = np.mean(x_s)
        c_lin = x_s - x_bar
        c_quad = (x_s - x_bar) ** 2 - np.mean((x_s - x_bar) ** 2)

        contrasts = [("Linear Trend", c_lin), ("Quadratic Trend", c_quad)]
        res_rows = []
        for cname, c_vec in contrasts:
            sum_c2 = np.sum(c_vec ** 2)
            if sum_c2 == 0:
                continue
            L = np.sum(c_vec * mu_s)
            ss_c = (r * (L ** 2)) / sum_c2
            f_c = ss_c / ms_e
            p_c = float(stats.f.sf(f_c, 1, df_e))
            res_rows.append({
                "Contrast": cname,
                "Contrast SS": round(ss_c, 3),
                "F-Ratio": round(f_c, 3),
                "P-Value": fmt_p(p_c),
                "Sig.": sig_stars(p_c, 0.05)
            })
        return pd.DataFrame(res_rows)
    except Exception:
        return None


# ───────────────────────── Simple-Effects Slicing ─────────────────────────
def compute_simple_effects(d, factor_a, factor_b, y_col, alpha):
    records = []
    for a_level, sub in d.groupby(factor_a):
        k_b = sub[factor_b].nunique()
        if k_b < 2:
            continue
        groups = [g[y_col].to_numpy() for _, g in sub.groupby(factor_b)]
        f_val, p_val = stats.f_oneway(*groups)
        records.append({
            f"Slice ({factor_a} = {a_level})": f"Effect of {factor_b}",
            "Levels of B": k_b,
            "F-Ratio": round(f_val, 3),
            "P-Value": fmt_p(p_val),
            "Sig.": sig_stars(p_val, alpha)
        })
    return pd.DataFrame(records)


# ───────────────────────── Genetic Parameters Module ─────────────────────────
def compute_genetic_parameters(model_fit, d, response_var):
    if model_fit["design"] != "RCBD":
        return None
    tbl = model_fit["tbl"]
    trt_row = tbl.loc[tbl["key"] == ("A",)]
    err_row = tbl.loc[tbl["key"] == "RES"]
    if trt_row.empty or err_row.empty:
        return None

    ms_g = trt_row["MS"].values[0]
    ms_e = err_row["MS"].values[0]
    r = len(d["REP"].unique())
    gm = float(d["Y"].mean())

    if gm == 0 or ms_e <= 0:
        return None

    var_g = max(0.0, (ms_g - ms_e) / r)
    var_e = ms_e
    var_p = var_g + var_e

    gcv = (np.sqrt(var_g) / gm) * 100
    pcv = (np.sqrt(var_p) / gm) * 100
    h2_b = (var_g / var_p) * 100 if var_p > 0 else 0.0

    k_sel = 2.06
    ga = k_sel * np.sqrt(var_p) * (h2_b / 100.0)
    gam = (ga / gm) * 100.0

    return pd.DataFrame({
        "Genetic Parameter": [
            "Genotypic Variance (σ²g)",
            "Phenotypic Variance (σ²p)",
            "Environmental Variance (σ²e)",
            "GCV (%)",
            "PCV (%)",
            "Broad-sense Heritability h²b (%)",
            "Genetic Advance (GA, 5%)",
            "Genetic Advance as % of Mean (GAM %)"
        ],
        "Estimate": [
            f"{var_g:.3f}", f"{var_p:.3f}", f"{var_e:.3f}",
            f"{gcv:.2f}%", f"{pcv:.2f}%", f"{h2_b:.2f}%",
            f"{ga:.3f}", f"{gam:.2f}%"
        ]
    })


# ───────────────────────── Diagnostics ─────────────────────────
def assumption_tests(d, fit, design):
    resid = fit["resid"]
    sh = stats.shapiro(resid) if 3 <= len(resid) <= 5000 else (np.nan, np.nan)
    grp_cols = ["A", "B"] if DESIGNS[design]["two_factor"] else ["A"]
    try:
        groups = [g["Y"].to_numpy() for _, g in d.groupby(grp_cols) if len(g) > 1]
        lev = stats.levene(*groups, center="median") if len(groups) >= 2 else (np.nan, np.nan)
    except Exception:
        lev = (np.nan, np.nan)
    try:
        kw = stats.kruskal(*[g["Y"].to_numpy() for _, g in d.groupby("A")])
    except Exception:
        kw = (np.nan, np.nan)
    return sh, lev, kw


def studentized_residuals(fit):
    h = np.clip(fit["hat"], 0, 0.999999)
    e, dfe, rss = fit["resid"], fit["dfe"], fit["rss"]
    if dfe <= 1:
        return np.full(len(e), np.nan)
    s2_i = (rss - e ** 2 / (1 - h)) / (dfe - 1)
    return e / np.sqrt(np.clip(s2_i, 1e-300, None) * (1 - h))


# ───────────────────────── Generic Agricultural Benchmark Datasets ─────────────────────────
def generate_sample_dataset(design):
    rng = np.random.default_rng(2026)
    if design == "CRD":
        # Tomato Cultivar Trial (5 cultivars x 5 replications = 25 observations)
        trts = ["Cultivar_Alpha", "Cultivar_Beta", "Cultivar_Gamma", "Cultivar_Delta", "Cultivar_Epsilon"]
        rows = []
        for t, base in zip(trts, [51.1, 61.3, 45.1, 70.1, 57.1]):
            for _ in range(5):
                rows.append({
                    "Cultivar": t,
                    "Fruit_Yield_kg": round(base + rng.normal(0, 1.2), 2),
                    "Brix_Percent": round(base * 0.08 + rng.normal(0, 0.15), 2)
                })
        return pd.DataFrame(rows)

    elif design == "RCBD":
        # Maize Nitrogen Trial (5 levels of N x 4 blocks)
        trts = ["N0_Control", "N60_kg_ha", "N120_kg_ha", "N180_kg_ha", "N240_kg_ha"]
        rows = []
        for r in range(4):
            blk = rng.normal(0, 0.25)
            for t, base in zip(trts, [3.2, 5.1, 7.4, 8.6, 8.1]):
                rows.append({
                    "Block": f"Block_{r+1}",
                    "Nitrogen_Level": t,
                    "Grain_Yield_t_ha": round(base + blk + rng.normal(0, 0.18), 2),
                    "Ear_Length_cm": round(base * 2.2 + blk * 0.5 + rng.normal(0, 0.4), 2)
                })
        return pd.DataFrame(rows)

    elif design == "FACT":
        # Rice Crop Spacing x Potassium Fertilizer
        spacing = ["Spacing_15cm", "Spacing_20cm", "Spacing_25cm"]
        k_levels = ["K0_Control", "K40_kg_ha", "K80_kg_ha"]
        rows = []
        for r in range(3):
            blk = rng.normal(0, 0.2)
            for sp, b_sp in zip(spacing, [4.0, 5.2, 4.8]):
                for k, b_k in zip(k_levels, [0.0, 0.8, 1.4]):
                    rows.append({
                        "Replication": f"Rep_{r+1}",
                        "Spacing": sp,
                        "Potassium": k,
                        "Yield_t_ha": round(b_sp + b_k + blk + rng.normal(0, 0.15), 2)
                    })
        return pd.DataFrame(rows)

    else:  # SPLIT
        # Soybean Irrigation (Main) x Sowing Date (Sub)
        main_plots = ["Irrig_Rainfed", "Irrig_Full"]
        sub_plots = ["Sowing_Early", "Sowing_Mid", "Sowing_Late"]
        rows = []
        for r in range(3):
            for mp, bmp in zip(main_plots, [2.1, 3.4]):
                err_a = rng.normal(0, 0.15)
                for sp, bsp in zip(sub_plots, [0.6, 0.2, -0.4]):
                    err_b = rng.normal(0, 0.08)
                    rows.append({
                        "Rep": f"Block_{r+1}",
                        "Irrigation_Main": mp,
                        "SowingDate_Sub": sp,
                        "Pod_Count": round((bmp + bsp + err_a + err_b) * 22, 1),
                        "Seed_Yield_t_ha": round(bmp + bsp + err_a + err_b, 2)
                    })
        return pd.DataFrame(rows)


def blank_template(design):
    s = lambda: pd.Series(dtype=str)
    f = lambda: pd.Series(dtype=float)
    if design == "CRD":
        return pd.DataFrame({"Treatment": s(), "Yield": f()})
    if design == "RCBD":
        return pd.DataFrame({"Replication": s(), "Treatment": s(), "Yield": f()})
    if design == "FACT":
        return pd.DataFrame({"Block": s(), "Factor_A": s(), "Factor_B": s(), "Yield": f()})
    return pd.DataFrame({"Rep": s(), "MainPlot": s(), "SubPlot": s(), "Yield": f()})


# ───────────────────────── Robust File Reading & Intelligent Column Mapping ─────────────────────────
def is_numeric_like(s):
    if pd.api.types.is_numeric_dtype(s):
        return True
    nn = int(s.notna().sum())
    return nn > 0 and pd.to_numeric(s, errors="coerce").notna().sum() / nn >= 0.85


def is_design_metadata_col(col_name: str) -> bool:
    """Detects whether a column is a design/experimental index rather than a measurement trait."""
    c = str(col_name).strip().lower()
    exact_match = bool(re.search(r"^(rep|replication|block|blk|sl|id|plot|entry|sno|serial|no)$", c))
    suffix_match = bool(re.search(r"(_id|plot_no|rep_no|block_no|sl_no)$", c))
    return exact_match or suffix_match


def guess_columns(df, spec):
    cols = list(df.columns)
    is_cat = {c: not pd.api.types.is_numeric_dtype(df[c]) for c in cols}
    rep = None
    if spec["needs_rep"]:
        rep = next((c for c in cols if re.search(r"^(rep|block|blk|replication)", c, re.I)), None)
        if rep is None:
            rep = next((c for c in cols if re.search(r"rep|block|blk", c, re.I)), None)
        if rep is None:
            rep = next((c for c in cols if is_cat[c]), cols[0])
    rest = [c for c in cols if c != rep]
    pref_a = [c for c in rest if re.search(r"treat|trt|nitro|cultivar|main|factor_a|spacing|variety", c, re.I)]
    cat_a = [c for c in rest if is_cat[c]]
    a = (pref_a or cat_a or rest or cols)[0]
    rest_b = [c for c in rest if c != a]
    pref_b = [c for c in rest_b if re.search(r"sub|factor_b|potassium|sowing|^b$", c, re.I)]
    cat_b = [c for c in rest_b if is_cat[c]]
    b = (pref_b or cat_b or rest_b or [None])[0]
    return {"rep": rep, "a": a, "b": b}


def dedupe_columns(cols):
    seen, out = {}, []
    for c in map(lambda x: str(x).strip(), cols):
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def read_csv_robust(upload):
    raw = upload.getvalue()
    last = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
        except Exception as ex:
            last = ex
    raise last


# ───────────────────────── Cached Pipeline Functions ─────────────────────────
@st.cache_data(show_spinner=False)
def run_model(df, design, cols, y, tf, avg, sstype, labels):
    d, info = prepare(df, design, cols, y, tf, avg)
    fit = fit_model(d, design, labels, sstype)
    return d, info, fit


@st.cache_data(show_spinner=False)
def run_batch(df, design, cols, traits, tf, avg, sstype, labels, alpha):
    rows, skipped = [], []
    for trait in traits:
        try:
            d, _ = prepare(df, design, cols, trait, tf, avg)
            fit = fit_model(d, design, labels, sstype)
            gm = float(d["Y"].mean())
            row = {"Trait": trait, "Grand Mean": round(gm, 3)}
            for _, r in fit["tbl"].iterrows():
                if pd.notna(r["F"]):
                    row[f"F: {r['Source']}"] = f"{r['F']:.2f} {sig_stars(r['P'], alpha)}"
            row["CV %"] = round(float(np.sqrt(fit["mse"]) / abs(gm) * 100), 2) if gm != 0 else np.nan
            row["R²"] = round(float(fit["r2"]), 3)
            rows.append(row)
        except AnalysisError as ae:
            skipped.append((trait, ae.en, ae.bn))
        except Exception as ex:
            skipped.append((trait, str(ex), str(ex)))
    return rows, skipped


# ───────────────────────── High-Resolution Scientific Plots ─────────────────────────
PALETTES = {
    "Academic Slate": ["#475569", "#334155", "#64748b", "#94a3b8"],
    "Steel Blue": ["#3b82f6", "#1d4ed8", "#60a5fa", "#93c5fd"],
    "Forest Emerald": ["#059669", "#047857", "#10b981", "#6ee7b7"],
    "Okabe-Ito (Colorblind-Safe)": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"],
}
SIZE_PRESETS = {"single": (3.35, 3.2), "double": (6.69, 4.2)}


def render_publication_plot(fam, d, response_name, palette_name, tf, show_letters=True, err_kind="SE",
                            color_mode="single", show_points=False, size="auto"):
    tbl = fam["table"]
    names = tbl[fam["label"]].tolist()
    means = tbl["LS-Mean"].to_numpy(float)
    groups = tbl["Group"].tolist()
    if err_kind == "SE":
        err = tbl["SEm ±"].to_numpy(float)
    elif err_kind == "SD":
        err = tbl["SD"].to_numpy(float)
    else:
        err = np.zeros(len(names))
    err = np.nan_to_num(err)

    x = np.arange(len(names))
    colors = PALETTES[palette_name]
    bar_colors = [colors[0]] * len(names) if color_mode == "single" else [colors[i % len(colors)] for i in range(len(names))]

    if size in SIZE_PRESETS:
        figsize = SIZE_PRESETS[size]
        fs_t, fs_l = 7.5, 8.5
    else:
        figsize = (max(5.5, 0.95 * len(names) + 1.8), 4.6)
        fs_t, fs_l = 9.5, 10.5

    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    ax.bar(x, means, yerr=err if err_kind != "None" else None, capsize=4, color=bar_colors, edgecolor="#0f172a",
           linewidth=1.0, alpha=0.9, width=0.55, error_kw={"elinewidth": 1.0, "ecolor": "#0f172a"})

    if show_points:
        key = d[list(fam["family"])].agg(" × ".join, axis=1) if len(fam["family"]) > 1 else d[fam["family"][0]]
        rng = np.random.default_rng(7)
        for xi, nm in zip(x, names):
            vals = d.loc[key == nm, "Y"].to_numpy(float)
            ax.scatter(xi + rng.uniform(-0.12, 0.12, len(vals)), vals, s=14, color="white",
                       edgecolor="#0f172a", linewidth=0.8, zorder=3)
        pts_hi, pts_lo = d["Y"].max(), d["Y"].min()
    else:
        pts_hi, pts_lo = -np.inf, np.inf

    hi = max((means + err).max(), pts_hi, 0.0)
    lo = min((means - err).min(), pts_lo, 0.0)
    span = (hi - lo) or 1.0

    if show_letters:
        for xi, mu, e, g in zip(x, means, err, groups):
            if g and g != "—":
                ytxt = max(mu + e, 0) + span * 0.03
                if show_points:
                    ytxt = max(ytxt, pts_hi + span * 0.02)
                ax.text(xi, ytxt, g, ha="center", va="bottom", fontsize=fs_l, fontweight="bold")

    ax.set_ylim(lo - (span * 0.05 if lo < 0 else 0), hi + span * 0.18)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=fs_t, fontweight="bold")
    ylab = f"{response_name} (LS-mean" + (f" ± {err_kind})" if err_kind != "None" else ")")
    if tf != "None":
        ylab += f" [{tf}]"
    ax.set_ylabel(ylab, fontsize=fs_l, fontweight="bold")
    ax.set_xlabel(fam["label"], fontsize=fs_l, fontweight="bold")

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#94a3b8")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def render_interaction_plot(fam, levels, a_lbl, b_lbl, response_name, palette_name, tf):
    colors = PALETTES[palette_name]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    mu = dict(zip(fam["combos"], fam["mu"]))
    xs = np.arange(len(levels["A"]))
    fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * len(xs) + 2.5), 4.4), dpi=120)
    for i, b in enumerate(levels["B"]):
        ys = np.array([mu[(a, b)] for a in levels["A"]], float)
        ax.errorbar(xs, ys, yerr=fam["se"], marker=markers[i % len(markers)], color=colors[i % len(colors)],
                    capsize=3, linewidth=1.6, markersize=6, label=b)
    ax.set_xticks(xs)
    ax.set_xticklabels(levels["A"], rotation=22, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_xlabel(a_lbl, fontsize=10.5, fontweight="bold")
    ylab = f"{response_name} (LS-mean ± SE)" + (f" [{tf}]" if tf != "None" else "")
    ax.set_ylabel(ylab, fontsize=10.5, fontweight="bold")
    ax.legend(title=b_lbl, frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#94a3b8")
    fig.tight_layout()
    return fig


def fig_bytes(fig, fmt, **kw):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight", **kw)
    return buf.getvalue()


def safe_name(s):
    return re.sub(r"[^\w\-]+", "_", str(s)).strip("_") or "figure"


def safe_sheet(name, used):
    s = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(name))[:31] or "Sheet"
    base, i = s, 1
    while s.lower() in used:
        suf = f"_{i}"
        s = base[:31 - len(suf)] + suf
        i += 1
    used.add(s.lower())
    return s


def cv_note(cv):
    if pd.isna(cv):
        return ""
    if cv < 10:
        return tr("excellent precision", "চমৎকার নির্ভুলতা")
    if cv < 20:
        return tr("acceptable precision", "গ্রহণযোগ্য নির্ভুলতা")
    if cv < 30:
        return tr("high variability", "উচ্চ পরিবর্তনশীলতা")
    return tr("very high variability", "অত্যধিক পরিবর্তনশীলতা")


# ───────────────────────── Executive Header ─────────────────────────
st.markdown(f"""
<div style="padding: 4px 0 16px 0; border-bottom: 1px solid rgba(128,128,128,0.25); margin-bottom: 20px;">
    <span style="font-size: 0.72rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; 
                 background: rgba(14,165,233,0.14); color: #0284c7; padding: 4px 10px; border-radius: 20px;">
        {tr("ENTERPRISE BIOMETRIC SUITE", "বায়োমেট্রিক রিসার্চ প্ল্যাটফর্ম")}
    </span>
    <h1 style="font-size: 2.2rem; font-weight: 800; margin: 8px 0 2px 0; letter-spacing: -0.5px;">
        AgriStat <span style="color: #0284c7;">Studio</span>
    </h1>
    <p style="font-size: 0.95rem; opacity: 0.8; margin: 0;">
        {tr("Design-Aware General Linear Model • Post-Hoc Groupings • Genetic Parameters • High-Precision Agronomic Analytics", 
            "ডিজাইন-সচেতন লিনিয়ার মডেলিং • পোস্ট-হক লেটার গ্রুপিং • জেনেটিক প্যারামিটার • পাবলিকেশন-রেডি বায়োমেট্রিক্স")}
    </p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────── Sidebar: Setup & Data ─────────────────────────
st.sidebar.markdown(f"### ⚙️ {tr('1. Experimental Setup', '১. পরীক্ষণ নকশা')}")
design_key = sb_select("Design Architecture", "নকশা নির্বাচন", list(DESIGNS.keys()), key="design_pick",
                       format_func=lambda k: DESIGNS[k]["label"])
spec = DESIGNS[design_key]

st.sidebar.markdown(f"### 📁 {tr('2. Data Intake', '২. ডেটা সোর্স')}")
SRC_LABELS = {"grid": ("Interactive Grid", "ইন্টারেক্টিভ গ্রিড"), "upload": ("File Upload", "ফাইল আপলোড"),
              "sample": ("Standard Benchmark Sample", "নমুনা ডেটাসেট")}
source_mode = sb_radio("Data Input Mode", "ইনপুট মোড", list(SRC_LABELS.keys()), key="src_mode_pick",
                       format_func=lambda m: tr(*SRC_LABELS[m]))

raw_df = None
if source_mode == "grid":
    blank = st.sidebar.checkbox(tr("Start from blank template", "ফাঁকা টেমপ্লেট দিয়ে শুরু করুন"), value=False, key="blank_tpl_pick")
    st.info(tr("💡 Directly edit cells in the spreadsheet below or paste directly from Excel.",
               "💡 নিচের গ্রিডের সেলে ক্লিক করে ডেটা এডিট করুন বা এক্সেল থেকে কপি-পেস্ট করুন।"))
    base_data = blank_template(design_key) if blank else generate_sample_dataset(design_key)
    raw_df = st.data_editor(base_data, num_rows="dynamic", key=f"grid_{design_key}_{int(blank)}", **_wide_kw(st.data_editor))

elif source_mode == "upload":
    up = st.sidebar.file_uploader(tr("Upload Spreadsheet (.csv, .xlsx)", "স্প্রেডশীট আপলোড"), type=["csv", "xlsx"])
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                raw_df = read_csv_robust(up)
            else:
                xls = pd.ExcelFile(up)
                sheet = xls.sheet_names[0]
                if len(xls.sheet_names) > 1:
                    sheet = sb_select("Worksheet", "ওয়ার্কশীট", xls.sheet_names, key="sheet_pick_opt")
                raw_df = xls.parse(sheet)
            raw_df.columns = dedupe_columns(raw_df.columns)
            with st.expander(tr("Preview (first 10 rows)", "প্রিভিউ (প্রথম ১০ সারি)"), expanded=False):
                show_df(raw_df.head(10))
        except Exception as ex:
            st.error(tr("File parsing error: ", "ফাইল পড়তে সমস্যা হয়েছে: ") + str(ex))
    else:
        st.info(tr("Awaiting dataset upload to commence analysis.", "বিশ্লেষণ শুরু করতে ফাইল আপলোড করুন।"))
else:
    raw_df = generate_sample_dataset(design_key)
    with st.expander(tr("Inspect Benchmark Sample", "নমুনা ডেটাসেট প্রিভিউ"), expanded=False):
        show_df(raw_df)

if raw_df is None:
    st.stop()
raw_df = raw_df.dropna(how="all").reset_index(drop=True)
if raw_df.empty:
    st.info(tr("No data available yet.", "কোনো ডেটা পাওয়া যায়নি।"))
    st.stop()
raw_df.columns = [str(c) for c in raw_df.columns]

# ───────────────────────── Sidebar: Variable Mapping ─────────────────────────
st.sidebar.markdown(f"### 🎯 {tr('3. Variable Mapping', '৩. ভ্যারিয়েবল ম্যাপিং')}")
cols_list = list(raw_df.columns)
guess_map = guess_columns(raw_df, spec)

col_map = {}
if spec["needs_rep"]:
    col_map["rep"] = sb_select("Block / Replication (Rep)", "ব্লক / রেপ্লিকেশন", cols_list,
                               key=f"map_rep_{design_key}", index=cols_list.index(guess_map["rep"]))

avail_for_a = [c for c in cols_list if c != col_map.get("rep")]
col_map["a"] = sb_select("Treatment A (Main-Plot)" if spec["two_factor"] else "Treatment Factor",
                         "ট্রিটমেন্ট ফ্যাক্টর (বা মেইন-প্লট)", avail_for_a, key=f"map_a_{design_key}",
                         index=avail_for_a.index(guess_map["a"]) if guess_map["a"] in avail_for_a else 0)

if spec["two_factor"]:
    avail_for_b = [c for c in cols_list if c not in (col_map.get("rep"), col_map["a"])]
    if not avail_for_b:
        st.error(tr("Not enough columns for Factor B.", "ফ্যাক্টর B এর জন্য পর্যাপ্ত কলাম নেই।"))
        st.stop()
    col_map["b"] = sb_select("Factor B (Sub-Plot)", "ফ্যাক্টর B (সাব-প্লট)", avail_for_b, key=f"map_b_{design_key}",
                             index=avail_for_b.index(guess_map["b"]) if guess_map["b"] in avail_for_b else 0)

# ── Intelligent Response Variable Filtering (Prevents Picking Replication/Block as Y) ──
all_numeric = [c for c in cols_list if c not in col_map.values() and is_numeric_like(raw_df[c])]

# Exclude structural columns from being candidate Y unless nothing else exists
clean_numeric = [c for c in all_numeric if not is_design_metadata_col(c)]
numeric_candidates = clean_numeric if clean_numeric else all_numeric

if not numeric_candidates:
    st.error(tr("No continuous response variable detected in the dataset.", "কোনো সংখ্যাভিত্তিক রেসপন্স ভ্যারিয়েবল পাওয়া যায়নি।"))
    st.stop()

# Prioritize agronomic trait keywords for default response variable
trait_keywords = r"yield|fruit|brix|grain|height|biomass|weight|length|count|pod|prod|area|chlorophyll|tiller"
pref_y = [c for c in numeric_candidates if re.search(trait_keywords, str(c), re.I)]
default_y_idx = numeric_candidates.index(pref_y[0]) if pref_y else 0

response_var = sb_select("Response Metric (Y)", "রেসপন্স ভ্যারিয়েবল (Y)", numeric_candidates, key="resp_pick", index=default_y_idx)

# ───────────────────────── Sidebar: Parameters ─────────────────────────
st.sidebar.markdown(f"### ⚙️ {tr('4. Model Parameters', '৪. মডেল ও টেস্ট সেটিংস')}")
comp_method = sb_select("Post-Hoc Mean Separation", "গড় পৃথকীকরণ পরীক্ষা", METHODS, key="method_pick")
alpha_level = sb_select("Significance Level (α)", "তাৎপর্য স্তর (α)", [0.05, 0.01, 0.001], key="alpha_pick")
ss_type = sb_select("Sum of Squares Type", "সাম অব স্কয়ার্স টাইপ", ["III", "I"], key="sstype_pick",
                    format_func=lambda s: f"Type {s} ({'Partial / Orthogonal' if s == 'III' else 'Sequential'})")
data_transform = sb_select("Mathematical Transformation", "ট্রান্সফরমেশন", list(TRANSFORMS.keys()), key="tf_pick",
                           help=tr("Use sqrt(x+0.5) for counts, log10(x+1) for zeros, Rank Transform for non-parametric ANOVA.",
                                   "গণনার ডেটায় sqrt(x+0.5), শূন্যযুক্ত ডেটায় log10(x+1), নন-প্যারামেট্রিকের জন্য র‍্যাঙ্ক ট্রান্সফর্ম।"))
auto_avg = st.sidebar.checkbox(tr("Average subsamples / duplicate cells", "সাব-স্যাম্পল গড় করুন"), value=False,
                               key="avg_dup_opt", disabled=(design_key == "CRD"),
                               help=tr("Available for blocked designs only.", "শুধুমাত্র ব্লকড ডিজাইনের জন্য প্রযোজ্য।"))
if design_key == "CRD":
    auto_avg = False

# Developer Profile
st.sidebar.markdown("---")
with st.sidebar.expander(f"👨‍💻 {tr('Scientific Developer', 'ডেভেলপার পরিচিতি')}", expanded=False):
    st.markdown(f"""
    **Md. Akib Rubayiet Tahmid**  
    *{tr('Biometrics & Agricultural Research Specialist', 'বায়োমেট্রিক্স ও এগ্রিকালচারাল রিসার্চ স্পেশালিস্ট')}*  

    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/akib-rubayiet-tahmid)
    """)

# ───────────────────────── Pipeline Execution ─────────────────────────
label_dict = {"REP": col_map.get("rep", "Block"), "A": col_map["a"], "B": col_map.get("b", "Factor_B")}

try:
    clean_d, d_info, model_fit = run_model(raw_df, design_key, col_map, response_var, data_transform,
                                           auto_avg, ss_type, label_dict)
except AnalysisError as ae:
    st.error(ae.bn if BN else ae.en)
    st.stop()
except Exception as e:
    st.error(tr("Computation engine error: ", "গণনা ইঞ্জিনে ত্রুটি: ") + str(e))
    st.stop()

# Data Integrity Expander
with st.expander(tr("🔎 Data Integrity & Layout Balance Diagnostics", "🔎 ডেটা অখণ্ডতা ও ফিল্ড ব্যালান্স রিপোর্ট"),
                 expanded=not d_info["balanced"] or d_info["n_dropped"] > 0):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Total Rows", "মোট সারি"), d_info["n_raw"])
    c2.metric(tr("Dropped Rows", "বাদ পড়া সারি"), d_info["n_dropped"])
    c3.metric(tr("Analyzed Observations", "ব্যবহৃত সারি"), d_info["n_used"])
    c4.metric(tr("Residual Error DF", "এরর DF"), model_fit["dfe"])

    if design_key == "CRD":
        show_df(clean_d.groupby("A").size().rename(tr("Replications", "রেপ্লিকেশন")).reset_index().rename(columns={"A": label_dict["A"]}), hide_index=True)
    elif design_key == "RCBD":
        show_df(pd.crosstab(clean_d["A"], clean_d["REP"]).rename_axis(label_dict["A"]).reset_index(), hide_index=True)
    else:
        show_df(pd.crosstab(clean_d["A"], clean_d["B"]).rename_axis(label_dict["A"]).reset_index(), hide_index=True)

    if not d_info["balanced"]:
        st.warning(tr(
            f"Unbalanced design detected ({d_info['n_missing']} missing, {d_info['n_dupes']} duplicate cells). "
            "Type III SS and LS-means are applied; pairwise tests utilize harmonic-mean replication approximation.",
            f"নকশায় ভারসাম্যহীনতা রয়েছে ({d_info['n_missing']}টি মিসিং, {d_info['n_dupes']}টি ডুপ্লিকেট ঘর)। "
            "Type III SS ও হারমোনিক-মিন অ্যাপ্রক্সিমেশন ব্যবহৃত হয়েছে।"))

# Mean Families
A_lbl, B_lbl = label_dict["A"], label_dict.get("B", "B")
if design_key in ("CRD", "RCBD"):
    fam_specs = [(("A",), "RES", A_lbl, True)]
elif design_key == "FACT":
    fam_specs = [(("A",), "RES", A_lbl, True), (("B",), "RES", B_lbl, True), (("A", "B"), "RES", f"{A_lbl} × {B_lbl}", True)]
else:  # SPLIT
    fam_specs = [(("A",), SPLIT_ERR_A, A_lbl, True), (("B",), "RES", B_lbl, True), (("A", "B"), "RES", f"{A_lbl} × {B_lbl}", False)]

# Dunnett control selector
ctrl_index_val = 0
if comp_method == "Dunnett (vs Control)":
    lvl_a = list(dict.fromkeys(clean_d["A"]))
    ctrl_pick = st.sidebar.selectbox(tr("Control Group Level (T₀)", "কন্ট্রোল গ্রুপ লেভেল (T₀)"), lvl_a, key="dunnett_ctrl_pick")
    ctrl_index_val = lvl_a.index(ctrl_pick)

families = []
for fam, ek, lab, comp in fam_specs:
    try:
        families.append(mean_family(clean_d, model_fit, fam, ek, lab, comp_method, alpha_level, data_transform,
                                    ctrl_idx=ctrl_index_val, comparable=comp))
    except Exception as ex:
        st.warning(f"{lab}: {ex}")

sh_res, lev_res, kw_res = assumption_tests(clean_d, model_fit, design_key)
stud_resid = studentized_residuals(model_fit)
grand_mean = float(clean_d["Y"].mean())
mse_main = model_fit["mse"]
cv_main = (np.sqrt(mse_main) / abs(grand_mean)) * 100 if grand_mean != 0 else np.nan

# ───────────────────────── KPI Performance Bar ─────────────────────────
kpi = st.columns(5)
scale_txt = tr("transformed scale", "ট্রান্সফর্ম করা স্কেল") if data_transform != "None" else tr("original scale", "মূল স্কেল")
kpi[0].metric(tr("Grand Mean (Ȳ)", "সামগ্রিক গড়"), f"{grand_mean:.2f}")
kpi[1].metric(tr("CV % (Error b)", "সিভি %") if design_key == "SPLIT" else tr("CV %", "সিভি %"),
              "—" if pd.isna(cv_main) else f"{cv_main:.2f}%",
              help=tr(f"CV = √MSE / Ȳ × 100 on {scale_txt}.", f"CV = √MSE / সামগ্রিক গড় × ১০০ ({scale_txt})।"))
kpi[2].metric(tr("Model R²", "মডেল R²"), f"{model_fit['r2']:.3f}", help=tr("Coefficient of Determination", "মডেল ব্যাখ্যা করার ক্ষমতা"))
kpi[3].metric("SEm ±", f"{families[0]['se']:.3f}", help=tr("Standard error of treatment mean", "গড়ের স্ট্যান্ডার্ড এরর"))
kpi[4].metric(f"CD / LSD ({int((1-alpha_level)*100)}%)", f"{families[0]['cd']:.3f}" if not np.isnan(families[0]['cd']) else "—",
              help=tr("Critical difference threshold", "গড় পৃথকীকরণের ন্যূনতম ব্যবধান"))

# ───────────────────────── Analytical Workspace Tabs ─────────────────────────
tab_anova, tab_means, tab_plots, tab_diag, tab_genetic, tab_batch, tab_rep = st.tabs([
    tr("📊 ANOVA Partitioning", "📊 অ্যানোভা টেবিল"),
    tr("📑 Mean Separation & Letters", "📑 গড় পৃথকীকরণ ও গ্রুপিং"),
    tr("📈 Publication Graphics", "📈 পাবলিকেশন গ্রাফ"),
    tr("🩺 Residual Diagnostics", "🩺 মডেল ডায়াগনস্টিকস"),
    tr("🧬 Genetic Parameters", "🧬 জেনেটিক প্যারামিটার"),
    tr("🧮 Multi-Trait Batch", "🧮 মাল্টি-ট্রেইট সারাংশ"),
    tr("📋 Thesis Methods & Synthesis", "📋 রিপোর্ট ও গবেষণার মেথডস"),
])

# ── 1. ANOVA Tab ──
with tab_anova:
    st.markdown(f"#### {tr('Analysis of Variance (General Linear Model)', 'অ্যানালাইসিস অব ভ্যারিয়েন্স টেবিল')}")
    show_df(display_anova(model_fit["tbl"], alpha_level), hide_index=True)
    st.caption(tr(f"ns = p ≥ {alpha_level}; * p < 0.05, ** p < 0.01, *** p < 0.001.",
                  f"ns = p ≥ {alpha_level}; * p < 0.05, ** p < 0.01, *** p < 0.001।"))

    if design_key == "SPLIT":
        st.info(tr("ℹ️ Split-plot EMS: Main-plot factors tested against Error (a); Sub-plot factors against Error (b).",
                   "ℹ️ স্প্লিট-প্লট: মেইন-প্লটের F-অনুপাত Error (a) এবং সাব-প্লটের F-অনুপাত Error (b) এর প্রেক্ষিতে পরীক্ষিত।"))

    # Auto-Generated Narrative Statement
    st.markdown("---")
    st.markdown(f"**📝 {tr('Auto-Generated Manuscript Statement:', 'স্বয়ংক্রিয় গবেষণার ফলাফল অনুচ্ছেদ:')}**")
    trt_row = model_fit["tbl"].loc[model_fit["tbl"]["key"] == ("A",)].iloc[0]
    f_stat, p_stat, df_trt = trt_row["F"], trt_row["P"], trt_row["DF"]
    dfe_used = model_fit["tbl"].loc[model_fit["tbl"]["key"] == "RES", "DF"].iloc[0]

    best_mean_row = families[0]["table"].sort_values("LS-Mean", ascending=False).iloc[0]
    worst_mean_row = families[0]["table"].sort_values("LS-Mean", ascending=True).iloc[0]

    statement_en = (
        f"The analysis of variance indicated that the effect of {label_dict['A']} on {response_var} was "
        f"{'statistically significant' if p_stat < alpha_level else 'not statistically significant'} "
        f"(F({int(df_trt)}, {int(dfe_used)}) = {f_stat:.2f}, {p_text(p_stat)}, Partial η² = {trt_row['EtaP']:.3f}). "
        f"The maximum value was attained by '{best_mean_row[label_dict['A']]}' ({best_mean_row['LS-Mean']:.2f} ± {best_mean_row['SEm ±']:.2f} {best_mean_row['Group']}), "
        f"whereas the lowest was recorded in '{worst_mean_row[label_dict['A']]}' ({worst_mean_row['LS-Mean']:.2f} ± {worst_mean_row['SEm ±']:.2f} {worst_mean_row['Group']}). "
        f"The experimental coefficient of variation was {cv_main:.2f}%."
    )
    statement_bn = (
        f"অ্যানালাইসিস অব ভ্যারিয়েন্স অনুযায়ী {response_var}-এর ওপর {label_dict['A']}-এর প্রভাব "
        f"{'পরিসংখ্যানগতভাবে তাৎপর্যপূর্ণ' if p_stat < alpha_level else 'তাৎপর্যপূর্ণ নয়'} "
        f"(F({int(df_trt)}, {int(dfe_used)}) = {f_stat:.2f}, {p_text(p_stat)}, আংশিক η² = {trt_row['EtaP']:.3f})। "
        f"সর্বোচ্চ মান পাওয়া গেছে '{best_mean_row[label_dict['A']]}' এ ({best_mean_row['LS-Mean']:.2f} ± {best_mean_row['SEm ±']:.2f} {best_mean_row['Group']}), "
        f"এবং সর্বনিম্ন মান '{worst_mean_row[label_dict['A']]}' এ ({worst_mean_row['LS-Mean']:.2f} ± {worst_mean_row['SEm ±']:.2f} {worst_mean_row['Group']})। "
        f"পরীক্ষণের সিভি (CV) ছিল {cv_main:.2f}%।"
    )
    st.info(statement_bn if BN else statement_en)

    # Orthogonal Polynomial Contrasts
    poly_df = compute_polynomial_contrasts(families[0])
    if poly_df is not None:
        st.markdown(f"**📈 {tr('Orthogonal Polynomial Trend Contrasts (Dose-Response):', 'অর্থোগোনাল পলিনোমিয়াল ট্রেন্ড অ্যানালাইসিস:')}**")
        show_df(poly_df, hide_index=True)

    # Simple-Effects Slicing
    if spec["two_factor"]:
        int_row = model_fit["tbl"].loc[model_fit["tbl"]["key"] == ("A", "B")]
        if not int_row.empty and int_row["P"].values[0] < alpha_level:
            st.markdown(f"**🔬 {tr('Interaction Slicing (Simple Effects of B within A):', 'ইন্টারঅ্যাকশন স্লাইসিং (ফ্যাক্টরের অভ্যন্তরীণ প্রভাব):')}**")
            slice_df = compute_simple_effects(clean_d, "A", "B", "Y", alpha_level)
            show_df(slice_df, hide_index=True)

# ── 2. Mean Separation Tab ──
with tab_means:
    if not families:
        st.info(tr("Mean separation unavailable.", "গড় পৃথকীকরণ সম্ভব নয়।"))
    else:
        st.markdown(f"#### {tr(f'Least-Squares Means via {comp_method} (α = {alpha_level})', f'গড় এবং {comp_method} গ্রুপিং')}")
        fi = st.selectbox(tr("Select Factor Family", "ফ্যাক্টর নির্বাচন"), range(len(families)),
                          format_func=lambda i: families[i]["label"], key=f"fam_sel_{len(families)}")
        fm = families[fi]
        show_df(fm["table"], hide_index=True)
        st.caption(tr("Means sharing the same letter do not differ significantly.",
                      "একই অক্ষরযুক্ত গড়গুলোর মধ্যে পরিসংখ্যানগত কোনো তাৎপর্যপূর্ণ পার্থক্য নেই।"))

        if fm["comparable"]:
            st.caption(f"**SEm ±** = {fm['se']:.3f} | **CD / LSD** = {fm['cd']:.3f} | " +
                       tr(f"Error Term: {fm['err_name']} (df = {fm['df_e']})", f"এরর টার্ম: {fm['err_name']} (df = {fm['df_e']})"))

        if fm["pair"] is not None:
            with st.expander(tr("Pairwise Contrasts & Adjusted P-values", "জোড়ায় জোড়ায় তুলনা ও পি-মান")):
                show_df(fm["pair"], hide_index=True)

        inter_fam = next((f for f in families if f["family"] == ("A", "B")), None)
        if spec["two_factor"] and inter_fam is not None:
            with st.expander(tr("Two-Way Pivot Table (Mean ± SE with Letter Grouping)", "দ্বিমুখী পিভট টেবিল (গড় ± SE ও লেটারিং)")):
                pv = build_pivot(inter_fam, model_fit["levels"]).rename_axis(A_lbl).reset_index()
                show_df(pv, hide_index=True)

# ── 3. Publication Graphics Tab ──
with tab_plots:
    if not families:
        st.info(tr("No means available to plot.", "প্লট করার মতো গড় নেই।"))
    else:
        p1, p2, p3 = st.columns(3)
        pi = p1.selectbox(tr("Plot Factor", "গ্রাফের ফ্যাক্টর"), range(len(families)),
                          format_func=lambda i: families[i]["label"], key=f"plt_fam_{len(families)}")
        pal = p2.selectbox(tr("Color Scheme", "কালার প্যালেট"), list(PALETTES.keys()), key="plt_pal_pick")
        err_kind = p3.selectbox(tr("Error Bar Metric", "এরর বার"), ["SE", "SD", "None"], key="plt_err_pick")

        p4, p5, p6 = st.columns(3)
        color_mode = p4.selectbox(tr("Bar Colors", "বার রঙ"), ["single", "group"], key="plt_cm_pick",
                                  format_func=lambda m: tr("Single Color", "একক রঙ") if m == "single" else tr("Color by Group", "গ্রুপ অনুযায়ী রঙ"))
        size_preset = p5.selectbox(tr("Journal Dimension", "জার্নাল মাপ"), ["auto", "single", "double"], key="plt_size_pick",
                                   format_func=lambda m: {"auto": tr("Auto", "অটো"), "single": tr("Single Column (85 mm)", "একক কলাম (৮৫ মিমি)"),
                                                          "double": tr("Double Column (170 mm)", "দ্বৈত কলাম (১৭০ মিমি)")}[m])
        show_letters = p6.checkbox(tr("Annotate Letter Groups", "অক্ষর গ্রুপিং দেখান"), value=True, key="plt_letters_pick")
        show_pts = st.checkbox(tr("Overlay Individual Observations", "প্রতিটি পর্যবেক্ষণ দেখান"), value=False, key="plt_pts_pick")

        fig = render_publication_plot(families[pi], clean_d, response_var, pal, data_transform, show_letters,
                                      err_kind, color_mode, show_pts, size_preset)
        st.pyplot(fig)

        inter_fam = next((f for f in families if f["family"] == ("A", "B")), None)
        fig2 = None
        if spec["two_factor"] and inter_fam is not None:
            st.markdown(f"##### {tr('Two-Factor Interaction Line Plot', 'ইন্টারঅ্যাকশন লাইন প্লট')}")
            fig2 = render_interaction_plot(inter_fam, model_fit["levels"], A_lbl, B_lbl, response_var, pal, data_transform)
            st.pyplot(fig2)

        # On-demand exports
        if st.checkbox(tr("Generate High-Resolution Exports (300 DPI)", "হাই-রেজোলিউশন এক্সপোর্ট ফাইল তৈরি করুন (৩০০ DPI)"),
                       value=False, key="plt_export_chk"):
            base_name = safe_name(f"{response_var}_{families[pi]['label']}")
            e1, e2, e3 = st.columns(3)
            e1.download_button("💾 PNG (300 DPI)", fig_bytes(fig, "png"), f"{base_name}.png", "image/png")
            e2.download_button("📄 PDF (Vector)", fig_bytes(fig, "pdf"), f"{base_name}.pdf", "application/pdf")
            try:
                e3.download_button("🖼️ TIFF (LZW)", fig_bytes(fig, "tiff", pil_kwargs={"compression": "tiff_lzw"}),
                                   f"{base_name}.tiff", "image/tiff")
            except Exception:
                pass
            if fig2 is not None:
                st.download_button("💾 " + tr("Interaction Plot PNG", "ইন্টারঅ্যাকশন প্লট PNG"), fig_bytes(fig2, "png"),
                                   f"{safe_name(response_var)}_interaction.png", "image/png")
        plt.close(fig)
        if fig2 is not None:
            plt.close(fig2)

# ── 4. Diagnostics Tab ──
with tab_diag:
    st.markdown(f"#### {tr('Model Assumption & Residual Diagnostics', 'মডেল এজাম্পশন ও নর্মালিটি টেস্ট')}")
    sh_p, lev_p = sh_res[1], lev_res[1]
    k1, k2 = st.columns(2)

    k1.metric("Shapiro-Wilk (Residual Normality)", "—" if pd.isna(sh_p) else f"{sh_p:.4f}",
              delta="Normal" if sh_p > alpha_level else "Non-normal Violation",
              delta_color="normal" if sh_p > alpha_level else "inverse")
    k2.metric("Levene Test (Variance Homogeneity)", "—" if pd.isna(lev_p) else f"{lev_p:.4f}",
              delta="Homoscedastic" if lev_p > alpha_level else "Heteroscedastic Violation",
              delta_color="normal" if lev_p > alpha_level else "inverse")

    if not pd.isna(kw_res[1]):
        st.caption(tr(f"Non-parametric check – Kruskal–Wallis on {A_lbl}: H = {kw_res[0]:.3f}, {p_text(kw_res[1])}.",
                      f"নন-প্যারামেট্রিক যাচাই – {A_lbl} এর উপর Kruskal–Wallis: H = {kw_res[0]:.3f}, {p_text(kw_res[1])}।"))

    fig_d, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4), dpi=110)
    ax1.scatter(model_fit["fitted"], model_fit["resid"], color="#0284c7", edgecolor="#0f172a", alpha=0.8)
    ax1.axhline(0, color="#ef4444", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("Fitted Values", fontweight="bold")
    ax1.set_ylabel("Residuals", fontweight="bold")
    ax1.set_title("Residuals vs Fitted", fontweight="bold")

    (osm, osr), (slope, intercept, _r) = stats.probplot(model_fit["resid"], dist="norm")
    ax2.scatter(osm, osr, color="#0284c7", edgecolor="#0f172a", alpha=0.8)
    ax2.plot(osm, slope * osm + intercept, color="#ef4444", linestyle="--")
    ax2.set_xlabel("Theoretical Quantiles", fontweight="bold")
    ax2.set_ylabel("Sample Quantiles", fontweight="bold")
    ax2.set_title("Normal Q-Q Plot", fontweight="bold")

    for a_ in (ax1, ax2):
        a_.spines["top"].set_visible(False)
        a_.spines["right"].set_visible(False)
    fig_d.tight_layout()
    st.pyplot(fig_d)
    plt.close(fig_d)

    out_mask = np.abs(np.nan_to_num(stud_resid)) > 3.0
    if np.any(out_mask):
        st.warning(tr(f"⚠️ {int(out_mask.sum())} observation(s) exhibit |Studentized Residual| > 3 (potential outliers).",
                      f"⚠️ {int(out_mask.sum())}টি সম্ভাব্য আউটলায়ার পাওয়া গেছে (|r| > 3.0)।"))
        show_df(clean_d[out_mask])

# ── 5. Genetic Parameters Tab ──
with tab_genetic:
    st.markdown(f"#### {tr('Biometrical Genetics & Heritability Estimates', 'বায়োমেট্রিকাল জেনেটিক্স ও বংশগতি পরিমাপ')}")
    if design_key != "RCBD":
        st.info(tr("Genetic parameters (GCV, PCV, Heritability) are computed specifically for RCBD varietal/germplasm trials.",
                   "জেনেটিক প্যারামিটার (GCV, PCV, Heritability) শুধুমাত্র RCBD জাত মূল্যায়ন ট্রায়ালের জন্য প্রযোজ্য।"))
    else:
        gen_df = compute_genetic_parameters(model_fit, clean_d, response_var)
        if gen_df is not None:
            show_df(gen_df, hide_index=True)
            st.caption(tr("Selection intensity k = 2.06 (5% selection). Genetic Advance as % of Mean (GAM): <10% Low, 10-20% Moderate, >20% High (Johnson et al., 1955).",
                          "বাছাই তীব্রতা k = ২.০৬ (৫%)। GAM মান: <১০% কম, ১০-২০% মাঝারি, >২০% উচ্চ (Johnson et al., 1955)।"))

# ── 6. Batch Processing Tab ──
with tab_batch:
    st.markdown(f"#### {tr('Batch Screening Across All Numerical Metrics', 'সব বৈশিষ্ট্যের এককালীন বিশ্লেষণ')}")
    rows_b, skipped_b = run_batch(raw_df, design_key, col_map, tuple(numeric_candidates), data_transform,
                                  auto_avg, ss_type, label_dict, alpha_level)
    if rows_b:
        show_df(pd.DataFrame(rows_b).fillna("—"), hide_index=True)
    for trait, en_msg, bn_msg in skipped_b:
        st.warning(tr(f"Skipped '{trait}': {en_msg}", f"'{trait}' বাদ পড়েছে: {bn_msg}"))

# ── 7. Reporting & Export Tab ──
with tab_rep:
    st.markdown(f"#### {tr('Materials & Methods Statement (Formatted for Thesis/Journal)', 'গবেষণাপত্রের জন্য মেথডলজি ড্রাফট')}")
    lv = model_fit["levels"]
    layout_bits = [f"{len(lv['A'])} levels of {label_dict['A']}"]
    if "B" in lv:
        layout_bits.append(f"{len(lv['B'])} levels of {label_dict['B']}")
    if "REP" in lv:
        layout_bits.append(f"{len(lv['REP'])} replications/blocks")
    err_desc = ("Main-plot effects were tested against Error (a) (Replication × Main-plot factor) and sub-plot effects against Error (b). "
                if design_key == "SPLIT" else "")
    tf_desc = "" if data_transform == "None" else f"Data were subjected to {data_transform} transformation prior to analysis to satisfy model assumptions. "
    avg_desc = "Duplicate subsamples were averaged prior to statistical modeling. " if d_info["averaged"] else ""

    method_text = (
        f"The experiment was conducted under a {DESIGNS[design_key]['label']} with {', '.join(layout_bits)}. "
        f"{tf_desc}{avg_desc}"
        f"Variance decomposition was evaluated via General Linear Model ({'Type III' if ss_type == 'III' else 'Type I'} sums of squares). {err_desc}"
        f"Post-hoc pairwise mean separation was evaluated using {comp_method} at α = {alpha_level}. "
        f"Model residuals were assessed for normality (Shapiro–Wilk) and homogeneity of variance (Levene's median-centered test). "
        f"The coefficient of variation was {cv_main:.2f}%. "
        f"Statistical analysis was executed via AgriStat Studio following biometric procedures outlined by "
        f"Gomez & Gomez (1984) and Steel, Torrie & Dickey (1997)."
    )
    st.code(method_text, language="markdown")

    if st.checkbox(tr("Generate Comprehensive Excel Workbook", "সম্পূর্ণ এক্সেল রিপোর্ট তৈরি করুন"), value=False, key="xl_export_chk"):
        try:
            excel_buf = io.BytesIO()
            used_names = set()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                clean_d.to_excel(writer, sheet_name=safe_sheet("Raw Data", used_names), index=False)
                anova_export(model_fit["tbl"], alpha_level).to_excel(writer, sheet_name=safe_sheet("ANOVA", used_names), index=False)
                for f in families:
                    f["table"].to_excel(writer, sheet_name=safe_sheet(f["label"], used_names), index=False)
                    if f["pair"] is not None:
                        f["pair"].to_excel(writer, sheet_name=safe_sheet("Pairs " + f["label"], used_names), index=False)
                if design_key == "RCBD":
                    gen_res = compute_genetic_parameters(model_fit, clean_d, response_var)
                    if gen_res is not None:
                        gen_res.to_excel(writer, sheet_name=safe_sheet("Genetics", used_names), index=False)
                pd.DataFrame({"Methods": [method_text]}).to_excel(writer, sheet_name=safe_sheet("Methods", used_names), index=False)

            st.download_button(
                label=tr("📊 Download Comprehensive Excel Workbook (.xlsx)", "📊 সম্পূর্ণ এক্সেল রিপোর্ট ডাউনলোড করুন"),
                data=excel_buf.getvalue(),
                file_name=f"{safe_name(response_var)}_{design_key}_AgriStat_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_excel_final"
            )
        except Exception as ex:
            st.error(tr("Could not build workbook: ", "ওয়ার্কবুক তৈরি করা যায়নি: ") + str(ex))

# ───────────────────────── Executive Footer ─────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; opacity: .75; font-size: 0.88rem; padding-bottom: 24px;'>"
    f"AgriStat Studio • {tr('Biometric System Architecture by', 'ডিজাইন ও তত্ত্বাবধানে')} <b>Md. Akib Rubayiet Tahmid</b> | "
    f"<a href='https://www.linkedin.com/in/akib-rubayiet-tahmid' target='_blank' "
    f"style='text-decoration: none; color: #0284c7; font-weight: 600;'>LinkedIn Profile</a></div>",
    unsafe_allow_html=True,
)
