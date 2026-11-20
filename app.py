"""
🌾 AgriStat Pro – RCBD Field Trial Analyzer
RCBD ANOVA | LSD / Tukey / Duncan with letter grouping | Assumption checks |
Transformations | Multi-trait summary | Excel + Word + PNG/PDF export | EN/BN
"""
import io
import itertools
import string
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AgriStat Pro – RCBD Analyzer", page_icon="🌾", layout="wide")

st.markdown("""<style>
@import url('https://fonts.maateen.me/kalpurush/font.css');
html, body, .stApp, p, label, li, td, th, h1, h2, h3, h4 {
    font-family: 'Times New Roman','Kalpurush','Liberation Serif',serif;
}
[data-testid="stMetric"] {background:rgba(74,112,139,.08); border:1px solid rgba(74,112,139,.25);
    border-radius:10px; padding:10px 14px;}
</style>""", unsafe_allow_html=True)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"]})

# ───────────────────────── Language ─────────────────────────
st.sidebar.markdown("### 🌐 Language / ভাষা")
BN = st.sidebar.radio("Language", ("English", "বাংলা"), horizontal=True,
                      label_visibility="collapsed") == "বাংলা"


def tr(en, bn):
    return bn if BN else en


# ───────────────────────── Statistics core ─────────────────────────
TRANSFORMS = {
    "None": lambda y: y,
    "log10": lambda y: np.log10(y),
    "sqrt": lambda y: np.sqrt(y),
    "arcsine (%)": lambda y: np.degrees(np.arcsin(np.sqrt(y / 100))),
}
METHODS = ["Tukey HSD", "LSD", "Duncan (DMRT)"]


def make_crit(method, alpha, mse, dfe, r, k):
    """Critical difference as a function of the number of means spanned (p)."""
    se = np.sqrt(mse / r)
    if method == "LSD":
        c = stats.t.ppf(1 - alpha / 2, dfe) * np.sqrt(2) * se
        return lambda p: c
    if method == "Tukey HSD":
        c = stats.studentized_range.ppf(1 - alpha, k, dfe) * se
        return lambda p: c
    return lambda p: stats.studentized_range.ppf((1 - alpha) ** (p - 1), p, dfe) * se


def letter_groups(means, crit):
    """Compact letter display (a, b, ab …) from means and a critical-difference function."""
    means = np.asarray(means, float)
    order = np.argsort(-means)
    m, k = means[order], len(means)
    groups = set()
    for i in range(k):
        j = i
        while j + 1 < k and m[i] - m[j + 1] < crit(j + 2 - i):
            j += 1
        groups.add((i, j))
    maximal = sorted(g for g in groups
                     if not any(h != g and h[0] <= g[0] and h[1] >= g[1] for h in groups))
    abc = string.ascii_lowercase + string.ascii_uppercase
    lab = [""] * k
    for n, (i, j) in enumerate(maximal):
        for t in range(i, j + 1):
            lab[t] += abc[n % 52]
    out = [""] * k
    for pos, idx in enumerate(order):
        out[idx] = lab[pos]
    return out


def sig_stars(p, alpha=0.05):
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < alpha else "ns"


def analyze(df, trt, rep, y, tf="None", alpha=0.05, method="Tukey HSD"):
    d = df[[trt, rep, y]].copy()
    d.columns = ["TRT", "REP", "Y0"]
    d["Y0"] = pd.to_numeric(d["Y0"], errors="coerce")
    d = d.dropna()
    d["TRT"] = d["TRT"].astype(str).str.strip()
    d["REP"] = d["REP"].astype(str).str.strip()
    if tf == "log10" and (d["Y0"] <= 0).any():
        raise ValueError(tr("log10 needs all values > 0.", "log10 এর জন্য সব মান > 0 হতে হবে।"))
    if tf == "sqrt" and (d["Y0"] < 0).any():
        raise ValueError(tr("sqrt needs all values ≥ 0.", "sqrt এর জন্য সব মান ≥ 0 হতে হবে।"))
    if tf == "arcsine (%)" and ((d["Y0"] < 0) | (d["Y0"] > 100)).any():
        raise ValueError(tr("arcsine needs values between 0 and 100.", "arcsine এর জন্য মান ০–১০০ এর মধ্যে হতে হবে।"))
    d["Y"] = TRANSFORMS[tf](d["Y0"])
    order = list(dict.fromkeys(d["TRT"]))
    k, nrep = len(order), d["REP"].nunique()
    if k < 2 or nrep < 2:
        raise ValueError(tr("Need ≥ 2 treatments and ≥ 2 replications.", "কমপক্ষে ২টি ট্রিটমেন্ট ও ২টি রেপ্লিকেশন লাগবে।"))

    model = ols("Y ~ C(REP) + C(TRT)", data=d).fit()
    a = sm.stats.anova_lm(model, typ=1)
    dfe = a.loc["Residual", "df"]
    if dfe <= 0:
        raise ValueError(tr("No error degrees of freedom – add more data.", "Error DF নেই – আরও ডেটা দিন।"))
    mse = a.loc["Residual", "sum_sq"] / dfe

    tbl = a.rename(index={"C(REP)": "Replication", "C(TRT)": "Treatment", "Residual": "Error"},
                   columns={"sum_sq": "SS", "mean_sq": "MS", "PR(>F)": "P"})
    tbl.loc["Total"] = [tbl["df"].sum(), tbl["SS"].sum(), np.nan, np.nan, np.nan]
    anova = pd.DataFrame({
        "Source": tbl.index, "DF": tbl["df"].astype(int).values,
        "SS": tbl["SS"].round(3).values, "MS": tbl["MS"].round(3).values,
        "F": tbl["F"].round(3).values, "P-value": tbl["P"].round(4).values,
        "Sig.": [sig_stars(p, alpha) for p in tbl["P"]],
    }).fillna("")

    cnt = d.groupby("TRT").size().reindex(order)
    r = k / np.sum(1 / cnt)                                   # harmonic mean replication
    mean_t = d.groupby("TRT")["Y"].mean().reindex(order)
    mean_o = d.groupby("TRT")["Y0"].mean().reindex(order)
    sd = d.groupby("TRT")["Y"].std().reindex(order)
    crit = make_crit(method, alpha, mse, dfe, r, k)
    se_pool = np.sqrt(mse / r)
    letters = letter_groups(mean_t.values, crit)
    means = pd.DataFrame({"Treatment": order, "Mean": mean_t.values.round(3),
                          "SD": sd.values.round(3), "SE": round(se_pool, 3),
                          "Group": letters})
    if tf != "None":
        means.insert(2, "Mean (original)", mean_o.values.round(3))

    pair = None
    if method in ("Tukey HSD", "LSD"):
        rows = []
        for u, v in itertools.combinations(order, 2):
            diff = mean_t[u] - mean_t[v]
            p = (stats.studentized_range.sf(abs(diff) / se_pool, k, dfe) if method == "Tukey HSD"
                 else 2 * stats.t.sf(abs(diff) / (np.sqrt(2) * se_pool), dfe))
            rows.append({"Comparison": f"{u} vs {v}", "Difference": round(diff, 3),
                         "P (adj.)" if method == "Tukey HSD" else "P": round(p, 4),
                         "Significant": "Yes" if p < alpha else "No"})
        pair = pd.DataFrame(rows)

    resid, fitted = model.resid, model.fittedvalues
    sh = stats.shapiro(resid) if len(resid) >= 3 else (np.nan, np.nan)
    try:
        lev = stats.levene(*[g["Y"].values for _, g in d.groupby("TRT") if len(g) > 1])
    except Exception:
        lev = (np.nan, np.nan)

    ms_rep = a.loc["C(REP)", "mean_sq"]
    re_cv = ((nrep - 1) * ms_rep + nrep * (k - 1) * mse) / ((nrep * k - 1) * mse)
    return dict(
        data=d, order=order, k=k, nrep=nrep, anova=anova, means=means, pair=pair,
        mse=mse, dfe=dfe, cv=np.sqrt(mse) / d["Y"].mean() * 100, grand=d["Y"].mean(),
        r2=model.rsquared, se_pool=se_pool, sem=se_pool, cd=crit(2), cd01=make_crit(method, 0.01, mse, dfe, r, k)(2),
        f_trt=a.loc["C(TRT)", "F"], p_trt=a.loc["C(TRT)", "PR(>F)"],
        f_rep=a.loc["C(REP)", "F"], p_rep=a.loc["C(REP)", "PR(>F)"],
        sh=sh, lev=lev, resid=resid, fitted=fitted, re=re_cv, counts=cnt,
        y=y, trt=trt, tf=tf, alpha=alpha, method=method,
    )


def interpret(r):
    sig = r["p_trt"] < r["alpha"]
    m = r["means"].sort_values("Mean", ascending=False)
    best, worst = m.iloc[0], m.iloc[-1]
    prec = ("good", "ভালো") if r["cv"] <= 10 else ("acceptable", "গ্রহণযোগ্য") if r["cv"] <= 15 else ("high (interpret with care)", "বেশি (সতর্কতার সাথে ব্যাখ্যা করুন)")
    if BN:
        s = (f"{r['y']} এর উপর {r['trt']} এর প্রভাব {'পরিসংখ্যানগতভাবে তাৎপর্যপূর্ণ' if sig else 'তাৎপর্যপূর্ণ নয়'} "
             f"(F = {r['f_trt']:.2f}, p = {r['p_trt']:.4f}). ")
        if sig:
            s += f"সর্বোচ্চ গড় ({best['Mean']}) পাওয়া গেছে '{best['Treatment']}' এ এবং সর্বনিম্ন ({worst['Mean']}) '{worst['Treatment']}' এ। "
        s += f"CV = {r['cv']:.2f}% – পরীক্ষণের নির্ভুলতা {prec[1]}।"
    else:
        s = (f"The effect of {r['trt']} on {r['y']} was {'statistically significant' if sig else 'not significant'} "
             f"(F = {r['f_trt']:.2f}, p = {r['p_trt']:.4f}). ")
        if sig:
            s += f"The highest mean ({best['Mean']}) was recorded for '{best['Treatment']}' and the lowest ({worst['Mean']}) for '{worst['Treatment']}'. "
        s += f"CV = {r['cv']:.2f}%, indicating {prec[0]} experimental precision."
    return s


def methods_text(r):
    tf_note = "" if r["tf"] == "None" else " after " + r["tf"] + " transformation"
    return (f"Data were analysed as a randomized complete block design (RCBD) with {r['k']} treatments and "
            f"{r['nrep']} replications using two-way ANOVA{tf_note}. "
            f"Treatment means were compared using {r['method']} at α = {r['alpha']}. Means followed by the same "
            f"letter(s) are not significantly different. Analysis was performed in Python (statsmodels, SciPy).")


# ───────────────────────── Plot ─────────────────────────
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
PALETTES = {"Steel blue": "#4A708B", "Forest green": "#2E7D32", "Earth brown": "#8D6E63",
            "Colorblind-safe (per bar)": None}


def make_plot(r, ylabel, xlabel, palette, show_letters, sort_bars):
    m = r["means"].copy()
    if sort_bars:
        m = m.sort_values("Mean", ascending=False)
    x = np.arange(len(m))
    colors = PALETTES[palette] or [OKABE[i % len(OKABE)] for i in range(len(m))]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(m) + 2), 5), dpi=300)
    ax.bar(x, m["Mean"], yerr=m["SE"], capsize=5, color=colors, edgecolor="black",
           linewidth=1.1, alpha=.9, width=.55, error_kw={"elinewidth": 1.2})
    top = (m["Mean"] + m["SE"]).max()
    if show_letters:
        for xi, mu, se, g in zip(x, m["Mean"], m["SE"], m["Group"]):
            ax.text(xi, mu + se + top * .02, g, ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(m["Treatment"], rotation=20, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylim(min(0, (m["Mean"] - m["SE"]).min() * 1.1), top * 1.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=.3)
    fig.tight_layout()
    return fig


def fig_bytes(fig, fmt):
    b = io.BytesIO()
    fig.savefig(b, format=fmt, dpi=300, bbox_inches="tight")
    return b.getvalue()


# ───────────────────────── Export ─────────────────────────
def to_excel(r, raw):
    b = io.BytesIO()
    with pd.ExcelWriter(b, engine="openpyxl") as w:
        raw.to_excel(w, sheet_name="Data", index=False)
        r["anova"].to_excel(w, sheet_name="ANOVA", index=False)
        r["means"].to_excel(w, sheet_name="Means", index=False)
        if r["pair"] is not None:
            r["pair"].to_excel(w, sheet_name="Pairwise", index=False)
        pd.DataFrame({"Statistic": ["CV %", "Grand mean", "R²", "SEm±", f"CD ({r['method']}, 5%)",
                                    "Shapiro-Wilk p", "Levene p", "RCBD efficiency"],
                      "Value": [r["cv"], r["grand"], r["r2"], r["sem"], r["cd"],
                                r["sh"][1], r["lev"][1], r["re"]]}).round(4).to_excel(
            w, sheet_name="Summary", index=False)
    return b.getvalue()


def to_word(r, fig):
    try:
        from docx import Document
        from docx.shared import Inches
    except ImportError:
        return None
    doc = Document()
    doc.add_heading(f"RCBD Analysis – {r['y']}", 1)
    doc.add_heading("Materials and Methods", 2)
    doc.add_paragraph(methods_text(r))

    def add_table(df):
        t = doc.add_table(rows=1, cols=len(df.columns))
        t.style = "Table Grid"
        for i, c in enumerate(df.columns):
            t.rows[0].cells[i].text = str(c)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = str(v)

    doc.add_heading("ANOVA", 2)
    add_table(r["anova"])
    doc.add_paragraph(f"CV = {r['cv']:.2f}% | SEm± = {r['sem']:.3f} | CD ({r['alpha']}) = {r['cd']:.3f} | R² = {r['r2']:.3f}")
    doc.add_heading("Treatment means", 2)
    add_table(r["means"])
    doc.add_heading("Figure", 2)
    doc.add_picture(io.BytesIO(fig_bytes(fig, "png")), width=Inches(5.8))
    doc.add_heading("Interpretation", 2)
    doc.add_paragraph(interpret(r))
    b = io.BytesIO()
    doc.save(b)
    return b.getvalue()


# ───────────────────────── Data helpers ─────────────────────────
def demo():
    rng = np.random.default_rng(42)
    trts = ["T0 (Control)", "T1 (50 mM NaCl)", "T2 (100 mM NaCl)", "T3 (150 mM NaCl)", "T4 (200 mM NaCl)"]
    rows = []
    for ri in range(4):
        blk = rng.normal(0, 1.2)
        for t, by, bh in zip(trts, [42, 37, 30, 24, 18], [88, 82, 75, 68, 60]):
            rows.append({"Replication": f"R{ri + 1}", "Treatment": t,
                         "Grain_Yield_g": round(by + blk + rng.normal(0, 1), 1),
                         "Plant_Height_cm": round(bh + blk + rng.normal(0, 1.5), 1)})
    return pd.DataFrame(rows)


def guess(cols, keys, default):
    for c in cols:
        if any(k in str(c).lower() for k in keys):
            return c
    return default


# ───────────────────────── UI ─────────────────────────
st.title(tr("🌾 AgriStat Pro – RCBD Field Trial Analyzer", "🌾 অ্যাগ্রিস্ট্যাট প্রো – RCBD ফিল্ড ট্রায়াল অ্যানালাইজার"))
st.caption(tr("ANOVA • LSD / Tukey / Duncan letters • Assumption checks • Publication plots • Word & Excel reports",
              "অ্যানোভা • LSD / Tukey / Duncan অক্ষর গ্রুপিং • অনুমান যাচাই • পাবলিকেশন প্লট • Word ও Excel রিপোর্ট"))

st.sidebar.header(tr("1. Data", "১. ডেটা"))
src = st.sidebar.radio(tr("Source", "সোর্স"), (tr("Editable sheet", "এডিটেবল শীট"), tr("Upload file", "ফাইল আপলোড"),
                                               tr("Demo data", "ডেমো ডেটা")))
df = None
if src == tr("Editable sheet", "এডিটেবল শীট"):
    st.info(tr("💡 Double-click a cell to edit; use the bottom row to add data.",
               "💡 সেলে ডাবল-ক্লিক করে এডিট করুন; নিচের সারিতে নতুন ডেটা যোগ করুন।"))
    df = st.data_editor(demo(), num_rows="dynamic", use_container_width=True, key="editor")
elif src == tr("Upload file", "ফাইল আপলোড"):
    up = st.sidebar.file_uploader(".csv / .xlsx", type=["csv", "xlsx"])
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up)
            else:
                xl = pd.ExcelFile(up)
                sheet = st.sidebar.selectbox("Sheet", xl.sheet_names) if len(xl.sheet_names) > 1 else xl.sheet_names[0]
                df = xl.parse(sheet)
            with st.expander(tr("Preview", "প্রিভিউ"), expanded=False):
                st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"{tr('Could not read file', 'ফাইল পড়া যায়নি')}: {e}")
    st.sidebar.download_button(tr("⬇️ Template CSV", "⬇️ টেমপ্লেট CSV"), demo().to_csv(index=False),
                               "rcbd_template.csv", "text/csv")
else:
    df = demo()
    with st.expander(tr("Demo dataset", "ডেমো ডেটাসেট"), expanded=True):
        st.dataframe(df, use_container_width=True)

if df is None or df.dropna(how="all").empty:
    st.info(tr("👈 Provide data to begin.", "👈 শুরু করতে ডেটা দিন।"))
    st.stop()

cols = list(df.columns)
st.sidebar.header(tr("2. Variables", "২. ভ্যারিয়েবল"))
trt_c = guess(cols, ["treat", "trt", "geno", "variet", "cultivar"], cols[min(1, len(cols) - 1)])
rep_c = guess(cols, ["rep", "block", "blk"], cols[0])
trt = st.sidebar.selectbox(tr("Treatment column", "ট্রিটমেন্ট কলাম"), cols, index=cols.index(trt_c))
rep = st.sidebar.selectbox(tr("Replication / block column", "রেপ্লিকেশন / ব্লক কলাম"), cols, index=cols.index(rep_c))
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in (trt, rep)]
if not num_cols or trt == rep:
    st.error(tr("Select different treatment/replication columns and ensure a numeric response exists.",
                "আলাদা ট্রিটমেন্ট/রেপ্লিকেশন কলাম নির্বাচন করুন এবং একটি সংখ্যাভিত্তিক রেসপন্স রাখুন।"))
    st.stop()
resp = st.sidebar.selectbox(tr("Response variable", "রেসপন্স ভ্যারিয়েবল"), num_cols)

st.sidebar.header(tr("3. Settings", "৩. সেটিংস"))
method = st.sidebar.selectbox(tr("Mean comparison", "গড় তুলনা"), METHODS)
alpha = st.sidebar.selectbox(tr("Significance level (α)", "তাৎপর্য স্তর (α)"), [0.05, 0.01, 0.10])
tf = st.sidebar.selectbox(tr("Data transformation", "ডেটা ট্রান্সফরমেশন"), list(TRANSFORMS))

try:
    res = analyze(df, trt, rep, resp, tf, alpha, method)
except Exception as e:
    st.error(f"{tr('Analysis error', 'বিশ্লেষণে সমস্যা')}: {e}")
    st.stop()

if res["counts"].nunique() > 1 or len(res["data"]) != res["k"] * res["nrep"]:
    st.warning(tr("⚠️ Design is unbalanced or has missing/duplicate cells; results use the harmonic-mean replication.",
                  "⚠️ ডিজাইন ভারসাম্যহীন বা মিসিং/ডুপ্লিকেট আছে; হারমোনিক-মিন রেপ্লিকেশন ব্যবহার হয়েছে।"))

st.subheader(f"📊 {resp}" + (f"  [{tf}]" if tf != "None" else ""))
c = st.columns(5)
c[0].metric("CV %", f"{res['cv']:.2f}")
c[1].metric(tr("Grand mean", "সামগ্রিক গড়"), f"{res['grand']:.2f}")
c[2].metric("R²", f"{res['r2']:.3f}")
c[3].metric("SEm ±", f"{res['sem']:.3f}")
c[4].metric(f"CD ({method.split()[0]}, {int(alpha * 100)}%)", f"{res['cd']:.3f}")
if res["cv"] > 15:
    st.warning(tr("⚠️ CV > 15% indicates high experimental variability.",
                  "⚠️ CV 15% এর বেশি হলে এক্সপেরিমেন্টাল এরর বেশি বলে গণ্য হয়।"))

t1, t2, t3, t4, t5 = st.tabs([tr("📋 Results", "📋 ফলাফল"), tr("🔍 Assumptions", "🔍 অনুমান যাচাই"),
                              tr("📈 Plot", "📈 প্লট"), tr("🧮 All traits", "🧮 সব বৈশিষ্ট্য"),
                              tr("💾 Export", "💾 এক্সপোর্ট")])

with t1:
    st.markdown("#### " + tr("A. ANOVA (RCBD)", "ক. ANOVA (RCBD)"))
    st.dataframe(res["anova"], use_container_width=True, hide_index=True)
    st.caption(f"RCBD efficiency vs CRD: {res['re']:.2f}× " +
               ("(blocking was effective)" if res["re"] > 1.1 else "(blocking gave little gain)"))
    st.markdown(f"#### B. {method} – " + tr("Treatment means", "ট্রিটমেন্ট গড়"))
    st.dataframe(res["means"], use_container_width=True, hide_index=True)
    st.caption(tr("Means with the same letter are not significantly different.",
                  "একই অক্ষরযুক্ত গড়গুলোর মধ্যে তাৎপর্যপূর্ণ পার্থক্য নেই।"))
    if res["pair"] is not None:
        with st.expander(tr("Pairwise comparisons", "জোড়ায় জোড়ায় তুলনা")):
            st.dataframe(res["pair"], use_container_width=True, hide_index=True)
    st.success(interpret(res))
    st.markdown("**" + tr("Methods paragraph (copy to thesis)", "মেথডস অনুচ্ছেদ (থিসিসে কপি করুন)") + "**")
    st.code(methods_text(res), language=None)

with t2:
    sh_p, lev_p = res["sh"][1], res["lev"][1]
    a1, a2 = st.columns(2)
    a1.metric("Shapiro-Wilk p (" + tr("normality", "নর্মালিটি") + ")", "—" if pd.isna(sh_p) else f"{sh_p:.4f}")
    a2.metric("Levene p (" + tr("equal variance", "সমান ভেরিয়েন্স") + ")", "—" if pd.isna(lev_p) else f"{lev_p:.4f}")
    bad = [n for n, p in (("normality", sh_p), ("variance homogeneity", lev_p)) if not pd.isna(p) and p < 0.05]
    if bad:
        st.warning(tr(f"Assumption violated: {', '.join(bad)}. Try log10 / sqrt / arcsine transformation in the sidebar, "
                      "or a non-parametric test.", "অনুমান ভঙ্গ হয়েছে। সাইডবার থেকে log10 / sqrt / arcsine ট্রান্সফরমেশন চেষ্টা করুন।"))
    else:
        st.success(tr("✅ Assumptions satisfied (small samples have low power – inspect plots too).",
                      "✅ অনুমানগুলো পূরণ হয়েছে (ছোট নমুনায় টেস্টের ক্ষমতা কম – প্লটও দেখুন)।"))
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.scatter(res["fitted"], res["resid"], color="#4A708B", edgecolor="k")
    ax1.axhline(0, color="red", ls="--", lw=1)
    ax1.set(xlabel="Fitted", ylabel="Residuals", title="Residuals vs Fitted")
    sm.qqplot(res["resid"], line="s", ax=ax2)
    ax2.set_title("Normal Q-Q")
    fig2.tight_layout()
    st.pyplot(fig2)

with t3:
    p1, p2, p3 = st.columns(3)
    pal = p1.selectbox(tr("Color", "রং"), list(PALETTES))
    show_l = p2.checkbox(tr("Show letters", "অক্ষর দেখান"), True)
    sort_b = p3.checkbox(tr("Sort by mean", "গড় অনুযায়ী সাজান"), False)
    q1, q2 = st.columns(2)
    ylab = q1.text_input(tr("Y-axis label", "Y অক্ষ"), f"{resp} (Mean ± SE)")
    xlab = q2.text_input(tr("X-axis label", "X অক্ষ"), trt)
    fig = make_plot(res, ylab, xlab, pal, show_l, sort_b)
    st.pyplot(fig)
    d1, d2 = st.columns(2)
    d1.download_button("💾 PNG (300 DPI)", fig_bytes(fig, "png"), f"{resp}_RCBD.png", "image/png")
    d2.download_button("💾 PDF (vector)", fig_bytes(fig, "pdf"), f"{resp}_RCBD.pdf", "application/pdf")

with t4:
    rows = []
    for v in num_cols:
        try:
            rv = analyze(df, trt, rep, v, "None", alpha, method)
            top = rv["means"].sort_values("Mean", ascending=False).iloc[0]
            rows.append({"Trait": v, "Grand mean": round(rv["grand"], 2), "F (trt)": round(rv["f_trt"], 2),
                         "P-value": round(rv["p_trt"], 4), "Sig.": sig_stars(rv["p_trt"], alpha),
                         "CV %": round(rv["cv"], 2), "Best treatment": top["Treatment"]})
        except Exception:
            rows.append({"Trait": v, "Sig.": "error"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(tr("Quick overview of every numeric trait (no transformation).",
                  "সব সংখ্যাভিত্তিক বৈশিষ্ট্যের দ্রুত সারাংশ (ট্রান্সফরমেশন ছাড়া)।"))

with t5:
    e1, e2 = st.columns(2)
    e1.download_button("📊 Excel (all tables)", to_excel(res, df), f"{resp}_RCBD_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    word = to_word(res, fig)
    if word:
        e2.download_button("📝 Word report", word, f"{resp}_RCBD_report.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        e2.info(tr("Add `python-docx` to requirements.txt to enable Word export.",
                   "Word এক্সপোর্টের জন্য requirements.txt এ `python-docx` যোগ করুন।"))

