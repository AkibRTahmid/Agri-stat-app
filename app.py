import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import io

# ১. পেজ কনফিগারেশন
st.set_page_config(page_title="RCBD AgriStat Analyzer", layout="wide")

# ২. কাস্টম সিএসএস (Times New Roman এবং SutonnyMJ/Kalpurush ফন্ট ইন্টিগ্রেশন)
font_css = """
<style>
@import url('https://fonts.maateen.me/kalpurush/font.css');

html, body, [class*="css"], .stMarkdown, .stSelectbox, .stDataFrame, .stTable, button, input, label {
    font-family: 'Times New Roman', 'SutonnyMJ', 'Kalpurush', serif !important;
}
</style>
"""
st.markdown(font_css, unsafe_allow_html=True)

# Matplotlib-এর জন্য ফন্ট কনফিগারেশন
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Kalpurush', 'DejaVu Serif']

# ৩. ভাষা নির্বাচন ও ডিকশনারি
st.sidebar.markdown("### 🌐 Language / ভাষা")
lang = st.sidebar.radio("Select Language / ভাষা নির্বাচন করুন:", ("English", "বাংলা"))

T = {
    "English": {
        "title": "🌾 RCBD Field Trial Analyzer & Plotter",
        "subtitle": "Direct Data Input, Two-Way ANOVA, CV%, and Publication-Ready Bar Plots.",
        "input_header": "1. Data Input Method",
        "data_source_label": "Choose Data Source:",
        "src_direct": "Direct Data Input (Editable Sheet)",
        "src_upload": "Upload CSV or Excel File",
        "src_demo": "Use Demo RCBD Data",
        "upload_label": "Upload .csv or .xlsx file",
        "preview_header": "📂 Dataset Preview / Editor",
        "preview_sub": "💡 You can directly double-click, edit, delete, or add new rows below:",
        "var_header": "2. Variable Selection",
        "trt_label": "Treatment Column:",
        "rep_label": "Replication / Block Column:",
        "resp_label": "Response Variable (e.g. Yield, Height):",
        "err_no_num": "No numerical response variable found!",
        "res_header": "📊 Analysis Results:",
        "anova_header": "A. RCBD ANOVA Table (Two-Way)",
        "cv_label": "Coefficient of Variation (CV %)",
        "cv_warn": "⚠️ Warning: CV > 15% indicates high experimental variability.",
        "tukey_header": "B. Tukey's HSD Post-hoc Test",
        "plot_header": "C. Publication Bar Plot (Mean ± SE)",
        "btn_download": "💾 Download Bar Plot (300 DPI PNG)"
    },
    "বাংলা": {
        "title": "🌾 আরসিবিডি ফিল্ড ট্রায়াল অ্যানালাইজার",
        "subtitle": "সরাসরি ডেটা ইনপুট, টু-ওয়ে অ্যানোভা, সিভি% এবং পাবলিকেশন বারপ্লট তৈরি করুন।",
        "input_header": "১. ডেটা ইনপুট পদ্ধতি",
        "data_source_label": "ডেটা সোর্স নির্বাচন করুন:",
        "src_direct": "সরাসরি ডেটা ইনপুট (এডিটেবল শীট)",
        "src_upload": "CSV বা Excel ফাইল আপলোড",
        "src_demo": "ডেমো RCBD ডেটা ব্যবহার করুন",
        "upload_label": ".csv বা .xlsx ফাইল আপলোড করুন",
        "preview_header": "📂 ডেটাসেট প্রিভিউ ও সরাসরি এডিটর",
        "preview_sub": "💡 নিচের টেবিলে ডাবল-ক্লিক করে সরাসরি ডেটা পরিবর্তন বা নতুন সারি যুক্ত করতে পারেন:",
        "var_header": "২. ভ্যারিয়েবল সিলেকশন",
        "trt_label": "ট্রিটমেন্ট কলাম (Treatment):",
        "rep_label": "ব্লক / রেপ্লিকেশন কলাম (Block/Rep):",
        "resp_label": "রেসপন্স ভ্যারিয়েবল (যেমন: Grain Yield, Plant Height):",
        "err_no_num": "কোনো সংখ্যাতাত্ত্বিক রেসপন্স ভ্যারিয়েবল পাওয়া যায়নি!",
        "res_header": "📊 অ্যানালাইসিস ফলাফল:",
        "anova_header": "ক. RCBD ANOVA টেবিল (Two-Way)",
        "cv_label": "Coefficient of Variation (CV %)",
        "cv_warn": "⚠️ সতর্কতা: মাঠ ফসলে CV 15% এর বেশি হলে এক্সপেরিমেন্টাল এরর বেশি বলে গণ্য হয়।",
        "tukey_header": "খ. Tukey's HSD গড় তুলনা পরীক্ষা",
        "plot_header": "গ. পাবলিকেশন বারপ্লট (Mean ± SE)",
        "btn_download": "💾 ডাউনলোড বারপ্লট (300 DPI PNG)"
    }
}[lang]

st.title(T["title"])
st.markdown(T["subtitle"])

# ৪. ডেটা ইনপুট ইন্টারফেস
st.sidebar.header(T["input_header"])
data_source = st.sidebar.radio(
    T["data_source_label"],
    (T["src_direct"], T["src_upload"], T["src_demo"])
)

df = None

# ডিফল্ট স্ট্রাকচার
default_template = {
    'Replication': ['R1', 'R2', 'R3'] * 3,
    'Treatment': ['T0 (Control)'] * 3 + ['T1 (50 mM NaCl)'] * 3 + ['T2 (100 mM NaCl)'] * 3,
    'Grain_Yield_g': [42.1, 41.5, 43.0, 36.8, 37.5, 35.9, 29.4, 30.2, 28.7],
    'Plant_Height_cm': [88.5, 87.2, 89.1, 82.3, 83.5, 81.0, 75.4, 76.8, 74.2]
}

if data_source == T["src_direct"]:
    st.markdown(f"### {T['preview_header']}")
    st.info(T["preview_sub"])
    # ইন্টারেক্টিভ ডেটা এডিটর (সরাসরি এক্সেলের মতো ইনপুট ও রো যোগ করার অপশন)
    init_df = pd.DataFrame(default_template)
    df = st.data_editor(init_df, num_rows="dynamic", use_container_width=True)

elif data_source == T["src_demo"]:
    df = pd.DataFrame(default_template)
    with st.expander(T["preview_header"], expanded=True):
        st.dataframe(df, use_container_width=True)

else:
    uploaded_file = st.sidebar.file_uploader(T["upload_label"], type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            with st.expander(T["preview_header"], expanded=True):
                st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")

# ৫. স্ট্যাটিস্টিক্যাল অ্যানালাইসিস ও ভিজ্যুয়ালাইজেশন
if df is not None and not df.empty:
    st.sidebar.header(T["var_header"])
    columns = list(df.columns)
    
    treatment_col = st.sidebar.selectbox(T["trt_label"], columns, index=1 if len(columns) > 1 else 0)
    rep_col = st.sidebar.selectbox(T["rep_label"], columns, index=0)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error(T["err_no_num"])
        st.stop()
        
    response_col = st.sidebar.selectbox(T["resp_label"], numeric_cols)

    st.subheader(f"{T['res_header']} {response_col}")
    col_stat, col_plot = st.columns([1.1, 0.9])

    with col_stat:
        st.markdown(f"### {T['anova_header']}")
        formula = f'Q("{response_col}") ~ C(Q("{rep_col}")) + C(Q("{treatment_col}"))'
        try:
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            anova_clean = anova_table.rename(index={
                f'C(Q("{rep_col}"))': f'Replication ({rep_col})',
                f'C(Q("{treatment_col}"))': f'Treatment ({treatment_col})',
                'Residual': 'Error (Residual)'
            }, columns={
                'sum_sq': 'Sum of Sq',
                'df': 'DF',
                'F': 'F-value',
                'PR(>F)': 'p-value'
            })
            
            def star_p(val):
                if pd.isna(val):
                    return ""
                if val < 0.001:
                    return f"{val:.4f} ***"
                elif val < 0.01:
                    return f"{val:.4f} **"
                elif val < 0.05:
                    return f"{val:.4f} *"
                return f"{val:.4f} (NS)"

            anova_display = anova_clean.copy()
            anova_display['DF'] = anova_display['DF'].astype(int)
            anova_display['Mean Sq'] = (anova_display['Sum of Sq'] / anova_display['DF']).round(3)
            anova_display['Sum of Sq'] = anova_display['Sum of Sq'].round(3)
            anova_display['F-value'] = anova_display['F-value'].round(3)
            anova_display['p-value'] = anova_display['p-value'].apply(star_p)
            anova_display = anova_display[['DF', 'Sum of Sq', 'Mean Sq', 'F-value', 'p-value']]
            
            st.dataframe(anova_display, use_container_width=True)

            # CV% নির্ণয়
            mse = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
            grand_mean = df[response_col].mean()
            cv = (np.sqrt(mse) / grand_mean) * 100
            
            st.metric(label=T["cv_label"], value=f"{cv:.2f} %")
            if cv > 15:
                st.warning(T["cv_warn"])

            # Tukey's HSD টেস্ট
            st.markdown(f"### {T['tukey_header']}")
            tukey = pairwise_tukeyhsd(endog=df[response_col], groups=df[treatment_col], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            st.dataframe(tukey_df, use_container_width=True)

        except Exception as err:
            st.error(f"Error executing ANOVA: {err}")

    with col_plot:
        st.markdown(f"### {T['plot_header']}")
        summary = df.groupby(treatment_col)[response_col].agg(['mean', 'std', 'count']).reset_index()
        summary['se'] = summary['std'] / np.sqrt(summary['count'])

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        x_positions = np.arange(len(summary))
        
        ax.bar(
            x_positions, 
            summary['mean'], 
            yerr=summary['se'], 
            capsize=5, 
            color='#4A708B', 
            edgecolor='black', 
            linewidth=1.2,
            alpha=0.85,
            width=0.52
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(summary[treatment_col], rotation=15, ha='right', fontsize=10, fontweight='bold')
        ax.set_ylabel(f"{response_col} (Mean ± SE)", fontsize=11, fontweight='bold')
        ax.set_xlabel(treatment_col, fontsize=11, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        y_max = (summary['mean'] + summary['se']).max()
        ax.set_ylim(0, y_max * 1.18)

        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button(
            label=T["btn_download"],
            data=buf.getvalue(),
            file_name=f"{response_col}_RCBD_Plot.png",
            mime="image/png"
        )
