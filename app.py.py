import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import io

# পেজ কনফিগারেশন
st.set_page_config(page_title="RCBD AgriStat Analyzer", layout="wide")

st.title("🌾 RCBD Field Trial Analyzer & Plotter")
st.markdown("ফিল্ড ডেটা থেকে স্বয়ংক্রিয় **RCBD Two-Way ANOVA**, **CV%** এবং **Publication-Ready Bar Plot** তৈরি করুন।")

# সাইডবার: ডেটা ইনপুট
st.sidebar.header("১. ডেটা ইনপুট")
data_source = st.sidebar.radio("ডেটা সোর্স নির্বাচন করুন:", ("ডেমো RCBD ডেটা ব্যবহার করুন", "CSV বা Excel ফাইল আপলোড করুন"))

df = None

if data_source == "ডেমো RCBD ডেটা ব্যবহার করুন":
    # ৩টি রেপ্লিকেশন এবং ৪টি ট্রিটমেন্টের ডেমো ডেটা
    demo_data = {
        'Replication': ['R1', 'R2', 'R3'] * 4,
        'Treatment': ['T0 (Control)'] * 3 + ['T1 (50 mM NaCl)'] * 3 + ['T2 (100 mM NaCl)'] * 3 + ['T3 (150 mM NaCl)'] * 3,
        'Plant_Height_cm': [88.5, 87.2, 89.1, 82.3, 83.5, 81.0, 75.4, 76.8, 74.2, 65.1, 64.3, 66.8],
        'Grain_Yield_g': [42.1, 41.5, 43.0, 36.8, 37.5, 35.9, 29.4, 30.2, 28.7, 20.1, 19.5, 21.3]
    }
    df = pd.DataFrame(demo_data)
else:
    uploaded_file = st.sidebar.file_uploader("আপনার ফাইল আপলোড করুন (.csv বা .xlsx)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"ফাইল পড়তে সমস্যা হয়েছে: {e}")

if df is not None:
    # ডেটা প্রিভিউ
    with st.expander("📂 ডেটাসেট প্রিভিউ (প্রথম ৫টি সারি)", expanded=True):
        st.dataframe(df.head(), use_container_width=True)

    # ভ্যারিয়েবল নির্বাচন
    st.sidebar.header("২. ভ্যারিয়েবল সিলেকশন")
    columns = list(df.columns)
    
    treatment_col = st.sidebar.selectbox("ট্রিটমেন্ট কলাম (Treatment):", columns, index=1 if len(columns) > 1 else 0)
    rep_col = st.sidebar.selectbox("ব্লক / রেপ্লিকেশন কলাম (Block/Rep):", columns, index=0)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("ডেটাসেটে কোনো নিউমেরিক (সংখ্যাতাত্ত্বিক) রেসপন্স ভ্যারিয়েবল পাওয়া যায়নি!")
        st.stop()
        
    response_col = st.sidebar.selectbox("রেসপন্স ভ্যারিয়েবল (যেমন: Plant Height, Yield):", numeric_cols)

    # অ্যানালাইসিস শুরু
    st.subheader(f"📊 ফলাফল: {response_col} (RCBD ডিজাইন)")
    
    col_stat, col_plot = st.columns([1, 1])

    with col_stat:
        st.markdown("### ক. ANOVA টেবিল (Two-Way)")
        # RCBD মডেল: Response ~ Block + Treatment
        formula = f'Q("{response_col}") ~ C(Q("{rep_col}")) + C(Q("{treatment_col}"))'
        try:
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # টেবিলের কলাম সুন্দর করা
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
            
            # p-value সিগনিফিকেন্স স্টার মার্কিং
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
            
            # কলামের ক্রম বিন্যাস
            anova_display = anova_display[['DF', 'Sum of Sq', 'Mean Sq', 'F-value', 'p-value']]
            st.dataframe(anova_display, use_container_width=True)

            # CV% গণনা
            mse = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
            grand_mean = df[response_col].mean()
            cv = (np.sqrt(mse) / grand_mean) * 100
            
            st.metric(label="Coefficient of Variation (CV %)", value=f"{cv:.2f} %")
            if cv > 15:
                st.warning("⚠️ সতর্কতা: ফিল্ড ট্রায়ালে CV 15% এর বেশি হলে এক্সপেরিমেন্টাল এরর বেশি বলে গণ্য হয়।")

            # Tukey's Post-hoc Test
            st.markdown("### খ. Tukey's HSD গড় তুলনা")
            tukey = pairwise_tukeyhsd(endog=df[response_col], groups=df[treatment_col], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            st.dataframe(tukey_df, use_container_width=True)

        except Exception as err:
            st.error(f"অ্যানালাইসিস করার সময় ত্রুটি হয়েছে: {err}")

    # বারপ্লট জেনারেশন
    with col_plot:
        st.markdown("### গ. পাবলিকেশন বারপ্লট (Mean ± SE)")
        
        # Treatment অনুযায়ী Mean এবং SE (Standard Error) বের করা
        summary = df.groupby(treatment_col)[response_col].agg(['mean', 'std', 'count']).reset_index()
        summary['se'] = summary['std'] / np.sqrt(summary['count'])

        # Matplotlib দিয়ে পাবলিকেশন স্টাইলের প্লট
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        
        x_positions = np.arange(len(summary))
        bars = ax.bar(
            x_positions, 
            summary['mean'], 
            yerr=summary['se'], 
            capsize=5, 
            color='#4A708B', 
            edgecolor='black', 
            linewidth=1.2,
            alpha=0.85,
            width=0.55
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(summary[treatment_col], rotation=15, ha='right', fontsize=10, fontweight='bold')
        ax.set_ylabel(f"{response_col} (Mean ± SE)", fontsize=11, fontweight='bold')
        ax.set_xlabel(treatment_col, fontsize=11, fontweight='bold')
        
        # গ্রাফের সৌন্দর্য ও পাবলিকেশন ফরম্যাটিং (Classic Border)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Y অক্ষের মার্জিন বাড়ানো যাতে এরর বার কেটে না যায়
        y_max = (summary['mean'] + summary['se']).max()
        ax.set_ylim(0, y_max * 1.18)

        st.pyplot(fig)

        # হাই-রেজ্যুলেশন ইমেজ ডাউনলোডের বাটন
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button(
            label="💾 ডাউনলোড বারপ্লট (300 DPI PNG)",
            data=buf.getvalue(),
            file_name=f"{response_col}_RCBD_Barplot.png",
            mime="image/png"
        )