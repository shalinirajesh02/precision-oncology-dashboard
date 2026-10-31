"""Precision Oncology Decision Support Dashboard.   Run:  streamlit run app.py

Data source (sidebar): a local .xlsx (default data.xlsx), an uploaded file, or a Google Sheet link
(share as 'Anyone with the link: Viewer'). With auto-refresh on, new form submissions appear without a manual reload.
"""
import io
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import etl, models, knowledge as K

st.set_page_config(page_title="Precision Oncology Dashboard", layout="wide", page_icon="🧬")
NAVY, TEAL, CORAL, AMBER, GREY = "#1B3A4B", "#3F7C82", "#E07A5F", "#E0A458", "#9AA7AE"
RESP_COL = {"CR": "#4C8C5B", "PR": TEAL, "SD": AMBER, "PD": CORAL, "Unknown": "#D5DBDE"}
st.markdown("""<style>
.block-container{padding-top:1.2rem}
div[data-testid="stMetric"]{background:#EAF2F1;border-radius:10px;padding:10px 14px}
.small{color:#5A6B72;font-size:.85rem}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data loading
def sheet_to_export(url):
    m = re.search(r"/spreadsheets/d/([\w-]+)", url)
    return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=xlsx" if m else url


@st.cache_data(ttl=60, show_spinner="Loading data...")
def load(kind, value, blob):
    if kind == "Google Sheet link":
        import requests
        r = requests.get(sheet_to_export(value), timeout=60); r.raise_for_status()
        src = r.content
    elif kind == "Upload":
        src = blob
    else:
        src = value
    out = etl.load_and_clean(src)
    return out["patients"], out["scans"], out["dq"]


@st.cache_data(show_spinner="Training prediction models...")
def fit(P):
    return models.run_models(P)


with st.sidebar:
    st.title("🧬 Oncology DSS")
    kind = st.radio("Data source", ["Local file", "Upload", "Google Sheet link"])
    value, blob = "data.xlsx", None
    if kind == "Local file":
        value = st.text_input("Path to .xlsx", "data.xlsx")
    elif kind == "Upload":
        up = st.file_uploader("Workbook (.xlsx)", type=["xlsx"])
        blob = up.getvalue() if up else None
    else:
        value = st.text_input("Google Sheet link")
    live = st.toggle("Auto-refresh (live mode)", value=False)
    every = st.select_slider("Refresh every", [30, 60, 120, 300], value=60, disabled=not live)
    st.caption("Patient names are never loaded into the dashboard. Free text is name-scrubbed.")

try:
    if kind == "Upload" and blob is None:
        st.info("Upload the workbook to begin."); st.stop()
    if kind == "Google Sheet link" and not value:
        st.info("Paste the Google Sheet link to begin."); st.stop()
    P_all, S_all, DQ = load(kind, value, blob)
except Exception as e:
    st.error(f"Could not load data: {e}"); st.stop()

with st.sidebar:
    st.markdown("**Filters**")
    cls = st.multiselect("Cancer class", sorted(P_all.cancer_class.unique()))
    sites = st.multiselect("Site group", sorted(P_all.site_group.unique()))
    stg = st.multiselect("Stage", ["I", "II", "III", "IV", "Not recorded"])
    sx = st.multiselect("Sex", sorted(P_all.sex.unique()))


def apply_filters(P):
    m = pd.Series(True, index=P.index)
    if cls: m &= P.cancer_class.isin(cls)
    if sites: m &= P.site_group.isin(sites)
    if sx: m &= P.sex.isin(sx)
    if stg:
        lab = P.stage.map({1: "I", 2: "II", 3: "III", 4: "IV"}).fillna("Not recorded")
        m &= lab.isin(stg)
    return m


def bar(df, x, y, color=TEAL, title="", horizontal=False, **kw):
    f = px.bar(df, x=y if horizontal else x, y=x if horizontal else y, orientation="h" if horizontal else "v", color_discrete_sequence=[color], text_auto=True, **kw)
    f.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), height=300, showlegend=False, plot_bgcolor="white")
    return f


def gauge(v, title, color):
    f = go.Figure(go.Indicator(mode="gauge+number", value=v, title={"text": title},
                  gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}, "steps": [{"range": [0, 33], "color": "#EAF2F1"}, {"range": [33, 66], "color": "#F6E7D2"}, {"range": [66, 100], "color": "#F7DAD1"}]}))
    f.update_layout(height=230, margin=dict(l=10, r=10, t=50, b=0))
    return f


def body():
    M = fit(P_all.reset_index(drop=True))
    R = M["scores"].set_index("patient_id")
    P = P_all.merge(M["scores"], on="patient_id")
    msk = apply_filters(P)
    P, PA = P[msk].copy(), P
    if P.empty:
        st.warning("No patients match the filters."); return
    st.title("Precision Oncology Decision Support")
    st.markdown(f"<span class='small'>{len(P)} of {len(PA)} patients shown · last data load {pd.Timestamp.now():%H:%M:%S} · prediction models are exploratory (small cohort)</span>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Executive", "Clinical", "Molecular", "AI Prediction"])

    # ---------------- Executive
    with t1:
        c = st.columns(5)
        c[0].metric("Total patients", len(P))
        c[1].metric("Active (living)", int((P.os_status == "Living").sum()))
        c[2].metric("High-risk (living)", int((P.ai_high_risk & (P.os_status != "Deceased")).sum()), help="Stage IV / M1 / imaging progression, or top third of AI composite risk")
        c[3].metric("Deceased", int((P.os_status == "Deceased").sum()))
        c[4].metric("Status unknown", int((P.os_status == "Unknown").sum()))
        a, b = st.columns(2)
        reg = P.assign(m=P.entry_date.dt.to_period("M").astype(str)).groupby("m").size().reset_index(name="patients")
        a.plotly_chart(bar(reg, "m", "patients", TEAL, "New registrations per month"), width="stretch")
        tier = P.risk_tier.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
        tier.columns = ["tier", "n"]
        f = px.pie(tier, names="tier", values="n", hole=.55, color="tier", color_discrete_map={"Low": TEAL, "Medium": AMBER, "High": CORAL})
        f.update_layout(title="AI risk tiers", height=300, margin=dict(l=0, r=0, t=40, b=0))
        b.plotly_chart(f, width="stretch")
        st.subheader("High-risk watch-list")
        w = P[P.ai_high_risk & (P.os_status != "Deceased")].sort_values("composite_risk", ascending=False).head(15)
        st.dataframe(w[["patient_id", "site_group", "stage", "age", "composite_risk", "prog_risk_6m", "io_score"]].rename(columns={
            "patient_id": "Patient", "site_group": "Site", "stage": "Stage", "age": "Age", "composite_risk": "Composite risk", "prog_risk_6m": "6-mo progression", "io_score": "Responder score"}).round(2), hide_index=True, width="stretch")

    # ---------------- Clinical
    with t2:
        a, b = st.columns(2)
        v = P.site_group.value_counts().reset_index(); v.columns = ["site", "n"]
        a.plotly_chart(bar(v.sort_values("n"), "site", "n", TEAL, "Cancer type distribution (site group)", True), width="stretch")
        s = P.stage.map({1: "I", 2: "II", 3: "III", 4: "IV"}).fillna("Not recorded").value_counts().reindex(["I", "II", "III", "IV", "Not recorded"]).fillna(0).reset_index()
        s.columns = ["stage", "n"]
        b.plotly_chart(bar(s, "stage", "n", CORAL, "Stage distribution"), width="stretch")
        a, b = st.columns(2)
        tu = pd.DataFrame({"Modality": ["Chemotherapy", "Surgery", "Targeted", "Immunotherapy", "Radiation", "Hormonal"],
                           "Patients": [P.chemo.sum(), P.surgery.sum(), P.targeted.sum(), P.immuno.sum(), P.radiation.sum(), P.hormonal.sum()]})
        a.plotly_chart(bar(tu.sort_values("Patients"), "Modality", "Patients", NAVY, "Treatment utilisation", True), width="stretch")
        rr = P.groupby(["site_group", "best_response"]).size().reset_index(name="n")
        f = px.bar(rr, x="site_group", y="n", color="best_response", color_discrete_map=RESP_COL, title="Response tracking: best imaging response by site",
                   category_orders={"best_response": ["CR", "PR", "SD", "PD", "Unknown"]})
        f.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor="white", legend_title="")
        b.plotly_chart(f, width="stretch")
        ls = P.last_status.value_counts().reindex(["CR", "PR", "SD", "PD", "Unknown"]).fillna(0)
        st.caption("Latest documented status: " + " · ".join(f"{k} {int(v)}" for k, v in ls.items()) + f" · patients with any progression: {int(P.ever_pd.sum())}")

    # ---------------- Molecular
    with t3:
        a, b = st.columns(2)
        g = P[P.genes_altered != ""].genes_altered.str.split(";").explode().value_counts().reset_index(); g.columns = ["gene", "n"]
        a.plotly_chart(bar(g.sort_values("n"), "gene", "n", CORAL, f"Genomic mutation frequency (n sequenced = {int(P.genomic_tested.sum())})", True), width="stretch") if len(g) else a.info("No gene-level alterations in the filtered cohort.")
        cat = np.select([P.pdl1_pct >= 50, P.pdl1_pct >= 1, P.pdl1_pct < 1, P.pdl1_status == "Positive", P.pdl1_status == "Negative"],
                        ["High (≥50%)", "Low (1–49%)", "Negative (<1%)", "Positive (no %)", "Negative (<1%)"], "Not tested")
        pl = pd.Series(cat).value_counts().reset_index(); pl.columns = ["PD-L1", "n"]
        b.plotly_chart(bar(pl, "PD-L1", "n", AMBER, "PD-L1 landscape"), width="stretch")
        a, b = st.columns(2)
        ms = P.msi.value_counts().reset_index(); ms.columns = ["MSI", "n"]
        f = px.pie(ms, names="MSI", values="n", hole=.5, color="MSI", color_discrete_map={"MSS": TEAL, "MSI-High": CORAL, "Unknown": GREY}, title="MSI status")
        f.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0)); a.plotly_chart(f, width="stretch")
        cnt = {}
        for _, r in P.iterrows():
            for m in set(K.patient_markers(r)):
                cnt[m] = cnt.get(m, 0) + 1
        ab = pd.DataFrame({"marker": list(cnt), "n": list(cnt.values())})
        if len(ab):
            b.plotly_chart(bar(ab.sort_values("n"), "marker", "n", TEAL, "Actionable biomarkers (gene level)", True), width="stretch")

    # ---------------- AI Prediction
    with t4:
        perf = M["perf"]
        st.subheader("Model validity")
        c = st.columns(3)
        def badge(v): return "🟢 moderate" if v >= .7 else ("🟠 weak" if v >= .6 else "🔴 ~chance")
        c[0].metric("Survival model C-index (CV)", f"{perf['os']['cindex_cv']:.2f} ± {perf['os']['cindex_sd']:.2f}", badge(perf['os']['cindex_cv']))
        c[0].caption(f"n={perf['os']['n']}, deaths={perf['os']['events']}")
        c[1].metric("Progression model C-index (CV)", f"{perf['pfs']['cindex_cv']:.2f} ± {perf['pfs']['cindex_sd']:.2f}", badge(perf['pfs']['cindex_cv']))
        c[1].caption(f"n={perf['pfs']['n']}, progressions={perf['pfs']['events']}")
        io = perf["io"]
        c[2].metric("Responder score AUC (LOO)", f"{io.get('auc_model_loo', float('nan')):.2f}", badge(io.get("auc_model_loo", 0.5)))
        c[2].caption(f"n={io['n']} immunotherapy-treated · exploratory")
        st.caption("Survival time is a proxy (first dated entry to last dated entry). Scores support triage and discussion; they are not clinical predictions.")
        st.divider()
        pid = st.selectbox("Select a patient", sorted(P.patient_id))
        row = P[P.patient_id == pid].iloc[0]
        base = P_all.reset_index(drop=True)
        i = base.index[base.patient_id == pid][0]
        c = st.columns([1.4, 1, 1])
        t = np.arange(0, 37)
        sv = M["models"]["os"].surv(M["X"].loc[[i]], t).iloc[0]
        f = go.Figure(go.Scatter(x=t, y=sv.values, mode="lines", line=dict(color=TEAL, width=3), fill="tozeroy", fillcolor="rgba(63,124,130,.12)"))
        f.update_layout(title="Predicted survival curve", xaxis_title="Months", yaxis_title="Probability", yaxis_range=[0, 1.02], height=300, margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor="white")
        c[0].plotly_chart(f, width="stretch")
        c[0].caption(f"Survival probability: 6 mo {row.os_s6:.0%} · 12 mo {row.os_s12:.0%} · 24 mo {row.os_s24:.0%}")
        c[1].plotly_chart(gauge(row.prog_risk_6m * 100, "Progression risk (6 mo, %)", CORAL), width="stretch")
        c[1].caption(f"Risk tier: **{row.risk_tier}** · stage {row.stage if pd.notna(row.stage) else 'n/r'} · {row.site_group}")
        c[2].plotly_chart(gauge(row.io_score, "Immunotherapy responder score", TEAL), width="stretch")
        c[2].caption("Biomarker prior (PD-L1, MSI-H, TMB, tumour phenotype) blended with a regularised model.")
        a, b = st.columns(2)
        a.subheader("Precision therapy matching")
        tm = pd.DataFrame(K.match_therapies(base.loc[i]))
        if len(tm):
            a.dataframe(tm.rename(columns={"marker": "Biomarker", "drugs": "Candidate therapies", "note": "Evidence note", "site_fit": "Site fit"}), hide_index=True, width="stretch")
            a.caption("Gene-level, illustrative knowledge base. Confirm the exact variant and guideline status before use.")
        else:
            a.info("No actionable biomarker recorded for this patient.")
        b.subheader("Clinical trial matching")
        tr = pd.DataFrame(K.match_trials(base.loc[i]))
        if len(tr):
            b.dataframe(tr.rename(columns={"trial_id": "Trial", "title": "Title", "match_score": "Score", "why": "Why it matches"}), hide_index=True, width="stretch")
            b.caption("Placeholder trial registry; connect ClinicalTrials.gov / CTRI for live matching.")
        else:
            b.info("No matching trial in the registry.")
        st.subheader("Similar patients")
        sp = models.similar_patients(base, pid, 5)
        st.dataframe(sp.rename(columns={"patient_id": "Patient", "site_group": "Site", "best_response": "Best response", "os_status": "Status", "os_months": "Follow-up (mo)"}), hide_index=True, width="stretch")
        st.divider()
        f = px.scatter(P, x="os_risk_pct", y="prog_risk_6m", color="risk_tier", size="io_score", hover_name="patient_id",
                       color_discrete_map={"Low": TEAL, "Medium": AMBER, "High": CORAL}, title="Cohort map: survival risk vs 6-month progression risk (bubble = responder score)")
        f.update_layout(height=380, plot_bgcolor="white", xaxis_title="Survival risk percentile", yaxis_title="6-month progression risk")
        st.plotly_chart(f, width="stretch")
        with st.expander("Data-quality log"):
            for d in DQ: st.write("• " + d)


if live:
    st.fragment(run_every=f"{every}s")(body)()
else:
    body()
