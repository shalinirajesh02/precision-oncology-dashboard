"""make_reports.py -- builds both analysis reports (.docx) from the workbook.

    python make_reports.py data.xlsx

Every number in the text is computed from the data, so re-running on the real workbook
regenerates figures, tables and narrative automatically. Review the wording before submission.
"""
import sys
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

import etl, models, knowledge as K, viz_static as V

NAVY, TEAL, CORAL = RGBColor(0x1B, 0x3A, 0x4B), RGBColor(0x3F, 0x7C, 0x82), RGBColor(0xE0, 0x7A, 0x5F)


# ---------------------------------------------------------------- docx helpers
class Rep:
    def __init__(self, title, subtitle):
        self.d = Document()
        sec = self.d.sections[0]
        sec.page_width, sec.page_height = Inches(8.5), Inches(11)
        for m in ("left_margin", "right_margin"):
            setattr(sec, m, Inches(0.9))
        sec.top_margin = sec.bottom_margin = Inches(0.8)
        st = self.d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
        for n, sz, col in (("Heading 1", 17, NAVY), ("Heading 2", 13, TEAL), ("Heading 3", 11, NAVY)):
            h = self.d.styles[n]; h.font.name = "Calibri"; h.font.size = Pt(sz); h.font.bold = True; h.font.color.rgb = col
            h.element.rPr.rFonts.set(qn("w:asciiTheme"), "minorHAnsi") if False else None
        self.fig_n = 0; self.tab_n = 0
        p = self.d.add_paragraph(); r = p.add_run("PRECISION ONCOLOGY ANALYSIS"); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = CORAL
        p = self.d.add_paragraph(); r = p.add_run(title); r.bold = True; r.font.size = Pt(23); r.font.color.rgb = NAVY
        p = self.d.add_paragraph(); r = p.add_run(subtitle); r.italic = True; r.font.size = Pt(11); r.font.color.rgb = RGBColor(0x5A, 0x6B, 0x72)

    def h1(self, t): self.d.add_heading(t, 1)
    def h2(self, t): self.d.add_heading(t, 2)
    def h3(self, t): self.d.add_heading(t, 3)

    def p(self, t, bold_lead=None, italic=False, size=None):
        para = self.d.add_paragraph()
        if bold_lead:
            r = para.add_run(bold_lead + " "); r.bold = True
        r = para.add_run(t); r.italic = italic
        if size: r.font.size = Pt(size)
        para.paragraph_format.space_after = Pt(6)
        return para

    def bullets(self, items):
        for it in items:
            para = self.d.add_paragraph(style="List Bullet")
            if isinstance(it, tuple):
                r = para.add_run(it[0] + " "); r.bold = True
                para.add_run(it[1])
            else:
                para.add_run(it)
            para.paragraph_format.space_after = Pt(3)

    def callout(self, title, text, color="EAF2F1"):
        t = self.d.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER
        c = t.cell(0, 0); _shade(c, color)
        c.paragraphs[0].add_run(title + "  ").bold = True
        c.paragraphs[0].add_run(text)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def fig(self, path, caption, width=6.6, read=None):
        self.fig_n += 1
        para = self.d.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.add_run().add_picture(path, width=Inches(width))
        para.paragraph_format.keep_with_next = True
        c = self.d.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(f"Figure {self.fig_n}. {caption}"); r.italic = True; r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x5A, 0x6B, 0x72)
        if read:
            self.p(read, bold_lead="How to read it:")

    def table(self, df, widths=None, caption=None, font=9):
        if caption:
            self.tab_n += 1
            c = self.d.add_paragraph(); r = c.add_run(f"Table {self.tab_n}. {caption}"); r.italic = True; r.font.size = Pt(9)
            c.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=1, cols=len(df.columns)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, col in enumerate(df.columns):
            c = t.rows[0].cells[i]; _shade(c, "1B3A4B")
            r = c.paragraphs[0].add_run(str(col)); r.bold = True; r.font.size = Pt(font); r.font.color.rgb = RGBColor(255, 255, 255)
        for k, (_, row) in enumerate(df.iterrows()):
            cells = t.add_row().cells
            for i, v in enumerate(row):
                r = cells[i].paragraphs[0].add_run("" if pd.isna(v) else str(v)); r.font.size = Pt(font)
                if k % 2 == 1: _shade(cells[i], "EAF2F1")
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths): row.cells[i].width = Inches(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def pagebreak(self): self.d.add_page_break()
    def save(self, path):
        # python-docx's default template omits w:percent on <w:zoom>; Word/validators expect it
        for z in self.d.settings.element.iter(qn("w:zoom")):
            if z.get(qn("w:percent")) is None:
                z.set(qn("w:percent"), "100")
        self.d.save(path)


def _shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcolor); tcPr.append(sh)


def pc(a, b): return f"{100*a/b:.0f}%" if b else "n/a"


# ---------------------------------------------------------------- analysis numbers
def compute(P, S, M):
    R = M["scores"]; X = {}
    n = len(P)
    X["n"] = n
    X["male"] = (P.sex.str.lower() == "male").sum(); X["female"] = (P.sex.str.lower() == "female").sum()
    X["age_med"] = P.age.median(); X["age_min"], X["age_max"] = P.age.min(), P.age.max()
    X["smoker"] = (P.smoking == "Yes").sum(); X["alcohol"] = (P.alcohol == "Yes").sum()
    X["class"] = P.cancer_class.value_counts().to_dict()
    X["site"] = P.site_group.value_counts()
    X["stage_known"] = P.stage.notna().sum(); X["stage4"] = (P.stage == 4).sum(); X["stage_imp"] = int(P.stage_imputed.sum())
    X["living"] = (P.os_status == "Living").sum(); X["dead"] = (P.os_status == "Deceased").sum(); X["unk"] = (P.os_status == "Unknown").sum()
    X["resp_known"] = (P.best_response != "Unknown").sum(); X["resp"] = P.best_response.value_counts().to_dict()
    X["orr"] = ((P.best_response.isin(["CR", "PR"])).sum())
    X["pd_ever"] = P.ever_pd.sum()
    X["msi_h"] = (P.msi == "MSI-High").sum(); X["msi_known"] = (P.msi != "Unknown").sum()
    X["pdl_tested"] = (P.pdl1_status != "Not performed").sum(); X["pdl_pos"] = (P.pdl1_status == "Positive").sum()
    X["pdl_high"] = (P.pdl1_pct >= 50).sum(); X["pdl_pct_n"] = P.pdl1_pct.notna().sum()
    X["gen_tested"] = P.genomic_tested.sum(); X["gen_alt"] = (P.n_altered > 0).sum()
    X["immuno"] = P.immuno.sum(); X["chemo"] = P.chemo.sum(); X["surg"] = P.surgery.sum(); X["rad"] = P.radiation.sum()
    X["hot"] = (P.tumor_type == "Hot").sum(); X["cold"] = (P.tumor_type == "Cold").sum()
    X["high_risk_rule"] = P.high_risk_rule.sum()
    X["ai_high"] = R.ai_high_risk.sum(); X["tier_high"] = (R.risk_tier == "High").sum()
    X["os_n"] = M["perf"]["os"]["n"]; X["os_ev"] = M["perf"]["os"]["events"]
    X["fu_med"] = P.os_months.median()
    X["actionable_pts"] = sum(1 for _, r in P.iterrows() if set(K.patient_markers(r)) & K.ACTIONABLE)
    return X


def treat_combos(P, top=8):
    def lab(t):
        parts = [x.strip() for x in str(t).split(",") if x.strip() and x.strip().lower() not in ("allied services", "nan")]
        return " + ".join(sorted(parts)) if parts else "Not recorded"
    s = P.treatment.map(lab).value_counts().head(top)
    return pd.DataFrame({"Treatment combination": s.index, "Patients": s.values, "Share": [pc(v, len(P)) for v in s.values]})


def therapy_summary(P):
    rows = []
    for _, r in P.iterrows():
        for m in K.match_therapies(r):
            rows.append(dict(marker=m["marker"], drugs=m["drugs"], fit=m["site_fit"], pid=r.patient_id))
    if not rows:
        return pd.DataFrame(columns=["Biomarker", "Candidate therapies (verify)", "Patients", "Site-appropriate"])
    D = pd.DataFrame(rows)
    g = D.groupby(["marker", "drugs"]).agg(n=("pid", "nunique"), ok=("fit", lambda s: (s == "Site-appropriate").sum())).reset_index()
    g = g.sort_values("n", ascending=False)
    return pd.DataFrame({"Biomarker": g.marker, "Candidate therapies (verify)": g.drugs, "Patients": g.n, "Site-appropriate": g.ok})


def trial_summary(P):
    cnt, best = {}, 0
    pts = 0
    for _, r in P.iterrows():
        m = K.match_trials(r)
        if m: pts += 1
        for t in m:
            cnt.setdefault(t["trial_id"], [t["title"], 0]); cnt[t["trial_id"]][1] += 1
    df = pd.DataFrame([(k, v[0], v[1]) for k, v in cnt.items()], columns=["Trial (placeholder)", "Title", "Matching patients"]).sort_values("Matching patients", ascending=False)
    return df, pts


# ---------------------------------------------------------------- report 1: manual-analysis report
def manual_report(P, S, M, X, out):
    V.OUT = "figures"
    r = Rep("Manual Analysis Report", f"Clinical, pathological, molecular and outcome patterns in {X['n']} cancer patients")
    r.callout("Scope.", "Descriptive, pattern-finding analysis of the combined patient workbook, with interpretation of each chart. "
              "This draft was produced on the sample (fake) dataset; numbers will change when the real workbook is run.")
    r.h1("1. Dataset and data quality")
    r.p(f"The workbook holds {X['n']} unique patients after cleaning. Each patient has demographics, pathology, staging, treatment, baseline laboratory "
        f"values (cycle 1), biomarker results, imaging narrative and survival status. A genomics sheet supplies sequencing text for a subset.")
    r.fig(V.f_completeness(P), "Share of patients with a usable value for each key field.",
          read="Bars show how many patients have each field filled in and interpretable. Red bars (<40%) mark fields too sparse for firm conclusions.")
    low = [k for k in ["Genomic report", "TMB", "PD-L1 %"] if True]
    r.bullets([("Staging:", f"{pc(X['stage_known'], X['n'])} of patients have a recorded stage ({X['stage_imp']} more were inferred as stage IV from M1 status)."),
               ("Response:", f"only {pc(X['resp_known'], X['n'])} have an imaging response that could be read from the report text."),
               ("Molecular data:", f"gene-level genomic results exist for {X['gen_tested']} patients ({pc(X['gen_tested'], X['n'])}); PD-L1 percentage for {X['pdl_pct_n']}.")])
    r.p("Cleaning actions recorded by the pipeline:", bold_lead="")
    r.bullets(etl_dq_short(P))

    r.h1("2. Who are the patients?")
    r.fig(V.f_demographics(P), "Age distribution by sex, with smoking and alcohol history.")
    r.p(f"Median age is {X['age_med']:.0f} years (range {X['age_min']:.0f}–{X['age_max']:.0f}); {X['male']} men and {X['female']} women. "
        f"{pc(X['smoker'], X['n'])} report a smoking history and {pc(X['alcohol'], X['n'])} alcohol use.", bold_lead="Pattern:")
    if X["age_max"] - X["age_min"] >= 30:
        r.p("Interpretation: the wide age range suggests a mixed cohort rather than a single disease population, so subgroup comparisons should be made within cancer site.", bold_lead="")

    r.h1("3. Cancer type and stage")
    r.fig(V.f_site(P), "Patients by cancer site group, stacked by cancer class.")
    top = X["site"].head(3)
    r.p(f"The most common site groups are {', '.join(f'{k} ({v})' for k, v in top.items())}. "
        f"{X['class'].get('Metastatic solid', 0)} patients are classified as metastatic solid disease and {X['class'].get('Hematologic', 0)} as hematologic cancers.", bold_lead="Pattern:")
    r.fig(V.f_stage(P), "Stage distribution and stage by site (solid tumours).")
    r.p(f"Among patients with a known stage, {pc(X['stage4'], X['stage_known'])} are stage IV, so the cohort is skewed toward advanced disease. "
        f"A large 'not recorded' group means stage-based comparisons cover only part of the cohort.", bold_lead="Pattern:")

    r.h1("4. Treatment patterns")
    r.fig(V.f_treatment(P), "Treatment modality utilisation overall and by stage group.")
    r.p(f"Chemotherapy is the most used modality ({pc(X['chemo'], X['n'])}), followed by immunotherapy ({pc(X['immuno'], X['n'])}), surgery ({pc(X['surg'], X['n'])}) and radiation ({pc(X['rad'], X['n'])}).", bold_lead="Pattern:")
    r.table(treat_combos(P), widths=[4.2, 1, 1], caption="Most frequent treatment combinations (as recorded).")

    r.h1("5. Response to treatment")
    r.fig(V.f_response(P), "Best imaging response documented in the report text, overall and by site.")
    r.p(f"Of {X['resp_known']} patients with a readable response, {X['orr']} ({pc(X['orr'], X['resp_known'])}) showed a complete or partial response; "
        f"{X['pd_ever']} patients have at least one report describing progression.", bold_lead="Pattern:")
    r.p("Interpretation: response is inferred from keywords in imaging narratives (e.g. 'reduction in size', 'new lesion'), so it approximates RECIST rather than replacing it. "
        "Response labels should be hand-checked on a sample before they are quoted.", bold_lead="")

    r.h1("6. Molecular and biomarker landscape")
    r.fig(V.f_biomarkers(P), "MSI status, PD-L1 expression and tumour immune phenotype.")
    r.p(f"MSI status is available for {X['msi_known']} patients, of whom {X['msi_h']} are MSI-high. PD-L1 was tested in {X['pdl_tested']} patients "
        f"({X['pdl_pos']} positive; {X['pdl_high']} with ≥50% where a percentage was recorded). Tumours are labelled hot in {X['hot']} and cold in {X['cold']} patients.", bold_lead="Pattern:")
    r.fig(V.f_genes(P), "Genes with an altered result among patients that have a gene-level report.")
    r.p("Interpretation: with so few sequenced patients, frequencies are indicative only; the sequencing column also holds non-genomic reports (IHC, flow cytometry), "
        "which the pipeline deliberately does not count as genomic tests.", bold_lead="")

    r.h1("7. Survival status and follow-up")
    P2 = P.assign(stage_grp=np.where(P.stage == 4, "Stage IV", np.where(P.stage.notna(), "Stage I–III", "Not recorded")))
    p1, _ = V.f_km(P, "os_months", "os_event", title="Overall survival (follow-up-time proxy)", name="km_all_manual")
    r.fig(p1, "Kaplan–Meier estimate for all patients with usable follow-up.", width=5.4)
    p2, pv = V.f_km(P2, "os_months", "os_event", by="stage_grp", order=["Stage I–III", "Stage IV", "Not recorded"], title="Survival by stage group", name="km_stage_manual")
    r.fig(p2, "Survival by stage group (log-rank test).", width=5.4)
    r.p(f"{X['living']} patients are recorded as living, {X['dead']} deceased and {X['unk']} 'not known'. Median follow-up is {X['fu_med']:.1f} months. "
        f"Stage-group log-rank p = {pv:.2f}.", bold_lead="Pattern:")
    r.callout("Caution.", "The sheet has no death date or last-contact date, so survival time is the span between the first dated treatment/imaging entry and the last dated entry. "
              "Treat curves as exploratory until true dates are available.", "FBEEE8")

    r.h1("8. Key patterns and interpretations")
    adv = X["stage4"] / max(X["stage_known"], 1) >= 0.5
    combo = X["chemo"] / X["n"] >= 0.5
    io_gap = X["immuno"] / X["n"] > X["pdl_pct_n"] / X["n"]
    miss = (X["n"] - X["resp_known"]) / X["n"]
    ins = [("Stage mix.", f"{pc(X['stage4'], X['stage_known'])} of staged patients are stage IV" + ("; findings therefore mostly describe advanced disease." if adv else "; the cohort is not dominated by advanced disease.")),
           ("Treatment intensity.", f"Chemotherapy appears in {pc(X['chemo'], X['n'])} of patients" + ("; most patients receive it, usually combined with another modality." if combo else ".")),
           ("Immunotherapy and biomarker testing.", f"{pc(X['immuno'], X['n'])} received immunotherapy while PD-L1 percentage is recorded for {pc(X['pdl_pct_n'], X['n'])}" + ("; testing appears to lag prescribing." if io_gap else ".")),
           ("Response data.", f"{pc(X['n']-X['resp_known'], X['n'])} of patients have no response that can be read from text" + ("; a structured response field would unlock much more analysis." if miss > 0.3 else ".")),
           ("Molecular data.", f"{X['actionable_pts']} patients carry at least one biomarker with an approved-therapy association at gene level (to be verified at variant level).")]
    r.bullets(ins)
    r.h1("9. Limitations and next steps")
    r.bullets(["Single-centre, small sample; subgroup results are hypothesis-generating.",
               "Free-text parsing (stage, response, genes) is rule-based; validate on a hand-reviewed sample.",
               "Add structured death date, last-contact date, RECIST response and variant-level genomics to the data entry form.",
               "Replace proxy survival with true time-to-event once dates are captured."])
    r.save(out)


def etl_dq_short(P):
    return [f"Patient names removed and scrubbed from free text; {int(P.note_scrubbed.str.contains(r'\[NAME\]').sum())} notes contained a name that was masked.",
            "Duplicate patient IDs collapsed to the most complete record.",
            "Lab values with mixed units (e.g. cells/µL, lakh/cmm, ×10³) standardised; implausible values set to missing.",
            "Dates in Excel-serial, dd/mm/yyyy and written-month formats parsed; dates after the record's entry time discarded."]


# ---------------------------------------------------------------- report 2: detailed AI-assisted report
def detailed_report(P, S, M, X, out):
    R = M["scores"]; perf = M["perf"]
    P2 = P.merge(R, on="patient_id")
    P2["stage_grp"] = np.where(P2.stage == 4, "Stage IV", np.where(P2.stage.notna(), "Stage I–III", "Not recorded"))
    r = Rep("Detailed Analysis Report — Precision Oncology Cohort",
            f"Clinical, molecular and AI-prediction insights from {X['n']} patients (generated with Claude)")
    r.h1("Executive summary")
    kpi = pd.DataFrame([["Patients", X["n"], "Living / active", X["living"]],
                        ["Deceased", X["dead"], "High-risk (rule or AI)", X["ai_high"]],
                        ["Stage IV (of staged)", pc(X["stage4"], X["stage_known"]), "Immunotherapy-treated", X["immuno"]],
                        ["Actionable biomarker carriers", X["actionable_pts"], "MSI-high", X["msi_h"]]], columns=["Metric", "Value", "Metric ", "Value "])
    r.table(kpi, widths=[2.0, 1.2, 2.2, 1.2], caption="Cohort at a glance.", font=10)
    r.p("Headline findings", bold_lead="")
    perf_os = perf["os"]; perf_pf = perf["pfs"]; io = perf["io"]
    r.bullets([
        ("Stage and treatment profile.", f"{pc(X['stage4'], X['stage_known'])} of staged patients are stage IV; {pc(X['chemo'], X['n'])} received chemotherapy and {pc(X['immuno'], X['n'])} immunotherapy."),
        ("Imaging response.", f"{X['orr']} of {X['resp_known']} patients with a readable response achieved CR/PR ({pc(X['orr'], X['resp_known'])}); {X['pd_ever']} show progression in at least one report."),
        ("Molecular data is the main opportunity.", f"{X['actionable_pts']} patients carry a biomarker with a known targeted or immunotherapy association; only {X['gen_tested']} have gene-level sequencing on file."),
        ("AI risk stratification is feasible but exploratory.", f"Cross-validated C-index is {perf_os['cindex_cv']:.2f} for survival and {perf_pf['cindex_cv']:.2f} for progression "
         f"(n={perf_os['n']}, {perf_os['events']} deaths): usable for triage and hypothesis generation, not yet for individual forecasts."),
        ("Responder score needs validation.", f"On {io['n']} immunotherapy-treated patients the blended score's leave-one-out AUC was {io.get('auc_model_loo', float('nan')):.2f}; "
         "a larger treated cohort is required before it informs decisions.")])
    r.callout("Data note.", "Figures in this report come from the sample dataset supplied in the same format as the original. Re-run make_reports.py on the real workbook to regenerate every chart, table and sentence.")

    r.pagebreak()
    r.h1("1. Cohort and data readiness")
    r.fig(V.f_completeness(P), "Completeness of analysis-ready fields.", width=5.6,
          read="Teal bars are well populated, amber are partial, red are sparse. Sparse fields limit what the AI layer can learn.")
    r.p("What the pipeline fixed automatically:", bold_lead="")
    r.bullets(etl_dq_short(P))

    r.h1("2. Clinical landscape")
    r.fig(V.f_demographics(P), "Demographics and lifestyle history.")
    r.fig(V.f_site(P), "Site and class distribution.", width=5.8)
    r.fig(V.f_stage(P), "Stage distribution and stage-by-site heat map.")
    top3 = ", ".join(f"{k} ({v})" for k, v in X["site"].head(3).items())
    r.p(f"The three most common site groups are {top3}. Stage is not recorded for {pc(X['n']-X['stage_known'], X['n'])} of patients, "
        "so every stage-stratified result below describes only the staged subset.", bold_lead="Insight:")

    r.h1("3. Treatment and response")
    r.fig(V.f_alluvial(P), "Flow of patients from stage group to treatment pattern to best imaging response.",
          read="Ribbon width is the number of patients. Follow a ribbon from left to right to see how a stage group was treated and how it responded.")
    r.fig(V.f_treatment(P), "Modality utilisation overall and by stage.")
    r.fig(V.f_response(P), "Imaging response overall and by site.")
    r.fig(V.f_swimmer(P, S), "Swimmer plot of the longest-followed patients with dated imaging outcomes.", width=6.0,
          read="Each bar is one patient's follow-up. Markers show imaging status at each dated report; a black cross marks death.")
    r.table(treat_combos(P), widths=[4.2, 1, 1], caption="Most frequent treatment combinations.")

    r.h1("4. Molecular landscape")
    r.fig(V.f_biomarkers(P), "MSI, PD-L1 and tumour immune phenotype.")
    r.fig(V.f_genes(P), "Gene-level alterations.", width=5.8)
    r.fig(V.f_oncoplot(P), "Oncoplot of patients with a notable molecular result.", width=6.4,
          read="Each column is a patient (coloured strip = tumour site); each orange cell is a detected alteration or biomarker.")
    r.fig(V.f_actionable(P), "Actionable biomarkers present at gene level.", width=5.8)
    r.p(f"{X['actionable_pts']} patients carry at least one biomarker with a published targeted or immunotherapy association. "
        "Matching is at gene level only; the exact variant must be confirmed in the sequencing report before any therapy discussion.", bold_lead="Insight:")

    r.h1("5. Outcomes and prognostic factors")
    p1, _ = V.f_km(P, "os_months", "os_event", title="Overall survival (follow-up-time proxy)", name="km_all_det")
    r.fig(p1, "Kaplan–Meier overall survival with 95% confidence band.", width=5.2)
    p2, pv2 = V.f_km(P2, "os_months", "os_event", by="stage_grp", order=["Stage I–III", "Stage IV", "Not recorded"], title="Survival by stage group", name="km_stage_det")
    r.fig(p2, f"Survival by stage group (log-rank p = {pv2:.2f}).", width=5.2)
    p3, pv3 = V.f_km(P2, "os_months", "os_event", by="cancer_class", title="Survival by cancer class", name="km_class_det")
    r.fig(p3, f"Survival by cancer class (log-rank p = {pv3:.2f}).", width=5.2)
    r.fig(V.f_forest(M["tables"]["os_hr"], "Baseline factors and overall survival", "forest_os_det"), "Hazard ratios from the penalised Cox model.", width=5.8,
          read="Dots right of the dashed line mean higher hazard of death per standard deviation (or for 'yes'); bars that cross 1 are not distinguishable from no effect.")
    r.fig(V.f_labs(P), "Baseline laboratory values by survival status.")
    r.fig(V.f_labcorr(P), "Correlation between labs, age and stage.", width=5.4)

    r.pagebreak()
    r.h1("6. AI prediction layer")
    r.p("The layer is built for a small cohort: ridge-penalised Cox models for survival and progression, a transparent biomarker-prior plus regularised logistic model "
        "for immunotherapy response, rule-based precision-therapy and trial matching, and a similar-patient lookup. All models use baseline information only.")
    r.fig(V.f_model_perf(perf), "Cross-validated vs apparent concordance for the two time-to-event models.", width=5.4,
          read="Dots show repeated 5-fold cross-validation (honest estimate); diamonds show performance on the training data (optimistic). A value of 0.5 equals chance.")
    r.fig(V.f_risk_tiers(P, R), "Distribution of AI risk tiers, progression risk and responder score.")
    pk, pk_p = V.f_km(P2, "os_months", "os_event", by="risk_tier", order=["Low", "Medium", "High"], title="Survival by AI risk tier (in-sample)", name="km_tier_det")
    r.fig(pk, f"Survival separated by AI risk tier (log-rank p = {pk_p:.3f}; in-sample, therefore optimistic).", width=5.2)
    r.fig(V.f_io(P, perf), "Immunotherapy benefit by biomarker group and responder-score validation.")
    hi = P2[P2.ai_high_risk & (P2.os_status != "Deceased")].sort_values("composite_risk", ascending=False).head(10)
    r.table(pd.DataFrame({"Patient": hi.patient_id, "Site": hi.site_group, "Stage": hi.stage.map({1: "I", 2: "II", 3: "III", 4: "IV"}).fillna("n/r"),
                          "6-mo progression risk": hi.prog_risk_6m.map(lambda v: f"{100*v:.0f}%"), "Composite risk": hi.composite_risk.round(0).astype(int),
                          "Responder score": hi.io_score.round(0).astype(int)}),
            widths=[1, 1.4, .7, 1.4, 1.1, 1.1], caption="Ten highest-risk living patients (watch-list).")
    r.h2("Precision therapy matching")
    r.p("Biomarker-to-therapy associations are drawn from a curated illustrative knowledge base and matched at gene level. Treat as candidates for tumour-board discussion.")
    r.table(therapy_summary(P), widths=[1.1, 3.6, .9, 1.0], caption="Candidate therapies by biomarker (verify against OncoKB / NCCN before use).", font=8.5)
    r.h2("Clinical trial matching")
    tdf, tn = trial_summary(P)
    r.p(f"{tn} living patients match at least one placeholder trial on site, stage, age and biomarker rules. In production, trials are pulled live from ClinicalTrials.gov / CTRI.")
    r.table(tdf.head(8), widths=[1.3, 4.0, 1.2], caption="Placeholder trials ranked by number of matching patients.", font=8.5)

    r.h1("7. Synthesis and recommendations")
    r.bullets([
        ("Capture the missing outcome fields.", "Add death date, last-contact date and a structured RECIST response. This single change would convert the proxy outcomes into true survival and response endpoints."),
        ("Expand molecular testing.", f"Only {X['gen_tested']} patients have gene-level results; broader panel testing would raise the number of patients with an actionable match."),
        ("Use the risk tiers for triage, not prognosis.", "Prioritise high-tier patients for review while collecting more events to firm up the models."),
        ("Validate the responder score prospectively.", "Record PD-L1 %, MSI and TMB before every immunotherapy start so benefit can be modelled properly."),
        ("Verify automated extraction.", "Spot-check stage, response and gene calls against source text for a random 20 patients; record accuracy in the methods appendix.")])
    r.h1("Appendix: methods and definitions")
    r.table(pd.DataFrame([
        ["Active patient", "Overall survival status = Living"], ["Deceased", "Overall survival status = Deceased; 'Not known' is treated as censored"],
        ["High-risk (rule)", "Living and (stage IV, M1 or imaging progression)"], ["High-risk (AI)", "Top third of composite risk (mean of survival and progression risk percentiles)"],
        ["Survival time (proxy)", "First dated treatment/imaging entry to last dated entry"], ["Progression", "Imaging text describing progression, new lesions or relapse (negations excluded)"],
        ["Responder score", "50% biomarker prior (PD-L1, MSI-H, TMB-H, hot tumour, albumin) + 50% regularised logistic model"],
        ["Validation", "Repeated 5-fold CV (Cox); leave-one-out AUC with bootstrap CI (responder score)"]], columns=["Term", "Definition"]), widths=[1.7, 5.0], font=9)
    r.save(out)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data.xlsx"
    o = etl.load_and_clean(src); P, S = o["patients"], o["scans"]
    M = models.run_models(P); X = compute(P, S, M)
    manual_report(P, S, M, X, "Manual_Analysis_Report.docx")
    detailed_report(P, S, M, X, "Claude_Detailed_Analysis_Report.docx")
    print("reports written")
