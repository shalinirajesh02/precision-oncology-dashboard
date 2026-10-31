"""
knowledge.py -- rule-based knowledge for precision-therapy and trial matching.

IMPORTANT
  * THERAPY_RULES is an ILLUSTRATIVE, hand-written summary of widely published biomarker-drug
    associations. It is matched at GENE level only (variant-level detail such as EGFR exon 19 vs
    T790M, or BRAF V600E vs other, is not captured). It MUST be verified against OncoKB / NCCN /
    ESMO ESCAT and by the treating oncologist before any clinical use.
  * SAMPLE_TRIALS are PLACEHOLDERS to demonstrate the matching engine. They are not real trials and
    have no NCT numbers. In production, replace with a live pull from ClinicalTrials.gov (API v2) and
    CTRI (India) -- see fetch_ctgov() below, which needs internet access (run it on your machine).
"""
import re
import pandas as pd

# gene/biomarker -> (cancer site groups where strongest, therapies, evidence note)
THERAPY_RULES = [
    dict(marker="EGFR", sites=["Lung"], drugs=["Osimertinib", "Erlotinib", "Gefitinib", "Afatinib"],
         note="EGFR-sensitising mutation in NSCLC: EGFR TKIs (standard of care)."),
    dict(marker="ALK", sites=["Lung"], drugs=["Alectinib", "Lorlatinib", "Brigatinib"], note="ALK fusion in NSCLC."),
    dict(marker="ROS1", sites=["Lung"], drugs=["Crizotinib", "Entrectinib", "Repotrectinib"], note="ROS1 fusion in NSCLC."),
    dict(marker="BRAF", sites=["Lung", "Colorectal", "Other"], drugs=["Dabrafenib + Trametinib", "Encorafenib + Cetuximab (colorectal)"],
         note="BRAF V600E (variant must be confirmed)."),
    dict(marker="KRAS", sites=["Lung", "Colorectal"], drugs=["Sotorasib / Adagrasib (only if KRAS G12C)"],
         note="KRAS G12C-specific; other KRAS variants predict resistance to anti-EGFR antibodies in colorectal."),
    dict(marker="HER2", sites=["Breast", "Upper GI", "Colorectal", "Gynecologic"],
         drugs=["Trastuzumab", "Pertuzumab", "Trastuzumab deruxtecan"], note="HER2-positive / amplified disease."),
    dict(marker="MET", sites=["Lung"], drugs=["Capmatinib", "Tepotinib"], note="MET exon-14 skipping (confirm variant)."),
    dict(marker="RET", sites=["Lung", "Other"], drugs=["Selpercatinib", "Pralsetinib"], note="RET fusion."),
    dict(marker="NTRK", sites=["Any"], drugs=["Larotrectinib", "Entrectinib"], note="NTRK fusion, tumour-agnostic."),
    dict(marker="BRCA1", sites=["Breast", "Gynecologic", "Prostate", "Hepatobiliary"], drugs=["Olaparib", "Niraparib", "Talazoparib"],
         note="Germline/somatic BRCA alteration: PARP inhibitors (site-specific indications)."),
    dict(marker="BRCA2", sites=["Breast", "Gynecologic", "Prostate", "Hepatobiliary"], drugs=["Olaparib", "Niraparib", "Talazoparib"],
         note="Germline/somatic BRCA alteration: PARP inhibitors (site-specific indications)."),
    dict(marker="PIK3CA", sites=["Breast"], drugs=["Alpelisib + Fulvestrant"], note="HR+/HER2- breast cancer with PIK3CA mutation."),
    dict(marker="IDH1", sites=["Hepatobiliary", "CNS", "Hematologic"], drugs=["Ivosidenib / Vorasidenib (tumour-specific)"], note="IDH1 mutation."),
    dict(marker="KIT", sites=["Upper GI", "Bone & Soft tissue", "Other"], drugs=["Imatinib"], note="KIT-mutant GIST."),
    dict(marker="PDGFRA", sites=["Upper GI", "Bone & Soft tissue", "Other"], drugs=["Imatinib / Avapritinib (D842V)"], note="PDGFRA-mutant GIST."),
    dict(marker="MSI-H", sites=["Any"], drugs=["Pembrolizumab", "Nivolumab +/- Ipilimumab", "Dostarlimab"], note="MSI-high / dMMR: checkpoint inhibitors (tumour-agnostic)."),
    dict(marker="TMB-H", sites=["Any"], drugs=["Pembrolizumab"], note="TMB >= 10 mut/Mb: pembrolizumab (tumour-agnostic)."),
    dict(marker="PD-L1 high", sites=["Lung"], drugs=["Pembrolizumab", "Atezolizumab", "Cemiplimab"], note="PD-L1 TPS >= 50% in NSCLC: checkpoint-inhibitor monotherapy."),
    dict(marker="PD-L1 positive", sites=["Any"], drugs=["Checkpoint inhibitor (site-specific; CPS/TPS cut-offs vary)"], note="PD-L1 expressing tumours."),
]
ACTIONABLE = {r["marker"] for r in THERAPY_RULES}


def patient_markers(row):
    """List of biomarker labels present for a patient (from the cleaned patient row)."""
    m = []
    for g in [x for x in str(row.get("genes_altered", "")).split(";") if x]:
        m.append(g)
    if row.get("msi") == "MSI-High":
        m.append("MSI-H")
    if pd.notna(row.get("tmb")) and row["tmb"] >= 10:
        m.append("TMB-H")
    if pd.notna(row.get("pdl1_pct")):
        if row["pdl1_pct"] >= 50:
            m.append("PD-L1 high")
        if row["pdl1_pct"] >= 1:
            m.append("PD-L1 positive")
    elif row.get("pdl1_status") == "Positive":
        m.append("PD-L1 positive")
    return m


def match_therapies(row):
    """Return list of dicts: marker, drugs, note, site_fit ('on-label site' / 'off-site: investigational')."""
    out = []
    for mk in patient_markers(row):
        for rule in THERAPY_RULES:
            if rule["marker"] != mk:
                continue
            fit = "Any" in rule["sites"] or row.get("site_group") in rule["sites"]
            out.append(dict(marker=mk, drugs=", ".join(rule["drugs"]), note=rule["note"],
                            site_fit="Site-appropriate" if fit else "Other tumour site: discuss at tumour board"))
    return out


# ------------------------------------------------------------------------------------
# Sample (placeholder) trial registry + matching engine
# ------------------------------------------------------------------------------------
SAMPLE_TRIALS = pd.DataFrame([
    dict(trial_id="SAMPLE-001", title="Checkpoint inhibitor + chemo, advanced lung cancer", sites="Lung", stage_min=3, stage_max=4,
         markers="", needs_io_naive=True, age_min=18, age_max=80, max_prior_lines=1, status="Recruiting"),
    dict(trial_id="SAMPLE-002", title="Targeted therapy for EGFR/ALK-altered lung cancer", sites="Lung", stage_min=3, stage_max=4,
         markers="EGFR|ALK|ROS1", needs_io_naive=False, age_min=18, age_max=85, max_prior_lines=3, status="Recruiting"),
    dict(trial_id="SAMPLE-003", title="HER2-directed therapy, HER2-positive solid tumours", sites="Breast|Upper GI|Colorectal|Gynecologic", stage_min=3, stage_max=4,
         markers="HER2", needs_io_naive=False, age_min=18, age_max=80, max_prior_lines=2, status="Recruiting"),
    dict(trial_id="SAMPLE-004", title="Immunotherapy for MSI-high / TMB-high tumours (any site)", sites="Any", stage_min=3, stage_max=4,
         markers="MSI-H|TMB-H", needs_io_naive=True, age_min=18, age_max=90, max_prior_lines=3, status="Recruiting"),
    dict(trial_id="SAMPLE-005", title="PARP inhibitor for BRCA-altered tumours", sites="Breast|Gynecologic|Prostate|Hepatobiliary", stage_min=2, stage_max=4,
         markers="BRCA1|BRCA2", needs_io_naive=False, age_min=18, age_max=80, max_prior_lines=2, status="Recruiting"),
    dict(trial_id="SAMPLE-006", title="First-line therapy, metastatic colorectal cancer", sites="Colorectal", stage_min=4, stage_max=4,
         markers="", needs_io_naive=False, age_min=18, age_max=80, max_prior_lines=0, status="Recruiting"),
    dict(trial_id="SAMPLE-007", title="Novel agents in relapsed/refractory myeloma or lymphoma", sites="Hematologic", stage_min=1, stage_max=4,
         markers="", needs_io_naive=False, age_min=18, age_max=85, max_prior_lines=4, status="Recruiting"),
    dict(trial_id="SAMPLE-008", title="Hormone-receptor-positive breast cancer, post-endocrine progression", sites="Breast", stage_min=3, stage_max=4,
         markers="PIK3CA", needs_io_naive=False, age_min=18, age_max=80, max_prior_lines=2, status="Recruiting"),
    dict(trial_id="SAMPLE-009", title="Supportive care / quality-of-life study, advanced cancer", sites="Any", stage_min=3, stage_max=4,
         markers="", needs_io_naive=False, age_min=18, age_max=95, max_prior_lines=9, status="Recruiting"),
])


def match_trials(row, trials=SAMPLE_TRIALS, min_score=0.5):
    """Score each trial 0-1 against a patient. Hard failures (site, age, stage) -> excluded.
    Biomarker match adds weight; IO-naive requirement is checked from treatment history."""
    pm = set(patient_markers(row))
    res = []
    stage = row.get("stage")
    age = row.get("age")
    for _, t in trials.iterrows():
        reasons, score = [], 0.0
        sites = t["sites"].split("|")
        if "Any" not in sites and row.get("site_group") not in sites:
            continue
        if pd.notna(age) and not (t["age_min"] <= age <= t["age_max"]):
            continue
        if pd.notna(stage) and not (t["stage_min"] <= stage <= t["stage_max"]):
            continue
        score += 0.4
        reasons.append("tumour site fits" if "Any" not in sites else "site-agnostic")
        score += 0.2 if pd.notna(stage) else 0.0
        reasons.append("stage fits" if pd.notna(stage) else "stage unknown (verify)")
        want = [m for m in t["markers"].split("|") if m]
        if want:
            hit = pm & set(want)
            if not hit:
                continue
            score += 0.3
            reasons.append("biomarker match: " + ", ".join(sorted(hit)))
        else:
            score += 0.1
        if t["needs_io_naive"] and row.get("immuno"):
            score -= 0.3
            reasons.append("prior immunotherapy: may be ineligible")
        if row.get("os_status") == "Deceased":
            continue
        if score >= min_score:
            res.append(dict(trial_id=t["trial_id"], title=t["title"], match_score=round(min(score, 1.0), 2), why="; ".join(reasons)))
    return sorted(res, key=lambda d: -d["match_score"])


def fetch_ctgov(condition, status="RECRUITING", page_size=20):
    """Pull recruiting trials from ClinicalTrials.gov API v2. Requires internet (run on your machine)."""
    import requests
    r = requests.get("https://clinicaltrials.gov/api/v2/studies",
                     params={"query.cond": condition, "filter.overallStatus": status, "pageSize": page_size}, timeout=30)
    r.raise_for_status()
    rows = []
    for st in r.json().get("studies", []):
        p = st["protocolSection"]
        rows.append(dict(trial_id=p["identificationModule"]["nctId"], title=p["identificationModule"].get("briefTitle", ""),
                         criteria=p.get("eligibilityModule", {}).get("eligibilityCriteria", "")))
    return pd.DataFrame(rows)
