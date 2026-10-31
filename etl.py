"""
etl.py -- loads the 3-sheet oncology workbook and builds clean analysis tables.

Outputs (all PHI-minimised: no patient names, names scrubbed from free text):
    patients : one row per patient (analysis table)
    scans    : one row per dated imaging / follow-up event with a classified status
    dq       : data-quality log (what was fixed, dropped or assumed)
"""
import io
import re
import datetime as dt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# Configuration -- the only place that depends on your exact header text
# ----------------------------------------------------------------------------------
C = dict(
    ts="Timestamp", pid="Patient ID", sex="Gender", age="Age (in years)",
    alc="Alcohol History", smk="Smoking history", cls="Cancer type",
    site="Please specify cancer site", lat="Cancer laterality",
    hist="Pease specify cancer histology", ihc="Immunohistochemistry (IHC) Findings",
    stage="Cancer stage", T="Primary tumor", M="Metastasis", N="Nodes",
    tx="Treatment details", allied="Allied Services", fu="Follow-up notes",
    drugs="Drug Name, Dose and Regimens", cbegin="Course Begin Date", cend="Course End Date",
    chemo_c="Chemotherapy Comments", cyc1="Cycle 1 investigations date",
    msi="MSI Status", pdl="PDL-1 Expression", pdlpct="How much percentage, If yes (PDL-1)",
    tumtype="Tumor Type", gen="Genomic Sequencing", omd="Other Molecular Data",
    hx="significant Past history", os="Overall Survival Status", rad="Radiogical Findings",
    extra="Column 6", pname="Patient Name",
)
LAB_COLS = {
    "wbc": "White Blood Cells (K/uL)", "anc": "Absolute Neutrophil Count (K/uL)",
    "hb": "Hemoglobin (g/dL)", "hct": "Hematocrit or PCV (%)", "plt": "Platelets (K/uL)",
    "creat": "Serum Creatinine", "bun": "Blood Urea Nitrogen (BUN)", "uric": "Serum Uric Acid",
    "ast": "AST (SGOT)", "alt": "ALT (SGPT)", "alp": "Alkaline Phosphatase (ALP)",
    "bili": "Total Bilirubin", "dbili": "Direct Bilirubin", "tprot": "Total Protein",
    "alb": "Serum Albumin", "glu": "Random Blood Glucose levels", "na": "Sodium (Na⁺)",
    "k": "Potassium (K⁺)", "ca": "Calcium (Ca²⁺)", "mg": "Magnesium (Mg²⁺)", "cl": "Chloride (Cl⁻)",
}
LAB_RANGE = dict(wbc=(0.1, 150), anc=(0.05, 100), hb=(3, 22), hct=(10, 70), plt=(10, 2000),
                 creat=(0.2, 20), bun=(2, 200), uric=(0.5, 20), ast=(1, 5000), alt=(1, 5000),
                 alp=(5, 5000), bili=(0.05, 40), dbili=(0.0, 30), tprot=(2, 12), alb=(1, 6.5),
                 glu=(30, 900), na=(100, 180), k=(1.5, 8.5), ca=(5, 15), mg=(0.5, 5), cl=(70, 130))
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
TODAY = pd.Timestamp(dt.date.today())

# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------
def norm_cols(df):
    df = df.copy()
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    return df


def s(v):
    """cell -> clean string or ''"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).strip()


def first_num(txt):
    m = re.search(r"[-+]?\d[\d,]*\.?\d*", txt or "")
    if not m:
        return np.nan
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return np.nan


def to_date(v):
    """Excel serial / datetime / text -> Timestamp (or NaT)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pd.NaT
    if isinstance(v, (dt.datetime, dt.date, pd.Timestamp)):
        return pd.Timestamp(v)
    if isinstance(v, (int, float)) and 20000 < float(v) < 70000:
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v))
    ds = find_dates(str(v))
    return ds[0][0] if ds else pd.NaT


def _mk(y, mth, d):
    try:
        if y < 100:
            y += 2000
        return pd.Timestamp(year=y, month=mth, day=d)
    except (ValueError, OverflowError):
        return None


def find_dates(text):
    """Return [(Timestamp, start_index)] for the date formats seen in clinical notes."""
    out = []
    t = text or ""
    for m in re.finditer(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})(?!\d)", t):
        d_, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ts = _mk(y, mo, d_) if mo <= 12 else None
        if ts is not None:
            out.append((ts, m.start()))
    for m in re.finditer(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", t):
        mo = MONTHS.get(m.group(1)[:3].lower())
        if mo:
            ts = _mk(int(m.group(3)), mo, int(m.group(2)))
            if ts is not None:
                out.append((ts, m.start()))
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})", t):
        mo = MONTHS.get(m.group(2)[:3].lower())
        if mo:
            ts = _mk(int(m.group(3)), mo, int(m.group(1)))
            if ts is not None:
                out.append((ts, m.start()))
    out.sort(key=lambda x: x[1])
    # drop duplicates that start at nearly the same index
    dedup = []
    for ts, i in out:
        if not dedup or i - dedup[-1][1] > 3:
            dedup.append((ts, i))
    return dedup


# ----------------------------------------------------------------------------------
# PHI scrubbing for free text
# ----------------------------------------------------------------------------------
TITLE_RE = re.compile(r"\b(?:MR|MRS|MS|MISS|SHRI|SMT|SRI|DR|MASTER|BABY)\.?\s+(?:[A-Z][A-Za-z]+\s?){1,3}", re.I)


def make_scrubber(names):
    toks = set()
    for n in names:
        for part in re.split(r"[\s,.]+", s(n)):
            if len(part) >= 3:
                toks.add(re.escape(part))
    name_re = re.compile(r"\b(" + "|".join(sorted(toks, key=len, reverse=True)) + r")\b", re.I) if toks else None

    def scrub(text):
        t = s(text)
        if not t:
            return t
        t = TITLE_RE.sub("[NAME] ", t)
        if name_re:
            t = name_re.sub("[NAME]", t)
        return t
    return scrub


# ----------------------------------------------------------------------------------
# Lab parsing
# ----------------------------------------------------------------------------------
def parse_lab(raw, kind):
    t = s(raw).lower()
    if not t:
        return np.nan
    v = first_num(t)
    if np.isnan(v):
        return np.nan
    sci = bool(re.search(r"[x×]\s*10|10\s*[\^*]|thou|\bk\s*/|k/u", t))
    if kind in ("wbc", "anc"):
        if not sci and v >= 100:
            v = v / 1000.0
    elif kind == "plt":
        if re.search(r"\dl\s*/|\d\s*l\s*/|lakh", t) or re.search(r"\d\s*l\b", t):
            v = v * 100.0
        elif not sci and v >= 1000:
            v = v / 1000.0
    elif kind == "hct":
        if v < 1.5:
            v = v * 100.0
    elif kind in ("alb", "tprot"):
        if v > 10 and kind == "alb":
            v = v / 10.0
        if v > 20 and kind == "tprot":
            v = v / 10.0
    lo, hi = LAB_RANGE[kind]
    return v if lo <= v <= hi else np.nan


# ----------------------------------------------------------------------------------
# Site / treatment / drug dictionaries
# ----------------------------------------------------------------------------------
SITE_GROUP = {
    "breast": "Breast", "lung": "Lung", "colon/rectum": "Colorectal", "liver": "Hepatobiliary",
    "gall bladder/biliary tree": "Hepatobiliary", "pancreas": "Hepatobiliary",
    "multiple myeloma": "Hematologic", "hodgkin lymphoma": "Hematologic",
    "non-hodgkin lymphoma": "Hematologic", "prostate": "Prostate", "ovary": "Gynecologic",
    "vagina/vulva": "Gynecologic", "stomach": "Upper GI", "esophagus": "Upper GI",
    "duodenum": "Upper GI", "brain": "CNS", "kidney": "Genitourinary", "salivary": "Head & Neck",
    "tongue": "Head & Neck", "bone": "Bone & Soft tissue",
}


def site_group(raw):
    parts = [p.strip() for p in s(raw).lower().split(",") if p.strip()]
    for p in parts:
        if p in SITE_GROUP:
            return SITE_GROUP[p]
    return "Other" if parts else "Unknown"


DRUG_CLASS = {
    "ici": ["pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "toripalimab", "ipilimumab",
            "tislelizumab", "camrelizumab", "sintilimab", "avelumab", "cemiplimab", "dostarlimab",
            "keytruda", "opdivo", "tecentriq", "imfinzi", "penpulimab"],
    "targeted": ["osimertinib", "erlotinib", "gefitinib", "afatinib", "crizotinib", "alectinib",
                 "lorlatinib", "trastuzumab", "pertuzumab", "cetuximab", "panitumumab", "imatinib",
                 "bevacizumab", "ramucirumab", "olaparib", "niraparib", "palbociclib", "ribociclib",
                 "abemaciclib", "sorafenib", "sunitinib", "lenvatinib", "regorafenib", "dabrafenib",
                 "trametinib", "bortezomib", "carfilzomib", "daratumumab", "rituximab", "ibrutinib",
                 "pomalidomide", "lenalidomide", "thalidomide", "everolimus"],
    "hormonal": ["letrozole", "anastrozole", "exemestane", "tamoxifen", "fulvestrant", "goserelin",
                 "leuprolide", "degarelix", "degardix", "enzalutamide", "x-tandi", "xtandi",
                 "abiraterone", "bicalutamide", "megestrol"],
    "chemo": ["carboplatin", "cisplatin", "oxaliplatin", "oxaliplatinum", "paclitaxel", "tortaxel",
              "docetaxel", "gemcitabine", "pemetrexed", "pemtrexed", "5fu", "5-fu", "fluorouracil",
              "capecitabine", "irinotecan", "onivyde", "cyclophosphamide", "endoxan", "bendamustine",
              "doxorubicin", "epirubicin", "etoposide", "vincristine", "methotrexate", "melphalan",
              "cytarabine", "temozolomide", "leucovorin", "eribulin", "ifosfamide", "dacarbazine",
              "chlorambucil", "vinorelbine", "mitomycin"],
}


def drug_flags(text):
    t = text.lower()
    f = {k: any(d in t for d in v) for k, v in DRUG_CLASS.items()}
    f["n_drugs"] = sum(1 for v in DRUG_CLASS.values() for d in v if d in t)
    return f


# ----------------------------------------------------------------------------------
# Imaging / follow-up event classification
# ----------------------------------------------------------------------------------
STRONG_PROG = [r"progress", r"relapse", r"new (?:lesion|nodule|metast|deposit|foci|focus|site|mass|lymph)",
               r"interval (?:increase|development|appearance|enlargement)", r"further increase", r"worsen"]
WEAK_PROG = [r"increase in (?:size|number|extent|metabolic|tracer|suv)", r"enlarg", r"recurren"]
STRONG_RESP = [r"partial (?:metabolic )?(?:response|remission)", r"good (?:response|control)", r"remission",
               r"complete (?:metabolic )?(?:response|resolution|remission)", r"(?:marked|significant) (?:reduction|regression)"]
WEAK_RESP = [r"(?:reduction|decrease|regression|resolution|resolved|shrink)"]
CR = [r"complete (?:metabolic )?(?:response|resolution|remission)",
      r"no (?:residual|evidence of) (?:disease|tumou?r|malignan|metabolically active)", r"resolution of (?:all|the)"]
STAB = [r"no significant (?:interval )?change", r"stable", r"unchanged", r"no (?:interval )?change", r"static"]
COMPARE = re.compile(r"interval|previous|prior|compared|comparison|as compared|since|follow[- ]?up|post[- ]?(?:chemo|treatment|therapy|op)|response|baseline")
NEG_BEFORE = re.compile(r"(?:\bno\b|\bwithout\b|absence of|\bnot\b|\bnor\b|\bnil\b|negative for)[^.]{0,30}$")


def _count(patterns, text):
    n = 0
    for p in patterns:
        for m in re.finditer(p, text):
            if not NEG_BEFORE.search(text[max(0, m.start() - 35):m.start()]):
                n += 1
    return n


def classify_segment(seg):
    """Rule-based status of one dated report. Comparison language is required for weak cues,
    so baseline staging scans ('enlarged node suggestive of metastasis') are not called progression."""
    t = seg.lower()
    cmp_ = bool(COMPARE.search(t))
    p = _count(STRONG_PROG, t) + (_count(WEAK_PROG, t) if cmp_ else 0)
    r = _count(STRONG_RESP, t) + (_count(WEAK_RESP, t) if cmp_ else 0)
    st = _count(STAB, t)
    cr = _count(CR, t)
    if p > 0 and p >= r:
        return "PD"
    if cr > 0 and p == 0:
        return "CR"
    if r > 0 and r > p:
        return "PR"
    if st > 0:
        return "SD"
    return "UNK"


def extract_events(text, pid, source, entry_ts):
    ev = []
    ds = find_dates(text)
    if not ds:
        return ev
    for k, (ts, idx) in enumerate(ds):
        end = ds[k + 1][1] if k + 1 < len(ds) else len(text)
        seg = text[idx:end]
        ev.append(dict(patient_id=pid, date=ts, source=source, status=classify_segment(seg)))
    return ev


# ----------------------------------------------------------------------------------
# Genomics
# ----------------------------------------------------------------------------------
GENES = {
    "EGFR": r"\bEGFR\b", "KRAS": r"\bKRAS\b", "NRAS": r"\bNRAS\b", "BRAF": r"\bBRAF\b", "ALK": r"\bALK\b",
    "ROS1": r"\bROS1\b", "MET": r"\bMET\b", "RET": r"\bRET\b", "NTRK": r"\bNTRK[123]?\b",
    "HER2": r"\b(?:HER2|HER-2|ERBB2)\b", "PIK3CA": r"\bPIK3CA\b", "PTEN": r"\bPTEN\b", "TP53": r"\b(?:TP53|P53)\b",
    "BRCA1": r"\bBRCA1\b", "BRCA2": r"\bBRCA2\b", "IDH1": r"\bIDH1\b", "IDH2": r"\bIDH2\b",
    "KIT": r"\bKIT\b", "PDGFRA": r"\bPDGFRA\b", "ATM": r"\bATM\b", "APC": r"\bAPC\b", "FGFR": r"\bFGFR[123]?\b",
    "CDKN2A": r"\bCDKN2A\b", "ARID1A": r"\bARID1A\b", "MDM4": r"\bMDM4\b", "SMAD4": r"\bSMAD4\b",
}
NEG_RES = re.compile(r"none detected|not detected|no (?:pathogenic |significant )?(?:mutation|variant|alteration)|wild[- ]?type|negative|absent|not amplified|no amplification", re.I)
VUS_RES = re.compile(r"uncertain significance|\bvus\b", re.I)
ALT_RES = re.compile(r"p\.[A-Za-z]|c\.[\d\-]|vaf|amplif|exon|deletion|fusion|copy number|mutation|variant|positive|\+ve|\b3\+|loss|rearrang", re.I)


def parse_genomics(text):
    """gene -> 'altered' | 'vus' | 'none'. Heuristic, window-based; verify with spot checks."""
    res = {}
    if not text:
        return res
    hits = []
    for g, pat in GENES.items():
        for m in re.finditer(pat, text):
            hits.append((m.start(), m.end(), g))
    hits.sort()
    for i, (st, en, g) in enumerate(hits):
        nxt = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        win = text[en:min(nxt, en + 160)]
        if NEG_RES.search(win):
            status = "none"
        elif VUS_RES.search(win):
            status = "vus"
        elif ALT_RES.search(win):
            status = "altered"
        else:
            continue
        rank = {"altered": 3, "vus": 2, "none": 1}
        if g not in res or rank[status] > rank[res[g]]:
            res[g] = status
    return res


def parse_tmb(*texts):
    for t in texts:
        m = re.search(r"(?:\bTMB\b|tumou?r mutational burden)[^0-9]{0,40}(\d+(?:\.\d+)?)", t or "", re.I)
        if m:
            v = float(m.group(1))
            if 0 <= v <= 500:
                return v
    return np.nan


def parse_pdl1_pct(raw, ihc, omd):
    r = s(raw).lower()
    if r:
        if "<1" in r or "minimal" in r:
            return 0.5
        if "not mentioned" in r:
            r = ""
        else:
            m = re.search(r"\d+(?:\.\d+)?", r)
            if m:
                v = float(m.group(0))
                if "%" not in r and v <= 1.0:
                    v = v * 100.0       # Excel stored 20% as 0.2
                return v if 0 <= v <= 100 else np.nan
    for t in (ihc, omd):
        m = re.search(r"(?:CPS|TPS|tumou?r proportion score|combined positive score)[^0-9]{0,40}(\d+(?:\.\d+)?)", t or "", re.I)
        if m:
            v = float(m.group(1))
            return v if 0 <= v <= 100 else np.nan
    return np.nan


# ----------------------------------------------------------------------------------
# Normalisers for categorical clinical fields
# ----------------------------------------------------------------------------------
def stage_num(v):
    t = s(v).lower()
    m = re.search(r"stage\s*(iv|iii|ii|i)\b", t)
    return {"i": 1, "ii": 2, "iii": 3, "iv": 4}.get(m.group(1)) if m else np.nan


def tnm_num(v, letter):
    t = s(v)
    m = re.match(rf"{letter}([0-4])", t)
    return int(m.group(1)) if m else np.nan


def yn(v):
    t = s(v).lower()
    if t.startswith("y"):
        return "Yes"
    if t.startswith("n") and "known" not in t:
        return "No"
    return "Unknown"


# ----------------------------------------------------------------------------------
# Main loader
# ----------------------------------------------------------------------------------
def read_workbook(src):
    """src: path, URL (Google Sheets export?format=xlsx) or bytes/file-like."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    return pd.read_excel(src, sheet_name=None, dtype=object)


def load_and_clean(src):
    sheets = {k: norm_cols(v) for k, v in read_workbook(src).items()}
    dq = []
    master = max((v for v in sheets.values() if C["cls"] in v.columns), key=len)
    names_sheet = next((v for v in sheets.values() if C["pname"] in v.columns), None)
    gen_sheet = names_sheet if names_sheet is not None else None

    scrub = make_scrubber(names_sheet[C["pname"]].dropna().tolist() if names_sheet is not None else [])
    if names_sheet is not None:
        dq.append("Patient Name column found and used ONLY to scrub names from free text; it is not kept.")

    df = master.copy()
    # ---- duplicates ----------------------------------------------------------------
    df["_ts"] = pd.to_datetime(df[C["ts"]], errors="coerce")
    df["_filled"] = df.notna().sum(axis=1)
    df[C["pid"]] = df[C["pid"]].astype(str).str.strip()
    dups = df[C["pid"]].duplicated(keep=False)
    if dups.any():
        ids = sorted(df.loc[dups, C["pid"]].unique())
        dq.append(f"{len(ids)} Patient ID(s) appear on more than one row ({', '.join(ids)}); kept the most complete / latest row.")
    df = df.sort_values(["_filled", "_ts"]).drop_duplicates(C["pid"], keep="last")

    # genomic text: master first, sheet 3 as fallback
    gen_fallback = {}
    if gen_sheet is not None and C["gen"] in gen_sheet.columns:
        gen_fallback = dict(zip(gen_sheet[C["pid"]].astype(str).str.strip(), gen_sheet[C["gen"]]))

    rows, events = [], []
    n_future = 0
    for _, r in df.iterrows():
        pid = r[C["pid"]]
        entry = pd.Timestamp(r["_ts"]) if pd.notna(r["_ts"]) else TODAY
        gen_txt = s(r.get(C["gen"])) or s(gen_fallback.get(pid))
        rad_txt = s(r.get(C["rad"]))
        ex_txt = s(r.get(C["extra"]))
        fu_txt = s(r.get(C["fu"]))
        hx_txt = s(r.get(C["hx"]))
        ihc_txt, omd_txt = s(r.get(C["ihc"])), s(r.get(C["omd"]))

        # ---- imaging / follow-up events
        evs = []
        evs += extract_events(rad_txt, pid, "radiology", entry)
        evs += extract_events(ex_txt, pid, "radiology", entry)
        evs += extract_events(fu_txt, pid, "followup", entry)
        good = []
        for e in evs:
            if e["date"] > entry + pd.Timedelta(days=1) or e["date"] < pd.Timestamp("1995-01-01"):
                n_future += 1
                continue
            good.append(e)
        events += good
        scan_all = [e for e in good if e["source"] == "radiology"] or good
        scan_all.sort(key=lambda e: e["date"])
        cbegin = to_date(r.get(C["cbegin"]))
        HISTORY_DAYS = 548        # imaging older than ~18 months before the course start is treated as prior history
        if pd.notna(cbegin) and cbegin <= entry + pd.Timedelta(days=1):
            scan_ev = [e for e in scan_all if e["date"] >= cbegin - pd.Timedelta(days=HISTORY_DAYS)]
            t0 = min([cbegin] + [e["date"] for e in scan_ev])
        else:
            scan_ev = scan_all
            t0 = scan_all[0]["date"] if scan_all else pd.NaT
        stat = [e["status"] for e in scan_ev if e["status"] != "UNK"]
        dates_all = [e["date"] for e in good]
        for c in (C["cbegin"], C["cend"], C["cyc1"]):
            d_ = to_date(r.get(c))
            if pd.notna(d_) and d_ <= entry + pd.Timedelta(days=1):
                dates_all.append(d_)
        for t in (hx_txt, s(r.get(C["chemo_c"])), s(r.get(C["chemo_c"]))):
            for d_, _ in find_dates(t):
                if d_ <= entry and d_ >= pd.Timestamp("1995-01-01"):
                    dates_all.append(d_)

        dates_all = [d_ for d_ in dates_all if pd.isna(t0) or d_ >= t0]
        last = max(dates_all) if dates_all else pd.NaT
        pd_dates = [e["date"] for e in scan_ev if e["status"] == "PD"]
        order = {"CR": 0, "PR": 1, "SD": 2, "PD": 3}
        best = min(stat, key=lambda x: order[x]) if stat else "Unknown"
        last_status = stat[-1] if stat else "Unknown"

        # ---- treatment
        tx = s(r.get(C["tx"]))
        drug_txt = s(r.get(C["drugs"])) + " " + s(r.get(C["chemo_c"]))
        dfl = drug_flags(drug_txt)
        tx_l = tx.lower()
        ici = dfl["ici"] or "immunotherapy" in tx_l
        # ---- genomics
        gstat = parse_genomics(gen_txt)
        altered = sorted(g for g, v in gstat.items() if v == "altered")
        tested = sorted(gstat)
        tmb = parse_tmb(gen_txt, omd_txt)
        pdl_pct = parse_pdl1_pct(r.get(C["pdlpct"]), ihc_txt, omd_txt)
        pdl_status = {"positive": "Positive", "negative": "Negative"}.get(s(r.get(C["pdl"])).lower(), "Not performed")
        if pdl_status == "Not performed" and not np.isnan(pdl_pct):
            pdl_status = "Positive" if pdl_pct >= 1 else "Negative"
        msi_raw = s(r.get(C["msi"])).lower()
        msi = "MSI-High" if msi_raw.startswith("unstable") else ("MSS" if msi_raw.startswith("stable") else "Unknown")
        stg, T_, N_, M_ = stage_num(r.get(C["stage"])), tnm_num(r.get(C["T"]), "T"), tnm_num(r.get(C["N"]), "N"), tnm_num(r.get(C["M"]), "M")
        stage_imp = False
        if np.isnan(stg) and M_ == 1:
            stg, stage_imp = 4, True
        age = first_num(s(r.get(C["age"])))
        age = age if 0 < age <= 105 else np.nan
        site_raw = s(r.get(C["site"]))
        os_raw = s(r.get(C["os"])).lower()
        os_status = "Deceased" if os_raw.startswith("dec") else ("Living" if os_raw.startswith("liv") else "Unknown")

        rec = dict(
            patient_id=pid, entry_date=entry, sex=s(r.get(C["sex"])) or "Unknown", age=age,
            alcohol=yn(r.get(C["alc"])), smoking=yn(r.get(C["smk"])),
            cancer_class={"solid cancer": "Solid", "metastatic disease (exclusively solid cancer)": "Metastatic solid",
                          "hematologic cancer": "Hematologic"}.get(s(r.get(C["cls"])).lower(), "Unknown"),
            site=site_raw.title() if site_raw else "Unknown", site_group=site_group(site_raw),
            histology=s(r.get(C["hist"])).title() or "Unknown", laterality=s(r.get(C["lat"])) or "Unknown",
            stage=stg, stage_imputed=stage_imp, T=T_, N=N_, M=M_,
            treatment=tx or "Unknown", surgery="surgery" in tx_l, chemo=("chemotherapy" in tx_l) or dfl["chemo"],
            radiation="radiation" in tx_l, hormonal=("hormonal" in tx_l) or dfl["hormonal"],
            immuno=bool(ici), targeted=dfl["targeted"], ici_drug=dfl["ici"], n_drugs=dfl["n_drugs"],
            n_modalities=sum(["surgery" in tx_l, "chemotherapy" in tx_l, "radiation" in tx_l, "immunotherapy" in tx_l, "hormonal" in tx_l]),
            allied=s(r.get(C["allied"])),
            msi=msi, pdl1_status=pdl_status, pdl1_pct=pdl_pct, tumor_type=s(r.get(C["tumtype"])) if s(r.get(C["tumtype"])).lower() in ("hot", "cold") else "Unknown",
            tmb=tmb, genomic_tested=bool(gstat), genes_tested=";".join(tested), genes_altered=";".join(altered),
            n_altered=len(altered), os_status=os_status, os_event=int(os_status == "Deceased"),
            n_events=len(good), n_scans=len(scan_ev), best_response=best, last_status=last_status,
            ever_pd=bool(pd_dates), first_pd=min(pd_dates) if pd_dates else pd.NaT,
            t0=t0, last_known=last, course_begin=cbegin,
            on_treatment_cycle1=to_date(r.get(C["cyc1"])),
            note_scrubbed=scrub(fu_txt)[:1500],
        )
        for k, col in LAB_COLS.items():
            rec[f"lab_{k}"] = parse_lab(r.get(col), k)
        rows.append(rec)

    P = pd.DataFrame(rows)
    # ---- time-to-event (proxy) --------------------------------------------------------
    P["os_months"] = ((P["last_known"] - P["t0"]).dt.days / 30.4375)
    P.loc[P["os_months"] <= 0, "os_months"] = np.nan
    prog_t = ((P["first_pd"] - P["t0"]).dt.days / 30.4375)
    last_scan_t = P["os_months"]
    P["pfs_event"] = P["ever_pd"].astype(int)
    P["pfs_months"] = np.where(P["ever_pd"], prog_t, last_scan_t)
    P.loc[P["pfs_months"] <= 0, "pfs_months"] = np.nan
    # derived flags
    P["sex_male"] = (P["sex"].str.lower() == "male").astype(int)
    P["high_risk_rule"] = ((P["stage"] == 4) | (P["M"] == 1) | P["ever_pd"]) & (P["os_status"] != "Deceased")
    P["active"] = P["os_status"] == "Living"
    P["io_benefit"] = np.where(P["best_response"].isin(["CR", "PR", "SD"]), 1, np.where(P["best_response"] == "PD", 0, np.nan))

    S = pd.DataFrame(events)
    dq.append(f"{len(P)} unique patients after de-duplication (from {len(master)} rows).")
    dq.append(f"{n_future} dated events were dropped because the date was after the record's entry time (likely typos) or before 1995.")
    dq.append("Survival time is a PROXY: time from course-begin (or first scan) to the last dated entry; the sheet has no death date or last-contact date.")
    dq.append("Imaging dated more than 18 months before the course start is treated as prior history and excluded from follow-up and response.")
    dq.append("Progression is inferred from imaging-text keywords (rule-based), not from an explicit RECIST field.")
    dq.append("PD-L1 values stored as 0-1 numbers are treated as fractions (0.2 -> 20%).")
    for k in ("lab_wbc", "lab_hb", "lab_plt", "lab_alb"):
        dq.append(f"{k}: parsed {P[k].notna().sum()}/{len(P)} values after unit normalisation.")
    return dict(patients=P, scans=S, dq=dq, sheets=list(sheets))


if __name__ == "__main__":
    import sys
    out = load_and_clean(sys.argv[1] if len(sys.argv) > 1 else "data.xlsx")
    P = out["patients"]
    print(P.shape)
    print("\n".join(out["dq"]))
