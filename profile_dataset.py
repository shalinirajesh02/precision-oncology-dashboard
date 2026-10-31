"""
profile_dataset.py  --  PHI-safe structure profiler
=====================================================
Run this LOCALLY on your real .xlsx. It prints a summary of the dataset's
STRUCTURE (column fill rates, data types, category labels with counts,
numeric ranges, date ranges by month, and keyword counts inside free-text
columns). It never prints free-text content, patient IDs, or patient names.

Usage:
    pip install pandas openpyxl
    python profile_dataset.py "your_file.xlsx"

Output: printed to screen AND saved to profile_summary.txt.
Skim profile_summary.txt before sharing it, then paste it back to Claude.
"""
import re
import sys
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

MAX_CATEGORIES = 15          # show value counts only if a column has <= this many distinct values
MAX_LABEL_LEN = 40           # ...and labels are short (so it's a category, not free text)
SKIP_NAME_COLS = ("patient name", "name of patient")   # never profiled or printed

KEYWORDS = [
    "EGFR", "KRAS", "ALK", "ROS1", "BRAF", "HER2", "BRCA", "TP53", "MET", "RET", "NTRK",
    "PIK3CA", "PD-L1", "PDL1", "MSI", "TMB", "ECOG",
    "progress", "response", "partial", "complete", "stable", "recurr", "relapse",
    "metast", "expired", "death", "deceased", "alive", "lost to follow",
    "CT", "PET", "MRI", "RECIST",
]
DATE_PATTERN = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b|\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b")

out_lines = []
def emit(s=""):
    print(s)
    out_lines.append(s)

def clean(h):
    return re.sub(r"\s+", " ", str(h)).strip()

def parse_dates(series):
    """Parse dates trying both day-first and month-first; keep whichever parses more."""
    a = pd.to_datetime(series, errors="coerce", dayfirst=True)
    b = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return a if a.notna().sum() >= b.notna().sum() else b

def classify(series):
    s = series.dropna()
    if s.empty:
        return "empty"
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.8:
        return "numeric"
    if parse_dates(s).notna().mean() >= 0.8:
        return "date"
    return "text"

def main(path):
    sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    emit(f"FILE STRUCTURE: {len(sheets)} sheet(s)")
    id_sets = {}

    for sname, df in sheets.items():
        df.columns = [clean(c) for c in df.columns]
        emit("\n" + "=" * 78)
        emit(f"SHEET: {sname}   rows={len(df)}   columns={len(df.columns)}")
        emit("=" * 78)

        id_col = next((c for c in df.columns if c.lower() == "patient id"), None)
        if id_col:
            ids = df[id_col].dropna().astype(str).str.strip()
            id_sets[sname] = set(ids)
            emit(f"Patient ID column: '{id_col}'  unique IDs={ids.nunique()}  "
                 f"rows per patient: max={ids.value_counts().max() if len(ids) else 0} "
                 f"(1 = one row per patient)")

        for col in df.columns:
            low = col.lower()
            if any(k in low for k in SKIP_NAME_COLS):
                emit(f"\n- {col}: [SKIPPED - patient name; remove this column before any analysis]")
                continue
            if low == "patient id":
                continue
            s = df[col]
            n = s.notna().sum()
            kind = classify(s)
            emit(f"\n- {col}   [{kind}]   filled {n}/{len(df)} ({n/len(df):.0%})")

            if kind == "numeric":
                num = pd.to_numeric(s, errors="coerce").dropna()
                emit(f"    min={num.min():.4g}  median={num.median():.4g}  max={num.max():.4g}")
                bad = s.notna().sum() - num.shape[0]
                if bad:
                    emit(f"    ({bad} non-numeric entries, e.g. units/text mixed in)")
            elif kind == "date":
                dt = parse_dates(s).dropna()
                emit(f"    range (month level): {dt.min():%Y-%m} to {dt.max():%Y-%m}")
            elif kind == "text":
                t = s.dropna().astype(str).str.strip()
                nuniq = t.nunique()
                avg_len = t.str.len().mean()
                if nuniq <= MAX_CATEGORIES and t.str.len().max() <= MAX_LABEL_LEN:
                    emit(f"    category with {nuniq} values:")
                    for val, cnt in t.value_counts().items():
                        emit(f"      {cnt:>3} x  {val}")
                else:
                    emit(f"    free text: {nuniq} distinct entries, avg length {avg_len:.0f} chars  (content NOT shown)")
                    has_date = t.apply(lambda x: bool(DATE_PATTERN.search(x))).sum()
                    emit(f"    rows containing a date-like pattern: {has_date}")
                    hits = []
                    for kw in KEYWORDS:
                        c = t.str.contains(re.escape(kw), case=(kw.isupper() and len(kw) <= 4), regex=True).sum()
                        if c:
                            hits.append(f"{kw}:{c}")
                    if hits:
                        emit("    keyword counts (rows mentioning): " + "  ".join(hits))

    if len(id_sets) > 1:
        emit("\n" + "=" * 78)
        emit("PATIENT ID OVERLAP BETWEEN SHEETS (counts only)")
        names = list(id_sets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = id_sets[names[i]], id_sets[names[j]]
                emit(f"  '{names[i]}' ({len(a)}) vs '{names[j]}' ({len(b)}): shared={len(a & b)}")

    with open("profile_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("\nSaved to profile_summary.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python profile_dataset.py "your_file.xlsx"')
    main(sys.argv[1])
