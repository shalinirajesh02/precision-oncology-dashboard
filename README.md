# Precision Oncology Decision Support — project package

## Run on your real data (locally, nothing leaves your machine)
1. `pip install -r requirements.txt`
2. Put the workbook in this folder as `data.xlsx` (3 sheets: master, genomics subset, genomics text).
3. Reports: `python make_reports.py data.xlsx`
   -> `Manual_Analysis_Report.docx` and `Claude_Detailed_Analysis_Report.docx` (all numbers/figures regenerate from the data)
4. Dashboard: `streamlit run app.py`
   -> sidebar: local file / upload / Google Sheet link. Turn on **Auto-refresh** for live mode.

## Files
| File | Purpose |
|---|---|
| etl.py | Cleans the workbook: de-duplicates, parses labs/dates/PD-L1/genomics, extracts imaging events, scrubs names from free text |
| models.py | AI layer: penalised Cox survival + progression, immunotherapy responder score, similar-patient lookup |
| knowledge.py | Therapy-matching rules + trial matching engine (placeholder trials; ClinicalTrials.gov fetch included) |
| viz_static.py | Report figures |
| make_reports.py | Builds both reports |
| app.py | Streamlit dashboard (Executive / Clinical / Molecular / AI Prediction) |
| profile_dataset.py | PHI-safe structure profiler |

## Things to know before submitting
* Survival time is a PROXY (no death/last-contact date in the sheet). Progression/response are inferred from imaging text.
* The therapy knowledge base is illustrative and gene-level; verify against OncoKB/NCCN. Trials are placeholders until connected to ClinicalTrials.gov/CTRI.
* Models are exploratory (small cohort); validity metrics are shown in the dashboard and reports.
* Spot-check automated extraction (stage, response, genes) on ~20 patients and note accuracy in your methods.
