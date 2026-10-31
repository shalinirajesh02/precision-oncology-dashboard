"""
models.py -- AI Prediction Layer, designed for SMALL cohorts (tens of patients).

Design choices (why these models):
  * Penalised (ridge) Cox regression instead of random forests / boosting: with ~15-20 events, flexible
    models overfit. Ridge keeps a handful of baseline predictors stable and interpretable.
  * Baseline predictors only (no 'ever progressed' etc.) so the scores do not leak the outcome.
  * Every model is validated with repeated cross-validation and reported with its uncertainty.
  * Immunotherapy responder score = transparent biomarker prior (PD-L1, MSI-H, TMB-H, hot tumour) blended
    with a heavily-regularised logistic model; validated by leave-one-out AUC on treated patients only.
  * Survival time in this dataset is a PROXY (see etl.py), so outputs are decision-support signals, not forecasts.
"""
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
RNG = 42


# ------------------------------------------------------------------ feature design
def make_X(P):
    X = pd.DataFrame(index=P.index)
    X["age"] = P["age"]
    X["stage"] = P["stage"]
    X["stage_missing"] = P["stage"].isna().astype(int)
    X["sex_male"] = P["sex_male"]
    X["albumin"] = P["lab_alb"]
    X["hemoglobin"] = P["lab_hb"]
    X["hematologic"] = (P["cancer_class"] == "Hematologic").astype(int)
    X["metastatic_class"] = (P["cancer_class"] == "Metastatic solid").astype(int)
    X["immunotherapy"] = P["immuno"].astype(int)
    X["smoker"] = (P["smoking"] == "Yes").astype(int)
    return X


FEATURE_LABELS = {
    "age": "Age", "stage": "Stage (I-IV)", "stage_missing": "Stage not recorded", "sex_male": "Male sex",
    "albumin": "Serum albumin", "hemoglobin": "Hemoglobin", "hematologic": "Hematologic cancer",
    "metastatic_class": "Metastatic solid disease", "immunotherapy": "Received immunotherapy", "smoker": "Smoker",
}


class PenCox:
    """Ridge-penalised Cox model with median imputation + standardisation done inside fit()."""

    def __init__(self, penalizer=0.5):
        self.pen = penalizer

    def fit(self, X, T, E):
        self.cols = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
        X = X[self.cols]
        self.med = X.median()
        Xf = X.fillna(self.med)
        self.mu, self.sd = Xf.mean(), Xf.std().replace(0, 1)
        Z = (Xf - self.mu) / self.sd
        d = Z.assign(T=np.asarray(T), E=np.asarray(E))
        self.m = CoxPHFitter(penalizer=self.pen)
        self.m.fit(d, "T", "E")
        return self

    def _z(self, X):
        return (X[self.cols].fillna(self.med) - self.mu) / self.sd

    def risk(self, X):
        return np.asarray(self.m.predict_partial_hazard(self._z(X))).ravel()

    def surv(self, X, times):
        return self.m.predict_survival_function(self._z(X), times=times).T

    def hazard_table(self):
        s = self.m.summary
        out = pd.DataFrame({"feature": s.index, "HR_per_SD": np.exp(s["coef"]),
                            "lo": np.exp(s["coef lower 95%"]), "hi": np.exp(s["coef upper 95%"]),
                            "p": s["p"]})
        out["label"] = out["feature"].map(FEATURE_LABELS)
        return out.reset_index(drop=True)


def cv_cindex(X, T, E, pen=0.5, k=5, repeats=10):
    scores = []
    E = np.asarray(E)
    for r in range(repeats):
        skf = StratifiedKFold(k, shuffle=True, random_state=RNG + r)
        for tr, te in skf.split(X, E):
            if E[te].sum() < 1 or E[tr].sum() < 3:
                continue
            try:
                m = PenCox(pen).fit(X.iloc[tr], np.asarray(T)[tr], E[tr])
                scores.append(concordance_index(np.asarray(T)[te], -m.risk(X.iloc[te]), E[te]))
            except Exception:
                continue
    return (float(np.mean(scores)), float(np.std(scores)), len(scores)) if scores else (np.nan, np.nan, 0)


# ------------------------------------------------------------------ immunotherapy responder
def io_prior(P):
    pts = pd.Series(0.0, index=P.index)
    pct = P["pdl1_pct"]
    pts += np.where(pct >= 50, 3.0, np.where(pct >= 1, 2.0, np.where(P["pdl1_status"] == "Positive", 1.5, 0.0)))
    pts += np.where(P["msi"] == "MSI-High", 3.0, 0.0)
    pts += np.where(P["tmb"] >= 10, 2.0, 0.0)
    pts += np.where(P["tumor_type"] == "Hot", 1.5, np.where(P["tumor_type"] == "Cold", -1.0, 0.0))
    pts += np.where(P["lab_alb"] < 3.5, -1.0, 0.0)
    return pts


def io_features(P):
    X = pd.DataFrame(index=P.index)
    X["prior_pts"] = io_prior(P)
    X["pdl1_pos"] = (P["pdl1_status"] == "Positive").astype(int)
    X["msi_h"] = (P["msi"] == "MSI-High").astype(int)
    X["stage"] = P["stage"].fillna(P["stage"].median())
    X["albumin"] = P["lab_alb"].fillna(P["lab_alb"].median())
    return X


def boot_auc(y, s, n=500):
    rng = np.random.default_rng(RNG)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


# ------------------------------------------------------------------ main entry
def run_models(P):
    P = P.reset_index(drop=True)
    X = make_X(P)
    perf, tables, models = {}, {}, {}

    # ---------------- Overall survival (proxy time) ----------------
    ok = P["os_months"].notna()
    T, E = P.loc[ok, "os_months"], P.loc[ok, "os_event"]
    perf["os"] = dict(n=int(ok.sum()), events=int(E.sum()))
    os_model = PenCox(0.5).fit(X[ok], T, E)
    cm, cs, nf = cv_cindex(X[ok], T, E, 0.5)
    perf["os"].update(cindex_cv=cm, cindex_sd=cs, folds=nf, cindex_apparent=float(concordance_index(T, -os_model.risk(X[ok]), E)))
    tables["os_hr"] = os_model.hazard_table()
    models["os"] = os_model

    times = [6, 12, 24]
    S = os_model.surv(X, times)
    S.columns = [f"os_s{t}" for t in times]
    r_os = pd.Series(os_model.risk(X), index=P.index)

    # ---------------- Progression (time to first imaging progression) ----------------
    okp = P["pfs_months"].notna()
    Tp, Ep = P.loc[okp, "pfs_months"], P.loc[okp, "pfs_event"]
    perf["pfs"] = dict(n=int(okp.sum()), events=int(Ep.sum()))
    pfs_model = PenCox(0.5).fit(X[okp], Tp, Ep)
    cm, cs, nf = cv_cindex(X[okp], Tp, Ep, 0.5)
    perf["pfs"].update(cindex_cv=cm, cindex_sd=cs, folds=nf, cindex_apparent=float(concordance_index(Tp, -pfs_model.risk(X[okp]), Ep)))
    tables["pfs_hr"] = pfs_model.hazard_table()
    models["pfs"] = pfs_model
    Sp = pfs_model.surv(X, [3, 6, 12])
    prog6 = 1 - Sp[6].values
    r_pfs = pd.Series(pfs_model.risk(X), index=P.index)

    # ---------------- Immunotherapy responder score ----------------
    pts = io_prior(P)
    prior_p = 1 / (1 + np.exp(-(pts - 1.0)))
    Xio = io_features(P)
    tr = P["immuno"] & P["io_benefit"].notna()
    perf["io"] = dict(n=int(tr.sum()), benefit=int(P.loc[tr, "io_benefit"].sum()))
    yio = P.loc[tr, "io_benefit"].astype(int)
    loo_pred, loo_prior = [], []
    cols = ["pdl1_pos", "msi_h", "stage", "albumin"]
    if tr.sum() >= 12 and yio.nunique() == 2:
        Xt = Xio.loc[tr, cols].values
        for a, b in LeaveOneOut().split(Xt):
            sc = StandardScaler().fit(Xt[a])
            lr = LogisticRegression(C=0.2, max_iter=500).fit(sc.transform(Xt[a]), yio.values[a])
            loo_pred.append(lr.predict_proba(sc.transform(Xt[b]))[0, 1])
        loo_pred = np.array(loo_pred)
        perf["io"]["auc_model_loo"] = float(roc_auc_score(yio, loo_pred))
        perf["io"]["auc_model_ci"] = boot_auc(yio, loo_pred)
        perf["io"]["auc_prior"] = float(roc_auc_score(yio, prior_p[tr]))
        perf["io"]["auc_prior_ci"] = boot_auc(yio, prior_p[tr].values)
        sc = StandardScaler().fit(Xio.loc[tr, cols].values)
        lr = LogisticRegression(C=0.2, max_iter=500).fit(sc.transform(Xio.loc[tr, cols].values), yio.values)
        model_p = lr.predict_proba(sc.transform(Xio[cols].values))[:, 1]
        io_score = 0.5 * prior_p.values + 0.5 * model_p
        perf["io"]["note"] = "Exploratory: blended biomarker prior + regularised logistic model."
    else:
        io_score = prior_p.values
        perf["io"]["note"] = "Too few treated patients with a documented response; biomarker prior only."

    # ---------------- Assemble per-patient scores ----------------
    def pct(v):
        return pd.Series(v, index=P.index).rank(pct=True) * 100

    R = pd.DataFrame({"patient_id": P["patient_id"]})
    R = pd.concat([R, S.reset_index(drop=True)], axis=1)
    R["os_risk_pct"] = pct(r_os).values
    R["prog_risk_6m"] = prog6
    R["prog_risk_pct"] = pct(r_pfs).values
    R["io_score"] = io_score * 100
    combined = 0.5 * R["os_risk_pct"] + 0.5 * R["prog_risk_pct"]
    R["composite_risk"] = combined
    R["risk_tier"] = pd.cut(combined, [-1, 33.3, 66.6, 101], labels=["Low", "Medium", "High"]).astype(str)
    R["ai_high_risk"] = (R["risk_tier"] == "High") | P["high_risk_rule"].values
    return dict(scores=R, perf=perf, tables=tables, models=models, X=X)


# ------------------------------------------------------------------ similar patients
def similar_patients(P, pid, k=5):
    P = P.reset_index(drop=True)
    X = pd.DataFrame({
        "age": P["age"].fillna(P["age"].median()), "stage": P["stage"].fillna(P["stage"].median()),
        "sex": P["sex_male"], "immuno": P["immuno"].astype(int), "chemo": P["chemo"].astype(int),
    })
    X = pd.concat([X, pd.get_dummies(P["site_group"], prefix="site").astype(int) * 3.0], axis=1)
    Z = (X - X.mean()) / X.std().replace(0, 1)
    Z.loc[:, [c for c in Z.columns if c.startswith("site_")]] = X[[c for c in X.columns if c.startswith("site_")]].values
    i = P.index[P["patient_id"] == pid]
    if len(i) == 0:
        return pd.DataFrame()
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(P))).fit(Z.values)
    d, idx = nn.kneighbors(Z.loc[[i[0]]].values)
    rows = [j for j in idx[0] if j != i[0]][:k]
    out = P.loc[rows, ["patient_id", "site_group", "stage", "age", "treatment", "best_response", "os_status", "os_months"]].copy()
    out["os_months"] = out["os_months"].round(1)
    return out


if __name__ == "__main__":
    import etl
    P = etl.load_and_clean("data.xlsx")["patients"]
    out = run_models(P)
    for k, v in out["perf"].items():
        print(k, v)
    print(out["scores"].describe().round(2).T.head(12))
    print(out["tables"]["os_hr"].round(2))
