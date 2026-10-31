"""viz_static.py -- publication-style figures (matplotlib) shared by the two reports."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Patch
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

NAVY, TEAL, CORAL, AMBER, GREY, GREEN, PURPLE = "#1B3A4B", "#3F7C82", "#E07A5F", "#E0A458", "#9AA7AE", "#4C8C5B", "#7B6D9C"
PAL = [TEAL, CORAL, AMBER, NAVY, GREEN, PURPLE, GREY, "#C4D7D9"]
RESP_COL = {"CR": GREEN, "PR": TEAL, "SD": AMBER, "PD": CORAL, "Unknown": "#D5DBDE"}
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 12, "figure.dpi": 130, "savefig.bbox": "tight", "axes.edgecolor": "#6B7A85"})
OUT = "figures"


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".png")
    fig.savefig(p, dpi=170, facecolor="white")
    plt.close(fig)
    return p


def _lab(ax, bars, fmt="{:.0f}", pad=0.3, horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if v and not np.isnan(v):
            if horizontal:
                ax.text(v + pad, b.get_y() + b.get_height() / 2, fmt.format(v), va="center", fontsize=8.5, color=NAVY)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + pad, fmt.format(v), ha="center", fontsize=8.5, color=NAVY)


# ---------------------------------------------------------------- data quality / cohort
def f_completeness(P):
    fields = {"Age": P.age.notna(), "Stage": P.stage.notna(), "T category": P["T"].notna(), "N category": P["N"].notna(),
              "M category": P["M"].notna(), "PD-L1 %": P.pdl1_pct.notna(), "MSI status": P.msi != "Unknown",
              "Tumour type (hot/cold)": P.tumor_type != "Unknown", "Genomic report": P.genomic_tested,
              "TMB": P.tmb.notna(), "Albumin": P.lab_alb.notna(), "Hemoglobin": P.lab_hb.notna(),
              "Platelets": P.lab_plt.notna(), "Imaging response": P.best_response != "Unknown",
              "Survival status": P.os_status != "Unknown", "Follow-up time": P.os_months.notna()}
    s = pd.Series({k: v.mean() * 100 for k, v in fields.items()}).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 5))
    cols = [CORAL if v < 40 else (AMBER if v < 75 else TEAL) for v in s]
    b = ax.barh(s.index, s.values, color=cols)
    _lab(ax, b, "{:.0f}%", 1, True)
    ax.set_xlim(0, 112); ax.set_xlabel("% of patients with a usable value")
    ax.set_title("Data completeness of key analysis fields")
    ax.legend(handles=[Patch(color=TEAL, label="≥75%"), Patch(color=AMBER, label="40–75%"), Patch(color=CORAL, label="<40%")],
              loc="lower right", frameon=False, fontsize=8)
    return _save(fig, "completeness")


def f_demographics(P):
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for sx, c in zip(P.sex.unique(), PAL):
        axs[0].hist(P.loc[P.sex == sx, "age"].dropna(), bins=np.arange(10, 100, 8), alpha=.75, label=f"{sx} (n={int((P.sex==sx).sum())})", color=c)
    axs[0].set_title("Age distribution by sex"); axs[0].set_xlabel("Age (years)"); axs[0].legend(frameon=False, fontsize=8)
    for ax, col, ttl in [(axs[1], "smoking", "Smoking history"), (axs[2], "alcohol", "Alcohol history")]:
        v = P[col].value_counts().reindex(["Yes", "No", "Unknown"]).fillna(0)
        b = ax.bar(v.index, v.values, color=[CORAL, TEAL, GREY]); _lab(ax, b)
        ax.set_title(ttl); ax.set_ylabel("Patients")
    fig.tight_layout()
    return _save(fig, "demographics")


def f_site(P):
    ct = pd.crosstab(P.site_group, P.cancer_class)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    left = np.zeros(len(ct))
    for c, col in zip(ct.columns, PAL):
        ax.barh(ct.index, ct[c], left=left, color=col, label=c); left += ct[c].values
    for i, t in enumerate(left):
        ax.text(t + .2, i, int(t), va="center", fontsize=8.5, color=NAVY)
    ax.set_title("Cancer site distribution (stacked by cancer class)"); ax.set_xlabel("Patients")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, "site")


def f_stage(P):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    v = P.stage.map({1: "I", 2: "II", 3: "III", 4: "IV"}).fillna("Not recorded").value_counts().reindex(["I", "II", "III", "IV", "Not recorded"]).fillna(0)
    b = axs[0].bar(v.index, v.values, color=[TEAL, TEAL, AMBER, CORAL, GREY]); _lab(axs[0], b)
    axs[0].set_title("Stage at presentation"); axs[0].set_ylabel("Patients")
    solid = P[P.cancer_class != "Hematologic"]
    h = pd.crosstab(solid.site_group, solid.stage.map({1: "I", 2: "II", 3: "III", 4: "IV"}).fillna("n/r"))
    h = h.reindex(columns=[c for c in ["I", "II", "III", "IV", "n/r"] if c in h.columns])
    sns.heatmap(h, annot=True, fmt="d", cmap=sns.light_palette(TEAL, as_cmap=True), cbar=False, ax=axs[1], linewidths=.5)
    axs[1].set_title("Stage by site (solid tumours)"); axs[1].set_xlabel(""); axs[1].set_ylabel("")
    fig.tight_layout()
    return _save(fig, "stage")


# ---------------------------------------------------------------- treatment & response
def f_treatment(P):
    mods = {"Chemotherapy": "chemo", "Surgery": "surgery", "Targeted agents": "targeted", "Immunotherapy": "immuno",
            "Radiation": "radiation", "Hormonal": "hormonal"}
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
    pct = pd.Series({k: P[v].mean() * 100 for k, v in mods.items()}).sort_values()
    b = axs[0].barh(pct.index, pct.values, color=TEAL); _lab(axs[0], b, "{:.0f}%", 1, True)
    axs[0].set_xlim(0, 100); axs[0].set_title("Treatment modality utilisation"); axs[0].set_xlabel("% of patients")
    g = P.assign(sg=P.stage.map({1: "I–II", 2: "I–II", 3: "III", 4: "IV"}).fillna("n/r"))
    rows = {k: g.groupby("sg")[v].mean() * 100 for k, v in mods.items() if k in ("Chemotherapy", "Surgery", "Immunotherapy", "Targeted agents")}
    D = pd.DataFrame(rows).reindex(["I–II", "III", "IV", "n/r"]).fillna(0)
    D.plot.bar(ax=axs[1], color=[TEAL, CORAL, AMBER, NAVY], width=.8, rot=0)
    axs[1].set_title("Utilisation by stage"); axs[1].set_ylabel("% of patients"); axs[1].set_xlabel("Stage group"); axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, "treatment")


def f_response(P):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.6]})
    order = ["CR", "PR", "SD", "PD", "Unknown"]
    v = P.best_response.value_counts().reindex(order).fillna(0)
    axs[0].pie(v.values, labels=[f"{k}\n({int(n)})" for k, n in v.items()], colors=[RESP_COL[k] for k in v.index], startangle=90,
               wedgeprops=dict(width=.45, edgecolor="white"), textprops={"fontsize": 8.5})
    axs[0].set_title("Best documented imaging response")
    k = P[P.best_response != "Unknown"]
    top = k.site_group.value_counts().head(7).index
    ct = pd.crosstab(k[k.site_group.isin(top)].site_group, k.best_response).reindex(columns=order[:4]).fillna(0)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    ctp = ct.div(ct.sum(axis=1), axis=0) * 100
    left = np.zeros(len(ctp))
    for c in ctp.columns:
        axs[1].barh(ctp.index, ctp[c], left=left, color=RESP_COL[c], label=c); left += ctp[c].values
    for i, (s_, n) in enumerate(zip(ctp.index, ct.sum(axis=1))):
        axs[1].text(101, i, f"n={int(n)}", va="center", fontsize=8)
    axs[1].set_xlim(0, 112); axs[1].set_xlabel("% of patients with a documented response"); axs[1].set_title("Response by site (sites with most patients)")
    axs[1].legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.18))
    fig.tight_layout()
    return _save(fig, "response")


def f_swimmer(P, S, n=28):
    d = P[P.t0.notna() & P.last_known.notna()].copy()
    d["dur"] = (d.last_known - d.t0).dt.days / 30.4375
    d = d.sort_values("dur", ascending=False).head(n).sort_values("dur")
    fig, ax = plt.subplots(figsize=(9.2, 0.26 * len(d) + 1.6))
    y = {pid: i for i, pid in enumerate(d.patient_id)}
    ax.barh(list(y.values()), d.dur.values, color="#C4D7D9", height=.55)
    mk = {"PD": (CORAL, "v"), "PR": (TEAL, "o"), "SD": (AMBER, "s"), "CR": (GREEN, "D")}
    if S is not None and len(S):
        s = S[S.patient_id.isin(y) & S.status.isin(mk)].merge(d[["patient_id", "t0"]], on="patient_id")
        s["m"] = (s.date - s.t0).dt.days / 30.4375
        for st, (c, m) in mk.items():
            q = s[s.status == st]
            ax.scatter(q.m, q.patient_id.map(y), c=c, marker=m, s=34, zorder=3, edgecolor="white", linewidth=.5, label=st)
    dead = d[d.os_status == "Deceased"]
    ax.scatter(dead.dur, dead.patient_id.map(y), marker="X", c="black", s=55, zorder=4, label="Deceased")
    ax.set_yticks(list(y.values())); ax.set_yticklabels(list(y.keys()), fontsize=7.5)
    ax.set_xlabel("Months from treatment start (negative = imaging before the course began)"); ax.set_title("Swimmer plot: follow-up, imaging status and outcome")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="lower right")
    return _save(fig, "swimmer")


def f_alluvial(P):
    """Stage group -> primary treatment pattern -> best response, drawn with bezier ribbons."""
    d = P.copy()
    d["A"] = d.stage.map({1: "Stage I–II", 2: "Stage I–II", 3: "Stage III", 4: "Stage IV"}).fillna("Stage n/r")
    def pat(r):
        if r.immuno and r.chemo: return "Chemo + Immuno"
        if r.immuno: return "Immuno"
        if r.targeted and r.chemo: return "Chemo + Targeted"
        if r.chemo: return "Chemo"
        if r.hormonal: return "Hormonal"
        return "Other / none"
    d["B"] = d.apply(pat, axis=1)
    d["C"] = d.best_response.replace({"Unknown": "No response data"})
    cols = [("A", ["Stage I–II", "Stage III", "Stage IV", "Stage n/r"]),
            ("B", ["Chemo + Immuno", "Immuno", "Chemo + Targeted", "Chemo", "Hormonal", "Other / none"]),
            ("C", ["CR", "PR", "SD", "PD", "No response data"])]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    n = len(d); gap = 0.025 * n
    pos = {}
    for ci, (c, cats) in enumerate(cols):
        tot = sum((d[c] == k).sum() for k in cats) + gap * (len(cats) - 1)
        y = tot
        for k in cats:
            h = (d[c] == k).sum()
            y -= h
            pos[(c, k)] = [ci, y, h, y]    # x, bottom, height, running pointer
            y -= gap
    ccol = {"Stage I–II": TEAL, "Stage III": AMBER, "Stage IV": CORAL, "Stage n/r": GREY}
    for ci in range(2):
        a, b = cols[ci][0], cols[ci + 1][0]
        flow = d.groupby([a, b]).size().reset_index(name="n")
        order_a = {k: i for i, k in enumerate(cols[ci][1])}; order_b = {k: i for i, k in enumerate(cols[ci + 1][1])}
        flow["oa"], flow["ob"] = flow[a].map(order_a), flow[b].map(order_b)
        # place ribbons from the top of each node downward
        ptr = {k: v[1] + v[2] for k, v in pos.items()}
        for _, f in flow.sort_values(["oa", "ob"]).iterrows():
            ka, kb = (a, f[a]), (b, f[b])
            ya_top, yb_top = ptr[ka], ptr[kb]
            ya_bot, yb_bot = ya_top - f.n, yb_top - f.n
            ptr[ka], ptr[kb] = ya_bot, yb_bot
            x0, x1 = ci + .06, ci + 1 - .06
            xm = (x0 + x1) / 2
            verts = [(x0, ya_top), (xm, ya_top), (xm, yb_top), (x1, yb_top), (x1, yb_bot), (xm, yb_bot), (xm, ya_bot), (x0, ya_bot), (x0, ya_top)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
            col = ccol.get(f[a], TEAL) if ci == 0 else RESP_COL.get(f[b], GREY)
            ax.add_patch(PathPatch(Path(verts, codes), fc=col, ec="none", alpha=.45))
    for (c, k), (x, bot, h, _) in pos.items():
        if h == 0: continue
        ax.add_patch(plt.Rectangle((x - .06, bot), .12, h, color=NAVY))
        right = x == 2
        ax.text(x + (.1 if right else -.1) if x != 1 else x, bot + h / 2, f"{k} ({h})", ha="left" if right else ("center" if x == 1 else "right"),
                va="center", fontsize=8.5, color=NAVY, bbox=dict(fc="white", ec="none", alpha=.7, pad=1) if x == 1 else None)
    ax.set_xlim(-.55, 2.55); ax.set_ylim(-1, max(v[1] + v[2] for v in pos.values()) + 2); ax.axis("off")
    for x, t in enumerate(["Stage", "Treatment pattern", "Best imaging response"]):
        ax.text(x, ax.get_ylim()[1], t, ha="center", fontweight="bold", color=TEAL)
    ax.set_title("Patient flow: stage → treatment → response", pad=22)
    return _save(fig, "alluvial")


# ---------------------------------------------------------------- survival
def f_km(P, time, event, by=None, order=None, title="", name="km", xlabel="Months", min_n=3):
    d = P[P[time].notna()].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.9 if by is None else 5.4))
    groups = [(None, d)] if by is None else [(g, d[d[by] == g]) for g in (order or sorted(d[by].dropna().unique(), key=str))]
    groups = [(g, x) for g, x in groups if len(x) >= min_n]
    fits = []
    for (g, x), c in zip(groups, PAL):
        k = KaplanMeierFitter().fit(x[time], x[event], label=f"{g} (n={len(x)}, events={int(x[event].sum())})" if g is not None else f"All (n={len(x)}, events={int(x[event].sum())})")
        k.plot_survival_function(ax=ax, ci_show=True, color=c, ci_alpha=.12, show_censors=True, censor_styles={"ms": 4, "marker": "|"})
        fits.append(k)
    p = np.nan
    if by is not None and len(groups) > 1:
        dd = pd.concat([x.assign(_g=str(g)) for g, x in groups])
        p = multivariate_logrank_test(dd[time], dd["_g"], dd[event]).p_value
        ax.text(.98, .96, f"log-rank p = {p:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=NAVY,
                bbox=dict(fc="white", ec="#C4D7D9", boxstyle="round,pad=.3"))
    ax.set_ylim(0, 1.02); ax.set_xlabel(xlabel); ax.set_ylabel("Probability"); ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return _save(fig, name), p


def f_forest(tbl, title, name):
    t = tbl.sort_values("HR_per_SD").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(t) + 1.5))
    ypos = np.arange(len(t))
    col = [CORAL if h > 1 else TEAL for h in t.HR_per_SD]
    ax.hlines(ypos, t.lo, t.hi, color=col, lw=2)
    ax.scatter(t.HR_per_SD, ypos, c=col, s=45, zorder=3)
    ax.axvline(1, color=GREY, ls="--")
    ax.set_yticks(ypos); ax.set_yticklabels(t.label); ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio per 1 SD (or per yes/no), 95% CI — ridge-penalised Cox")
    ax.set_title(title)
    for i, r in t.iterrows():
        ax.text(ax.get_xlim()[1] * 1.0, i, f" {r.HR_per_SD:.2f}", va="center", fontsize=8, color=NAVY)
    return _save(fig, name)


# ---------------------------------------------------------------- molecular
def f_biomarkers(P):
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8))
    v = P.msi.value_counts().reindex(["MSS", "MSI-High", "Unknown"]).fillna(0)
    b = axs[0].bar(v.index, v.values, color=[TEAL, CORAL, GREY]); _lab(axs[0], b); axs[0].set_title("MSI status")
    cat = pd.Series(np.where(P.pdl1_pct >= 50, "High (≥50%)", np.where(P.pdl1_pct >= 1, "Low (1–49%)", np.where(P.pdl1_pct < 1, "Negative (<1%)",
          np.where(P.pdl1_status == "Positive", "Positive (n/s)", np.where(P.pdl1_status == "Negative", "Negative (<1%)", "Not tested"))))))
    v = cat.value_counts().reindex(["High (≥50%)", "Low (1–49%)", "Positive (n/s)", "Negative (<1%)", "Not tested"]).fillna(0)
    b = axs[1].bar(range(len(v)), v.values, color=[CORAL, AMBER, AMBER, TEAL, GREY]); _lab(axs[1], b)
    axs[1].set_xticks(range(len(v))); axs[1].set_xticklabels(["High\n≥50%", "Low\n1–49%", "Positive\n(no %)", "Negative\n<1%", "Not\ntested"], fontsize=8)
    axs[1].set_title("PD-L1 landscape")
    t = P.tumor_type.value_counts().reindex(["Hot", "Cold", "Unknown"]).fillna(0)
    b = axs[2].bar(t.index, t.values, color=[CORAL, TEAL, GREY]); _lab(axs[2], b); axs[2].set_title("Tumour immune phenotype")
    fig.tight_layout()
    return _save(fig, "biomarkers")


def gene_table(P):
    g = P[P.genes_altered != ""].genes_altered.str.split(";").explode()
    tested = P.genes_tested[P.genes_tested != ""].str.split(";").explode().value_counts()
    alt = g.value_counts()
    return pd.DataFrame({"altered": alt, "tested": tested.reindex(alt.index)}).fillna(0).astype(int)


def f_genes(P):
    gt = gene_table(P)
    nt = int(P.genomic_tested.sum())
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    if len(gt):
        gt = gt.sort_values("altered")
        b = ax.barh(gt.index, gt.altered, color=CORAL)
        for i, (g, r) in enumerate(gt.iterrows()):
            ax.text(r.altered + .05, i, f"{r.altered}/{max(r.tested, r.altered)} tested", va="center", fontsize=8.5, color=NAVY)
        ax.set_xlim(0, gt.altered.max() + 2)
    ax.set_xlabel("Patients with an altered result"); ax.set_title(f"Genomic alteration frequency (patients with a gene-level report: {nt})")
    return _save(fig, "genes")


def f_oncoplot(P):
    d = P[(P.n_altered > 0) | (P.msi == "MSI-High") | (P.tmb >= 10) | (P.pdl1_pct >= 50)].copy()
    genes = [g for g in gene_table(P).sort_values("altered", ascending=False).index]
    rows = genes + ["MSI-High", "TMB ≥10", "PD-L1 ≥50%"]
    M = pd.DataFrame(0, index=rows, columns=d.patient_id)
    for _, r in d.iterrows():
        for g in str(r.genes_altered).split(";"):
            if g in M.index: M.loc[g, r.patient_id] = 1
        M.loc["MSI-High", r.patient_id] = int(r.msi == "MSI-High")
        M.loc["TMB ≥10", r.patient_id] = int(pd.notna(r.tmb) and r.tmb >= 10)
        M.loc["PD-L1 ≥50%", r.patient_id] = int(pd.notna(r.pdl1_pct) and r.pdl1_pct >= 50)
    if M.shape[1] == 0:
        fig, ax = plt.subplots(figsize=(6, 2)); ax.text(.5, .5, "No molecular alterations recorded", ha="center"); ax.axis("off")
        return _save(fig, "oncoplot")
    M = M.loc[:, M.sum().sort_values(ascending=False).index]
    sg = d.set_index("patient_id").loc[M.columns, "site_group"]
    fig, axs = plt.subplots(2, 1, figsize=(max(7, 0.34 * M.shape[1] + 2.5), 0.38 * len(rows) + 2), gridspec_kw={"height_ratios": [.6, len(rows)]}, sharex=True)
    cm = {k: PAL[i % len(PAL)] for i, k in enumerate(sorted(sg.unique()))}
    for i, k in enumerate(sg): axs[0].add_patch(plt.Rectangle((i, 0), .92, 1, color=cm[k]))
    axs[0].set_xlim(0, M.shape[1]); axs[0].set_ylim(0, 1); axs[0].axis("off")
    axs[0].legend(handles=[Patch(color=c, label=k) for k, c in cm.items()], ncol=min(5, len(cm)), frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(.5, 1.0))
    sns.heatmap(M, cmap=sns.color_palette(["#EEF2F3", CORAL]), cbar=False, ax=axs[1], linewidths=.6, linecolor="white", xticklabels=True)
    axs[1].tick_params(axis="x", labelsize=7, rotation=90); axs[1].set_xlabel(""); axs[1].set_ylabel("")
    fig.suptitle("Oncoplot: molecular profile of patients with an actionable or notable result", y=1.0, fontweight="bold", fontsize=11)
    return _save(fig, "oncoplot")


def f_labs(P):
    labs = [("lab_alb", "Albumin (g/dL)"), ("lab_hb", "Hemoglobin (g/dL)"), ("lab_anc", "ANC (K/µL)"), ("lab_plt", "Platelets (K/µL)")]
    d = P[P.os_status.isin(["Living", "Deceased"])]
    fig, axs = plt.subplots(1, 4, figsize=(11.8, 3.6))
    for ax, (c, t) in zip(axs, labs):
        sns.boxplot(data=d, x="os_status", y=c, hue="os_status", palette={"Living": TEAL, "Deceased": CORAL}, ax=ax, width=.55, legend=False, fliersize=0)
        sns.stripplot(data=d, x="os_status", y=c, color=NAVY, size=3, alpha=.6, ax=ax)
        ax.set_title(t, fontsize=10); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Baseline laboratory values by survival status", y=1.03, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, "labs")


def f_labcorr(P):
    cols = [c for c in P.columns if c.startswith("lab_") and P[c].notna().sum() >= 25]
    C = P[cols + ["age", "stage"]].corr(method="spearman")
    C.index = C.columns = [c.replace("lab_", "").upper() if c.startswith("lab_") else c.title() for c in C.columns]
    fig, ax = plt.subplots(figsize=(7.4, 6))
    sns.heatmap(C, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".1f", annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": .7})
    ax.set_title("Spearman correlation of baseline labs, age and stage")
    return _save(fig, "labcorr")


def f_io(P, perf):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4))
    t = P[P.immuno & P.io_benefit.notna()].copy()
    cat = np.where(t.pdl1_pct >= 50, "PD-L1 ≥50%", np.where(t.pdl1_pct >= 1, "PD-L1 1–49%", np.where((t.pdl1_pct < 1) | (t.pdl1_status == "Negative"), "PD-L1 neg", "PD-L1 n/t")))
    t["cat"] = np.where(t.msi == "MSI-High", "MSI-High", cat)
    g = t.groupby("cat").io_benefit.agg(["mean", "count"]).sort_values("mean")
    b = axs[0].barh(g.index, g["mean"] * 100, color=TEAL)
    for i, (k, r) in enumerate(g.iterrows()): axs[0].text(r["mean"] * 100 + 1, i, f"{r['mean']*100:.0f}%  (n={int(r['count'])})", va="center", fontsize=8.5)
    axs[0].set_xlim(0, 130); axs[0].set_xlabel("% with clinical benefit (CR/PR/SD) on immunotherapy")
    axs[0].set_title("Immunotherapy benefit by biomarker group")
    io = perf.get("io", {})
    if "auc_model_loo" in io:
        names = ["Biomarker prior", "Logistic model\n(leave-one-out)"]
        auc = [io["auc_prior"], io["auc_model_loo"]]
        ci = [io["auc_prior_ci"], io["auc_model_ci"]]
        axs[1].bar(names, auc, color=[AMBER, TEAL], width=.5)
        for i, (a, c) in enumerate(zip(auc, ci)):
            axs[1].errorbar(i, a, yerr=[[a - c[0]], [c[1] - a]], color=NAVY, capsize=5)
            axs[1].text(i, c[1] + .03, f"AUC {a:.2f}", ha="center", fontsize=9)
        axs[1].axhline(.5, color=GREY, ls="--"); axs[1].set_ylim(0, 1.15); axs[1].set_ylabel("AUC (95% bootstrap CI)")
        axs[1].set_title(f"Responder-score validation (n={io['n']} treated)")
    fig.tight_layout()
    return _save(fig, "io")


def f_model_perf(perf):
    rows = [("Overall survival\n(proxy time)", perf["os"]), ("Time to progression", perf["pfs"])]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for i, (n, p) in enumerate(rows):
        ax.errorbar(p["cindex_cv"], i, xerr=p["cindex_sd"], fmt="o", color=TEAL, capsize=5, ms=8)
        ax.scatter(p["cindex_apparent"], i, marker="D", color=AMBER, zorder=3)
        ax.text(1.02, i, f"n={p['n']}, events={p['events']}", va="center", fontsize=8.5)
    ax.axvline(.5, color=GREY, ls="--"); ax.set_xlim(.3, 1.2); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Concordance index (0.5 = chance)"); ax.set_title("Survival-model performance")
    ax.legend(handles=[plt.Line2D([], [], marker="o", color=TEAL, ls="", label="Cross-validated (±SD)"), plt.Line2D([], [], marker="D", color=AMBER, ls="", label="Apparent (optimistic)")],
              frameon=False, fontsize=8, loc="lower left")
    return _save(fig, "modelperf")


def f_risk_tiers(P, R):
    d = P.merge(R[["patient_id", "composite_risk", "risk_tier", "prog_risk_6m", "io_score"]], on="patient_id")
    fig, axs = plt.subplots(1, 3, figsize=(11.8, 3.8))
    v = d.risk_tier.value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    b = axs[0].bar(v.index, v.values, color=[TEAL, AMBER, CORAL]); _lab(axs[0], b); axs[0].set_title("AI risk tiers")
    axs[1].hist(d.prog_risk_6m * 100, bins=12, color=CORAL, alpha=.85); axs[1].set_title("Predicted 6-month progression risk"); axs[1].set_xlabel("%")
    axs[2].hist(d.io_score, bins=12, color=TEAL, alpha=.85); axs[2].set_title("Immunotherapy responder score"); axs[2].set_xlabel("score (0–100)")
    fig.tight_layout()
    return _save(fig, "risk_tiers")


def f_actionable(P):
    import knowledge as K
    cnt = {}
    for _, r in P.iterrows():
        for m in set(K.patient_markers(r)):
            cnt[m] = cnt.get(m, 0) + 1
    s = pd.Series(cnt).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 0.4 * max(len(s), 3) + 1.3))
    if len(s):
        b = ax.barh(s.index, s.values, color=[CORAL if i in K.ACTIONABLE else GREY for i in s.index]); _lab(ax, b, "{:.0f}", .1, True)
    ax.set_xlabel("Patients"); ax.set_title("Actionable biomarkers present (gene-level)")
    return _save(fig, "actionable")
