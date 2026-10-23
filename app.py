import os
import numpy as np, pandas as pd, joblib, shap, sklearn, streamlit as st
import db

st.set_page_config(page_title="Explainable and Fair Credit Risk Assessment", layout="wide")
ART_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifact.joblib")


@st.cache_resource
def load_artifact():
    return joblib.load(ART_PATH)


@st.cache_resource
def get_explainer(_art):
    return shap.TreeExplainer(_art["model"].named_steps["clf"])


@st.cache_resource
def get_db(_art):
    eng = db.get_engine()
    return eng, db.ensure_model(eng, _art)


# ----------------------------------------------------------------- feature construction (single row or batch)
def derive(df, art):
    d = df.copy()
    for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "BUR_LOAN_COUNT", "BUR_ACTIVE_COUNT",
              "BUR_MAX_DAYS_OVERDUE", "BUR_TOTAL_OVERDUE"):
        if c not in d.columns:
            d[c] = np.nan
    d["CREDIT_INCOME"] = d["AMT_CREDIT"] / d["AMT_INCOME_TOTAL"]
    d["ANNUITY_INCOME"] = d["AMT_ANNUITY"] / d["AMT_INCOME_TOTAL"]
    d["IMPLIED_TERM"] = d["AMT_CREDIT"] / d["AMT_ANNUITY"]
    d["CREDIT_GOODS"] = d["AMT_CREDIT"] / d["AMT_GOODS_PRICE"]
    ext = d[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].astype(float)
    d["EXT_MEAN"], d["EXT_MIN"] = ext.mean(axis=1), ext.min(axis=1)
    if "BUR_COVERAGE" not in d.columns:
        d["BUR_COVERAGE"] = d[["BUR_LOAN_COUNT", "BUR_ACTIVE_COUNT"]].notna().any(axis=1).astype(float)
    d = d.replace([np.inf, -np.inf], np.nan)
    out = d.reindex(columns=art["features"])
    for c in out.columns:
        if c not in art["categories"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def band_and_decision(p, thr):
    if p < thr / 2:
        return "Low", "Suggest approval (standard checks)"
    if p < thr:
        return "Medium", "Suggest manual review"
    return "High", "Suggest detailed review before any approval"


def base_feature(name, feats):
    n = name.split("__", 1)[-1]
    if n.startswith("missingindicator_"):
        n = n[len("missingindicator_"):]
    if n in feats:
        return n
    for f in sorted(feats, key=len, reverse=True):
        if n.startswith(f + "_"):
            return f
    return n


def explain(row, art, k=3):
    pre = art["model"].named_steps["pre"]
    Xt = pre.transform(row)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    sv = get_explainer(art).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = np.asarray(sv)
    vec = sv[0, :, -1] if sv.ndim == 3 else sv[0]
    agg = {}
    for nm, v in zip(pre.get_feature_names_out(), vec):
        f = base_feature(nm, art["features"])
        agg[f] = agg.get(f, 0.0) + float(v)
    items = []
    for f, v in sorted(agg.items(), key=lambda kv: -abs(kv[1])):
        txt = art["reason_text"].get(f, (f"{f} raises risk", f"{f} lowers risk"))
        items.append({"feature": f, "shap": v, "text": txt[0] if v > 0 else txt[1]})
    up = [i for i in items if i["shap"] > 0][:k]
    down = [i for i in items if i["shap"] < 0][:k]
    return up, down


# ----------------------------------------------------------------- app
art = load_artifact()
eng, model_id = get_db(art)
thr = art["threshold"]
cats = art["categories"]

st.title("Explainable and Fair Credit Risk Assessment")
st.warning("Decision-support prototype built on public Home Credit data. It is advisory only and must not be used "
           "for real lending decisions without local validation. A human reviewer makes the final decision.")
if art["env"]["scikit-learn"] != sklearn.__version__:
    st.info(f"Model trained with scikit-learn {art['env']['scikit-learn']}; this app runs {sklearn.__version__}. "
            "If scoring fails, pin the training version in requirements.txt.")

page = st.sidebar.radio("Page", ["Score an applicant", "Batch scoring", "Model information", "Monitoring", "Audit log"])
reviewer = st.sidebar.text_input("Reviewer name", value="analyst")


def choice(col, default=None):
    opts = cats.get(col, [])
    i = opts.index(default) if default in opts else 0
    return st.selectbox(col.replace("_", " ").title(), opts, index=i)


if page == "Score an applicant":
    st.header("Score an applicant")
    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        with c1:
            contract = choice("NAME_CONTRACT_TYPE", "Cash loans")
            age = st.number_input("Age (years)", 18, 100, 40)
            employed = st.number_input("Years in current employment", 0.0, 60.0, 5.0)
            children = st.number_input("Number of children", 0, 20, 0)
            fam_members = st.number_input("Family members", 1, 20, 2)
            gender = st.selectbox("Gender (optional; used only for fairness auditing, never for scoring)", ["Not provided", "F", "M"])
        with c2:
            income = st.number_input("Annual income", 1.0, 1e9, 150000.0, step=5000.0)
            credit = st.number_input("Loan amount", 1.0, 1e9, 500000.0, step=10000.0)
            annuity = st.number_input("Periodic repayment (annuity)", 1.0, 1e8, 25000.0, step=1000.0)
            goods = st.number_input("Price of goods financed", 1.0, 1e9, 450000.0, step=10000.0)
            region = st.selectbox("Region rating (1 best, 3 weakest)", [1, 2, 3], index=1)
        with c3:
            inc_type = choice("NAME_INCOME_TYPE", "Working")
            edu = choice("NAME_EDUCATION_TYPE", "Secondary / secondary special")
            fam = choice("NAME_FAMILY_STATUS", "Married")
            house = choice("NAME_HOUSING_TYPE", "House / apartment")
            car = st.selectbox("Owns a car", ["N", "Y"])
            realty = st.selectbox("Owns property", ["Y", "N"])
        has_ext = st.checkbox("External risk scores are available")
        e1 = e2 = e3 = np.nan
        if has_ext:
            e1 = st.slider("External score 1", 0.0, 1.0, 0.5); e2 = st.slider("External score 2", 0.0, 1.0, 0.5)
            e3 = st.slider("External score 3", 0.0, 1.0, 0.5)
        bur = dict(BUR_LOAN_COUNT=np.nan, BUR_ACTIVE_COUNT=np.nan, BUR_MAX_DAYS_OVERDUE=np.nan, BUR_TOTAL_OVERDUE=np.nan)
        has_bur = False
        if art["uses_bureau"]:
            has_bur = st.checkbox("Applicant has credit bureau records")
            if has_bur:
                b1, b2, b3, b4 = st.columns(4)
                bur = dict(BUR_LOAN_COUNT=b1.number_input("Previous loans", 0, 100, 2),
                           BUR_ACTIVE_COUNT=b2.number_input("Active loans", 0, 100, 1),
                           BUR_MAX_DAYS_OVERDUE=b3.number_input("Max days overdue", 0, 5000, 0),
                           BUR_TOTAL_OVERDUE=b4.number_input("Total overdue amount", 0.0, 1e9, 0.0))
        submitted = st.form_submit_button("Score applicant")
    if submitted:
        if annuity > credit * 2:
            st.error("The repayment looks implausibly high compared with the loan amount. Please check the inputs.")
            st.stop()
        inputs = {"NAME_CONTRACT_TYPE": contract, "FLAG_OWN_CAR": car, "FLAG_OWN_REALTY": realty, "CNT_CHILDREN": children,
                  "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity, "AMT_GOODS_PRICE": goods,
                  "NAME_INCOME_TYPE": inc_type, "NAME_EDUCATION_TYPE": edu, "NAME_FAMILY_STATUS": fam,
                  "NAME_HOUSING_TYPE": house, "CNT_FAM_MEMBERS": fam_members, "REGION_RATING_CLIENT": region,
                  "AGE_YEARS": age, "EMPLOYED_YEARS": employed, "EXT_SOURCE_1": e1, "EXT_SOURCE_2": e2, "EXT_SOURCE_3": e3,
                  "BUR_COVERAGE": 1.0 if has_bur else 0.0, **bur}
        row = derive(pd.DataFrame([inputs]), art)
        raw = art["model"].predict_proba(row)[:, 1]
        p = float(art["iso"].predict(raw)[0])
        band, decision = band_and_decision(p, thr)
        up, down = explain(row, art)
        pid = db.log_scoring(eng, model_id, inputs, None if gender == "Not provided" else gender, p, band, decision,
                             up + down, reviewer)
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated probability of default", f"{p:.1%}")
        m2.metric("Risk band", band)
        m3.metric("Decision support", decision)
        {"Low": st.success, "Medium": st.warning, "High": st.error}[band](f"{band} risk. {decision}. This is advisory only.")
        l, r = st.columns(2)
        l.subheader("Factors raising risk")
        for i in up or [{"text": "None identified", "feature": ""}]:
            l.write("• " + i["text"])
        r.subheader("Factors lowering risk")
        for i in down or [{"text": "None identified", "feature": ""}]:
            r.write("• " + i["text"])
        st.caption(f"Saved as prediction #{pid} in the audit trail. Model version {art['version']}.")

elif page == "Batch scoring":
    st.header("Batch scoring")
    need = ["NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT",
            "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "AGE_YEARS", "EMPLOYED_YEARS"]
    st.write("Upload a CSV with these columns (EXT_SOURCE_1..3 and BUR_* columns are optional):")
    st.code(", ".join(need))
    tmpl = pd.DataFrame([{c: "" for c in need}])
    st.download_button("Download empty template", tmpl.to_csv(index=False), "batch_template.csv")
    up_file = st.file_uploader("CSV file", type="csv")
    if up_file is not None:
        df = pd.read_csv(up_file)
        miss = [c for c in need if c not in df.columns]
        if miss:
            st.error("Missing columns: " + ", ".join(miss))
        else:
            X = derive(df, art)
            p = art["iso"].predict(art["model"].predict_proba(X)[:, 1])
            res = df.copy(); res["PD"] = p
            res["RISK_BAND"] = [band_and_decision(v, thr)[0] for v in p]
            res["DECISION_SUPPORT"] = [band_and_decision(v, thr)[1] for v in p]
            st.dataframe(res)
            st.download_button("Download results", res.to_csv(index=False), "scored.csv")
            db.log_audit(eng, reviewer, f"batch_score rows={len(df)}")

elif page == "Model information":
    st.header("Model information")
    m = art["metrics"]
    c = st.columns(4)
    c[0].metric("ROC-AUC (test)", f"{m['roc_auc_test']:.3f}"); c[1].metric("KS (test)", f"{m['ks_test']:.3f}")
    c[2].metric("Brier, calibrated", f"{m['brier_calibrated_test']:.3f}"); c[3].metric("Approval rate (test)", f"{m['approval_rate_test']:.1%}")
    st.write(f"Version **{art['version']}**. Decision threshold **{thr:.3f}** chosen on validation data assuming a missed "
             f"default costs {m['cost_ratio_fn_over_fp']:.0f} times a wrongly rejected good applicant.")
    st.subheader("Fairness audit by gender (gender is not a model input)")
    st.dataframe(pd.DataFrame(art["fairness_gender"]).set_index("group"))
    st.subheader("Intended use and limitations")
    st.markdown("- Decision support for a human reviewer, not an automated decision-maker.\n"
                "- Trained on public Home Credit data from a different market; validate locally before any real use.\n"
                "- Target is payment difficulty in an observation window, not legal default.\n"
                "- Explanations show which inputs moved the score; they do not prove causation.\n"
                "- Fairness metrics describe outcomes; they do not by themselves show a process is fair.")

elif page == "Monitoring":
    st.header("Monitoring")
    pr = db.read_table(eng, "predictions")
    if pr.empty:
        st.info("No applicants scored yet. Score one on the first page.")
    else:
        a, b = st.columns(2)
        a.metric("Applicants scored", len(pr)); b.metric("Average probability of default", f"{pr['pd_score'].mean():.1%}")
        st.subheader("Risk band distribution")
        st.bar_chart(pr["risk_band"].value_counts())
        st.subheader("Probability of default over time")
        st.line_chart(pr.sort_values("scored_at").set_index("scored_at")["pd_score"])
    st.subheader("Fairness metrics stored for this model")
    st.dataframe(db.read_table(eng, "fairness_metrics"))

elif page == "Audit log":
    st.header("Audit log")
    log = db.read_table(eng, "audit_log").sort_values("logged_at", ascending=False)
    q = st.text_input("Filter by user or action")
    if q:
        log = log[log.apply(lambda r: q.lower() in f"{r['user_name']} {r['action']}".lower(), axis=1)]
    st.dataframe(log)
    st.subheader("Explanations stored")
    st.dataframe(db.read_table(eng, "explanations").sort_values("explanation_id", ascending=False).head(100))
