"""Database layer. SQLite by default (zero setup); MySQL when DB_URL is set, e.g.
   mysql+pymysql://user:password@host:3306/credit_risk
Set DB_URL in .streamlit/secrets.toml (local) or in the Streamlit Cloud 'Secrets' box (deployed)."""
import os, json, datetime
import pandas as pd
from sqlalchemy import (create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text,
                        ForeignKey, select, insert, func)

metadata = MetaData()
model_versions = Table("model_versions", metadata,
    Column("model_id", Integer, primary_key=True, autoincrement=True), Column("version", String(60), nullable=False),
    Column("algorithm", String(50)), Column("trained_on", String(20)), Column("roc_auc_test", Float),
    Column("threshold", Float), Column("artifact_path", String(255)))
applicants = Table("applicants", metadata,
    Column("applicant_id", Integer, primary_key=True, autoincrement=True), Column("created_at", DateTime),
    Column("gender_audit_only", String(12)), Column("family_status", String(40)), Column("education", String(60)),
    Column("income_total", Float), Column("credit_amount", Float), Column("annuity", Float), Column("children", Integer),
    Column("employment_years", Float), Column("inputs_json", Text))
predictions = Table("predictions", metadata,
    Column("prediction_id", Integer, primary_key=True, autoincrement=True),
    Column("applicant_id", Integer, ForeignKey("applicants.applicant_id"), nullable=False),
    Column("model_id", Integer, ForeignKey("model_versions.model_id"), nullable=False),
    Column("pd_score", Float, nullable=False), Column("risk_band", String(10), nullable=False),
    Column("decision_support", String(60)), Column("reviewer", String(60)), Column("scored_at", DateTime))
explanations = Table("explanations", metadata,
    Column("explanation_id", Integer, primary_key=True, autoincrement=True),
    Column("prediction_id", Integer, ForeignKey("predictions.prediction_id"), nullable=False),
    Column("feature_name", String(80), nullable=False), Column("shap_value", Float), Column("reason_text", String(200)),
    Column("rank_order", Integer))
audit_log = Table("audit_log", metadata,
    Column("log_id", Integer, primary_key=True, autoincrement=True), Column("user_name", String(60), nullable=False),
    Column("action", String(80), nullable=False), Column("prediction_id", Integer, ForeignKey("predictions.prediction_id")),
    Column("logged_at", DateTime))
fairness_metrics = Table("fairness_metrics", metadata,
    Column("metric_id", Integer, primary_key=True, autoincrement=True),
    Column("model_id", Integer, ForeignKey("model_versions.model_id"), nullable=False),
    Column("group_name", String(30)), Column("metric_name", String(60)), Column("metric_value", Float),
    Column("computed_at", DateTime))


def _now():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def get_engine(db_url=None):
    url = db_url or os.environ.get("DB_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("DB_URL")
        except Exception:
            url = None
    url = url or "sqlite:///credit_risk.db"
    eng = create_engine(url, pool_pre_ping=True)
    metadata.create_all(eng)
    return eng


def ensure_model(eng, art, artifact_path="model_artifact.joblib"):
    with eng.begin() as c:
        row = c.execute(select(model_versions.c.model_id).where(model_versions.c.version == art["version"])).first()
        if row:
            return row[0]
        r = c.execute(insert(model_versions).values(version=art["version"], algorithm="LightGBM + isotonic calibration",
                      trained_on=art["version"].split("-")[-1], roc_auc_test=art["metrics"]["roc_auc_test"],
                      threshold=art["threshold"], artifact_path=artifact_path))
        mid = r.inserted_primary_key[0]
        for g in art["fairness_gender"]:
            for k in ("approval_rate", "approval_among_repaid", "approval_among_defaulters"):
                c.execute(insert(fairness_metrics).values(model_id=mid, group_name=g["group"], metric_name=k,
                                                          metric_value=g[k], computed_at=_now()))
        return mid


def log_scoring(eng, model_id, inputs, gender, pd_score, band, decision, reasons, reviewer):
    """reasons: list of dicts with feature, shap, text. Returns prediction_id."""
    with eng.begin() as c:
        a = c.execute(insert(applicants).values(
            created_at=_now(), gender_audit_only=gender, family_status=str(inputs.get("NAME_FAMILY_STATUS")),
            education=str(inputs.get("NAME_EDUCATION_TYPE")), income_total=float(inputs["AMT_INCOME_TOTAL"]),
            credit_amount=float(inputs["AMT_CREDIT"]), annuity=float(inputs["AMT_ANNUITY"]),
            children=int(inputs["CNT_CHILDREN"]), employment_years=float(inputs["EMPLOYED_YEARS"]),
            inputs_json=json.dumps({k: (None if pd.isna(v) else v) for k, v in inputs.items()}, default=str)))
        aid = a.inserted_primary_key[0]
        p = c.execute(insert(predictions).values(applicant_id=aid, model_id=model_id, pd_score=float(pd_score),
                      risk_band=band, decision_support=decision, reviewer=reviewer, scored_at=_now()))
        pid = p.inserted_primary_key[0]
        for i, r in enumerate(reasons, 1):
            c.execute(insert(explanations).values(prediction_id=pid, feature_name=r["feature"], shap_value=float(r["shap"]),
                                                  reason_text=r["text"], rank_order=i))
        c.execute(insert(audit_log).values(user_name=reviewer, action="score_applicant", prediction_id=pid, logged_at=_now()))
        return pid


def log_audit(eng, user, action):
    with eng.begin() as c:
        c.execute(insert(audit_log).values(user_name=user, action=action, logged_at=_now()))


def read_table(eng, table_name):
    return pd.read_sql_table(table_name, eng)
