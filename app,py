"""
Predictive Maintenance Dashboard
--------------------------------
A real-world-style Streamlit app for monitoring machine health, comparing
ML models (Decision Tree / Random Forest / XGBoost), and generating
maintenance recommendations for a fleet of machines.

Run with:
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLS = [
    "Air_Temp", "Process_Temp", "RPM", "Torque",
    "Tool_Wear", "Machine_Age", "Vibration", "Voltage",
]
TARGET_COL = "Failure"

# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_sample_data(n=1000, seed=42):
    """Generates a realistic synthetic predictive-maintenance dataset,
    used only when the user hasn't uploaded their own CSV."""
    rng = np.random.default_rng(seed)

    machine_age = rng.uniform(0, 15, n)
    tool_wear = np.clip(rng.normal(50, 25, n) + machine_age * 2, 0, 250)
    rpm = rng.normal(1500, 300, n)
    torque = rng.normal(40, 12, n)
    air_temp = rng.normal(298, 4, n)
    process_temp = air_temp + rng.normal(10, 2, n)
    vibration = np.clip(rng.normal(0.4, 0.2, n) + tool_wear / 500, 0, None)
    voltage = rng.normal(220, 8, n)

    # Failure probability rises with tool wear, vibration, torque and age
    risk_score = (
        0.05 * tool_wear
        + 4.5 * vibration
        + 0.05 * torque
        + 0.18 * machine_age
        + 0.15 * (process_temp - 305).clip(min=0)
        - 12.5
    )
    prob_fail = np.clip(1 / (1 + np.exp(-risk_score)), 0.01, 0.97)
    failure = rng.binomial(1, prob_fail)
    failure_label = np.where(failure == 1, "Yes", "No")

    machine_id = [f"MC-{1000 + i}" for i in range(n)]

    df = pd.DataFrame({
        "Machine_ID": machine_id,
        "Air_Temp": air_temp.round(2),
        "Process_Temp": process_temp.round(2),
        "RPM": rpm.round(0),
        "Torque": torque.round(2),
        "Tool_Wear": tool_wear.round(1),
        "Machine_Age": machine_age.round(1),
        "Vibration": vibration.round(3),
        "Voltage": voltage.round(1),
        "Failure": failure_label,
    })
    return df


def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df, "Uploaded dataset"
    else:
        return generate_sample_data(), "Synthetic demo dataset (1,000 machines)"


# ----------------------------------------------------------------------
# MODEL TRAINING (cached)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def train_models(df: pd.DataFrame):
    data = df.copy()

    # Keep only usable feature columns that actually exist in this data
    available_features = [c for c in FEATURE_COLS if c in data.columns]
    if len(available_features) < 3:
        # fall back: use all numeric columns except target
        available_features = [c for c in data.select_dtypes(include=np.number).columns
                               if c != TARGET_COL]

    le = LabelEncoder()
    if data[TARGET_COL].dtype == object:
        data[TARGET_COL] = le.fit_transform(data[TARGET_COL])
        classes_ = le.classes_
    else:
        classes_ = np.array(["No", "Yes"])

    X = data[available_features]
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss", random_state=42, verbosity=0
        )

    trained = {}
    preds = {}
    probs = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        preds[name] = model.predict(X_test)
        try:
            probs[name] = model.predict_proba(X_test)[:, 1]
        except Exception:
            probs[name] = preds[name].astype(float)

    return {
        "trained": trained,
        "preds": preds,
        "probs": probs,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "features": available_features,
        "classes_": classes_,
    }


def compute_metrics(y_test, preds_dict):
    rows = []
    for name, pred in preds_dict.items():
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, pred, zero_division=0), 4),
        })
    return pd.DataFrame(rows).sort_values("F1 Score", ascending=False)


def risk_band(prob):
    if prob >= 0.75:
        return "🔴 Critical"
    elif prob >= 0.45:
        return "🟠 High"
    elif prob >= 0.20:
        return "🟡 Medium"
    else:
        return "🟢 Low"


def maintenance_recommendation(row, prob):
    """Generates a plain-English, real-world style maintenance action."""
    reasons = []
    if "Tool_Wear" in row and row["Tool_Wear"] > 150:
        reasons.append("tool wear is critically high — schedule tool/insert replacement")
    elif "Tool_Wear" in row and row["Tool_Wear"] > 100:
        reasons.append("tool wear is elevated — inspect cutting tool")

    if "Vibration" in row and row["Vibration"] > 0.7:
        reasons.append("vibration levels exceed safe threshold — check bearings/alignment")

    if "Process_Temp" in row and row["Process_Temp"] > 312:
        reasons.append("process temperature is above normal range — check cooling system")

    if "Torque" in row and row["Torque"] > 60:
        reasons.append("torque readings are unusually high — inspect drive train")

    if not reasons:
        if prob >= 0.45:
            reasons.append("model flags elevated failure risk — recommend general inspection")
        else:
            return "No action needed. Machine operating within normal parameters."

    band = risk_band(prob)
    urgency = {
        "🔴 Critical": "IMMEDIATE action required (within 24 hours):",
        "🟠 High": "Schedule maintenance within this week:",
        "🟡 Medium": "Plan maintenance within 2-4 weeks:",
        "🟢 Low": "Monitor during routine checks:",
    }[band]
    return urgency + " " + "; ".join(reasons) + "."


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
st.sidebar.title("🛠️ Control Panel")
st.sidebar.markdown("Upload your machine sensor CSV, or explore with demo data.")

uploaded_file = st.sidebar.file_uploader("Upload CSV (must include a 'Failure' column)", type=["csv"])
df, data_source_label = load_data(uploaded_file)

st.sidebar.info(f"📊 Data source: **{data_source_label}**")
st.sidebar.metric("Total Machines / Records", f"{len(df):,}")

if TARGET_COL not in df.columns:
    st.sidebar.error("No 'Failure' column found — please upload a valid dataset or use demo data.")
    st.stop()

fail_rate = (df[TARGET_COL].astype(str).str.lower().isin(["yes", "1", "true"])).mean()
st.sidebar.metric("Historical Failure Rate", f"{fail_rate*100:.1f}%")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · scikit-learn · XGBoost")

# ----------------------------------------------------------------------
# HEADER / KPIs
# ----------------------------------------------------------------------
st.title("🛠️ Predictive Maintenance Dashboard")
st.caption("Monitor machine health, predict failures before they happen, and prioritize maintenance action.")

model_bundle = train_models(df)
metrics_df = compute_metrics(model_bundle["y_test"], model_bundle["preds"])
best_model_name = metrics_df.iloc[0]["Model"]
best_model = model_bundle["trained"][best_model_name]
best_probs = model_bundle["probs"][best_model_name]

# Fleet-wide risk scoring using the best model on the FULL dataset
X_full = df[model_bundle["features"]].copy()
full_probs = best_model.predict_proba(X_full)[:, 1] if hasattr(best_model, "predict_proba") else best_model.predict(X_full)
df_scored = df.copy()
df_scored["Failure_Probability"] = full_probs
df_scored["Risk_Level"] = df_scored["Failure_Probability"].apply(risk_band)

critical_count = (df_scored["Risk_Level"] == "🔴 Critical").sum()
high_count = (df_scored["Risk_Level"] == "🟠 High").sum()
avg_wear = df_scored["Tool_Wear"].mean() if "Tool_Wear" in df_scored else np.nan
est_cost_avoided = critical_count * 15000 + high_count * 5000  # illustrative downtime-cost model

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Machines Monitored", f"{len(df_scored):,}")
k2.metric("🔴 Critical Risk", f"{critical_count}")
k3.metric("🟠 High Risk", f"{high_count}")
k4.metric("Avg Tool Wear", f"{avg_wear:.1f}" if not np.isnan(avg_wear) else "N/A")
k5.metric("Est. Downtime Cost at Risk", f"${est_cost_avoided:,.0f}",
          help="Illustrative: assumes $15k avg cost per unplanned failure for critical-risk machines, $5k for high-risk.")

if critical_count > 0:
    st.error(f"⚠️ {critical_count} machine(s) are at CRITICAL risk of failure. See the Fleet Monitor tab for details.")
elif high_count > 0:
    st.warning(f"{high_count} machine(s) are at HIGH risk — plan maintenance soon.")
else:
    st.success("✅ No machines currently flagged at critical or high risk.")

st.markdown("---")

# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab_overview, tab_eda, tab_models, tab_fleet, tab_predict = st.tabs(
    ["📋 Data Overview", "📈 EDA", "🤖 Model Comparison", "🚨 Fleet Risk Monitor", "🔮 Predict & Recommend"]
)

# ---- TAB 1: OVERVIEW ----
with tab_overview:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    with c2:
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values detected.")
        else:
            st.dataframe(missing.rename("Missing Count"), use_container_width=True)

    st.subheader("Failure Class Distribution")
    fail_counts = df[TARGET_COL].value_counts().reset_index()
    fail_counts.columns = ["Failure", "Count"]
    fig = px.pie(fail_counts, names="Failure", values="Count", hole=0.45,
                 color="Failure", color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71", 1: "#e74c3c", 0: "#2ecc71"})
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 2: EDA ----
with tab_eda:
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != TARGET_COL]

    st.subheader("Feature Distribution")
    sel_col = st.selectbox("Choose a feature to explore", numeric_cols)
    fig = px.histogram(df, x=sel_col, color=TARGET_COL, marginal="box", barmode="overlay", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    if "Tool_Wear" in df.columns:
        st.subheader("Tool Wear vs Failure")
        fig2 = px.box(df, x=TARGET_COL, y="Tool_Wear", color=TARGET_COL)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig3, use_container_width=True)

    if len(numeric_cols) >= 2:
        st.subheader("Feature Relationship Explorer")
        cc1, cc2 = st.columns(2)
        x_axis = cc1.selectbox("X-axis", numeric_cols, index=0)
        y_axis = cc2.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
        fig4 = px.scatter(df, x=x_axis, y=y_axis, color=TARGET_COL, opacity=0.6)
        st.plotly_chart(fig4, use_container_width=True)

# ---- TAB 3: MODEL COMPARISON ----
with tab_models:
    st.subheader("Model Performance Comparison")
    st.dataframe(metrics_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score"], color="#2ecc71"),
                 use_container_width=True)

    fig = px.bar(metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                 x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"🏆 Best performing model: **{best_model_name}** (highest F1 Score)")

    st.subheader("Confusion Matrices")
    cols = st.columns(len(model_bundle["preds"]))
    for col, (name, pred) in zip(cols, model_bundle["preds"].items()):
        cm = confusion_matrix(model_bundle["y_test"], pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual"),
                            x=["No Failure", "Failure"], y=["No Failure", "Failure"])
        fig_cm.update_layout(title=name, height=350)
        col.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    for name, prob in model_bundle["probs"].items():
        fpr, tpr, _ = roc_curve(model_bundle["y_test"], prob)
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={roc_auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=450)
    st.plotly_chart(fig_roc, use_container_width=True)

    if hasattr(best_model, "feature_importances_"):
        st.subheader(f"Feature Importance — {best_model_name}")
        imp_df = pd.DataFrame({
            "Feature": model_bundle["features"],
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig_imp, use_container_width=True)

# ---- TAB 4: FLEET RISK MONITOR ----
with tab_fleet:
    st.subheader("Fleet-Wide Risk Assessment")
    st.caption(f"Every machine scored using the best model ({best_model_name}) on its current sensor readings.")

    risk_filter = st.multiselect(
        "Filter by risk level",
        options=["🔴 Critical", "🟠 High", "🟡 Medium", "🟢 Low"],
        default=["🔴 Critical", "🟠 High"]
    )

    display_cols = [c for c in ["Machine_ID"] if c in df_scored.columns] + \
                    model_bundle["features"] + ["Failure_Probability", "Risk_Level"]
    filtered = df_scored[display_cols].sort_values("Failure_Probability", ascending=False)
    if risk_filter:
        filtered = filtered[filtered["Risk_Level"].isin(risk_filter)]

    st.dataframe(
        filtered.style.format({"Failure_Probability": "{:.1%}"}),
        use_container_width=True, height=420
    )

    csv_buf = io.StringIO()
    filtered.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download Risk Report (CSV)", csv_buf.getvalue(),
                        file_name="fleet_risk_report.csv", mime="text/csv")

    st.subheader("Risk Level Breakdown")
    risk_counts = df_scored["Risk_Level"].value_counts().reindex(
        ["🔴 Critical", "🟠 High", "🟡 Medium", "🟢 Low"]).fillna(0).reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    fig_risk = px.bar(risk_counts, x="Risk Level", y="Count", color="Risk Level",
                       color_discrete_map={"🔴 Critical": "#e74c3c", "🟠 High": "#e67e22",
                                           "🟡 Medium": "#f1c40f", "🟢 Low": "#2ecc71"})
    st.plotly_chart(fig_risk, use_container_width=True)

# ---- TAB 5: PREDICT & RECOMMEND ----
with tab_predict:
    st.subheader("Simulate a Machine Reading")
    st.caption("Enter live sensor values (or pick a machine from the dataset) to get an instant risk assessment and maintenance recommendation.")

    source_choice = st.radio("Input method", ["Manual entry", "Pick existing machine"], horizontal=True)

    input_row = {}
    if source_choice == "Pick existing machine" and "Machine_ID" in df.columns:
        machine_pick = st.selectbox("Machine ID", df["Machine_ID"].unique())
        base_row = df[df["Machine_ID"] == machine_pick].iloc[0]
        for f in model_bundle["features"]:
            input_row[f] = float(base_row[f])
    else:
        cols = st.columns(4)
        defaults = df[model_bundle["features"]].mean()
        for i, f in enumerate(model_bundle["features"]):
            with cols[i % 4]:
                lo, hi = float(df[f].min()), float(df[f].max())
                input_row[f] = st.slider(f, lo, hi, float(defaults[f]))

    input_df = pd.DataFrame([input_row])[model_bundle["features"]]

    if st.button("🔍 Predict Failure Risk", type="primary"):
        prob = best_model.predict_proba(input_df)[0, 1] if hasattr(best_model, "predict_proba") else float(best_model.predict(input_df)[0])
        band = risk_band(prob)
        rec = maintenance_recommendation(input_row, prob)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Failure Probability", f"{prob*100:.1f}%")
            st.metric("Risk Level", band)
        with c2:
            st.info(f"**Recommended Action:** {rec}")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={"text": "Failure Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 20], "color": "#2ecc71"},
                    {"range": [20, 45], "color": "#f1c40f"},
                    {"range": [45, 75], "color": "#e67e22"},
                    {"range": [75, 100], "color": "#e74c3c"},
                ],
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")
st.caption("⚠️ This dashboard uses illustrative cost figures and synthetic demo data when no CSV is uploaded. "
           "For production use, connect to your real sensor/IoT data pipeline and validate model thresholds with domain experts.")
