import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from data_preprocessing import load_or_generate_dataset
from feature_engineering import build_user_profiles, engineer_features
from models import ModelBundle, predict_for_features, train_models
from simulation import build_stream, next_batch
from explainability import explain_transaction


st.set_page_config(
    page_title="Behavioral FinTech Fraud Detection",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return load_or_generate_dataset()


@st.cache_data(show_spinner=False)
def prepare_features(df: pd.DataFrame):
    profiles = build_user_profiles(df)
    X, y = engineer_features(df, profiles)
    return profiles, X, y


@st.cache_resource(show_spinner=False)
def train_bundle(X: pd.DataFrame, y: pd.Series):
    bundle, eval_df = train_models(X, y)
    return bundle, eval_df


def render_overview_tab(df: pd.DataFrame, bundle: ModelBundle, eval_df: pd.DataFrame):
    st.subheader("Dataset & Model Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total transactions", f"{len(df):,}")
    with col2:
        fraud_rate = df["is_fraud"].mean() if "is_fraud" in df.columns else 0.0
        st.metric("Fraud rate", f"{100*fraud_rate:.2f}%")
    with col3:
        st.metric("Users", df["user_id"].nunique())

    st.markdown("### Model Performance (Test Set)")
    metrics_df = []
    for name, m in bundle.metrics.items():
        metrics_df.append(
            {
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
            }
        )
    st.dataframe(pd.DataFrame(metrics_df).set_index("Model").style.format("{:.3f}"))

    st.markdown("#### Confusion Matrices")
    cm_cols = st.columns(3)
    for (name, m), c in zip(bundle.metrics.items(), cm_cols):
        with c:
            cm = np.array(m["confusion_matrix"])
            cm_df = pd.DataFrame(
                cm,
                index=["Actual 0", "Actual 1"],
                columns=["Pred 0", "Pred 1"],
            )
            fig = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Blues",
                title=name,
            )
            st.plotly_chart(fig, width="stretch")


def render_live_tab(
    df: pd.DataFrame,
    profiles,
    X: pd.DataFrame,
    bundle: ModelBundle,
):
    st.subheader("Live-style Transaction Stream & Alerts")
    st.markdown(
        "Simulated stream of transactions with **behavioral fraud probability** and explanations."
    )

    if "stream_df" not in st.session_state:
        st.session_state["stream_df"] = build_stream(df, X)
        st.session_state["stream_idx"] = 0
        st.session_state["history"] = pd.DataFrame()

    batch_size = st.slider("Transactions per step", 1, 20, 5, 1)
    proba_threshold = st.slider(
        "Fraud probability threshold",
        0.1,
        0.9,
        0.6,
        0.05,
    )

    if st.button("Next transactions"):
        batch_df, next_idx = next_batch(
            st.session_state["stream_df"],
            st.session_state["stream_idx"],
            batch_size=batch_size,
        )
        st.session_state["stream_idx"] = next_idx

        feature_cols = bundle.feature_columns
        feat_batch = batch_df[feature_cols]
        preds = predict_for_features(bundle, feat_batch)

        explanations = []
        for i in range(len(batch_df)):
            idx = batch_df.index[i]
            row = batch_df.iloc[i]
            feat_row = feat_batch.iloc[i]
            user_profile = profiles.get(row["user_id"])
            pred_row = preds.iloc[i]

            exp = explain_transaction(
                tx_row=row,
                feat_row=feat_row,
                user_profile=user_profile,
                lr_prob=float(pred_row["lr_prob"]),
                rf_prob=float(pred_row["rf_prob"]),
                iso_score=float(pred_row["iso_score"]),
                iso_flag=int(pred_row["iso_flag"]),
                proba_threshold=proba_threshold,
            )
            explanations.append(exp)

        batch_view = batch_df[
            ["user_id", "amount", "timestamp", "location", "device_id", "is_fraud"]
        ].copy()
        batch_view["fraud_prob"] = [e["fraud_probability"] for e in explanations]
        batch_view["iso_score"] = [e["iso_score"] for e in explanations]
        batch_view["flagged"] = [e["is_flagged"] for e in explanations]
        batch_view["reasons"] = ["; ".join(e["reasons"]) for e in explanations]

        st.session_state["history"] = pd.concat(
            [st.session_state["history"], batch_view], axis=0
        ).tail(200)

    history = st.session_state["history"]
    if not history.empty:
        st.markdown("### Recent Transactions")
        st.dataframe(
            history.sort_index(ascending=False),
            use_container_width=True,
        )

        flagged = history[history["flagged"] == True]
        if not flagged.empty:
            st.markdown("### Latest Fraud Alerts")
            for _, row in flagged.tail(10).iterrows():
                st.markdown(
                    f"- **User** `{row['user_id']}` | **₹{row['amount']:.0f}`` "
                    f"| **Prob** {row['fraud_prob']:.2f} | **Reasons:** {row['reasons']}"
                )


def render_behavior_tab(df: pd.DataFrame):
    st.subheader("Behavioral Analytics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Amount distribution by fraud label")
        fig1 = px.histogram(
            df,
            x="amount",
            color="is_fraud",
            nbins=50,
            marginal="box",
            log_y=True,
            title="Transaction amounts (log scale)",
        )
        st.plotly_chart(fig1, width="stretch")

    with col2:
        st.markdown("#### Transaction hour distribution")
        df_hour = df.copy()
        df_hour["hour"] = df_hour["timestamp"].dt.hour
        fig2 = px.histogram(
            df_hour,
            x="hour",
            color="is_fraud",
            nbins=24,
            title="Hourly activity (fraud vs normal)",
        )
        st.plotly_chart(fig2, width="stretch")

    st.markdown("#### Top users by average spend")
    top_users = (
        df.groupby("user_id")["amount"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    fig3 = px.bar(
        top_users,
        x="user_id",
        y="amount",
        title="Top 15 users by average spend",
    )
    st.plotly_chart(fig3, width="stretch")


def main():
    st.title("Behavioral FinTech Fraud Detection Dashboard")
    st.caption(
        "AI-powered behavioral fraud detection that learns user transaction patterns "
        "and flags suspicious activity with explainable real-time insights."
    )

    with st.spinner("Loading data and training models..."):
        df = load_data()
        profiles, X, y = prepare_features(df)
        bundle, eval_df = train_bundle(X, y)

    tab_overview, tab_live, tab_beh = st.tabs(
        ["📊 Overview", "🚨 Live Simulation", "📈 Behavioral Analytics"]
    )

    with tab_overview:
        render_overview_tab(df, bundle, eval_df)

    with tab_live:
        render_live_tab(df, profiles, X, bundle)

    with tab_beh:
        render_behavior_tab(df)


if __name__ == "__main__":
    main()

