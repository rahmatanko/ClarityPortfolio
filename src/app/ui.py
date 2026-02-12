import sys
import os

# make project root importable when running via Streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import streamlit as st

from src.app.wrapper import generate_recommendation


st.set_page_config(
    page_title="Confidence-Aware Financial Advisor",
    page_icon="📈",
    layout="wide",
)


def make_allocation_df(allocation: dict[str, float]) -> pd.DataFrame:
    # convert allocation mapping into a clean dataframe for plotting and display
    df = pd.DataFrame(
        {
            "Asset": list(allocation.keys()),
            "Weight": list(allocation.values()),
        }
    )
    df["Weight (%)"] = df["Weight"] * 100.0
    return df.sort_values("Weight", ascending=False).reset_index(drop=True)


def confidence_color(label: str) -> str:
    # simple color mapping for confidence badge styling
    if label == "High":
        return "#16a34a"
    if label == "Moderate":
        return "#d97706"
    return "#dc2626"


def explanation_card(group: dict) -> str:
    # render one explanation group as lightweight HTML
    increased = ", ".join(group["increased_assets"]) if group["increased_assets"] else "None"
    decreased = ", ".join(group["decreased_assets"]) if group["decreased_assets"] else "None"

    return f"""
    <div style="
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 16px;
        background: #ffffff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    ">
        <div style="font-size: 1.05rem; font-weight: 600; margin-bottom: 8px;">
            {group['group'].title()}
        </div>
        <div style="font-size: 0.95rem; color: #374151; margin-bottom: 6px;">
            <strong>Effect size:</strong> {group['effect_size']:.4f}
        </div>
        <div style="font-size: 0.95rem; color: #065f46; margin-bottom: 4px;">
            <strong>Increased:</strong> {increased}
        </div>
        <div style="font-size: 0.95rem; color: #991b1b;">
            <strong>Decreased:</strong> {decreased}
        </div>
    </div>
    """


def main():
    st.markdown(
        """
        <h1 style="margin-bottom: 0.2rem;">Confidence-Aware Financial Advisor</h1>
        <p style="color: #4b5563; margin-top: 0;">
            Ensemble portfolio recommendation with disagreement-based confidence and perturbation-based explanations.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Inference Settings")
        split_name = st.selectbox("Dataset split", ["val", "test", "train"], index=0)
        row_mode = st.selectbox("Observation", ["Latest", "Custom index"], index=0)

        if row_mode == "Latest":
            row_idx = -1
        else:
            row_idx = st.number_input("Row index", min_value=0, value=0, step=1)

        confidence_alpha = st.slider("Confidence sensitivity", 1.0, 20.0, 10.0, 0.5)
        perturbation_delta = st.slider("Perturbation delta", 0.01, 0.50, 0.10, 0.01)
        explanation_top_k = st.slider("Top explanation groups", 1, 5, 3, 1)

        run_button = st.button("Generate Recommendation", use_container_width=True)

    if not run_button:
        st.info("Choose settings in the sidebar, then click **Generate Recommendation**.")
        return

    with st.spinner("Running ensemble inference..."):
        recommendation = generate_recommendation(
            split_name=split_name,
            row_idx=row_idx,
            confidence_alpha=confidence_alpha,
            perturbation_delta=perturbation_delta,
            explanation_top_k=explanation_top_k,
            device="cpu",
        )

    allocation_df = make_allocation_df(recommendation["allocation"])

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Timestamp", str(recommendation["timestamp"]).split(" ")[0])
    col2.metric("Confidence", f"{recommendation['confidence'] * 100:.1f}%")
    col3.metric("Confidence Band", recommendation["confidence_label"])
    col4.metric("Disagreement", f"{recommendation['disagreement']:.4f}")

    badge_color = confidence_color(recommendation["confidence_label"])
    st.markdown(
        f"""
        <div style="
            display: inline-block;
            padding: 10px 14px;
            border-radius: 999px;
            background: {badge_color};
            color: white;
            font-weight: 600;
            margin-top: 8px;
            margin-bottom: 8px;
        ">
            {recommendation["confidence_label"]} Confidence
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(recommendation["confidence_note"])

    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Recommended Allocation")
        st.bar_chart(
            allocation_df.set_index("Asset")["Weight (%)"],
            use_container_width=True,
        )
        st.dataframe(
            allocation_df[["Asset", "Weight (%)"]],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Recommendation Summary")
        st.markdown(
            f"""
            <div style="
                border: 1px solid #e5e7eb;
                border-radius: 18px;
                padding: 18px;
                background: #ffffff;
                box-shadow: 0 2px 10px rgba(0,0,0,0.04);
                line-height: 1.6;
            ">
                {recommendation["summary"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Top Explanation Groups")
    for group in recommendation["top_explanation_groups"]:
        st.markdown(explanation_card(group), unsafe_allow_html=True)


if __name__ == "__main__":
    main()