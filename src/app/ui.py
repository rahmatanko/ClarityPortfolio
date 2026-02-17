import sys
import os

# make project root importable when running via Streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import streamlit as st

from src.app.wrapper import generate_recommendation
from src.eval.explanation_validation import validate_explanations_across_regimes
from src.models.ensemble import prepare_inference_splits


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


def regime_note(row: pd.Series) -> str:
    # lightweight interpretation helper for the validation tab
    notes = []

    if row["confidence"] < 0.2:
        notes.append("low confidence")
    elif row["confidence"] < 0.5:
        notes.append("moderate confidence")
    else:
        notes.append("higher confidence")

    if row["top_group_1"] == "volatility":
        notes.append("volatility-driven explanation")

    if row["top_alloc_1"] == "CASH":
        notes.append("strong risk-off allocation")

    return ", ".join(notes)


@st.cache_data(show_spinner=False)
def cached_regime_validation(confidence_alpha: float, perturbation_delta: float, top_k: int):
    # cache the expensive regime validation so the UI stays responsive
    return validate_explanations_across_regimes(
        confidence_alpha=confidence_alpha,
        perturbation_delta=perturbation_delta,
        top_k=top_k,
    )


@st.cache_data(show_spinner=False)
def cached_splits():
    # cache prepared splits so date controls stay responsive
    return prepare_inference_splits()


def resolve_row_idx(
    features: pd.DataFrame,
    selection_mode: str,
    selected_date,
    slider_index: int,
) -> int:
    # convert the user's date/slider selection into the row index expected by the wrapper
    if selection_mode == "Latest":
        return -1

    if selection_mode == "Specific date":
        ts = pd.Timestamp(selected_date)
        return int(features.index.get_indexer([ts], method="nearest")[0])

    return int(slider_index)


def render_recommendation_tab():
    st.subheader("Live Recommendation")
    st.caption(
        "Choose a dataset split and a point in time, then generate a portfolio recommendation "
        "with confidence and explanation."
    )

    splits = cached_splits()

    col1, col2 = st.columns([1, 1])

    with col1:
        split_name = st.selectbox(
            "Dataset split",
            ["val", "test", "train"],
            index=0,
            help="Validation usually shows the latest model behaviour most clearly.",
        )

    features = splits[f"{split_name}_features"]
    prices = splits[f"{split_name}_prices"]

    min_date = features.index.min().date()
    max_date = features.index.max().date()

    with col2:
        selection_mode = st.radio(
            "Choose observation by",
            ["Latest", "Specific date", "Slider"],
            horizontal=True,
            help="Latest = most recent row in the selected split.",
        )

    st.info(
        f"Available dates for **{split_name}**: "
        f"**{min_date}** to **{max_date}** "
        f"({len(features)} observations)"
    )

    selected_date = None
    slider_index = len(features) - 1

    if selection_mode == "Specific date":
        selected_date = st.date_input(
            "Pick a date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            help="The nearest available trading date will be used.",
        )
    elif selection_mode == "Slider":
        slider_index = st.slider(
            "Observation index",
            min_value=0,
            max_value=len(features) - 1,
            value=len(features) - 1,
            step=1,
            help="Moves through the selected split chronologically.",
        )
        preview_date = features.index[slider_index].date()
        st.caption(f"Selected date: {preview_date}")

    with st.expander("Advanced settings", expanded=False):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            confidence_alpha = st.slider(
                "Confidence sensitivity",
                1.0, 20.0, 10.0, 0.5,
                help="Higher values make confidence drop faster as disagreement rises.",
            )
        with col_b:
            perturbation_delta = st.slider(
                "Perturbation delta",
                0.01, 0.50, 0.10, 0.01,
                help="Controls how strongly feature groups are perturbed during explanation.",
            )
        with col_c:
            explanation_top_k = st.slider(
                "Top explanation groups",
                1, 5, 3, 1,
                help="Number of explanation groups shown in the output.",
            )

    if "confidence_alpha" not in locals():
        confidence_alpha = 10.0
    if "perturbation_delta" not in locals():
        perturbation_delta = 0.10
    if "explanation_top_k" not in locals():
        explanation_top_k = 3

    run_button = st.button("Generate Recommendation", use_container_width=True, type="primary")

    if not run_button:
        st.info(
            "Pick a split and a date selection mode, then click **Generate Recommendation**. "
            "If you keep **Latest**, the timestamp will remain the latest available date for that split."
        )
        return

    row_idx = resolve_row_idx(
        features=features,
        selection_mode=selection_mode,
        selected_date=selected_date,
        slider_index=slider_index,
    )

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

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Timestamp", str(recommendation["timestamp"]).split(" ")[0])
    metric2.metric("Confidence", f"{recommendation['confidence'] * 100:.1f}%")
    metric3.metric("Confidence Band", recommendation["confidence_label"])
    metric4.metric("Disagreement", f"{recommendation['disagreement']:.4f}")

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


def render_validation_tab():
    st.subheader("Explanation Validation Across Market Regimes")
    st.caption(
        "This view checks whether the explanation engine remains financially coherent "
        "across selected crash, uncertainty, and recovery periods."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        confidence_alpha = st.slider(
            "Validation confidence sensitivity",
            1.0, 20.0, 10.0, 0.5,
            key="val_conf_alpha",
        )
    with col2:
        perturbation_delta = st.slider(
            "Validation perturbation delta",
            0.01, 0.50, 0.10, 0.01,
            key="val_perturb_delta",
        )
    with col3:
        top_k = st.slider(
            "Validation top explanation groups",
            1, 5, 3, 1,
            key="val_top_k",
        )

    if st.button("Run Regime Validation", use_container_width=True):
        with st.spinner("Validating explanation coherence across regimes..."):
            result = cached_regime_validation(
                confidence_alpha=confidence_alpha,
                perturbation_delta=perturbation_delta,
                top_k=top_k,
            )

        df = result["table"].copy()
        df["interpretation"] = df.apply(regime_note, axis=1)

        display_cols = [
            "regime",
            "actual_timestamp",
            "confidence",
            "disagreement",
            "top_group_1",
            "top_group_2",
            "top_group_3",
            "top_alloc_1",
            "top_alloc_1_weight",
            "top_alloc_2",
            "top_alloc_2_weight",
            "top_alloc_3",
            "top_alloc_3_weight",
            "interpretation",
        ]

        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        st.markdown("### Regime Notes")
        for _, row in df.iterrows():
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #e5e7eb;
                    border-radius: 16px;
                    padding: 16px;
                    background: #ffffff;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
                    margin-bottom: 12px;
                ">
                    <div style="font-size: 1.05rem; font-weight: 600; margin-bottom: 6px;">
                        {row['regime']} ({row['actual_timestamp']})
                    </div>
                    <div style="margin-bottom: 4px;">
                        <strong>Confidence:</strong> {row['confidence']:.3f}
                    </div>
                    <div style="margin-bottom: 4px;">
                        <strong>Top explanation groups:</strong> {row['top_group_1']}, {row['top_group_2']}, {row['top_group_3']}
                    </div>
                    <div style="margin-bottom: 4px;">
                        <strong>Top allocations:</strong> {row['top_alloc_1']} ({row['top_alloc_1_weight']:.1%}), {row['top_alloc_2']} ({row['top_alloc_2_weight']:.1%}), {row['top_alloc_3']} ({row['top_alloc_3_weight']:.1%})
                    </div>
                    <div>
                        <strong>Interpretation:</strong> {row['interpretation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.success(f"Saved artifacts: {result['csv_path']} and {result['json_path']}")


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

    tab1, tab2 = st.tabs(["Live Recommendation", "Explanation Validation"])

    with tab1:
        render_recommendation_tab()

    with tab2:
        render_validation_tab()


if __name__ == "__main__":
    main()