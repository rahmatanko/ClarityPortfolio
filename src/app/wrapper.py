from typing import Any

from src.explain.sensitivity import build_explanation_payload, sensitivity_analysis
from src.models.ensemble import (
    build_model_paths,
    infer_on_split_row,
    load_ensemble_models,
    prepare_inference_splits,
    pretty_allocation_dict,
)


def confidence_label(confidence: float) -> str:
    # convert numeric confidence into a simple human-readable band
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")

    if confidence >= 0.70:
        return "High"
    if confidence >= 0.40:
        return "Moderate"
    return "Low"


def confidence_agreement_phrase(confidence: float) -> str:
    # make the summary wording feel more natural than repeating "degree of agreement"
    label = confidence_label(confidence)

    if label == "High":
        return "strong agreement across independently trained policies"
    if label == "Moderate":
        return "moderate agreement across independently trained policies"
    return "limited agreement across independently trained policies"


def confidence_caution_note(confidence: float) -> str:
    # add a short user-facing interpretation of what the confidence means
    label = confidence_label(confidence)

    if label == "High":
        return "The recommendation appears relatively stable across ensemble members."
    if label == "Moderate":
        return "The recommendation should be treated with some caution because model agreement is mixed."
    return "The recommendation should be treated cautiously because the ensemble members disagree substantially."


def format_percentage(x: float) -> str:
    # nice readable formatting for allocations and confidence
    return f"{100.0 * x:.1f}%"


def prettify_group_name(group_name: str) -> str:
    # convert internal feature-group keys into nicer user-facing labels
    mapping = {
        "short_term_returns": "short-term returns",
        "medium_term_returns": "medium-term returns",
        "long_term_returns": "long-term returns",
        "volatility": "volatility",
        "rsi": "relative strength (RSI)",
    }

    return mapping.get(group_name, group_name.replace("_", " "))


def summarize_allocation(
    allocation: dict[str, float],
    top_k: int = 3,
    include_cash: bool = False,
) -> list[tuple[str, float]]:
    # return the top-k largest weights sorted descending
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    items = allocation.items()
    if not include_cash:
        items = [(asset, weight) for asset, weight in items if asset != "CASH"]

    ranked = sorted(items, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def sorted_allocation(
    allocation: dict[str, float],
) -> dict[str, float]:
    # keep allocation display ordered from largest to smallest weight
    ranked = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
    return dict(ranked)


def natural_language_summary(
    allocation: dict[str, float],
    explanation_payload: dict,
    confidence: float,
    top_k_allocations: int = 3,
) -> str:
    # build one concise paragraph summarizing the recommendation, confidence, and drivers
    top_allocs = summarize_allocation(
        allocation,
        top_k=top_k_allocations,
        include_cash=False,
    )
    alloc_text = ", ".join([f"{asset} ({format_percentage(weight)})" for asset, weight in top_allocs])

    conf_text = format_percentage(confidence)
    agreement_text = confidence_agreement_phrase(confidence)
    caution_text = confidence_caution_note(confidence)

    cash_weight = allocation.get("CASH", 0.0)

    top_groups = explanation_payload["top_groups"]
    if top_groups:
        pretty_groups = [prettify_group_name(group["group"]) for group in top_groups]
        if len(pretty_groups) == 1:
            group_text = pretty_groups[0]
        elif len(pretty_groups) == 2:
            group_text = f"{pretty_groups[0]} and {pretty_groups[1]}"
        else:
            group_text = ", ".join(pretty_groups[:-1]) + f", and {pretty_groups[-1]}"

        explanation_text = f"The most influential feature groups were {group_text}."
    else:
        explanation_text = "No dominant explanatory feature groups were identified."

    cash_text = ""
    if cash_weight > 0.0:
        cash_text = f" The portfolio also retains {format_percentage(cash_weight)} in cash."

    return (
        f"The recommended portfolio places the largest weights on {alloc_text}.{cash_text} "
        f"Model confidence is {confidence_label(confidence).lower()} ({conf_text}), indicating {agreement_text}. "
        f"{explanation_text} {caution_text}"
    )


def build_recommendation_output(
    timestamp: Any,
    allocation: dict[str, float],
    ensemble_result: dict,
    explanation_payload: dict,
) -> dict:
    # package everything into one clean response object for demos or later UI work
    confidence = float(ensemble_result["confidence"])
    pretty_top_groups = []

    for group in explanation_payload["top_groups"]:
        pretty_top_groups.append(
            {
                "group": prettify_group_name(group["group"]),
                "effect_size": float(group["effect_size"]),
                "increased_assets": group["increased_assets"],
                "decreased_assets": group["decreased_assets"],
            }
        )

    ordered_allocation = sorted_allocation(allocation)

    return {
        "timestamp": timestamp,
        "allocation": ordered_allocation,
        "confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "confidence_note": confidence_caution_note(confidence),
        "disagreement": float(ensemble_result["disagreement"]),
        "top_explanation_groups": pretty_top_groups,
        "summary": natural_language_summary(
            allocation=ordered_allocation,
            explanation_payload={"top_groups": pretty_top_groups},
            confidence=confidence,
        ),
    }


def generate_recommendation(
    split_name: str = "val",
    row_idx: int = -1,
    confidence_alpha: float = 10.0,
    perturbation_delta: float = 0.1,
    explanation_top_k: int = 3,
    device: str = "cpu",
) -> dict:
    # end-to-end wrapper:
    # load ensemble, prepare data, run inference, compute explanation, return one clean recommendation object
    if split_name not in {"train", "val", "test"}:
        raise ValueError("split_name must be one of: train, val, test")

    model_paths = build_model_paths()
    models = load_ensemble_models(model_paths, device=device)
    splits = prepare_inference_splits()

    prices_key = f"{split_name}_prices"
    features_key = f"{split_name}_features"

    prices = splits[prices_key]
    features = splits[features_key]

    ensemble_result = infer_on_split_row(
        models=models,
        features=features,
        row_idx=row_idx,
        confidence_alpha=confidence_alpha,
    )

    explanation_result = sensitivity_analysis(
        models=models,
        observation=ensemble_result["observation"],
        features=features,
        asset_names=list(prices.columns),
        row_idx=row_idx,
        perturbation_delta=perturbation_delta,
        confidence_alpha=confidence_alpha,
    )

    explanation_payload = build_explanation_payload(
        explanation_result,
        top_k=explanation_top_k,
    )

    allocation = pretty_allocation_dict(
        ensemble_result["ensemble_weights"],
        asset_names=list(prices.columns),
    )

    output = build_recommendation_output(
        timestamp=ensemble_result["timestamp"],
        allocation=allocation,
        ensemble_result=ensemble_result,
        explanation_payload=explanation_payload,
    )

    return output


def print_recommendation(recommendation: dict) -> None:
    # simple terminal-friendly printer for demos
    print(f"Timestamp: {recommendation['timestamp']}")
    print(f"Confidence: {recommendation['confidence_label']} ({format_percentage(recommendation['confidence'])})")
    print(f"Confidence note: {recommendation['confidence_note']}")
    print(f"Disagreement: {recommendation['disagreement']:.4f}")

    print("Allocation:")
    for asset, weight in recommendation["allocation"].items():
        print(f"  - {asset}: {format_percentage(weight)}")

    print("Top explanation groups:")
    for group in recommendation["top_explanation_groups"]:
        print(
            f"  - {group['group']} | effect={group['effect_size']:.4f} | "
            f"increased={group['increased_assets']} | decreased={group['decreased_assets']}"
        )

    print("Summary:")
    print(recommendation["summary"])


if __name__ == "__main__":
    recommendation = generate_recommendation(
        split_name="val",
        row_idx=-1,
        confidence_alpha=10.0,
        perturbation_delta=0.1,
        explanation_top_k=3,
        device="cpu",
    )
    print_recommendation(recommendation)