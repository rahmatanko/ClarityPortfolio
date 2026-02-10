import numpy as np

from src.app.wrapper import (
    build_recommendation_output,
    confidence_label,
    format_percentage,
    natural_language_summary,
    summarize_allocation,
)


def test_confidence_label_returns_expected_band():
    assert confidence_label(0.8) == "High"
    assert confidence_label(0.5) == "Moderate"
    assert confidence_label(0.1) == "Low"


def test_format_percentage_formats_cleanly():
    assert format_percentage(0.1234) == "12.3%"
    assert format_percentage(0.0) == "0.0%"


def test_summarize_allocation_returns_top_k():
    allocation = {
        "AAPL": 0.10,
        "MSFT": 0.25,
        "NVDA": 0.30,
        "GOOGL": 0.15,
        "TSLA": 0.05,
        "CASH": 0.15,
    }

    top = summarize_allocation(allocation, top_k=3)

    assert len(top) == 3
    assert top[0][0] == "NVDA"
    assert top[1][0] == "MSFT"


def test_natural_language_summary_contains_key_parts():
    allocation = {
        "AAPL": 0.10,
        "MSFT": 0.25,
        "NVDA": 0.30,
        "GOOGL": 0.15,
        "TSLA": 0.05,
        "CASH": 0.15,
    }

    explanation_payload = {
        "top_groups": [
            {"group": "volatility", "effect_size": 0.01, "increased_assets": ["CASH"], "decreased_assets": ["NVDA"]},
            {"group": "long_term_returns", "effect_size": 0.008, "increased_assets": ["AAPL"], "decreased_assets": ["CASH"]},
        ]
    }

    summary = natural_language_summary(
        allocation=allocation,
        explanation_payload=explanation_payload,
        confidence=0.2,
    )

    assert "NVDA" in summary
    assert "low" in summary.lower()
    assert "volatility" in summary
    assert "long_term_returns" in summary


def test_build_recommendation_output_has_expected_keys():
    allocation = {
        "AAPL": 0.10,
        "MSFT": 0.25,
        "NVDA": 0.30,
        "GOOGL": 0.15,
        "TSLA": 0.05,
        "CASH": 0.15,
    }

    ensemble_result = {
        "confidence": 0.2,
        "disagreement": 0.3,
    }

    explanation_payload = {
        "top_groups": [
            {"group": "volatility", "effect_size": 0.01, "increased_assets": ["CASH"], "decreased_assets": ["NVDA"]}
        ]
    }

    output = build_recommendation_output(
        timestamp="2022-12-30",
        allocation=allocation,
        ensemble_result=ensemble_result,
        explanation_payload=explanation_payload,
    )

    assert output["timestamp"] == "2022-12-30"
    assert output["confidence_label"] == "Low"
    assert "summary" in output
    assert "allocation" in output