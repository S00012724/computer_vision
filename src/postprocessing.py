"""
Post-processing utilities for collection-priority decisions

The CNN outputs class scores, but the app needs an action: collect now, leave
the bin, or send the case to a review queue. The project routes every decision
from one traceable score: P(needs_collection)
"""


URGENT_RISK_THRESHOLD = 0.70
LOW_PRIORITY_RISK_THRESHOLD = 0.15
HIGH_RISK_THRESHOLD = URGENT_RISK_THRESHOLD
MEDIUM_RISK_THRESHOLD = 0.35

# The lower threshold allows clear has_space cases to remain automatic. The
# upper threshold sends clear needs_collection cases to collection priority. The
# middle band goes to review because it is operationally uncertain


def _score_for(
    class_scores: dict[str, float] | None,
    class_name: str,
) -> float | None:
    """Read one class score from the model output if it is available

    Args:
        class_scores: Mapping from class name to probability or None
        class_name: Class label whose score should be read

    Returns:
        Class score as a float or None when unavailable
    """

    if not class_scores:
        return None
    score = class_scores.get(class_name)
    return None if score is None else float(score)


def _needs_collection_score(
    class_name: str,
    confidence: float | None,
    class_scores: dict[str, float] | None,
) -> float | None:
    """Return the score used as collection risk

    Prefer the explicit needs_collection softmax score. If a caller only passes
    a predicted class and confidence, estimate the opposite-class score for a
    binary classifier

    Args:
        class_name: Predicted class label
        confidence: Probability assigned to the predicted class
        class_scores: Mapping from class name to probability or None

    Returns:
        needs_collection risk score or None when it cannot be estimated
    """

    score = _score_for(class_scores, "needs_collection")
    if score is not None:
        return score
    if confidence is None:
        return None
    return float(confidence) if class_name == "needs_collection" else 1.0 - float(confidence)


def _risk_level(risk_score: float | None) -> str:
    """Convert a numeric collection-risk score to a readable API label

    Args:
        risk_score: Numeric needs_collection risk score or None

    Returns:
        Risk label for API output
    """

    if risk_score is None:
        return "unknown"
    if risk_score >= HIGH_RISK_THRESHOLD:
        return "high"
    if risk_score >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def postprocess_prediction(
    class_name: str,
    confidence: float | None = None,
    class_scores: dict[str, float] | None = None,
    urgent_risk_threshold: float = URGENT_RISK_THRESHOLD,
    low_priority_risk_threshold: float = LOW_PRIORITY_RISK_THRESHOLD,
) -> dict[str, object]:
    """Convert model output into an operational collection decision

    The function keeps the model prediction intact and routes the operational
    decision from the needs_collection risk score. Low risk becomes a low
    priority decision, high risk becomes urgent, and the uncertain middle band
    goes to review

    Args:
        class_name: Predicted class label
        confidence: Probability assigned to the predicted class
        class_scores: Mapping from class name to probability or None
        urgent_risk_threshold: needs_collection score for urgent decisions
        low_priority_risk_threshold: needs_collection score for low-priority decisions

    Returns:
        Collection priority, priority reason, and risk level
    """

    risk_score = _needs_collection_score(class_name, confidence, class_scores)
    risk_level = _risk_level(risk_score)

    if risk_score is None:
        priority = "review"
        reason = "Model did not provide a needs_collection risk score."
    elif risk_score >= urgent_risk_threshold:
        priority = "urgent"
        reason = (
            f"Needs_collection risk {risk_score:.2f} meets "
            f"urgent threshold {urgent_risk_threshold:.2f}."
        )
    elif risk_score <= low_priority_risk_threshold:
        priority = "low"
        reason = (
            f"Needs_collection risk {risk_score:.2f} meets "
            f"low-priority threshold {low_priority_risk_threshold:.2f}."
        )
    else:
        priority = "review"
        reason = (
            f"Needs_collection risk {risk_score:.2f} is between "
            f"{low_priority_risk_threshold:.2f} and {urgent_risk_threshold:.2f}."
        )

    return {
        "collection_priority": priority,
        "priority_reason": reason,
        "risk_level": risk_level,
    }
