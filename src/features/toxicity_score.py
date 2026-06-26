def compute_toxicity_score(
    probabilities: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute toxicity score from the probability dictionary of the 3 classes.
    Purely mathematical, no forward pass or model calls.
    """
    if weights is None:
        weights = {"OFFENSIVE": 0.4, "HATE": 1.0}
    
    score = sum(
        probabilities.get(label, 0.0) * w 
        for label, w in weights.items()
    )
    return float(min(max(score, 0.0), 1.0))
