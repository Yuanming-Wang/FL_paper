"""Parameter-free privacy cost used by Adap-CTA decision making."""

import math


def privacy_cost(correction_ratio: float) -> float:
    """Return the parameter-free ``exp(-ratio)`` privacy proxy.

    The paper writes this term with tunable ``a,b``. Those parameters are
    intentionally removed here, so the fixed unit-slope form is used.
    """

    ratio = min(max(float(correction_ratio), 0.0), 1.0)
    return math.exp(-ratio)
