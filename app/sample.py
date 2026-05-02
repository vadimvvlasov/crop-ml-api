import numpy as np

T, C = 26, 15
RNG = np.random.default_rng(42)


def make_sample(n: int = 2) -> dict:
    """Deterministic fake HLS-like input for demo (raw reflectance scale ~1e4)."""
    features = RNG.uniform(500, 5000, size=(n, T, C)).astype(np.float32)
    week_of_year = [((40 + i - 1) % 52) + 1 for i in range(T)]
    location = [[-29.5 + i * 0.1, -53.5 + i * 0.1] for i in range(n)]
    return {
        "features": features.tolist(),
        "week_of_year": week_of_year,
        "location": location,
    }
