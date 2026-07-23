"""Synthetic well log data generator for training and testing."""

import numpy as np
import pandas as pd


LITHOLOGY_LABELS = ["sandstone", "limestone", "shale", "dolomite", "anhydrite"]

LITHOLOGY_PARAMS = {
    "sandstone": {
        "gamma_ray": (40, 15),
        "resistivity": (50, 20),
        "neutron_porosity": (25, 5),
        "density_porosity": (22, 4),
        "sonic": (75, 8),
        "caliper": (8.5, 0.3),
    },
    "limestone": {
        "gamma_ray": (30, 10),
        "resistivity": (200, 50),
        "neutron_porosity": (15, 4),
        "density_porosity": (12, 3),
        "sonic": (55, 6),
        "caliper": (8.0, 0.2),
    },
    "shale": {
        "gamma_ray": (120, 25),
        "resistivity": (15, 5),
        "neutron_porosity": (35, 6),
        "density_porosity": (30, 5),
        "sonic": (95, 10),
        "caliper": (9.5, 0.5),
    },
    "dolomite": {
        "gamma_ray": (25, 8),
        "resistivity": (150, 40),
        "neutron_porosity": (10, 3),
        "density_porosity": (8, 2),
        "sony": (45, 5),
        "caliper": (8.2, 0.2),
    },
    "anhydrite": {
        "gamma_ray": (15, 5),
        "resistivity": (500, 100),
        "neutron_porosity": (2, 1),
        "density_porosity": (1, 1),
        "sonic": (50, 4),
        "caliper": (8.0, 0.1),
    },
}

LITHOLOGY_PARAMS["dolomite"]["sonic"] = LITHOLOGY_PARAMS["dolomite"].pop(
    "sony", (45, 5)
)


def generate_well_log_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic well log data with realistic geological patterns.

    Args:
        n_samples: Number of data points to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns: depth, gamma_ray, resistivity,
        neutron_porosity, density_porosity, sonic, caliper, lithology.
    """
    rng = np.random.RandomState(random_state)

    lithology_counts = rng.multinomial(
        n_samples, [0.30, 0.25, 0.25, 0.12, 0.08]
    )

    records = []
    depth_cursor = 100.0
    lithology_order = ["sandstone", "limestone", "shale", "dolomite", "anhydrite"]

    for lith_idx, lith_name in enumerate(lithology_order):
        count = lithology_counts[lith_idx]
        if count == 0:
            continue
        params = LITHOLOGY_PARAMS[lith_name]
        thickness = count * 0.5
        depths = np.linspace(depth_cursor, depth_cursor + thickness, count)
        depth_cursor += thickness + rng.uniform(2, 10)

        for depth in depths:
            base_porosity = max(0.01, rng.normal(
                (params["neutron_porosity"][0] + params["density_porosity"][0]) / 200,
                0.02,
            ))
            gr = rng.normal(*params["gamma_ray"])
            res = max(0.1, rng.normal(*params["resistivity"]))
            np_val = max(0.01, rng.normal(*params["neutron_porosity"]))
            dp_val = max(0.01, rng.normal(*params["density_porosity"]))
            sonic = max(30, rng.normal(*params["sonic"]))
            caliper = max(6, rng.normal(*params["caliper"]))

            records.append({
                "depth": round(depth, 2),
                "gamma_ray": round(gr, 2),
                "resistivity": round(res, 2),
                "neutron_porosity": round(np_val, 2),
                "density_porosity": round(dp_val, 2),
                "sonic": round(sonic, 2),
                "caliper": round(caliper, 2),
                "lithology": lith_name,
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df
