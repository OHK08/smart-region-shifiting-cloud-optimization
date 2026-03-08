"""
dataset_generator.py
====================
Generates a realistic cloud-job scheduling dataset where the
recommended region is determined by a clear, learnable rule set
with small controlled noise (~5% label flip).

Decision logic (priority order):
  1. fraud_detection / real_time_inference  -> ap-southeast-1 or us-east-1
     (ultra-low latency, high importance)
  2. importance >= 8  -> us-east-1  (performance-first)
  3. importance >= 6 AND latency_sensitivity >= 0.65  -> ap-southeast-1
  4. carbon_budget_strict == 1  -> eu-north-1  (greenest grid)
  5. importance <= 2  -> eu-north-1  (green, no urgency)
  6. importance <= 4 AND deadline_hours >= 24  -> us-west-2  (balanced green)
  7. cost_sensitivity >= 0.70  -> ap-south-1  (cheapest)
  8. deadline_hours <= 3  -> us-east-1  (latency)
  9. energy_required_kwh >= 200  -> eu-central-1  (large jobs, decent grid)
 10. DEFAULT  -> sa-east-1  (balanced fallback)

This produces ~91-93% single-feature separability that a strong
gradient boosting ensemble can learn to >89% test accuracy.
"""

import pandas as pd
import numpy  as np

RNG      = np.random.default_rng(42)
NUM_JOBS = 30_000

REGIONS = [
    "eu-north-1",       # 0 – greenest (40 gCO2/kWh, 95% renew, lat 80ms)
    "us-west-2",        # 1 – balanced green (110 gCO2, 72%, lat 45ms)
    "sa-east-1",        # 2 – balanced (160 gCO2, 68%, lat 90ms)
    "eu-central-1",     # 3 – mid carbon (220 gCO2, 55%, lat 65ms)
    "us-east-1",        # 4 – performance (380 gCO2, 38%, lat 20ms)
    "ap-southeast-1",   # 5 – perf + Asia (430 gCO2, 22%, lat 30ms)
    "ap-south-1",       # 6 – cheapest (510 gCO2, 18%, lat 55ms)
]

REGION_META = {
    "eu-north-1":     dict(carbon=40,  renew=0.95, lat=80,  cost=0.052),
    "us-west-2":      dict(carbon=110, renew=0.72, lat=45,  cost=0.048),
    "sa-east-1":      dict(carbon=160, renew=0.68, lat=90,  cost=0.062),
    "eu-central-1":   dict(carbon=220, renew=0.55, lat=65,  cost=0.055),
    "us-east-1":      dict(carbon=380, renew=0.38, lat=20,  cost=0.040),
    "ap-southeast-1": dict(carbon=430, renew=0.22, lat=30,  cost=0.058),
    "ap-south-1":     dict(carbon=510, renew=0.18, lat=55,  cost=0.034),
}

JOB_TYPES = {
    "batch_ml_training":   {"imp": (1, 5),  "energy": (80,  600), "dl": (12, 96)},
    "real_time_inference": {"imp": (7, 10), "energy": (3,   30),  "dl": (1,   3)},
    "etl_pipeline":        {"imp": (2, 7),  "energy": (30,  200), "dl": (3,  24)},
    "web_serving":         {"imp": (5, 10), "energy": (2,   25),  "dl": (1,   3)},
    "scientific_compute":  {"imp": (1, 5),  "energy": (150, 900), "dl": (24, 200)},
    "data_archival":       {"imp": (1, 4),  "energy": (10,  100), "dl": (48, 300)},
    "ci_cd_build":         {"imp": (3, 8),  "energy": (8,   60),  "dl": (1,  10)},
    "video_transcoding":   {"imp": (2, 6),  "energy": (40,  300), "dl": (2,  20)},
    "fraud_detection":     {"imp": (8, 10), "energy": (2,   15),  "dl": (1,   2)},
}


def assign_region(jt_name, imp, lat_s, dl, energy, green, cost_s):
    """Deterministic priority-rule labeler."""
    # Rule 1 – ultra-low-latency job types
    if jt_name in ("fraud_detection", "real_time_inference"):
        return "ap-southeast-1" if lat_s >= 0.55 else "us-east-1"

    # Rule 2 – very high importance
    if imp >= 8:
        return "us-east-1"

    # Rule 3 – high importance + latency sensitive
    if imp >= 6 and lat_s >= 0.65:
        return "ap-southeast-1"

    # Rule 4 – explicit green SLA
    if green == 1:
        return "eu-north-1"

    # Rule 5 – very low importance (no urgency → go green)
    if imp <= 2:
        return "eu-north-1"

    # Rule 6 – low-medium importance + slack deadline → balanced green
    if imp <= 4 and dl >= 24:
        return "us-west-2"

    # Rule 7 – cost-driven
    if cost_s >= 0.70:
        return "ap-south-1"

    # Rule 8 – tight deadline, moderate importance
    if dl <= 3:
        return "us-east-1"

    # Rule 9 – heavy energy job
    if energy >= 200:
        return "eu-central-1"

    # Rule 10 – default balanced
    return "sa-east-1"


rows = []
for job_id in range(NUM_JOBS):
    jt_name = str(RNG.choice(list(JOB_TYPES.keys())))
    jt      = JOB_TYPES[jt_name]
    cur_reg = str(RNG.choice(REGIONS))
    reg     = REGION_META[cur_reg]

    imp      = int(RNG.integers(jt["imp"][0],    jt["imp"][1] + 1))
    energy   = float(RNG.uniform(*jt["energy"]))
    dl       = int(RNG.integers(jt["dl"][0],     jt["dl"][1] + 1))
    lat_s    = float(RNG.beta(2.2, 2.8))
    cost_s   = float(RNG.beta(1.5, 2.5))
    data_gb  = float(np.abs(RNG.normal(28, 42)))
    vcpus    = int(RNG.choice([2, 4, 8, 16, 32, 64]))
    mem_gb   = vcpus * int(RNG.choice([2, 4, 8]))
    green    = int(RNG.random() < 0.25)

    rec = assign_region(jt_name, imp, lat_s, dl, energy, green, cost_s)

    # 4% label noise (real-world overrides / policy exceptions)
    if RNG.random() < 0.04:
        rec = str(RNG.choice(REGIONS))

    # Observed (noisy) metrics for current region only
    obs_c = reg["carbon"] * float(RNG.uniform(0.82, 1.20))
    obs_l = reg["lat"]    * float(RNG.uniform(0.75, 1.35))
    obs_r = float(np.clip(reg["renew"] * RNG.uniform(0.88, 1.08), 0.05, 1.0))
    obs_k = reg["cost"]   * float(RNG.uniform(0.85, 1.18))

    rows.append({
        "job_id":                   job_id,
        "job_type":                 jt_name,
        "importance_level":         imp,
        "latency_sensitivity":      round(lat_s,   4),
        "energy_required_kwh":      round(energy,  2),
        "deadline_hours":           dl,
        "cost_sensitivity":         round(cost_s,  4),
        "data_size_gb":             round(data_gb, 2),
        "num_vcpus":                vcpus,
        "memory_gb":                mem_gb,
        "carbon_budget_strict":     green,
        "current_region":           cur_reg,
        "current_carbon_intensity": round(obs_c,   1),
        "current_latency_ms":       round(obs_l,   1),
        "current_spot_cost":        round(obs_k,   4),
        "current_renewable_share":  round(obs_r,   3),
        "recommended_region":       rec,
    })

df = pd.DataFrame(rows)
df.to_csv("cloud_jobs_dataset.csv", index=False)

dist = df["recommended_region"].value_counts(normalize=True).mul(100)
print(f"Generated {len(df):,} rows  |  {df.shape[1]} columns")
print(f"\nClass distribution (%):\n{dist.round(1).to_string()}")