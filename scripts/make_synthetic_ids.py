"""Write a small synthetic id bearing ads log in the Taobao file format.

This exists to build and test the retrieval code without the real data. Users
and ads are both Zipf distributed in how often they appear. Every ad belongs to
one category, every user carries a hidden preference over a handful of
categories, and the click probability rises with that affinity. That gives the
two tower model something real to learn and the tests something to check: a
trained model should score a user's own clicked ads above random ones.

Nothing this produces is a result. Every number measured on it is written with
the SYNTHETIC label and none of them reaches NUMBERS.md.

Usage:
    python scripts/make_synthetic_ids.py --out data/synthetic_ids
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.data import BEIJING_OFFSET_S, TAOBAO_FIRST_DAY, TAOBAO_N_DAYS  # noqa: E402


def zipf_weights(n: int, a: float, rng: np.random.Generator) -> np.ndarray:
    w = 1.0 / np.arange(1, n + 1) ** a
    rng.shuffle(w)
    return w / w.sum()


def make(
    out: str,
    n_users: int = 20_000,
    n_ads: int = 5_000,
    n_cates: int = 50,
    n_impressions: int = 400_000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)

    # Ads: a category each, then campaign, customer, brand nested under it.
    ad_ids = rng.choice(np.arange(1, n_ads * 20), size=n_ads, replace=False)
    ad_cate = rng.integers(0, n_cates, size=n_ads)
    ad_brand = ad_cate * 10 + rng.integers(0, 10, size=n_ads)
    ad_customer = ad_brand * 3 + rng.integers(0, 3, size=n_ads)
    ad_campaign = ad_customer * 2 + rng.integers(0, 2, size=n_ads)
    price = np.round(np.exp(rng.normal(4.0, 1.0, size=n_ads)), 2)
    ad_pop = zipf_weights(n_ads, 1.0, rng)

    # Users: a hidden preference over three categories, and a profile that is
    # weakly tied to the first of them so profile fields carry some signal.
    user_ids = rng.choice(np.arange(1, n_users * 20), size=n_users, replace=False)
    pref = rng.integers(0, n_cates, size=(n_users, 3))
    user_pop = zipf_weights(n_users, 0.8, rng)
    cms_segid = pref[:, 0] % 13
    age_level = rng.integers(0, 7, size=n_users)
    gender = rng.integers(1, 3, size=n_users)

    ads_by_cate = [np.flatnonzero(ad_cate == c) for c in range(n_cates)]

    u = rng.choice(n_users, size=n_impressions, p=user_pop)
    # Half the impressions come from the user's own categories, half from the
    # popularity distribution, which is roughly how a targeted log looks.
    own = rng.random(n_impressions) < 0.5
    a = rng.choice(n_ads, size=n_impressions, p=ad_pop)
    pick = rng.integers(0, 3, size=n_impressions)
    for i in np.flatnonzero(own):
        pool = ads_by_cate[pref[u[i], pick[i]]]
        if len(pool):
            a[i] = pool[rng.integers(0, len(pool))]

    affinity = (ad_cate[a][:, None] == pref[u]).any(axis=1)
    p_click = np.where(affinity, 0.12, 0.015)
    clk = (rng.random(n_impressions) < p_click).astype(np.int8)

    first = pd.Timestamp(TAOBAO_FIRST_DAY).value // 10**9 - BEIJING_OFFSET_S
    ts = first + rng.integers(0, TAOBAO_N_DAYS * 86400, size=n_impressions)
    pids = np.array(["430548_1007", "430539_1007"])[rng.integers(0, 2, size=n_impressions)]

    raw = pd.DataFrame({
        "user": user_ids[u],
        "time_stamp": ts,
        "adgroup_id": ad_ids[a],
        "pid": pids,
        "nonclk": 1 - clk,
        "clk": clk,
    }).sort_values("time_stamp")
    raw.to_csv(os.path.join(out, "raw_sample.csv"), index=False)

    brand = ad_brand.astype(float)
    brand[rng.random(n_ads) < 0.2] = np.nan
    pd.DataFrame({
        "adgroup_id": ad_ids,
        "cate_id": ad_cate,
        "campaign_id": ad_campaign,
        "customer": ad_customer,
        "brand": brand,
        "price": price,
    }).to_csv(os.path.join(out, "ad_feature.csv"), index=False)

    has_profile = rng.random(n_users) < 0.9
    pvalue = rng.integers(1, 4, size=n_users).astype(float)
    pvalue[rng.random(n_users) < 0.5] = np.nan
    pd.DataFrame({
        "userid": user_ids,
        "cms_segid": cms_segid,
        "cms_group_id": rng.integers(0, 13, size=n_users),
        "final_gender_code": gender,
        "age_level": age_level,
        "pvalue_level": pvalue,
        "shopping_level": rng.integers(1, 4, size=n_users),
        "occupation": rng.integers(0, 2, size=n_users),
        # The real file's last header has a trailing space. Keep it.
        "new_user_class_level ": rng.integers(1, 5, size=n_users),
    })[has_profile].to_csv(os.path.join(out, "user_profile.csv"), index=False)

    # Throw away any cache from an earlier run with different parameters.
    cache = os.path.join(out, "cache")
    if os.path.isdir(cache):
        for f in os.listdir(cache):
            os.remove(os.path.join(cache, f))

    return {"impressions": n_impressions, "clicks": int(clk.sum()), "users": n_users, "ads": n_ads}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/synthetic_ids")
    ap.add_argument("--users", type=int, default=20_000)
    ap.add_argument("--ads", type=int, default=5_000)
    ap.add_argument("--cates", type=int, default=50)
    ap.add_argument("--impressions", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    stats = make(args.out, args.users, args.ads, args.cates, args.impressions, args.seed)
    print(f"SYNTHETIC sample written to {args.out}: {stats}")


if __name__ == "__main__":
    main()
