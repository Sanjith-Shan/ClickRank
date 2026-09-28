"""Load an id bearing ads dataset into one canonical shape, with its time split.

Criteo cannot support retrieval because it has no user or ad identifiers, so the
retrieval stage runs on a second dataset that has both. Three sources are
supported and all of them land in the same three tables:

impressions  one row per ad shown: user, ad, ts (unix seconds), day (1 based),
             pid (the placement, as an integer code), clk (0 or 1).
ads          one row per ad in the corpus: ad, cate, campaign, customer, brand,
             price. Ad attributes are raw ids, price is a float.
users        one row per user with a profile: user plus the profile fields.

Taobao (Alibaba display ads, Tianchi dataset 56). raw_sample.csv holds about
26.6M impressions over eight days, ad_feature.csv holds 846,811 ads and
user_profile.csv holds about 1.06M users. The official split is days 1 to 7
train and day 8 test. Days are computed in Beijing time because that is the
clock the logs were written in, and the eight days are 2017-05-06 to
2017-05-13. Rows that fall outside those eight days are dropped and counted.

Avazu (Kaggle avazu-ctr-prediction train.gz, 40.4M rows over ten days). It has
no user id and no ad id, so both are proxies: the user is device_id, falling
back to device_ip where device_id is the shared placeholder a99f214a, and the ad
is the anonymised C14 column. The split is days 1 to 9 train and day 10 test.
Anything measured on Avazu says so.

Synthetic (scripts/make_synthetic_ids.py). Written in the Taobao file format so
the same loader reads it. Used to build and test the code and nothing else.

Neither real dataset is redistributed. The raw files stay under data/, which is
git ignored, and only code and aggregates are committed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Profile fields in user_profile.csv, in file order. The last header in the
# real file carries a trailing space ("new_user_class_level "), which is why
# the loader strips column names.
TAOBAO_USER_FIELDS = [
    "cms_segid",
    "cms_group_id",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
]
AD_FIELDS = ["cate", "campaign", "customer", "brand"]

# The published counts for the Taobao files. The loader reports what it found
# next to these so a truncated download is caught at load time, not at M2.
TAOBAO_CARD = {
    "raw_sample_rows": 26_557_961,
    "ad_feature_rows": 846_811,
    "user_profile_rows": 1_061_768,
}

BEIJING_OFFSET_S = 8 * 3600
TAOBAO_FIRST_DAY = "2017-05-06"
TAOBAO_N_DAYS = 8

AVAZU_PLACEHOLDER_DEVICE = "a99f214a"


@dataclass
class RetrievalData:
    """The three canonical tables plus what is known about where they came from."""

    name: str
    impressions: pd.DataFrame
    ads: pd.DataFrame
    users: pd.DataFrame
    train_days: tuple
    test_day: int
    synthetic: bool = False
    proxy_ids: bool = False
    notes: Dict[str, object] = field(default_factory=dict)

    @property
    def train(self) -> pd.DataFrame:
        d = self.impressions["day"]
        return self.impressions[(d >= self.train_days[0]) & (d <= self.train_days[1])]

    @property
    def test(self) -> pd.DataFrame:
        return self.impressions[self.impressions["day"] == self.test_day]

    def label(self) -> str:
        """How every result row names its data. Synthetic is always said."""
        if self.synthetic:
            return f"SYNTHETIC ({self.name})"
        if self.proxy_ids:
            return f"{self.name} (user and ad ids are proxies)"
        return self.name

    def summary(self) -> Dict[str, object]:
        imp = self.impressions
        tr, te = self.train, self.test
        out = {
            "dataset": self.label(),
            "impressions": int(len(imp)),
            "clicks": int(imp["clk"].sum()),
            "ctr": float(imp["clk"].mean()) if len(imp) else float("nan"),
            "users_in_log": int(imp["user"].nunique()),
            "ads_in_log": int(imp["ad"].nunique()),
            "ads_in_corpus": int(len(self.ads)),
            "users_with_profile": int(len(self.users)),
            "train_impressions": int(len(tr)),
            "train_clicks": int(tr["clk"].sum()),
            "test_impressions": int(len(te)),
            "test_clicks": int(te["clk"].sum()),
            "impressions_per_day": {
                int(k): int(v) for k, v in imp["day"].value_counts().sort_index().items()
            },
        }
        out.update(self.notes)
        return out


# ---------------------------------------------------------------------------
# Taobao, and the synthetic sample that shares its file format


def _read_csv(path: str, **kw) -> pd.DataFrame:
    df = pd.read_csv(path, **kw)
    df.columns = [c.strip() for c in df.columns]
    return df


def _cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, "cache")


def load_taobao(
    data_dir: str,
    *,
    synthetic: bool = False,
    use_cache: bool = True,
    name: Optional[str] = None,
) -> RetrievalData:
    """Read raw_sample, ad_feature and user_profile from data_dir.

    The first call writes parquet copies under data_dir/cache so later calls
    take seconds instead of the minute the 1 GB csv takes to parse.
    """
    name = name or ("taobao" if not synthetic else "synthetic_ids")
    cache = _cache_path(data_dir)
    files = {k: os.path.join(cache, f"{k}.parquet") for k in ("impressions", "ads", "users")}
    notes_path = os.path.join(cache, "notes.json")

    if use_cache and all(os.path.exists(p) for p in files.values()):
        imp = pd.read_parquet(files["impressions"])
        ads = pd.read_parquet(files["ads"])
        users = pd.read_parquet(files["users"])
        notes = json.load(open(notes_path)) if os.path.exists(notes_path) else {}
    else:
        raw = _read_csv(
            os.path.join(data_dir, "raw_sample.csv"),
            dtype={"user": np.int64, "time_stamp": np.int64, "adgroup_id": np.int64,
                   "pid": str, "nonclk": np.int8, "clk": np.int8},
        )
        ad_raw = _read_csv(os.path.join(data_dir, "ad_feature.csv"))
        user_raw = _read_csv(os.path.join(data_dir, "user_profile.csv"))
        notes = {
            "raw_sample_rows_read": int(len(raw)),
            "ad_feature_rows_read": int(len(ad_raw)),
            "user_profile_rows_read": int(len(user_raw)),
        }

        first = pd.Timestamp(TAOBAO_FIRST_DAY).value // 10**9
        first_local_day = (first + BEIJING_OFFSET_S) // 86400
        # The synthetic generator writes the same calendar, so the same rule works.
        day = (raw["time_stamp"].to_numpy() + BEIJING_OFFSET_S) // 86400 - first_local_day + 1
        keep = (day >= 1) & (day <= TAOBAO_N_DAYS)
        notes["rows_outside_eight_days_dropped"] = int((~keep).sum())

        pid_codes, pid_uniques = pd.factorize(raw["pid"])
        notes["pids"] = [str(p) for p in pid_uniques]
        imp = pd.DataFrame({
            "user": raw["user"].to_numpy(np.int64),
            "ad": raw["adgroup_id"].to_numpy(np.int64),
            "ts": raw["time_stamp"].to_numpy(np.int64),
            "day": day.astype(np.int8),
            "pid": pid_codes.astype(np.int16),
            "clk": raw["clk"].to_numpy(np.int8),
        })[keep].reset_index(drop=True)
        del raw

        ads = pd.DataFrame({
            "ad": ad_raw["adgroup_id"].astype(np.int64),
            "cate": ad_raw["cate_id"].astype(np.int64),
            "campaign": ad_raw["campaign_id"].astype(np.int64),
            "customer": ad_raw["customer"].astype(np.int64),
            # brand is missing for a share of ads. -1 keeps it an integer id.
            "brand": ad_raw["brand"].fillna(-1).astype(np.int64),
            "price": ad_raw["price"].astype(np.float32),
        }).drop_duplicates("ad").reset_index(drop=True)

        users = pd.DataFrame({"user": user_raw["userid"].astype(np.int64)})
        for f in TAOBAO_USER_FIELDS:
            users[f] = user_raw[f].fillna(-1).astype(np.int64)
        users = users.drop_duplicates("user").reset_index(drop=True)

        if use_cache:
            os.makedirs(cache, exist_ok=True)
            imp.to_parquet(files["impressions"], index=False)
            ads.to_parquet(files["ads"], index=False)
            users.to_parquet(files["users"], index=False)
            json.dump(notes, open(notes_path, "w"), indent=1)

    if not synthetic:
        notes = dict(notes)
        notes["dataset_card"] = TAOBAO_CARD
    return RetrievalData(
        name=name,
        impressions=imp,
        ads=ads,
        users=users,
        train_days=(1, 7),
        test_day=8,
        synthetic=synthetic,
        notes=notes,
    )


def check_against_card(data: RetrievalData) -> Dict[str, object]:
    """Compare the rows read with the published Taobao counts.

    Returns a dict of field -> (read, published, matches). A download that was
    cut short shows up here as a mismatch rather than as a quietly smaller run.
    """
    out = {}
    for key, published in TAOBAO_CARD.items():
        read = data.notes.get(key.replace("_rows", "_rows_read"))
        out[key] = {"read": read, "published": published, "matches": read == published}
    return out


# ---------------------------------------------------------------------------
# Avazu


def load_avazu(path: str, *, nrows: Optional[int] = None) -> RetrievalData:
    """Read the Avazu train file (csv or csv.gz) into the canonical tables.

    Both ids are proxies. The user is device_id, or device_ip where device_id
    is the placeholder that about 80 percent of rows carry. The ad is C14. Ad
    attributes come from the other anonymised C columns, which the competition
    describes only as categorical, so the mapping to cate, campaign, customer
    and brand is a naming convenience, not a claim about what they mean.
    """
    cols = ["click", "hour", "banner_pos", "site_id", "app_id", "device_id",
            "device_ip", "device_model", "device_type", "device_conn_type",
            "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
    df = pd.read_csv(path, usecols=cols, nrows=nrows, dtype=str)

    user_key = np.where(df["device_id"] == AVAZU_PLACEHOLDER_DEVICE, "ip:" + df["device_ip"],
                        "id:" + df["device_id"])
    user_codes, _ = pd.factorize(user_key)
    ad_codes = df["C14"].astype(np.int64).to_numpy()

    hour = pd.to_datetime(df["hour"], format="%y%m%d%H")
    day0 = hour.min().normalize()
    day = ((hour - day0).dt.days + 1).to_numpy(np.int8)
    ts = (hour.astype("int64") // 10**9).to_numpy(np.int64)
    placement = df["site_id"].where(df["site_id"] != "85f751fd", "app:" + df["app_id"])
    pid_codes, _ = pd.factorize(placement.astype(str) + "|" + df["banner_pos"])

    imp = pd.DataFrame({
        "user": user_codes.astype(np.int64),
        "ad": ad_codes,
        "ts": ts,
        "day": day,
        "pid": (pid_codes % 32000).astype(np.int16),
        "clk": df["click"].astype(np.int8).to_numpy(),
    })

    # One row per ad. An ad id can appear with several attribute values in the
    # log, and the most frequent one is taken.
    adf = pd.DataFrame({
        "ad": ad_codes,
        "cate": df["C17"].astype(np.int64),
        "campaign": df["C19"].astype(np.int64),
        "customer": df["C20"].astype(np.int64),
        "brand": df["C21"].astype(np.int64),
        "price": (df["C15"].astype(np.float32) * df["C16"].astype(np.float32)),
    })
    ads = adf.groupby("ad", sort=False).agg(lambda s: s.mode().iloc[0]).reset_index()

    uf = pd.DataFrame({
        "user": user_codes.astype(np.int64),
        "device_model": pd.factorize(df["device_model"])[0],
        "device_type": df["device_type"].astype(np.int64),
        "device_conn_type": df["device_conn_type"].astype(np.int64),
    })
    users = uf.drop_duplicates("user").reset_index(drop=True)

    n_days = int(day.max())
    return RetrievalData(
        name="avazu",
        impressions=imp,
        ads=ads,
        users=users,
        train_days=(1, n_days - 1),
        test_day=n_days,
        proxy_ids=True,
        notes={"rows_read": int(len(df)), "days": n_days},
    )


def user_fields(data: RetrievalData) -> list:
    """The profile columns of the users table, whatever the dataset."""
    return [c for c in data.users.columns if c != "user"]


def load(source: str, path: str, **kw) -> RetrievalData:
    """Dispatch on the source name: taobao, avazu or synthetic."""
    if source == "taobao":
        return load_taobao(path, **kw)
    if source == "synthetic":
        return load_taobao(path, synthetic=True, **kw)
    if source == "avazu":
        return load_avazu(path, **kw)
    raise ValueError(f"unknown source {source!r}")
