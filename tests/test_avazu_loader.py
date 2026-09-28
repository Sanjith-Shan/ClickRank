"""The Avazu adapter maps proxy ids into the canonical tables.

Avazu is not on the development machine, so this builds a few rows in the
competition's file format and checks the id proxies and the day split.
"""

import pandas as pd

from src.retrieval.data import AVAZU_PLACEHOLDER_DEVICE, load_avazu


def _rows():
    base = {
        "banner_pos": "0", "site_id": "1fbe01fe", "app_id": "ecad2386",
        "device_ip": "ddd2926e", "device_model": "44956a24", "device_type": "1",
        "device_conn_type": "2", "C15": "320", "C16": "50", "C17": "1722",
        "C18": "0", "C19": "35", "C20": "-1", "C21": "79",
    }
    rows = []
    for i, (hour, dev, ip, c14, click) in enumerate([
        ("14102100", "aaaa0001", "ip000001", "15706", "0"),
        ("14102100", AVAZU_PLACEHOLDER_DEVICE, "ip000002", "15704", "1"),
        ("14102205", AVAZU_PLACEHOLDER_DEVICE, "ip000002", "15706", "0"),
        ("14102312", "aaaa0001", "ip000003", "15704", "1"),
    ]):
        r = dict(base, id=str(i), click=click, hour=hour, device_id=dev, C14=c14)
        r["device_ip"] = ip
        rows.append(r)
    return pd.DataFrame(rows)


def test_avazu_proxies_and_split(tmp_path):
    path = tmp_path / "train.csv"
    _rows().to_csv(path, index=False)
    d = load_avazu(str(path))

    assert d.proxy_ids and "proxies" in d.label()
    imp = d.impressions
    # device_id identifies the user where it is real, device_ip where it is
    # the placeholder, so rows 0 and 3 share a user and rows 1 and 2 share one.
    assert imp["user"][0] == imp["user"][3]
    assert imp["user"][1] == imp["user"][2]
    assert imp["user"][0] != imp["user"][1]
    assert list(imp["day"]) == [1, 1, 2, 3]
    assert d.train_days == (1, 2) and d.test_day == 3
    assert set(d.ads["ad"]) == {15704, 15706}
    assert len(d.users) == 2
