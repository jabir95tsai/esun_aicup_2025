import polars as pl

from esun_aicup_2025.preprocess.data_preprocess import create_observation_points


def test_create_observation_points_includes_multi_pit_and_negatives():
    acc_alert = pl.DataFrame({"acct": ["A1"], "event_date": [10]})
    acc_predict = pl.DataFrame({"acct": ["A2", "A3"], "label": [0, 1]})

    result = create_observation_points(acc_alert, acc_predict, max_date=20, offsets=[1, 3])

    rows = {(r[0], r[1], r[2]) for r in result.select(["acct", "event_date", "label"]).iter_rows()}
    assert ("A1", 10, 1) in rows
    assert ("A1", 9, 1) in rows
    assert ("A1", 7, 1) in rows
    assert ("A2", 20, 0) in rows
