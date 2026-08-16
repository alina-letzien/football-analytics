from calibrate_smoke_thresholds import BAND_COMMENTS, derive_bands


def test_derive_bands_basic():
    stats = {
        "avg_players_per_frame": 20.0,
        "avg_referees_per_frame": 2.0,
        "ball_detection_rate": 0.6,
        "player_track_id_count": 24,
        "speed_kmh_max": 30.0,
    }
    bands = derive_bands(stats)
    assert bands["AVG_PLAYERS_LOW"] == 14.0
    assert bands["AVG_PLAYERS_HIGH"] == 23.0
    assert bands["AVG_REFEREES_LOW"] == 1.0
    assert bands["AVG_REFEREES_HIGH"] == 3.0
    assert bands["BALL_DETECTION_RATE_MIN"] == 0.36
    assert bands["PLAYER_TRACK_ID_COUNT_LOW"] == 18
    assert bands["PLAYER_TRACK_ID_COUNT_HIGH"] == 34
    assert bands["SPEED_KMH_MAX_CEILING"] == 45.0


def test_derive_bands_avg_referees_high_has_floor_when_baseline_is_zero():
    # This pipeline currently detects zero referees on the baseline video.
    # Without a floor, AVG_REFEREES_HIGH would be 0.0 * 1.5 == 0.0, collapsing
    # the band to [0.0, 0.0] and asserting referees are always exactly zero —
    # any future detection improvement would then false-fail CI.
    stats = {
        "avg_players_per_frame": 12.0,
        "avg_referees_per_frame": 0.0,
        "ball_detection_rate": 0.5,
        "player_track_id_count": 20,
        "speed_kmh_max": 25.0,
    }
    bands = derive_bands(stats)
    assert bands["AVG_REFEREES_LOW"] == 0.0
    assert bands["AVG_REFEREES_HIGH"] == 1.0  # floor, not 0.0


def test_derive_bands_ball_detection_rate_has_no_hard_floor():
    # Purely relative to baseline — a hard floor above baseline*0.6 would make
    # the derived threshold impossible to pass against the very run it was
    # calibrated from whenever the baseline rate is naturally low (e.g. a
    # video where the ball is small/fast/frequently occluded).
    stats = {
        "avg_players_per_frame": 10.0,
        "avg_referees_per_frame": 1.0,
        "ball_detection_rate": 0.1,
        "player_track_id_count": 12,
        "speed_kmh_max": 20.0,
    }
    bands = derive_bands(stats)
    assert bands["BALL_DETECTION_RATE_MIN"] == 0.06  # 0.1 * 0.6, no floor
    assert bands["PLAYER_TRACK_ID_COUNT_LOW"] == 10  # absolute floor still applies, not round(12 * 0.75) = 9
    assert bands["SPEED_KMH_MAX_CEILING"] == 30.0  # 20.0 * 1.5


def test_band_comments_only_reference_real_band_keys():
    # BAND_COMMENTS is printed alongside derive_bands()'s output so a blind
    # paste into test_smoke.py doesn't drop the rationale for non-obvious
    # bands. If a key here stops matching a real band (e.g. renamed in
    # derive_bands), the comment would silently stop printing.
    bands = derive_bands({
        "avg_players_per_frame": 1.0,
        "avg_referees_per_frame": 1.0,
        "ball_detection_rate": 1.0,
        "player_track_id_count": 1,
        "speed_kmh_max": 1.0,
    })
    assert set(BAND_COMMENTS) <= set(bands)
