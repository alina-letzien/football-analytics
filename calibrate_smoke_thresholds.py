"""
Derive tolerance bands for tests/test_smoke.py from a baseline
output/analysis_summary.json. Re-run after any intentional change to
detection/tracking behavior (model version, thresholds, dependency bumps
to ultralytics/torch/supervision) and paste the printed block into
test_smoke.py.

Usage: python calibrate_smoke_thresholds.py output/analysis_summary.json
"""
import json
import sys


def derive_bands(stats: dict) -> dict:
    return {
        "AVG_PLAYERS_LOW": round(stats["avg_players_per_frame"] * 0.7, 1),
        "AVG_PLAYERS_HIGH": round(stats["avg_players_per_frame"] * 1.15, 1),
        "AVG_REFEREES_LOW": round(stats["avg_referees_per_frame"] * 0.5, 1),
        # Floored at 1.0 so a baseline of 0.0 referees (this pipeline
        # currently detects none) still leaves headroom — otherwise the band
        # collapses to [0.0, 0.0] and asserts referees are always exactly
        # zero, failing CI the moment detection improves.
        "AVG_REFEREES_HIGH": max(1.0, round(stats["avg_referees_per_frame"] * 1.5, 1)),
        # Purely relative to baseline (no hard floor) — a hard floor above
        # baseline*0.6 would make the derived threshold impossible to pass
        # against the very run it was calibrated from whenever the baseline
        # rate is naturally below that floor.
        "BALL_DETECTION_RATE_MIN": round(stats["ball_detection_rate"] * 0.6, 2),
        "PLAYER_TRACK_ID_COUNT_LOW": max(10, round(stats["player_track_id_count"] * 0.75)),
        "PLAYER_TRACK_ID_COUNT_HIGH": round(stats["player_track_id_count"] * 1.4),
        # Relative to baseline, not a hardcoded physiological limit — this
        # pipeline has no perspective calibration configured
        # (FIELD_CONFIG["calibration_points"] is None), so absolute speed
        # values are known-approximate (see README limitations). This catches
        # a further regression in speed calculation, not physiological
        # implausibility.
        "SPEED_KMH_MAX_CEILING": round(stats["speed_kmh_max"] * 1.5, 1),
    }


# Explanatory comments for bands whose formula isn't self-evident from the
# value alone — printed alongside the constant so a blind paste into
# test_smoke.py doesn't silently drop the rationale.
BAND_COMMENTS = {
    "AVG_REFEREES_HIGH": (
        "# Floored at 1.0 so a baseline of 0.0 referees still leaves headroom —\n"
        "# otherwise the band collapses to [0.0, 0.0] and asserts referees are\n"
        "# always exactly zero, failing CI the moment detection improves."
    ),
    "SPEED_KMH_MAX_CEILING": (
        "# Relative to baseline, not a hardcoded physiological limit — this\n"
        "# pipeline has no perspective calibration configured\n"
        "# (FIELD_CONFIG[\"calibration_points\"] is None), so absolute speed\n"
        "# values are known-approximate (see README limitations). This catches\n"
        "# a further regression in speed calculation, not physiological\n"
        "# implausibility."
    ),
}


def main():
    if len(sys.argv) != 2:
        print("Usage: python calibrate_smoke_thresholds.py <path-to-analysis_summary.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        stats = json.load(f)

    bands = derive_bands(stats)
    print("# Paste into tests/test_smoke.py, replacing the existing constants block:")
    for name, value in bands.items():
        if name in BAND_COMMENTS:
            print(BAND_COMMENTS[name])
        print(f"{name} = {value}")
    print("TEAM_SPLIT_MAX_SKEW = 0.30  # max fractional imbalance between team_split[0] and team_split[1]")


if __name__ == "__main__":
    main()
