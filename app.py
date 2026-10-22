def _fixed_gop_args(info: dict[str, Any], clip_length: float, total_duration: float) -> list[str]:
    """Use fixed GOPs only when cumulative boundary drift stays within tolerance.

    Fixed GOPs are faster than forcing keyframes per segment, but a tiny
    per-GOP duration mismatch accumulates over long videos. If that accumulated
    drift approaches the segment muxer's tolerance, use the existing forced-
    keyframe route instead so long clips don't lose segment boundaries.
    """
    fps = _source_fps(info)
    if fps is None or clip_length <= 0.05 or total_duration <= 0:
        return []

    frames = max(1, int(round(fps * clip_length)))
    if frames < 2:
        return []

    gop_duration = frames / fps
    segment_count = max(
        1,
        int(math.ceil((total_duration - 1e-6) / clip_length)),
    )
    boundary_count = max(0, segment_count - 1)
    cumulative_drift = abs(gop_duration - clip_length) * boundary_count

    # The existing segment_time_delta is 0.02 seconds.
    # Reserve 25% as a safety margin.
    max_cumulative_drift = 0.015

    if cumulative_drift > max_cumulative_drift:
        log.info(
            "FIXED_GOP_DISABLED fps=%.6f clip_length=%.3f segments=%d "
            "gop_duration=%.6f cumulative_drift=%.6f budget=%.6f; "
            "using forced-keyframe fallback",
            fps,
            clip_length,
            segment_count,
            gop_duration,
            cumulative_drift,
            max_cumulative_drift,
        )
        return []

    return [
        "-g", str(frames),
        "-keyint_min", str(frames),
        "-sc_threshold", "0",
    ]
