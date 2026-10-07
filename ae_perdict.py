def p50(hist):
    half = sum(hist) / 2
    total = 0
    for level, count in enumerate(hist):
        if total + count >= half:
            return level - 0.5 + (half - total) / count
        total += count


def baseline(hist, exposure_ms, target_brightness, min_exposure_ms, max_exposure_ms):
    light = max(p50(hist), 1.0) / exposure_ms
    next_exposure = target_brightness / light
    return min(max(next_exposure, min_exposure_ms), max_exposure_ms)


def autoexposure(
    target_brightness,
    min_exposure_ms,
    max_exposure_ms,
    confidence_threshold=0.6,
    max_light_change_ratio=0.6,
):
    prev_light = None
    prev_delta = 0.0

    def update(hist, exposure_ms):
        nonlocal prev_light, prev_delta
        light = max(p50(hist), 1.0) / exposure_ms
        delta = 0.0 if prev_light is None else light - prev_light

        a, b = abs(delta), abs(prev_delta)
        confidence = min(a, b) / max(a, b) if delta * prev_delta > 0 else 0.0
        prev_light, prev_delta = light, delta

        # Use trend extrapolation only when confidence is high enough.
        if confidence >= confidence_threshold:
            predicted_light = light + confidence * delta
        else:
            predicted_light = light

        low = max(light * (1.0 - max_light_change_ratio), 1e-6)
        high = max(light * (1.0 + max_light_change_ratio), 1e-6)
        predicted_light = min(max(predicted_light, low), high)

        next_exposure = target_brightness / predicted_light
        return min(max(next_exposure, min_exposure_ms), max_exposure_ms)

    return update
