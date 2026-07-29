from rollout.grasp_ablation import AblationSampleResult, select_method


def _sample(name, expected, white, geometry, hybrid, latency=1.0):
    return AblationSampleResult(
        name,
        expected,
        {
            "WHITE_WINDOW": white,
            "MASK_GEOMETRY": geometry,
            "HYBRID": hybrid,
        },
        {
            "WHITE_WINDOW": latency,
            "MASK_GEOMETRY": latency + 1,
            "HYBRID": latency + 2,
        },
    )


def test_ablation_prefers_fail_closed_hybrid_when_equally_accurate():
    samples = [
        _sample("success", True, True, True, True),
        _sample("false-grasp", False, True, False, False),
    ]
    result = select_method(samples)
    assert result.selected_method == "HYBRID"


def test_ablation_uses_hybrid_fail_closed_fallback():
    samples = [
        _sample("negative-a", False, True, False, False),
        _sample("negative-b", False, False, True, False),
        _sample("negative-c", False, True, True, True),
    ]
    result = select_method(samples)
    assert result.selected_method == "HYBRID"
