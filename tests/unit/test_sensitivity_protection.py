from cnn_compression_toolkit.sensitivity.analyzer import (
    SensitivityResult,
    protected_pruning_ratios,
)


def test_lowest_sqnr_layers_receive_protected_ratio() -> None:
    results = [
        SensitivityResult("conv_a", "Conv2d", 1.0, 2.0, 8),
        SensitivityResult("conv_b", "Conv2d", 0.5, 4.0, 8),
        SensitivityResult("conv_c", "Conv2d", 0.1, 8.0, 8),
    ]

    ratios, marked = protected_pruning_ratios(results, 0.5, protected_count=2)

    assert ratios == {"conv_a": 0.25, "conv_b": 0.25}
    assert [result.protected for result in marked] == [True, True, False]