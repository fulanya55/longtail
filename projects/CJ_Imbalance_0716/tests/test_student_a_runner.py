from experiments.run_student_a import student_a_variant_matrix


def test_student_a_variant_matrix_forms_minimum_ablation_loop():
    variants = student_a_variant_matrix()

    assert list(variants) == ["A0", "A2", "A3", "A4"]
    assert variants["A0"] == {"prototype": False, "power": False}
    assert variants["A2"] == {"prototype": True, "power": False}
    assert variants["A3"] == {"prototype": False, "power": True}
    assert variants["A4"] == {"prototype": True, "power": True}
