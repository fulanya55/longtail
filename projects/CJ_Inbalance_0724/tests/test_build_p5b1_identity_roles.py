from cj0724.build_p5b1_identity_roles import COUNTS


def test_p5b1_roles_use_all_sixty_identities_per_combination():
    assert COUNTS == {"train": 36, "early": 12, "confirm": 12}
    assert sum(COUNTS.values()) == 60
