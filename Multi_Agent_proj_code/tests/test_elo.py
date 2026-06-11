import pytest

from arena.elo import EloSystem, _expected_score, _new_ratings


def make_system():
    elo = EloSystem(k_factor=32, initial_rating=1200)
    elo.register_attacker("atk", "Attacker")
    elo.register_defender("dfn", "Defender")
    return elo


def test_expected_score_symmetric():
    assert _expected_score(1200, 1200) == pytest.approx(0.5)
    assert _expected_score(1400, 1200) + _expected_score(1200, 1400) == pytest.approx(1.0)


def test_ratings_are_zero_sum():
    a, b = _new_ratings(1300, 1100, a_won=True, k=32)
    assert a + b == pytest.approx(1300 + 1100)


def test_update_winner_gains_loser_drops():
    elo = make_system()
    new_atk, new_dfn = elo.update("atk", "dfn", attacker_won=True)
    assert new_atk > 1200
    assert new_dfn < 1200
    assert elo.get_attacker("atk").wins == 1
    assert elo.get_defender("dfn").losses == 1


def test_upset_moves_ratings_more_than_expected_win():
    # An underdog win should move ratings more than a favorite win.
    favorite_delta = _new_ratings(1400, 1000, a_won=True, k=32)[0] - 1400
    underdog_delta = _new_ratings(1000, 1400, a_won=True, k=32)[0] - 1000
    assert underdog_delta > favorite_delta


def test_leaderboard_sorted_desc():
    elo = EloSystem()
    elo.register_attacker("a1", "A1")
    elo.register_attacker("a2", "A2")
    elo.register_defender("d", "D")
    elo.update("a1", "d", attacker_won=True)
    elo.update("a2", "d", attacker_won=False)
    board = elo.attacker_leaderboard()
    assert board[0]["entity_id"] == "a1"
    assert board[0]["rating"] >= board[1]["rating"]


def test_serialization_round_trip():
    elo = make_system()
    elo.update("atk", "dfn", attacker_won=True)
    restored = EloSystem.from_dict(elo.to_dict())
    assert restored.get_attacker("atk").rating == pytest.approx(elo.get_attacker("atk").rating)
    assert restored.get_defender("dfn").losses == 1
