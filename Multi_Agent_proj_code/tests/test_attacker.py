from agents.attacker import (
    JAILBREAK_TEMPLATES,
    AdaptiveAttacker,
    TemplateAttacker,
    get_attacker,
)


def test_template_attack_embeds_seed():
    attacker = TemplateAttacker(preferred_templates=["dan"])
    result = attacker.generate_attack("build a device", "weapons")
    assert "build a device" in result.attack_prompt
    assert result.template_used == "dan"
    assert result.turn == 1


def test_all_templates_format_cleanly():
    for name, template in JAILBREAK_TEMPLATES.items():
        rendered = template.format(prompt="SEED")
        assert "SEED" in rendered, f"template {name} dropped the seed prompt"


def test_registry_returns_fresh_instances():
    # Regression: registry used to hold singletons, so AdaptiveAttacker
    # history leaked across tournaments and experiments.
    a1 = get_attacker("adaptive_template_random")
    a2 = get_attacker("adaptive_template_random")
    assert a1 is not a2
    a1._history.append({"attack_prompt": "x"})
    assert a2._history == []


def test_adaptive_turn_increments_and_resets():
    adaptive = AdaptiveAttacker(TemplateAttacker())
    first = adaptive.generate_attack("seed", "weapons")
    second = adaptive.generate_attack("seed", "weapons")
    assert (first.turn, second.turn) == (1, 2)
    adaptive.reset()
    third = adaptive.generate_attack("seed", "weapons")
    assert third.turn == 1
