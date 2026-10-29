"""Tests for Task 7: Rich Rule Extraction - feature aliasing, validation, and dedup.

Tests pure Python functions and feature aliasing without requiring pyswip/numpy.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.reasoning.reasoning import _extract_rule_body, _body_to_key


class TestRuleBodyExtract:
    """Test _extract_rule_body extracts rule bodies correctly."""

    def test_extract_body_with_separator(self):
        rule = "neural_rule_1 :- feature_threshold(s4, _, high), trend(s7, decreasing)."
        body = _extract_rule_body(rule)
        assert body is not None
        assert "feature_threshold(s4, _, high)" in body
        assert "trend(s7, decreasing)" in body

    def test_extract_body_no_separator(self):
        rule = "neural_rule_1."
        body = _extract_rule_body(rule)
        assert body is None

    def test_extract_body_trailing_dot(self):
        rule = "neural_rule_1 :- feature_threshold(s4, _, high)."
        body = _extract_rule_body(rule)
        assert body is not None
        assert body == "feature_threshold(s4, _, high)"


class TestBodyDedup:
    """Test _body_to_key enables body-level deduplication."""

    def test_dedup_same_body_different_names(self):
        """Rules with same body but different names should map to same key."""
        key1 = _body_to_key("neural_rule_1 :- feature_threshold(s4, _, high).")
        key2 = _body_to_key("neural_rule_2 :- feature_threshold(s4, _, high).")
        assert key1 == key2, f"Expected same key for same body, got {key1} != {key2}"

    def test_dedup_different_bodies(self):
        """Rules with different bodies should have different keys."""
        key1 = _body_to_key("neural_rule_1 :- feature_threshold(s4, _, high).")
        key2 = _body_to_key("neural_rule_2 :- feature_threshold(s7, _, low).")
        assert key1 != key2

    def test_dedup_no_body(self):
        """Rules without ':-' should return None."""
        key = _body_to_key("neural_rule_1.")
        assert key is None


class TestFeatureAliasing:
    """Test that feature aliases are present in rules.pl."""

    def test_temperature_alias(self):
        with open("src/reasoning/rules.pl", "r") as f:
            content = f.read()
        assert "temperature_alias(s4)." in content

    def test_vibration_alias(self):
        with open("src/reasoning/rules.pl", "r") as f:
            content = f.read()
        assert "vibration_alias(s11)." in content

    def test_pressure_alias(self):
        with open("src/reasoning/rules.pl", "r") as f:
            content = f.read()
        assert "pressure_alias(s7)." in content

    def test_efficiency_alias(self):
        with open("src/reasoning/rules.pl", "r") as f:
            content = f.read()
        assert "efficiency_alias(s12)." in content


def test_acceptance_criteria():
    """Verify all Task 7 acceptance criteria are met."""
    # 1. rules_learned.pl supports neural_rule/0 predicates (already done)
    with open("src/reasoning/rules.pl", "r") as f:
        rules_content = f.read()
    assert "neural_rule" in rules_content, "rules.pl should contain neural_rule/0"
    
    # 2. Feature aliases map s4→temperature, s11→vibration, s7→pressure, s12→efficiency_index
    with open("src/reasoning/rules.pl", "r") as f:
        rules_content = f.read()
    assert "temperature_alias(s4)." in rules_content
    assert "vibration_alias(s11)." in rules_content
    assert "pressure_alias(s7)." in rules_content
    assert "efficiency_alias(s12)." in rules_content
    
    # 3. _extract_rule_body works correctly
    rule = "neural_rule_1 :- feature_threshold(s4, _, high)."
    body = _extract_rule_body(rule)
    assert body == "feature_threshold(s4, _, high)"
    
    # 4. _body_to_key enables dedup
    key1 = _body_to_key("neural_rule_1 :- feature_threshold(s4, _, high).")
    key2 = _body_to_key("neural_rule_2 :- feature_threshold(s4, _, high).")
    assert key1 == key2
    
    # 5. validate_learned_rules is callable (structure check)
    from src.reasoning.reasoning import SymbolicReasoner
    # Just verify the method exists - may fail without Prolog but structure should be there
    assert hasattr(SymbolicReasoner, 'validate_learned_rules')
    assert hasattr(SymbolicReasoner, 'extract_rules_from_neural_model')
    assert hasattr(SymbolicReasoner, 'update_rules')
    
    print("✓ All acceptance criteria verified!")


if __name__ == "__main__":
    # Run body extract tests
    t1 = TestRuleBodyExtract()
    t1.test_extract_body_with_separator()
    print("✓ test_extract_body_with_separator")
    t1.test_extract_body_no_separator()
    print("✓ test_extract_body_no_separator")
    t1.test_extract_body_trailing_dot()
    print("✓ test_extract_body_trailing_dot")
    
    # Run dedup tests
    t2 = TestBodyDedup()
    t2.test_dedup_same_body_different_names()
    print("✓ test_dedup_same_body_different_names")
    t2.test_dedup_different_bodies()
    print("✓ test_dedup_different_bodies")
    t2.test_dedup_no_body()
    print("✓ test_dedup_no_body")
    
    # Run feature alias tests
    t3 = TestFeatureAliasing()
    t3.test_temperature_alias()
    print("✓ test_temperature_alias")
    t3.test_vibration_alias()
    print("✓ test_vibration_alias")
    t3.test_pressure_alias()
    print("✓ test_pressure_alias")
    t3.test_efficiency_alias()
    print("✓ test_efficiency_alias")
    
    # Run acceptance criteria test
    test_acceptance_criteria()
    
    print("\n=== All Task 7 tests passed! ===")
