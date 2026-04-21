"""
test_anonymizer.py — Tests for the PHI Anonymizer

These tests run automatically every time we push code.
If any test FAILS, GitHub Actions blocks the deploy.
This protects patient data from our own future mistakes.

Each test follows the AAA pattern:
  ARRANGE — set up the test data
  ACT     — call the function being tested
  ASSERT  — verify the result

Run all tests:    uv run pytest
Run with detail:  uv run pytest -v
Run one test:     uv run pytest tests/test_anonymizer.py::test_aadhaar_stripped
"""
import sys
from pathlib import Path

# Add src/ to Python's path so we can import harivaidya
# This is needed because tests live OUTSIDE src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from harivaidya.anonymizer import PHIAnonymizer, AnonymizationResult

# ─────────────────────────────────────────────────────────────────
# pytest fixture — reusable setup for tests
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def anonymizer():
    """
    Provides a fresh anonymizer to each test.
    Tests can use this by adding `anonymizer` as a parameter.
    """
    return PHIAnonymizer()

# ─────────────────────────────────────────────────────────────────
# CRITICAL TESTS — These MUST pass for patient safety
# ─────────────────────────────────────────────────────────────────

class TestPHIStripping:
    """Tests that PHI is correctly removed from text."""
    def test_aadhaar_stripped(self, anonymizer):
        """Aadhaar (12 digits, with or without spaces) must be removed."""
        # ARRANGE
        text = "Patient Aadhaar: 1234 5678 9012"
        # ACT
        result = anonymizer.anonymize(text)
        
        assert "1234 5678 9012" not in result.anonymized_text
        assert "[AADHAAR_1]" in result.anonymized_text
        assert result.phi_count >= 1
    def test_phone_stripped(self, anonymizer):
        """Indian mobile numbers (+91 followed by 10 digits) must be removed."""
        text = "Contact: +91 9876543210"
        result = anonymizer.anonymize(text)

        assert "9876543210" not in result.anonymized_text
        assert "[PHONE_1]" in result.anonymized_text
    def test_email_stripped(self, anonymizer):
        """Email addresses must be removed."""
        text = "Send report to ramesh@gmail.com"
        result = anonymizer.anonymize(text)

        assert "ramesh@gmail.com" not in result.anonymized_text
        assert "[EMAIL_1]" in result.anonymized_text
    def test_doctor_name_stripped(self, anonymizer):
        """Doctor names (Dr. Firstname Lastname) must be removed."""
        text = "Referred by Dr. Priya Singh"
        result = anonymizer.anonymize(text)

        assert "Priya Singh" not in result.anonymized_text
        assert "[DOCTOR_1]" in result.anonymized_text
    def test_patient_name_stripped(self, anonymizer):
        """Patient names after 'Patient Name:' label must be removed."""
        text = "Patient Name: Ramesh Kumar Sharma\nAadhaar: 1234 5678 9012"
        result = anonymizer.anonymize(text)

        assert "Ramesh Kumar" not in result.anonymized_text
        assert "[NAME_1]" in result.anonymized_text


# ─────────────────────────────────────────────────────────────────
# CRITICAL TESTS — Clinical content MUST be preserved
# ─────────────────────────────────────────────────────────────────
class TestClinicalPreservation:
    """Tests that medical findings are NOT accidentally stripped."""

    def test_findings_preserved(self, anonymizer):
        """Words like 'consolidation' must NOT be stripped."""
        text = "FINDINGS: Right lower lobe consolidation"
        result = anonymizer.anonymize(text)

        assert "consolidation" in result.anonymized_text
        assert "Right lower lobe" in result.anonymized_text

    def test_measurements_preserved(self, anonymizer):
        """CTR and other measurements must stay (they're clinical, not PHI)."""
        text = "CTR is 0.55. Hemoglobin 12.1 g/dL"
        result = anonymizer.anonymize(text)

        assert "CTR" in result.anonymized_text
        assert "0.55" in result.anonymized_text
        assert "12.1" in result.anonymized_text

    def test_diseases_preserved(self, anonymizer):
        """Disease names must NEVER be stripped."""
        text = "Diagnosis: pneumonia, cardiomegaly, diabetes"
        result = anonymizer.anonymize(text)

        assert "pneumonia" in result.anonymized_text
        assert "cardiomegaly" in result.anonymized_text
        assert "diabetes" in result.anonymized_text


# ─────────────────────────────────────────────────────────────────
# EDGE CASES — Things that often crash code
# ─────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for unusual inputs that often break code."""

    def test_empty_text(self, anonymizer):
        """Empty input should not crash."""
        result = anonymizer.anonymize("")

        assert result.anonymized_text == ""
        assert result.phi_count == 0

    def test_no_phi_in_text(self, anonymizer):
        """Text without any PHI should pass through unchanged."""
        text = "Right lower lobe consolidation"
        result = anonymizer.anonymize(text)

        assert result.anonymized_text == text
        assert result.phi_count == 0

    def test_content_hash_is_consistent(self, anonymizer):
        """Same input must always produce same hash (for audit trail)."""
        text = "Patient: Ramesh"
        r1 = anonymizer.anonymize(text)
        r2 = anonymizer.anonymize(text)

        assert r1.content_hash == r2.content_hash
        assert len(r1.content_hash) == 16  # we truncated to 16 chars


# ─────────────────────────────────────────────────────────────────
# INTEGRATION TEST — Real-world report
# ─────────────────────────────────────────────────────────────────

class TestRealWorldReport:
    """Test on a complete realistic medical report."""

    def test_full_report_anonymization(self, anonymizer):
        """A complete report must have all PHI stripped, all findings kept."""
        # ARRANGE - real report style with PHI
        report = """
Patient Name: Ramesh Kumar Sharma
Aadhaar: 1234 5678 9012
Phone: +91 9876543210
Date: 15/11/2024
Ref. Doctor: Dr. Priya Singh

FINDINGS: Right lower lobe consolidation with mild cardiomegaly.
CTR 0.55. Features suggestive of pneumonia.

IMPRESSION:
1. Right lower lobe consolidation — pneumonia
2. Mild cardiomegaly
"""

        # ACT
        result = anonymizer.anonymize(report)

        # ASSERT - PHI is GONE
        assert "Ramesh Kumar" not in result.anonymized_text
        assert "1234 5678 9012" not in result.anonymized_text
        assert "9876543210" not in result.anonymized_text
        assert "Priya Singh" not in result.anonymized_text

        # ASSERT - Clinical content is PRESERVED
        assert "consolidation" in result.anonymized_text
        assert "cardiomegaly" in result.anonymized_text
        assert "pneumonia" in result.anonymized_text
        assert "CTR 0.55" in result.anonymized_text

        # ASSERT - We caught multiple PHI items
        assert result.phi_count >= 5
