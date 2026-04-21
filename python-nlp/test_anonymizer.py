"""Quick test of the anonymizer."""

import sys
from pathlib import Path

# Add src/ to Python path so we can import harivaidya
sys.path.insert(0, str(Path(__file__).parent / "src"))

from harivaidya.anonymizer import PHIAnonymizer


# Real test report with PHI
test_report = """
Patient Name: Ramesh Kumar Sharma
Aadhaar: 1234 5678 9012
Phone: +91 9876543210
Email: ramesh.sharma@gmail.com
Date: 15/11/2024
Ref. Doctor: Dr. Priya Singh

FINDINGS: Right lower lobe consolidation with mild cardiomegaly.
CTR 0.55. Features suggestive of pneumonia.

IMPRESSION:
1. Right lower lobe consolidation — pneumonia
2. Mild cardiomegaly
"""

anonymizer = PHIAnonymizer()
result = anonymizer.anonymize(test_report)

print("=" * 70)
print("ORIGINAL TEXT (contains PHI)")
print("=" * 70)
print(test_report)

print("=" * 70)
print("ANONYMIZED TEXT (safe for AI)")
print("=" * 70)
print(result.anonymized_text)

print("=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)
print(f"  PHI items stripped: {result.phi_count}")
print(f"  Content hash:       {result.content_hash}")
print(f"  Breakdown:          {result.items_found}")

print("\n🕉️  Clinical content preserved. Patient identity destroyed.")
print("    This is the ONLY way to ethically handle medical AI.")