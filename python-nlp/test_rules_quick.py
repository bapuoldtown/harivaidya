"""Quick test of the clinical rules engine."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from harivaidya.clinical_rules import ClinicalRuleEngine, SeverityLevel

engine = ClinicalRuleEngine()

# Test cases
test_reports = [
    ("Normal chest X-ray. Clear lung fields. No abnormalities.",
     "Expected: NORMAL"),

    ("Right lower lobe consolidation with mild cardiomegaly. Pneumonia.",
     "Expected: MODERATE"),

    ("Tension pneumothorax with mediastinal shift. Emergency decompression needed.",
     "Expected: CRITICAL"),

    ("Mild cardiomegaly. CTR 0.53. No acute findings.",
     "Expected: MILD"),
]

for text, expected in test_reports:
    result = engine.assess(text)
    print(f"\n📋 Text: {text[:60]}...")
    print(f"   {expected}")
    print(f"   ✓ Severity: {result.severity.value}")
    print(f"   ✓ Action:   {result.action}")
    print(f"   ✓ Why:      {result.reasoning}")