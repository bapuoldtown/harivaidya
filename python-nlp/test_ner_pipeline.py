"""Quick check that NER works — with DEBUG output."""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from harivaidya.ner_pipeline import MedicalNERPipeline
from harivaidya.anonymizer import PHIAnonymizer


report = """
Patient Name: Ramesh Kumar
Phone: 9876543210

FINDINGS: Right lower lobe consolidation with mild cardiomegaly.
CTR 0.55. No pleural effusion.

IMPRESSION: Pneumonia in right lower lobe. Mild cardiomegaly.
"""

# Step 1: Anonymize
anonymizer = PHIAnonymizer()
clean = anonymizer.anonymize(report)

print("=" * 70)
print("📄 TEXT BEING SENT TO NER MODEL:")
print("=" * 70)
print(clean.anonymized_text)
print("=" * 70)

# Step 2: Load NER
print("\nLoading NER model...")
ner = MedicalNERPipeline()
ner.load()

# Step 3: DEBUG — see RAW output before any filtering
print("\n" + "=" * 70)
print("🔍 RAW MODEL OUTPUT (no filtering applied)")
print("=" * 70)
raw = ner._model(clean.anonymized_text)
print(f"Total raw entities: {len(raw)}\n")
for r in raw:
    print(f"  score={r['score']:.3f}  label={r['entity_group']:25s}  word='{r['word']}'")

# Step 4: Filtered output
print("\n" + "=" * 70)
print("🎯 AFTER OUR CONFIDENCE FILTER")
print("=" * 70)
result = ner.extract(clean.anonymized_text)
print(f"Filtered entities: {result.total_entities}")
print(f"Processing time: {result.processing_time_ms} ms\n")
for entity in result.entities:
    bar = "█" * int(entity.confidence * 20)
    print(f"  [{entity.label:25s}] '{entity.text}' {bar} {entity.confidence:.2%}")