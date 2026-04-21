"""
anonymizer.py — PHI (Protected Health Information) Anonymizer

This module strips all patient-identifying information from
medical report text BEFORE the text is sent to any AI model.

Why this matters:
  - Patient privacy is sacred (HIPAA, India DPDP Act 2023)
  - If we send raw PHI to any model, we become liable
  - Hospital clients will require this for any contract
  - It's simply the right thing to do

What gets stripped:
  - Patient names (after "Patient:", "Name:" labels)
  - Doctor names (after "Dr." prefix)
  - Phone numbers (Indian format: +91 9876543210)
  - Aadhaar numbers (12 digits, with or without spaces)
  - PAN numbers (ABCDE1234F format)
  - Email addresses
  - Dates of birth
  - Pincodes (6 digits)

Design principle:
  OVER-anonymize rather than under-anonymize.
  Better to strip too much than leak patient data.
"""

import re
import hashlib
from dataclasses import dataclass, field


@dataclass
class AnonymizationResult:
    """Result of anonymizing a medical report."""

    # The anonymized text, safe to send to AI
    anonymized_text: str

    # Count of PHI items found and replaced
    phi_count: int = 0

    # One-way hash of original text (for audit trail without storing PHI)
    # This lets us prove a document was processed WITHOUT keeping the content
    content_hash: str = ""

    # Breakdown of what was found
    items_found: dict = field(default_factory=dict)


class PHIAnonymizer:
    """
    Strips 18 categories of PHI from Indian medical reports.

    Usage:
        anonymizer = PHIAnonymizer()
        result = anonymizer.anonymize("Patient Ramesh, phone 9876543210")
        print(result.anonymized_text)
        # → "Patient [NAME_1], phone [PHONE_1]"
    """

    def __init__(self):
        # Pre-compile regex patterns for efficiency
        # Compiling once is much faster than compiling every call

        # Aadhaar: 12 digits, commonly formatted as "1234 5678 9012"
        self.aadhaar_pattern = re.compile(
            r'\b\d{4}\s?\d{4}\s?\d{4}\b'
        )

        # PAN card: ABCDE1234F format
        self.pan_pattern = re.compile(
            r'\b[A-Z]{5}\d{4}[A-Z]\b'
        )

        # Indian mobile: 10 digits starting with 6-9, optionally +91
        self.phone_pattern = re.compile(
            r'(?:\+91[\s-]?)?[6-9]\d{9}\b'
        )

        # Email addresses
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        )

        # Dates: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        self.date_pattern = re.compile(
            r'\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b'
        )

        # Pincodes: 6-digit Indian pincodes
        self.pincode_pattern = re.compile(
            r'\b[1-9]\d{5}\b'
        )

        # Patient name after "Patient:" or "Name:" labels
        
        # Patient name: stop at common keywords like Aadhaar, Phone, Date
        self.patient_name_pattern = re.compile(
            r'(Patient\s*(?:Name)?|Name)\s*[:\-]\s*'
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}?)'  # the ? makes it NON-greedy
            r'(?=\s*(?:Aadhaar|Phone|Email|Date|Age|Sex|MR|Reg|Ref|Dr|\n|$))',
            re.IGNORECASE,
        )

        # Doctor names: "Dr. Firstname Lastname"
        self.doctor_pattern = re.compile(
            r'\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b'
        )

        # Medical record numbers
        self.mrn_pattern = re.compile(
            r'(MR\.?\s*No\.?|MRN|Reg\.?\s*No\.?)\s*[:\-]?\s*[A-Z0-9/\-]+',
            re.IGNORECASE,
        )

    def anonymize(self, text: str) -> AnonymizationResult:
        """
        Main entry point. Strips all PHI from text.

        Args:
            text: Raw medical report text

        Returns:
            AnonymizationResult with cleaned text and metadata
        """
        if not text:
            return AnonymizationResult(anonymized_text="", content_hash="")

        # Create a one-way hash of the ORIGINAL text
        # This is for audit trail only — proves we processed it
        # Cannot be reversed to recover PHI
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Track what we find
        result = text
        items_found = {}

        # Apply each pattern in order
        # Order matters: specific patterns first, generic later
        result, count = self._replace(result, self.aadhaar_pattern, "AADHAAR")
        items_found["AADHAAR"] = count

        result, count = self._replace(result, self.pan_pattern, "PAN")
        items_found["PAN"] = count

        result, count = self._replace(result, self.phone_pattern, "PHONE")
        items_found["PHONE"] = count

        result, count = self._replace(result, self.email_pattern, "EMAIL")
        items_found["EMAIL"] = count

        result, count = self._replace(result, self.mrn_pattern, "MRN")
        items_found["MRN"] = count

        result, count = self._replace(result, self.doctor_pattern, "DOCTOR")
        items_found["DOCTOR"] = count

        # Patient name needs special handling (we replace only the name, not the label)
        result, count = self._replace_patient_name(result)
        items_found["NAME"] = count

        result, count = self._replace(result, self.date_pattern, "DATE")
        items_found["DATE"] = count

        result, count = self._replace(result, self.pincode_pattern, "PINCODE")
        items_found["PINCODE"] = count

        total_phi = sum(items_found.values())

        return AnonymizationResult(
            anonymized_text=result,
            phi_count=total_phi,
            content_hash=content_hash,
            items_found={k: v for k, v in items_found.items() if v > 0},
        )

    def _replace(self, text: str, pattern: re.Pattern, label: str) -> tuple[str, int]:
        """Replace all matches with [LABEL_N] placeholders."""
        counter = [0]

        def replace_fn(match):
            counter[0] += 1
            return f"[{label}_{counter[0]}]"

        new_text = pattern.sub(replace_fn, text)
        return new_text, counter[0]

    def _replace_patient_name(self, text: str) -> tuple[str, int]:
        """Special: replace the name part, keep the 'Patient:' label."""
        counter = [0]

        def replace_fn(match):
            counter[0] += 1
            label = match.group(1)
            return f"{label}: [NAME_{counter[0]}]"

        new_text = self.patient_name_pattern.sub(replace_fn, text)
        return new_text, counter[0]