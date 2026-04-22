"""
clinical_rules.py — Rule-Based Severity & Action Engine

The third layer of Harivaidya's brain:
  Layer 1: Anonymizer       → strips PHI
  Layer 2: NER              → extracts medical entities
  Layer 3: Clinical Rules   → classifies severity + actions (THIS FILE)
  Layer 4: BioMistral       → deep reasoning (later)

Design philosophy:
  - HIGHEST severity wins — worst finding determines urgency
  - If patient has pneumonia + mild cardiomegaly, severity = MODERATE
  - Critical findings ALWAYS trigger alerts
  - No double-counting: "mild cardiomegaly" doesn't also count as "cardiomegaly"
"""

from dataclasses import dataclass, field
from enum import Enum


# ─────────────────────────────────────────────────────────────
# SEVERITY LEVELS
# ─────────────────────────────────────────────────────────────

class SeverityLevel(str, Enum):
    """4 severity levels from most to least urgent."""
    CRITICAL = "CRITICAL"
    MODERATE = "MODERATE"
    MILD = "MILD"
    NORMAL = "NORMAL"


# ─────────────────────────────────────────────────────────────
# CRITICAL FINDINGS — always trigger CRITICAL severity
# ─────────────────────────────────────────────────────────────

CRITICAL_KEYWORDS = {
    # Respiratory emergencies
    "pneumothorax",
    "tension pneumothorax",
    "massive pleural effusion",
    "respiratory failure",
    "pulmonary embolism",

    # Cardiac emergencies
    "myocardial infarction",
    "acute mi",
    "cardiac tamponade",
    "aortic dissection",
    "cardiac arrest",

    # Neurological emergencies
    "stroke",
    "cerebral hemorrhage",
    "subarachnoid hemorrhage",
    "intracranial bleed",
    "brain herniation",

    # Abdominal emergencies
    "bowel perforation",
    "ruptured aneurysm",
    "ectopic pregnancy",
    "acute appendicitis",

    # Trauma signals
    "tension",
    "perforation",
    "rupture",
}


# ─────────────────────────────────────────────────────────────
# MILD FINDINGS — specific qualifiers, downgrade severity
# ─────────────────────────────────────────────────────────────

MILD_KEYWORDS = {
    "mild cardiomegaly",
    "mild anaemia",
    "mild anemia",
    "mildly elevated",               # ← NEW (catches "mildly elevated creatinine")
    "bronchovascular markings",
    "atelectasis",
    "pre-diabetic",
    "prediabetic",
    "borderline",
}


# ─────────────────────────────────────────────────────────────
# MODERATE FINDINGS — need attention within days
# ─────────────────────────────────────────────────────────────

MODERATE_KEYWORDS = {
    # Respiratory
    "pneumonia",
    "consolidation",
    "pleural effusion",
    "bronchitis",
    "tuberculosis",

    # Cardiac
    "cardiomegaly",
    "heart failure",
    "arrhythmia",

    # Abdominal
    "cholecystitis",
    "pancreatitis",
    "hepatitis",

    # Endocrine / Metabolic (EXPANDED)
    "hyperglycemia",
    "diabetic ketoacidosis",
    "dka",
    "diabetes",                     # ← NEW
    "poorly controlled diabetes",   # ← NEW
    "diabetic range",               # ← NEW
    "uncontrolled diabetes",        # ← NEW

    # Hematological (NEW SECTION)
    "anaemia",                      # ← NEW (British spelling)
    "anemia",                       # ← NEW (American spelling)
    "progressive anaemia",          # ← NEW
    "progressive anemia",           # ← NEW

    # Kidney (NEW SECTION)
    "kidney involvement",           # ← NEW
    "kidney disease",               # ← NEW
    "nephropathy",                  # ← NEW
    "elevated creatinine",          # ← NEW
    "renal insufficiency",          # ← NEW
}


# ─────────────────────────────────────────────────────────────
# ACTION RECOMMENDATIONS per severity
# ─────────────────────────────────────────────────────────────

ACTION_MESSAGES = {
    SeverityLevel.CRITICAL: (
        "🚨 EMERGENCY — Call ambulance or go to nearest ER immediately. "
        "This cannot wait."
    ),
    SeverityLevel.MODERATE: (
        "⚠️ URGENT — See a doctor within 24-48 hours. "
        "Do not wait, but you don't need an ambulance."
    ),
    SeverityLevel.MILD: (
        "📋 MONITOR — Discuss with your doctor at next scheduled visit. "
        "No immediate action needed."
    ),
    SeverityLevel.NORMAL: (
        "✅ NORMAL — No concerning findings detected. "
        "Continue routine care."
    ),
}


# ─────────────────────────────────────────────────────────────
# THE RESULT DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class ClinicalAssessment:
    """Complete clinical assessment of a medical report."""
    severity: SeverityLevel = SeverityLevel.NORMAL
    trigger_findings: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    action: str = ""
    color: str = "green"
    reasoning: str = ""


# ─────────────────────────────────────────────────────────────
# THE MAIN ENGINE
# ─────────────────────────────────────────────────────────────

class ClinicalRuleEngine:
    """
    Assesses severity using deterministic rules.
    HIGHEST severity wins — worst finding determines urgency.
    """

    SEVERITY_COLORS = {
        SeverityLevel.CRITICAL: "red",
        SeverityLevel.MODERATE: "orange",
        SeverityLevel.MILD: "yellow",
        SeverityLevel.NORMAL: "green",
    }

    def assess(self, text: str, entities: list[str] = None) -> ClinicalAssessment:
        """
        Assess severity from report text + optional entity list.

        Logic: Find ALL severity matches, return HIGHEST severity found.
        If patient has pneumonia + mild cardiomegaly → MODERATE (pneumonia wins).
        """
        if not text:
            return self._build_result(SeverityLevel.NORMAL, [], [])

        text_lower = text.lower()
        all_sources = [text_lower]

        if entities:
            entity_text = " ".join(e.lower() for e in entities)
            all_sources.append(entity_text)

        combined = " ".join(all_sources)

        # Find ALL matches across all severity levels
        critical_matches = self._find_matches(combined, CRITICAL_KEYWORDS)
        moderate_matches = self._find_matches(combined, MODERATE_KEYWORDS)
        mild_matches = self._find_matches(combined, MILD_KEYWORDS)

        # Remove moderate matches already covered by mild qualifiers
        # e.g. "cardiomegaly" removed if "mild cardiomegaly" was found
        moderate_matches = self._filter_out_mild_qualifiers(
            moderate_matches, mild_matches
        )

        # Rule 1: CRITICAL wins above all (life-threatening)
        if critical_matches:
            return self._build_result(
                severity=SeverityLevel.CRITICAL,
                triggers=critical_matches,
                red_flags=critical_matches,
                reasoning=(
                    f"Critical finding(s) detected: {', '.join(critical_matches)}. "
                    f"These indicate potentially life-threatening conditions."
                ),
            )

        # Rule 2: MODERATE wins over MILD (worst finding matters)
        if moderate_matches:
            return self._build_result(
                severity=SeverityLevel.MODERATE,
                triggers=moderate_matches,
                red_flags=[],
                reasoning=(
                    f"Moderate finding(s): {', '.join(moderate_matches)}. "
                    f"These require medical attention within 1-2 days."
                ),
            )

        # Rule 3: Only MILD findings → MILD
        if mild_matches:
            return self._build_result(
                severity=SeverityLevel.MILD,
                triggers=mild_matches,
                red_flags=[],
                reasoning=(
                    f"Mild finding(s): {', '.join(mild_matches)}. "
                    f"Monitor but not urgent."
                ),
            )

        # Rule 4: Nothing concerning → NORMAL
        return self._build_result(
            severity=SeverityLevel.NORMAL,
            triggers=[],
            red_flags=[],
            reasoning="No concerning findings detected in report.",
        )

    def _find_matches(self, text: str, keywords: set) -> list[str]:
        """Find which keywords appear in text."""
        return [kw for kw in keywords if kw in text]

    def _filter_out_mild_qualifiers(
        self, moderate: list[str], mild: list[str]
    ) -> list[str]:
        """
        Prevent double-counting: if 'mild cardiomegaly' is in mild list,
        remove bare 'cardiomegaly' from moderate list (same finding).
        """
        result = []
        for mod in moderate:
            is_qualified = any(mod in mild_kw for mild_kw in mild)
            if not is_qualified:
                result.append(mod)
        return result

    def _build_result(
        self,
        severity: SeverityLevel,
        triggers: list[str],
        red_flags: list[str],
        reasoning: str = "",
    ) -> ClinicalAssessment:
        """Construct the final assessment."""
        return ClinicalAssessment(
            severity=severity,
            trigger_findings=triggers,
            red_flags=red_flags,
            action=ACTION_MESSAGES[severity],
            color=self.SEVERITY_COLORS[severity],
            reasoning=reasoning,
        )
        