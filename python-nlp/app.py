"""
app.py — Harivaidya Web Demo

A Gradio web interface that demonstrates:
  1. PHI anonymization (patient safety)
  2. Medical entity extraction (AI understanding)
  3. Clinical severity assessment (rule-based triage)

Design philosophy:
  - Severity banner FIRST (most important output)
  - Show the JOURNEY: PHI stripped → entities found → severity assessed
  - Everything visible, nothing hidden

To run locally:
    uv run python app.py
    Opens http://localhost:7860

🕉️  Dedicated to Lord Hari and Lord Vaidyanath
    For every patient who ever held a report
    and did not know what it meant.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from harivaidya.anonymizer import PHIAnonymizer
from harivaidya.ner_pipeline import MedicalNERPipeline
from harivaidya.clinical_rules import ClinicalRuleEngine


# ─────────────────────────────────────────────────────────────
# INITIALIZE MODELS (load once at startup)
# ─────────────────────────────────────────────────────────────

print("🕉️  Initializing Harivaidya...")
print("   Loading anonymizer...")
anonymizer = PHIAnonymizer()

print("   Loading NER model (first time downloads ~250MB)...")
ner = MedicalNERPipeline()
ner.load()

print("   Loading clinical rule engine...")
rule_engine = ClinicalRuleEngine()

print("✅ Harivaidya ready.\n")


# ─────────────────────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────

def analyze_report(report_text: str):
    """
    Returns 4 outputs:
      1. Severity banner (HTML)
      2. Anonymized text
      3. Entity table
      4. Summary markdown
    """

    if not report_text or not report_text.strip():
        return (
            "<div style='text-align:center;padding:20px;color:#888'>⚪ Please paste a medical report to analyze.</div>",
            "",
            [],
            "No analysis performed.",
        )

    # Layer 1: Strip PHI
    anon_result = anonymizer.anonymize(report_text)

    # Layer 2: Run NER
    ner_result = ner.extract(anon_result.anonymized_text)

    # Layer 3: Clinical rules
    entity_texts = [e.text for e in ner_result.entities]
    assessment = rule_engine.assess(
        text=anon_result.anonymized_text,
        entities=entity_texts,
    )

    # Build severity banner
    severity_emojis = {
        "CRITICAL": "🚨",
        "MODERATE": "⚠️",
        "MILD": "📋",
        "NORMAL": "✅",
    }
    severity_colors = {
        "CRITICAL": "#CC0000",
        "MODERATE": "#FF9900",
        "MILD": "#FFCC00",
        "NORMAL": "#00AA00",
    }
    emoji = severity_emojis.get(assessment.severity.value, "⚪")
    color = severity_colors.get(assessment.severity.value, "#888888")

    severity_banner = f"""
<div style="background: {color}15; border: 3px solid {color}; border-radius: 12px; padding: 20px; text-align: center;">
    <div style="font-size: 48px;">{emoji}</div>
    <div style="font-size: 28px; font-weight: bold; color: {color}; margin: 8px 0;">
        {assessment.severity.value}
    </div>
    <div style="font-size: 16px; color: #333; margin-top: 12px;">
        {assessment.action}
    </div>
    <div style="font-size: 13px; color: #666; margin-top: 12px; font-style: italic;">
        Why: {assessment.reasoning}
    </div>
</div>
"""

    # Build entity table
    entity_rows = []
    for entity in ner_result.entities:
        entity_rows.append([
            entity.text,
            entity.label.replace("_", " ").title(),
            f"{entity.confidence:.1%}",
        ])

    # Build summary
    red_flag_text = ""
    if assessment.red_flags:
        red_flag_text = f"\n- **🚩 RED FLAGS:** {', '.join(assessment.red_flags)}"

    summary = f"""
**📊 Analysis Summary**

- **Severity:** {assessment.severity.value}
- **Trigger findings:** {', '.join(assessment.trigger_findings) if assessment.trigger_findings else 'None'}{red_flag_text}
- **PHI items stripped:** {anon_result.phi_count}
- **Medical entities found:** {ner_result.total_entities}
- **Processing time:** {ner_result.processing_time_ms:.0f} ms
- **Audit hash:** `{anon_result.content_hash}`

**🔒 Privacy guarantee:** The anonymized text above is what the AI actually analyzed.
Your original patient data never touched the AI model.

🕉️  *Powered by Harivaidya — Hari + Vaidyanath, the Divine Physician.*
"""

    return (
        severity_banner,
        anon_result.anonymized_text,
        entity_rows,
        summary,
    )


# ─────────────────────────────────────────────────────────────
# EXAMPLES
# ─────────────────────────────────────────────────────────────

EXAMPLE_REPORTS = [
    ["""Patient Name: Ramesh Kumar Sharma
Aadhaar: 1234 5678 9012
Phone: +91 9876543210
Date: 15/11/2024
Ref. Doctor: Dr. Priya Singh

FINDINGS: Right lower lobe consolidation with mild cardiomegaly.
CTR 0.55. No pleural effusion.

IMPRESSION: Pneumonia in right lower lobe. Mild cardiomegaly."""],

    ["""Patient: Suresh Patel
Phone: 9988776655
Date: 10/01/2024

BLOOD TEST RESULTS:
HbA1c: 8.3% (diabetic range)
Fasting Glucose: 187 mg/dL
Hemoglobin: 9.8 g/dL (low)
Creatinine: 1.4 mg/dL (mildly elevated)

INTERPRETATION: Poorly controlled diabetes with
progressive anaemia and early kidney involvement."""],

    ["""IMPRESSION: Bilateral pneumothorax with mediastinal shift.
Severe respiratory compromise. Emergency chest tube placement required.
Blood pressure dropping — likely tension pneumothorax."""],

    ["""IMPRESSION: Normal chest X-ray. Clear lung fields.
No abnormalities detected. Routine screening."""],
]


# ─────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    font-family: 'Segoe UI', Tahoma, sans-serif;
}
#title {
    text-align: center;
    background: linear-gradient(135deg, #FF9933, #138808);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em;
    font-weight: bold;
}
"""

with gr.Blocks(title="Harivaidya — Medical AI for India", css=custom_css) as demo:

    # Header
    gr.HTML("""
    <div id="title">🕉️ Harivaidya — हरिवैद्य</div>
    <p style="text-align:center; color: #888; margin-top: 0;">
        AI Medical Report Analyzer for India<br>
        <em>Hari (Narayan) + Vaidyanath (Shiva) — The Divine Physician</em>
    </p>
    """)

    # Input
    gr.Markdown("### 📋 Paste a medical report below")
    gr.Markdown(
        "Try an example below, or paste any X-ray, blood test, or clinical report. "
        "Patient data will be automatically stripped before AI analysis."
    )

    input_text = gr.Textbox(
        label="Medical report text",
        lines=12,
        placeholder="Paste any medical report here...",
    )

    analyze_btn = gr.Button("🔍 Analyze with Harivaidya", variant="primary", size="lg")

    # SEVERITY BANNER (most important output, shown first)
    gr.Markdown("---")
    gr.Markdown("### 🎯 Clinical Severity Assessment")
    output_severity = gr.HTML()

    # Anonymized text
    gr.Markdown("---")
    gr.Markdown("### 🔒 Anonymized Text (PHI Stripped)")
    gr.Markdown("This is what the AI actually sees — all patient identifiers replaced.")

    output_anon = gr.Textbox(
        label="Anonymized version",
        lines=10,
        interactive=False,
    )

    # Entity table
    gr.Markdown("### 🧠 Medical Entities Extracted")
    gr.Markdown("Our AI found these medical concepts automatically.")

    output_entities = gr.Dataframe(
        headers=["Entity", "Category", "Confidence"],
        label="Medical entities",
        wrap=True,
    )

    # Summary
    gr.Markdown("### 📊 Summary")
    output_summary = gr.Markdown()

    # Examples
    gr.Examples(
        examples=EXAMPLE_REPORTS,
        inputs=input_text,
        label="💡 Try these example reports",
    )

    # Wire up the button
    analyze_btn.click(
        fn=analyze_report,
        inputs=[input_text],
        outputs=[output_severity, output_anon, output_entities, output_summary],
    )

    # Footer
    gr.Markdown("""
---
**🕉️ About Harivaidya**
- Built with love for the people of India 🇮🇳
- Open source: [github.com/bapuoldtown/harivaidya](https://github.com/bapuoldtown/harivaidya)
- Privacy first: PHI stripped before any AI sees your data
- Free for patients, sustainable for hospitals
- Dedicated to Lord Jagannath and Lord Lingaraj of Odisha
""")


# ─────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )