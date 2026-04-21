"""
ner_pipeline.py — Medical Named Entity Recognition

Uses d4data/biomedical-ner-all (HuggingFace) to extract medical
entities from text: DISEASES, ANATOMY, SYMPTOMS, PROCEDURES, etc.

This is the second layer of Harivaidya's brain:
  Layer 1: Anonymizer  → strips PHI before AI sees text
  Layer 2: NER         → finds medical entities (THIS FILE)
  Layer 3: BioMistral  → clinical reasoning (later)

Design notes:
  - Singleton pattern: model loads ONCE at startup, not per request
  - Confidence threshold: filters out uncertain extractions
  - Returns dataclass: structured, type-safe output
"""
from dataclasses import dataclass, field
from typing import Optional
import logging

# transformers and torch are big imports — log so we know it's happening
logger = logging.getLogger(__name__)
@dataclass
class MedicalEntity:
    """A single medical entity found in text."""

    text: str           # the actual word(s): "consolidation"
    label: str          # the type: "Sign_symptom"
    confidence: float   # how sure the AI is: 0.0 to 1.0
    start: int          # character position in text where entity starts
    end: int            # character position where entity ends


@dataclass
class NERResult:
    """Result of running NER on a piece of text."""

    entities: list[MedicalEntity] = field(default_factory=list)
    total_entities: int = 0
    processing_time_ms: float = 0.0

    def by_label(self, label: str) -> list[MedicalEntity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities if e.label == label]
    
    def unique_diseases(self) -> list[str]:
        """Get list of unique disease names found."""
        diseases = self.by_label("Disease_disorder")
        return list({e.text.lower() for e in diseases})

class MedicalNERPipeline:
    """
    Singleton wrapper around the HuggingFace NER pipeline.

    Usage:
        pipeline = MedicalNERPipeline()
        pipeline.load()  # call once at app startup

        result = pipeline.extract("Patient has cardiomegaly")
        for entity in result.entities:
            print(f"{entity.label}: {entity.text}")
    """
    _instance = None
    _model = None
    MODEL_NAME = "d4data/biomedical-ner-all"
    MIN_CONFIDENCE = 0.5  # ignore entities below this confidence
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MedicalNERPipeline, cls).__new__(cls)
        return cls._instance
    def load(self):
        """
        Load the model into memory. Call ONCE at app startup.
        Subsequent calls are no-ops (model already loaded).
        """
        if self._model is not None:
            logger.info("Medical NER model already loaded, skipping.")
            return
        logger.info(f"Loading Medical NER model: {self.MODEL_NAME}")
        from transformers import pipeline
        self._model = pipeline("ner", model=self.MODEL_NAME, aggregation_strategy="simple")
        logger.info("Medical NER model loaded successfully.")
    def is_loaded(self) -> bool:
        return self._model is not None
    
    def extract(self, text:str) -> NERResult:
        """
        Extract medical entities from text.

        Args:
            text: Anonymized medical report text (PHI already stripped)

        Returns:
            NERResult with all entities found above confidence threshold
        """
        if not text or not text.strip():
            return NERResult()
        if not self.is_loaded():
            raise RuntimeError(
                "NER model not loaded. Call .load() first at app startup."
            )
        
        import time
        start_time = time.perf_counter()

        # Run the actual model
        raw_results = self._model(text)
        
        entities = []
        for raw in raw_results:
            confidence = float(raw["score"])
            # Filter out low-confidence extractions
            if confidence < self.MIN_CONFIDENCE:
                continue
            entities.append(MedicalEntity(
                text=raw["word"],
                label=raw["entity_group"],
                confidence=round(confidence, 3),
                start=int(raw["start"]),
                end=int(raw["end"]),
            ))
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
        return NERResult(
        entities=entities,
        total_entities=len(entities),
        processing_time_ms=round(elapsed_ms, 1),
    )
