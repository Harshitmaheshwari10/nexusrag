"""
NexusRAG Entity Extractor
Rule-based NER + spaCy (optional) for knowledge graph population.
Falls back to regex patterns if spaCy isn't available.
"""

import re
from typing import List, Dict


CONCEPT_PATTERNS = [
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b',  # Title Case phrases
]

TECH_KEYWORDS = [
    "machine learning", "deep learning", "neural network", "transformer",
    "llm", "rag", "embedding", "vector", "api", "database", "algorithm",
    "python", "javascript", "react", "fastapi", "langchain", "faiss",
    "reinforcement learning", "natural language processing", "nlp",
    "computer vision", "generative ai", "large language model",
    "retrieval augmented generation", "fine-tuning", "inference",
]

DATE_PATTERN = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    r'|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    r'|\b(?:Q[1-4]\s+)?\d{4}\b',
    re.IGNORECASE
)

ORG_SUFFIXES = r'\b\w+(?:\s+\w+)?\s+(?:Inc|Corp|Ltd|LLC|Co|Company|University|Institute|Foundation|Group|Agency|Department|Ministry|Bureau)\b'


class EntityExtractor:
    """Extracts named entities from text chunks."""

    def __init__(self):
        self._nlp = None
        self._use_spacy = self._try_load_spacy()

    def _try_load_spacy(self) -> bool:
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
                return True
            except OSError:
                return False
        except ImportError:
            return False

    def extract_from_chunks(self, chunks: List[str]) -> Dict[str, List[str]]:
        """Extract entities from a list of text chunks."""
        persons = set()
        orgs = set()
        concepts = set()
        dates = set()

        # Sample a subset for speed (max 20 chunks)
        sample = chunks[:20]
        full_text = " ".join(sample)

        if self._use_spacy and self._nlp:
            entities = self._spacy_extract(full_text)
            persons.update(entities.get("persons", []))
            orgs.update(entities.get("orgs", []))
            dates.update(entities.get("dates", []))
        else:
            # Regex-based fallback
            persons.update(self._extract_persons_regex(full_text))
            orgs.update(self._extract_orgs_regex(full_text))

        concepts.update(self._extract_concepts(full_text))
        dates.update(self._extract_dates(full_text))

        return {
            "persons": sorted(list(persons))[:15],
            "orgs": sorted(list(orgs))[:15],
            "concepts": sorted(list(concepts))[:15],
            "dates": sorted(list(dates))[:10],
        }

    def _spacy_extract(self, text: str) -> Dict[str, List[str]]:
        """Use spaCy NER."""
        # Truncate for speed
        doc = self._nlp(text[:5000])
        persons, orgs = [], []
        for ent in doc.ents:
            if ent.label_ in ("PERSON",):
                if len(ent.text.split()) >= 2:
                    persons.append(ent.text.strip())
            elif ent.label_ in ("ORG", "GPE", "FAC"):
                orgs.append(ent.text.strip())
        return {"persons": persons, "orgs": orgs, "dates": []}

    def _extract_persons_regex(self, text: str) -> List[str]:
        """Simple person extraction: 2-3 capitalized words in sequence."""
        pattern = re.compile(
            r'\b([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,15})(?:\s+([A-Z][a-z]{1,15}))?\b'
        )
        # Filter common false positives
        stopwords = {"The", "In", "On", "At", "By", "For", "And", "Or", "But",
                     "With", "From", "This", "That", "These", "Those", "Is", "Are",
                     "Was", "Were", "Has", "Have", "Had", "Can", "Will", "Would"}
        persons = []
        for m in pattern.finditer(text):
            first, last = m.group(1), m.group(2)
            if first not in stopwords and last not in stopwords:
                name = f"{first} {last}"
                if m.group(3) and m.group(3) not in stopwords:
                    name = f"{name} {m.group(3)}"
                persons.append(name)
        return list(set(persons))[:15]

    def _extract_orgs_regex(self, text: str) -> List[str]:
        """Extract organization-like strings."""
        pattern = re.compile(ORG_SUFFIXES, re.IGNORECASE)
        orgs = [m.group().strip() for m in pattern.finditer(text)]
        return list(set(orgs))[:10]

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract domain concepts and tech keywords."""
        text_lower = text.lower()
        found = []
        for kw in TECH_KEYWORDS:
            if kw in text_lower:
                found.append(kw.title())
        # Also find recurring noun phrases (2-3 words, non-stopword)
        pattern = re.compile(r'\b([a-z]+(?:\s+[a-z]+){1,2})\b')
        word_freq: Dict[str, int] = {}
        for m in pattern.finditer(text.lower()):
            phrase = m.group().strip()
            if len(phrase) > 8 and not any(sw in phrase.split()[0] for sw in
                                             {"the", "and", "for", "with", "this", "that", "from", "into"}):
                word_freq[phrase] = word_freq.get(phrase, 0) + 1
        # Take phrases that appear 3+ times
        frequent = [p for p, c in sorted(word_freq.items(), key=lambda x: -x[1]) if c >= 3][:10]
        found.extend([p.title() for p in frequent])
        return list(set(found))[:15]

    def _extract_dates(self, text: str) -> List[str]:
        """Extract date patterns."""
        dates = [m.group().strip() for m in DATE_PATTERN.finditer(text)]
        return list(set(dates))[:10]
