"""Disease knowledge base with structured treatment protocols.

Provides lookup by exact name, fuzzy matching, and CLIP-based semantic
retrieval for mapping classifier output to treatment recommendations.
"""

import json
import os
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Optional


_KB_PATH = os.path.join(
    os.path.dirname(__file__), "data", "disease_knowledge.json"
)


@dataclass
class DiseaseInfo:
    key: str
    common_name: str
    scientific_name: str
    pathogen_type: str
    affected_crops: list
    symptoms: dict
    treatments: dict
    prevention: list
    environmental_risk: dict


@dataclass
class TreatmentPlan:
    disease: DiseaseInfo
    severity_level: str
    symptoms_at_level: str
    chemical_treatments: list
    biological_treatments: list
    cultural_treatments: list
    prevention: list
    urgency: str


class DiseaseKnowledgeBase:
    """Lookup and retrieval over the disease knowledge base."""

    def __init__(self, kb_path: Optional[str] = None):
        path = kb_path or _KB_PATH
        with open(path) as f:
            raw = json.load(f)

        self._diseases: dict[str, DiseaseInfo] = {}
        self._alias_map: dict[str, str] = {}

        for key, entry in raw["diseases"].items():
            info = DiseaseInfo(
                key=key,
                common_name=entry["common_name"],
                scientific_name=entry["scientific_name"],
                pathogen_type=entry["pathogen_type"],
                affected_crops=entry["affected_crops"],
                symptoms=entry["symptoms"],
                treatments=entry["treatments"],
                prevention=entry["prevention"],
                environmental_risk=entry["environmental_risk"],
            )
            self._diseases[key] = info
            self._alias_map[key.lower()] = key
            self._alias_map[info.common_name.lower()] = key
            for alias in entry.get("aliases", []):
                self._alias_map[alias.lower()] = key

    @property
    def disease_names(self) -> list[str]:
        return [d.common_name for d in self._diseases.values()]

    def lookup(self, query: str) -> Optional[DiseaseInfo]:
        """Exact or alias-based lookup.

        Args:
            query: disease name or alias (case-insensitive).

        Returns:
            DiseaseInfo if found, else None.
        """
        key = self._alias_map.get(query.lower())
        if key:
            return self._diseases[key]
        return None

    def fuzzy_lookup(self, query: str, cutoff: float = 0.4) -> Optional[DiseaseInfo]:
        """Fuzzy match against all known aliases.

        Args:
            query:  approximate disease name.
            cutoff: minimum similarity ratio (0-1).

        Returns:
            Best-matching DiseaseInfo, or None if no match above cutoff.
        """
        exact = self.lookup(query)
        if exact:
            return exact

        candidates = list(self._alias_map.keys())
        matches = get_close_matches(query.lower(), candidates, n=1, cutoff=cutoff)
        if matches:
            key = self._alias_map[matches[0]]
            return self._diseases[key]
        return None

    def get_treatment_plan(
        self, disease_name: str, severity: str = "moderate"
    ) -> Optional[TreatmentPlan]:
        """Build a structured treatment plan for a disease at a severity level.

        Args:
            disease_name: name or alias of the disease.
            severity:     one of 'mild', 'moderate', 'severe'.

        Returns:
            TreatmentPlan, or None if disease not found.
        """
        info = self.fuzzy_lookup(disease_name)
        if info is None:
            return None

        severity = severity.lower()
        if severity not in ("mild", "moderate", "severe"):
            severity = "moderate"

        urgency_map = {"mild": "low", "moderate": "medium", "severe": "high"}

        return TreatmentPlan(
            disease=info,
            severity_level=severity,
            symptoms_at_level=info.symptoms.get(severity, "Unknown"),
            chemical_treatments=info.treatments.get("chemical", []),
            biological_treatments=info.treatments.get("biological", []),
            cultural_treatments=info.treatments.get("cultural", []),
            prevention=info.prevention,
            urgency=urgency_map.get(severity, "medium"),
        )

    def get_all_diseases(self) -> list[DiseaseInfo]:
        return list(self._diseases.values())

    def format_report_section(self, plan: TreatmentPlan) -> str:
        """Format a treatment plan as a readable text block."""
        lines = [
            f"Disease: {plan.disease.common_name}",
            f"Pathogen: {plan.disease.scientific_name} ({plan.disease.pathogen_type})",
            f"Severity: {plan.severity_level.upper()}",
            f"Symptoms: {plan.symptoms_at_level}",
            f"Urgency: {plan.urgency.upper()}",
            "",
            "Treatment Options:",
        ]
        if plan.chemical_treatments:
            lines.append("  Chemical:")
            for t in plan.chemical_treatments:
                lines.append(f"    - {t}")
        if plan.biological_treatments:
            lines.append("  Biological:")
            for t in plan.biological_treatments:
                lines.append(f"    - {t}")
        if plan.cultural_treatments:
            lines.append("  Cultural:")
            for t in plan.cultural_treatments:
                lines.append(f"    - {t}")
        if plan.prevention:
            lines.append("")
            lines.append("Prevention:")
            for p in plan.prevention:
                lines.append(f"  - {p}")

        return "\n".join(lines)
