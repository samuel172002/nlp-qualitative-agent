from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CodeType(str, Enum):
    DESCRIPTIVE = "descriptive"
    IN_VIVO = "in_vivo"
    PROCESS = "process"
    INITIAL = "initial"
    STRUCTURAL = "structural"
    EMOTION = "emotion"
    VALUES = "values"
    VERSUS = "versus"
    EVALUATION = "evaluation"
    PATTERN = "pattern"


@dataclass
class TextSegment:
    text: str
    segment_id: str
    source_file: str


@dataclass
class Code:
    label: str           # UPPER CASE
    code_type: CodeType
    description: str
    excerpt: str
    confidence: float    # 0.0 – 1.0


@dataclass
class CodedSegment:
    segment: TextSegment
    codes: list[Code] = field(default_factory=list)


@dataclass
class FirstCycleResult:
    coded_segments: list[CodedSegment] = field(default_factory=list)
    code_frequencies: dict[str, int] = field(default_factory=dict)   # label → count
    all_codes: dict[str, list[Code]] = field(default_factory=dict)   # label → all Code instances


@dataclass
class Category:
    name: str
    description: str
    codes: list[str] = field(default_factory=list)       # member code labels
    frequency: int = 0
    properties: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)


@dataclass
class AxialRelationship:
    source_category: str
    target_category: str
    relationship_type: str    # leads_to | causes | constrains | enables
    description: str
    conditions: list[str] = field(default_factory=list)
    consequences: list[str] = field(default_factory=list)


@dataclass
class Theme:
    statement: str            # full sentence making a theoretical claim
    categories: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    level: str = "manifest"   # "manifest" | "latent"


@dataclass
class CoreCategory:
    name: str
    description: str
    related_categories: list[str] = field(default_factory=list)
    theoretical_statement: str = ""


@dataclass
class SecondCycleResult:
    categories: list[Category] = field(default_factory=list)
    axial_relationships: list[AxialRelationship] = field(default_factory=list)
    themes: list[Theme] = field(default_factory=list)
    core_category: Optional[CoreCategory] = None
    pattern_codes: dict[str, list[str]] = field(default_factory=dict)  # pattern → [code labels]
