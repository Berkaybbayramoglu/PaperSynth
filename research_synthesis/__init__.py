"""Multi-agent research paper synthesis system.

Pipeline:
    Reader -> Analyzer -> Synthesizer -> Writer
                                          ^
                                          |
              Reviewer / Chief Reviewer / Validator
"""
from research_synthesis.agents import (
    AnalyzerAgent,
    ChiefReviewerAgent,
    MemoryAgent,
    MetricsExtractorAgent,
    OrderVerificationAgent,
    ReaderAgent,
    ReviewerAgent,
    SynthesizerAgent,
    ValidationAgent,
    WriterAgent,
)
from research_synthesis.core import ProgressTracker, ResearchSynthesisBehaviorTree
from research_synthesis.output import DOCX_AVAILABLE, save_report_as_word

__all__ = [
    "AnalyzerAgent",
    "ChiefReviewerAgent",
    "MemoryAgent",
    "MetricsExtractorAgent",
    "OrderVerificationAgent",
    "ReaderAgent",
    "ReviewerAgent",
    "SynthesizerAgent",
    "ValidationAgent",
    "WriterAgent",
    "ProgressTracker",
    "ResearchSynthesisBehaviorTree",
    "save_report_as_word",
    "DOCX_AVAILABLE",
]
