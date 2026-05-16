"""Renderers for the final report (currently: Word)."""
from research_synthesis.output.word_export import DOCX_AVAILABLE, save_report_as_word

__all__ = ["save_report_as_word", "DOCX_AVAILABLE"]
