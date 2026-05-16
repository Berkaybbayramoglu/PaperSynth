"""Entry point: wire up agents, run the pipeline against a directory of papers."""
import asyncio
import sys
from datetime import datetime
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from research_synthesis import (
    AnalyzerAgent,
    ChiefReviewerAgent,
    DOCX_AVAILABLE,
    MemoryAgent,
    MetricsExtractorAgent,
    OrderVerificationAgent,
    ProgressTracker,
    ReaderAgent,
    ResearchSynthesisBehaviorTree,
    ReviewerAgent,
    SynthesizerAgent,
    ValidationAgent,
    WriterAgent,
    save_report_as_word,
)


LLM_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL_ID = "qwen3-vl-8b"
SUPPORTED_EXTENSIONS = ("*.pdf", "*.txt", "*.md")


def build_chat_service(service_id: str) -> OpenAIChatCompletion:
    """Connect to the local LLM (LM Studio by default) with no overall timeout."""
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0))
    openai_client = AsyncOpenAI(
        base_url=LLM_BASE_URL, api_key="NA", http_client=http_client,
    )
    return OpenAIChatCompletion(
        service_id=service_id, ai_model_id=LLM_MODEL_ID, async_client=openai_client,
    )


def build_pipeline(kernel: Kernel, service_id: str) -> ResearchSynthesisBehaviorTree:
    """Instantiate every agent and return a fully wired-up behavior tree."""
    print("  - Memory subsystem...")
    memory_agent = MemoryAgent("analysis_cache.json")

    print("  - Core pipeline agents (6)...")
    reader = ReaderAgent(kernel, service_id)
    analyzer = AnalyzerAgent(kernel, service_id, memory_agent)
    synthesizer = SynthesizerAgent(kernel, service_id)
    writer = WriterAgent(kernel, service_id)
    reviewer = ReviewerAgent(kernel, service_id)
    chief_reviewer = ChiefReviewerAgent(kernel, service_id)

    print("  - Enhancement agents (4)...")
    validator = ValidationAgent(kernel, service_id)
    metrics_extractor = MetricsExtractorAgent(kernel, service_id)
    progress_tracker = ProgressTracker("synthesis_progress.json")
    order_verifier = OrderVerificationAgent(kernel, service_id)

    print("All 11 agents created")
    return ResearchSynthesisBehaviorTree(
        reader, analyzer, synthesizer, writer, reviewer, chief_reviewer,
        validator, metrics_extractor, memory_agent, progress_tracker, order_verifier,
    )


def discover_papers(papers_dir: Path) -> list:
    """Return absolute paths for every supported file in ``papers_dir``."""
    if not papers_dir.exists():
        return []
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(str(p) for p in papers_dir.glob(ext))
    return paths


def save_outputs(report: str, timestamp: str) -> None:
    """Write the report to ``research_report_<timestamp>.txt`` (and ``.docx`` if available)."""
    txt_path = Path(f"research_report_{timestamp}.txt")
    txt_path.write_text(report)
    print(f"\nText report saved to: {txt_path}")

    if DOCX_AVAILABLE:
        docx_path = Path(f"research_report_{timestamp}.docx")
        save_report_as_word(report, docx_path)
        print(f"Word report saved to: {docx_path}")
    else:
        print("WARNING: Word export skipped (install python-docx to enable)")


async def main(papers_directory: str = None) -> None:
    print(
        "\nMulti-Agent Research Paper Synthesis System\n"
        "Pipeline: Reader -> Analyzer -> Synthesizer -> Writer -> Reviewer -> Chief Reviewer\n"
        "Enhancements: Validator + MetricsExtractor + Memory + ProgressTracker + OrderVerifier\n"
    )

    print("Initializing Semantic Kernel...")
    kernel = Kernel()
    service_id = "local-gpt"
    kernel.add_service(build_chat_service(service_id))
    print("Connected to Local LLM (no timeout - unlimited generation time)")

    print("Creating agents...")
    tree = build_pipeline(kernel, service_id)

    papers_dir = Path(papers_directory) if papers_directory else Path("sample_papers")
    print(f"\nLooking for papers in: {papers_dir.absolute()}")

    paper_paths = discover_papers(papers_dir)
    if not paper_paths:
        print(f"\n{'=' * 70}")
        print(f"ERROR: No papers found in {papers_dir.absolute()}")
        print(f"{'=' * 70}")
        print(
            "\nPlease ensure your papers directory contains PDF, TXT, or MD files.\n\n"
            "Usage:\n"
            "  1. Create a directory with your papers\n"
            "  2. Add PDF, TXT, or MD research papers\n"
            "  3. Run: python main.py <directory_name>\n\n"
            "Example:\n"
            "  python main.py sample_papers\n\n"
            "Default: Looks for 'sample_papers' directory if no argument given"
        )
        return

    print(f"\nFound {len(paper_paths)} papers:")
    for path in paper_paths:
        print(f"   - {Path(path).name}")

    result = await tree.run(paper_paths, max_iterations=3, parallel_processing=False)

    if not result or not isinstance(result, dict):
        print("\nERROR: Pipeline returned invalid result")
        return

    if not result.get("success"):
        print(f"\nPipeline completed with issues: {result.get('error', 'Unknown error')}")
        if result.get("details"):
            print(f"   Details: {result.get('details')}")
        print("\nProgress summary:")
        print(result.get("progress_summary", "No progress data"))
        print("\nReport will still be saved (last version available)...")

    if result.get("report") and isinstance(result["report"], str) and len(result["report"]) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_outputs(result["report"], timestamp)

    print(f"\n{'=' * 70}")
    print("SYNTHESIS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Success: {result.get('success', False)}")
    print(f"Iterations: {result.get('iterations', 0)}")

    if result.get("progress_summary"):
        print(f"\n{result['progress_summary']}")

    print("\nFinal Report Preview:")
    print("-" * 70)
    report_text = result.get("report", "")
    if report_text:
        print(report_text[:500] + "..." if len(report_text) > 500 else report_text)
    else:
        print("(No report generated)")
    print("-" * 70)


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(directory))
