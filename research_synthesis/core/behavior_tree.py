"""Top-level orchestrator that runs the full Read → Analyze → Synthesize → Write → Review loop."""
from pathlib import Path
from typing import Dict, List

from research_synthesis.agents.analyzer import AnalyzerAgent
from research_synthesis.agents.chief_reviewer import ChiefReviewerAgent
from research_synthesis.agents.memory import MemoryAgent
from research_synthesis.agents.metrics_extractor import MetricsExtractorAgent
from research_synthesis.agents.order_verifier import OrderVerificationAgent
from research_synthesis.agents.reader import ReaderAgent
from research_synthesis.agents.reviewer import ReviewerAgent
from research_synthesis.agents.synthesizer import SynthesizerAgent
from research_synthesis.agents.validator import ValidationAgent
from research_synthesis.agents.writer import WriterAgent
from research_synthesis.core.progress_tracker import ProgressTracker


class ResearchSynthesisBehaviorTree:
    """Glues every agent together; ``run()`` is the only public method.

    The pipeline rewrites the report up to ``max_iterations`` times until
    both the Reviewer and Chief Reviewer approve and the Validator finds no
    hallucinations.
    """

    def __init__(
        self,
        reader: ReaderAgent,
        analyzer: AnalyzerAgent,
        synthesizer: SynthesizerAgent,
        writer: WriterAgent,
        reviewer: ReviewerAgent,
        chief_reviewer: ChiefReviewerAgent,
        validator: ValidationAgent,
        metrics_extractor: MetricsExtractorAgent,
        memory_agent: MemoryAgent,
        progress_tracker: ProgressTracker,
        order_verifier: OrderVerificationAgent,
    ):
        self.reader = reader
        self.analyzer = analyzer
        self.synthesizer = synthesizer
        self.writer = writer
        self.reviewer = reviewer
        self.chief_reviewer = chief_reviewer
        self.validator = validator
        self.metrics_extractor = metrics_extractor
        self.memory_agent = memory_agent
        self.progress_tracker = progress_tracker
        self.order_verifier = order_verifier

    async def run(
        self,
        paper_paths: List[str],
        max_iterations: int = 3,
        parallel_processing: bool = False,
        read_batch_size: int = 6,
        analyze_batch_size: int = 4,
    ) -> Dict:
        """Run the pipeline end-to-end and return ``{success, report, ...}``.

        ``parallel_processing=False`` is recommended for local LLMs like
        LM Studio which queue requests anyway.
        """
        print("\n" + "=" * 70)
        mode = "PARALLEL" if parallel_processing else "SEQUENTIAL"
        print(f"RESEARCH SYNTHESIS BEHAVIOR TREE ({mode} MODE)")
        print("=" * 70)
        if parallel_processing:
            print(f"Parallelization: Read batch={read_batch_size}, Analyze batch={analyze_batch_size}")

        await self.progress_tracker.set_expected_papers(len(paper_paths))
        await self.progress_tracker.log_stage(
            "pipeline_start", "started",
            {"paper_count": len(paper_paths), "parallel": parallel_processing},
        )

        # Phase 1: Read
        await self.progress_tracker.log_stage("reading", "in_progress")
        papers = await self.reader.read_multiple_papers(
            paper_paths, parallel=parallel_processing, batch_size=read_batch_size,
        )
        if not papers:
            await self.progress_tracker.log_stage("reading", "failed")
            return self._failure("No papers read successfully")

        invalid = [i for i, p in enumerate(papers) if not isinstance(p, dict) or not p.get("extracted")]
        if invalid:
            print(f"WARNING: {len(invalid)} papers have invalid structure")

        await self.progress_tracker.set_received_papers(len(papers))
        await self.progress_tracker.log_stage("reading", "completed", {"papers_read": len(papers)})

        # Phase 1.5: Verify ordering
        read_verification = await self.order_verifier.verify_order(paper_paths, papers, "after_reading")
        await self.progress_tracker.log_order_verification("reading", read_verification)
        if not read_verification["verified"]:
            print("CRITICAL: Paper order corrupted after reading!")
            return self._failure(
                "Order verification failed after reading", details=read_verification["details"],
            )

        # Recover any missing papers (best-effort)
        papers = await self._maybe_recover_missing_papers(paper_paths, papers)

        # Phase 2: Analyze
        await self.progress_tracker.log_stage("analyzing", "in_progress")
        analysis = await self.analyzer.analyze_papers(
            papers, parallel=parallel_processing, batch_size=analyze_batch_size,
        )
        if not analysis:
            await self.progress_tracker.log_stage("analyzing", "failed")
            return self._failure("Analysis failed - no results returned")
        await self.progress_tracker.log_stage("analyzing", "completed")

        # Phase 2.7: Metrics
        await self.progress_tracker.log_stage("metrics_extraction", "in_progress")
        metrics_summary = await self.metrics_extractor.extract_metrics(analysis)
        await self.progress_tracker.log_stage("metrics_extraction", "completed")

        # Phase 3: Synthesize
        await self.progress_tracker.log_stage("synthesizing", "in_progress")
        synthesis = await self.synthesizer.synthesize(analysis)
        if not synthesis:
            await self.progress_tracker.log_stage("synthesizing", "failed")
            return self._failure("Synthesis failed - no output generated")
        synthesis_with_metrics = f"{synthesis}\n\n{metrics_summary}"
        analysis = None
        await self.progress_tracker.log_stage("synthesizing", "completed")

        # Phase 4-7: Write/Validate/Review loop
        return await self._write_and_review_loop(synthesis_with_metrics, metrics_summary, max_iterations)

    async def _maybe_recover_missing_papers(
        self, paper_paths: List[str], papers: List[Dict],
    ) -> List[Dict]:
        expected_count = len(paper_paths)
        actual_count = len(papers)
        if actual_count == expected_count:
            return papers

        print(f"WARNING: Paper count mismatch! Expected {expected_count} papers but got {actual_count}")
        print(f"Expected papers: {[Path(p).name for p in paper_paths]}")
        print(f"Received papers: {[p.get('filename', 'UNKNOWN') for p in papers]}")

        received_filenames = {p.get("filename", "") for p in papers}
        missing_paths = [p for p in paper_paths if Path(p).name not in received_filenames]
        if not missing_paths:
            return papers

        print(f"Attempting to re-read {len(missing_paths)} missing paper(s)...")
        print(f"Missing: {[Path(p).name for p in missing_paths]}")

        retry_papers = await self.reader.read_multiple_papers(missing_paths, parallel=False)
        if retry_papers:
            valid_retry = [p for p in retry_papers if isinstance(p, dict) and p.get("extracted")]
            if valid_retry:
                papers.extend(valid_retry)
                print(f"Successfully recovered {len(valid_retry)} missing paper(s)")
                print(f"Total papers now: {len(papers)}")

        if len(papers) != expected_count:
            print(f"CRITICAL: Still missing {expected_count - len(papers)} paper(s) after retry!")
            print(f"Continuing with {len(papers)} paper(s) - report quality will be affected")
        else:
            print(f"All {expected_count} papers recovered successfully")
        return papers

    async def _write_and_review_loop(
        self, synthesis_with_metrics: str, metrics_summary: str, max_iterations: int,
    ) -> Dict:
        report = None
        chief_feedback = None
        validation_feedback = None
        reviewer_feedback = None

        for iteration in range(1, max_iterations + 1):
            print(f"\n{'-' * 70}")
            print(f"Writing Iteration {iteration}/{max_iterations}")
            print(f"{'-' * 70}")

            await self.progress_tracker.log_stage(f"writing_iteration_{iteration}", "in_progress")

            if iteration == 1:
                report = await self.writer.write_report(synthesis_with_metrics)
            else:
                parts: List[str] = []
                sources: Dict[str, bool] = {}
                if reviewer_feedback:
                    parts.append(
                        f"Reviewer notes: {reviewer_feedback.replace('=== REVIEWER FEEDBACK ===', '').strip()}"
                    )
                    sources["reviewer"] = True
                if chief_feedback:
                    parts.append(
                        f"Chief reviewer notes: {chief_feedback.replace('=== CHIEF REVIEWER FEEDBACK ===', '').strip()}"
                    )
                    sources["chief"] = True
                if validation_feedback:
                    parts.append(
                        f"Validation notes: {validation_feedback.replace('=== VALIDATION FEEDBACK ===', '').strip()}"
                    )
                    sources["validator"] = True
                combined = "\n\n".join(parts) if parts else None
                report = await self.writer.write_report(
                    synthesis_with_metrics, combined, sources, previous_report=report,
                )

            # Phase 4.5: Validate
            await self.progress_tracker.log_stage(f"validation_iteration_{iteration}", "in_progress")
            validation_result = await self.validator.validate(report, synthesis_with_metrics)
            if not validation_result["valid"]:
                print(f"WARNING: Iteration {iteration}: Validation failed - hallucinations detected!")
                validation_feedback = validation_result["feedback"]
                await self.progress_tracker.log_stage(
                    f"validation_iteration_{iteration}", "failed",
                    {"issues": validation_result["issues_found"]},
                )
                continue
            validation_feedback = None
            await self.progress_tracker.log_stage(f"validation_iteration_{iteration}", "completed")

            # Phase 5: First review
            await self.progress_tracker.log_stage(f"review_iteration_{iteration}", "in_progress")
            reviewer_result = await self.reviewer.review(report)
            if not reviewer_result["approved"]:
                print(f"WARNING: Iteration {iteration}: Reviewer rejected, retrying...")
                reviewer_feedback = reviewer_result["feedback"]
                await self.progress_tracker.log_stage(f"review_iteration_{iteration}", "rejected")
                continue
            reviewer_feedback = None
            await self.progress_tracker.log_stage(f"review_iteration_{iteration}", "approved")

            # Phase 6: Chief review
            await self.progress_tracker.log_stage(f"chief_review_iteration_{iteration}", "in_progress")
            chief_result = await self.chief_reviewer.final_review(report)
            if chief_result["approved"]:
                print(f"\n{'=' * 70}")
                print(f"SUCCESS! Report approved on iteration {iteration}")
                print(f"{'=' * 70}")
                await self.progress_tracker.log_stage(f"chief_review_iteration_{iteration}", "approved")
                await self.progress_tracker.log_stage(
                    "pipeline_complete", "success", {"iterations": iteration},
                )
                return {
                    "success": True,
                    "iterations": iteration,
                    "report": report,
                    "metrics_summary": metrics_summary,
                    "final_feedback": chief_result["feedback"],
                    "progress_summary": self.progress_tracker.get_summary(),
                }
            print(f"WARNING: Iteration {iteration}: Chief Reviewer rejected, retrying...")
            chief_feedback = chief_result["feedback"]
            await self.progress_tracker.log_stage(f"chief_review_iteration_{iteration}", "rejected")

        print(f"\n{'=' * 70}")
        print("WARNING: Max iterations reached. Using last version.")
        print(f"{'=' * 70}")
        await self.progress_tracker.log_stage("pipeline_complete", "max_iterations_reached")
        return {
            "success": False,
            "iterations": max_iterations,
            "report": report if report else "Report generation incomplete",
            "metrics_summary": metrics_summary if metrics_summary else "No metrics",
            "final_feedback": "Maximum iterations reached",
            "progress_summary": self.progress_tracker.get_summary(),
        }

    def _failure(self, error: str, details: str = None) -> Dict:
        payload: Dict = {
            "success": False,
            "error": error,
            "report": None,
            "metrics_summary": None,
            "iterations": 0,
            "progress_summary": self.progress_tracker.get_summary(),
        }
        if details:
            payload["details"] = details
        return payload
