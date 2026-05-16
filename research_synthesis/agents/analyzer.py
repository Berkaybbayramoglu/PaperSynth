"""Analyzer agent: turns each paper's extraction into a detailed prose analysis."""
import asyncio
from typing import Dict, List, Optional

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory

from research_synthesis.agents.memory import MemoryAgent


class AnalyzerAgent:
    """Produces a paragraph-style analysis for each paper, optionally caching results."""

    def __init__(self, kernel: Kernel, service_id: str, memory_agent: Optional[MemoryAgent] = None):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
        self.memory_agent = memory_agent

    async def _analyze_single_paper(
        self, paper: Dict[str, str], paper_num: int, total_papers: int,
    ) -> str:
        if not paper or not isinstance(paper, dict) or not paper.get("extracted") or not paper.get("title"):
            return f"PAPER {paper_num}: ERROR - Invalid paper structure\n"

        if self.memory_agent:
            paper_hash = MemoryAgent.compute_hash(paper["extracted"])
            cached = await self.memory_agent.get_analysis(paper["title"], paper_hash)
            if cached and isinstance(cached, dict) and cached.get("analysis"):
                print(f"   Using cached analysis for paper {paper_num}/{total_papers}: {paper['title'][:50]}")
                return f"PAPER {paper_num}: {paper['title']}\n{'='*80}\n{cached['analysis']}\n"

        print(f"   Analyzing paper {paper_num}/{total_papers}: {paper['title'][:50]}...")

        history = ChatHistory()
        history.add_system_message(
            "You are a research analyst. Write an EXHAUSTIVE, DETAILED analysis in clear paragraphs.\n\n"
            "FORMAT RULES:\n"
            "- Write in plain, flowing paragraphs\n"
            "- NO numbered lists, NO bullets, NO symbols (-, *, >)\n"
            "- Section headers in ALL CAPS (PAPER DETAILS, APPROACH, FINDINGS, etc.)\n"
            "- Natural, readable text flow\n\n"
            "REQUIRED COMPREHENSIVE CONTENT (use EXACT terms from paper):\n\n"
            "PAPER DETAILS\n"
            "Full title, all authors, year, publication venue, DOI if available, paper type (empirical/theoretical/review/commentary/position)\n\n"
            "WHAT PAPER DOES\n"
            "Write 2-3 detailed paragraphs describing the paper's main focus, objectives, and contributions. Use EXACT terminology. Explain the research question or problem addressed.\n\n"
            "APPROACH\n"
            "Write 2-3 paragraphs describing WHAT THE PAPER ACTUALLY DID (not what's missing):\n"
            "IF empirical: Describe the research design used, the sample characteristics (who participated, how many, how recruited), the exact measures and instruments employed (with names/versions), the procedures followed step-by-step, and the statistical analyses performed (with exact test names). Focus on WHAT WAS DONE.\n"
            "IF conceptual: Describe the theoretical framework proposed, the components of the conceptual model, how the argument is structured, and what sources ground the theory. Focus on WHAT WAS PROPOSED.\n"
            "IF review: Describe the scope covered (databases searched, years included, inclusion/exclusion criteria), how many sources were reviewed, and the synthesis methodology used. Focus on WHAT WAS REVIEWED.\n"
            "IF other: Describe what the paper actually does - what position it argues, what it critiques, what ideas it proposes, what evidence it uses. Focus on WHAT THE PAPER ACCOMPLISHES.\n\n"
            "FINDINGS AND RESULTS\n"
            "Write 2-3 paragraphs describing WHAT THE PAPER FOUND OR CONCLUDED (not what's missing):\n"
            "IF empirical: Describe ALL key findings with EXACT numbers (means, standard deviations, percentages, p-values, effect sizes, confidence intervals), report patterns across analyses, subgroup differences, and main outcomes. Focus on WHAT WAS DISCOVERED.\n"
            "IF conceptual: Describe the frameworks or models proposed in detail, explain the key insights with full depth, present the specific recommendations made. Focus on WHAT WAS PROPOSED.\n"
            "IF review: Describe the synthesized conclusions, identify patterns across studies, present research gaps identified. Focus on WHAT WAS SYNTHESIZED.\n"
            "IF other: Describe the arguments made, positions taken, proposals presented in full detail. Focus on WHAT WAS ARGUED OR PROPOSED.\n"
            "NEVER start with 'no empirical data' or 'no results' - ALWAYS describe what the paper DOES contain.\n\n"
            "LIMITATIONS AND SCOPE\n"
            "Write 1-2 paragraphs describing the paper's scope and boundaries. Focus PRIMARILY on what the paper covers and accomplishes (70%), then briefly note areas beyond its scope (30%). Use constructive language: 'focuses on X', 'addresses Y', 'covers Z context', 'scope limited to A'.\n\n"
            "CONTRIBUTION AND VALUE\n"
            "Write 1 paragraph explaining why this paper matters: what gap it fills, practical applications, who benefits, theoretical/methodological advances\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Extract EVERY detail from the paper - be thorough\n"
            "- Use EXACT wording from paper for methods, measures, findings\n"
            "- Write in complete, detailed paragraphs\n"
            "- Be comprehensive but organized - cover everything systematically\n"
            "- NO repetition, but include ALL important details\n"
            "- Balanced tone: strengths AND limitations"
        )
        history.add_user_message(
            f"Analyze this paper in complete detail:\n\n"
            f"Paper Title: {paper['title']}\n\n"
            f"Content:\n{paper['extracted'][:6000]}"
        )

        settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id, max_tokens=2500, temperature=0.2,
        )

        try:
            result = await self.chat_service.get_chat_message_contents(
                chat_history=history, settings=settings, kernel=self.kernel,
            )
            analysis = result[0].content if result else f"Analysis failed for {paper['title']}"

            if self.memory_agent and analysis and isinstance(analysis, str) and len(analysis) > 50:
                try:
                    paper_hash = MemoryAgent.compute_hash(paper["extracted"])
                    await self.memory_agent.store_analysis(paper["title"], paper_hash, analysis)
                except Exception as cache_error:
                    print(f"   WARNING: Cache storage failed: {str(cache_error)[:100]}")

            return f"PAPER {paper_num}: {paper['title']}\n{'='*80}\n{analysis}\n"

        except Exception as e:
            msg = str(e)[:200]
            print(f"   WARNING: Analysis failed for paper {paper_num}: {msg}")
            return f"PAPER {paper_num}: {paper['title']}\n{'='*80}\nERROR: Analysis failed - {msg}\n"

    async def analyze_papers(
        self, papers: List[Dict[str, str]], parallel: bool = True, batch_size: int = 4,
    ) -> Optional[str]:
        if not parallel or len(papers) <= 1:
            print(f"\nAnalyzer: Analyzing {len(papers)} papers sequentially...")
            paper_analyses = []
            for i, paper in enumerate(papers, 1):
                paper_analyses.append(await self._analyze_single_paper(paper, i, len(papers)))
        else:
            print(f"\nAnalyzer: Analyzing {len(papers)} papers in parallel (batch_size={batch_size})...")
            paper_analyses: List[Optional[str]] = [None] * len(papers)
            for batch_start in range(0, len(papers), batch_size):
                batch_papers = papers[batch_start : batch_start + batch_size]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (len(papers) + batch_size - 1) // batch_size
                if total_batches > 1:
                    print(f"   Analyzing batch {batch_num}/{total_batches} ({len(batch_papers)} papers)...")
                tasks = [
                    self._analyze_single_paper(p, batch_start + i + 1, len(papers))
                    for i, p in enumerate(batch_papers)
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                print(f"      Batch {batch_num} completed: {len(batch_results)} papers processed")
                for i, r in enumerate(batch_results):
                    paper_analyses[batch_start + i] = r

        valid: List[str] = []
        for i, r in enumerate(paper_analyses, 1):
            if isinstance(r, Exception):
                print(f"   ERROR: Paper {i} analysis failed: {str(r)[:100]}")
                valid.append(f"PAPER {i}: Analysis failed - {str(r)}\n")
            elif r and isinstance(r, str) and len(r.strip()) > 0:
                valid.append(r)
            else:
                print(f"   WARNING: Paper {i} returned empty analysis")
                valid.append(f"PAPER {i}: Empty analysis result\n")

        if not valid:
            return None

        full = "\n\n".join(valid)
        valid.clear()
        mode = "parallel" if parallel else "sequential"
        print(f"Analyzer: Completed {mode} analysis of {len(papers)} papers")
        return full
