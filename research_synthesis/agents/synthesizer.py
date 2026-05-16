"""Synthesizer agent: merges per-paper analyses into a single thematic summary."""
from typing import List, Optional

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory


class SynthesizerAgent:
    """Combines analyses into a synthesis grouped by paper and by theme.

    For inputs over ~8k chars, splits along ``PAPER N:`` boundaries and
    synthesizes each chunk separately, then concatenates them in order.
    """

    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)

    def _chunk_analysis(self, analysis: str, max_chars: int = 8000) -> List[str]:
        papers = analysis.split("PAPER ")
        chunks: List[str] = []
        current = ""
        paper_count = 0

        for i, p in enumerate(papers):
            if not p.strip():
                continue
            paper_text = f"PAPER {p}" if i > 0 else p
            if i > 0:
                paper_count += 1
            if len(current) + len(paper_text) > max_chars and current:
                chunks.append(current + "\n[END OF CHUNK - Continued in next chunk]\n")
                current = f"[CONTINUED FROM PREVIOUS CHUNK - Paper {paper_count}]\n\n" + paper_text
            else:
                current += ("\n\n" + paper_text) if current else paper_text

        if current:
            chunks.append(current)
        return chunks if chunks else [analysis[:max_chars]]

    async def synthesize(self, analysis: str) -> Optional[str]:
        if not analysis or len(analysis.strip()) == 0:
            print("\nSynthesizer: ERROR - Empty analysis provided")
            return None

        print("\nSynthesizer: Creating synthesis...")
        size = len(analysis)
        print(f"   Analysis size: {size} characters")

        if size < 8000:
            history = ChatHistory()
            history.add_system_message(
                "Organize analyses into clear, readable sections. Write in simple paragraphs.\n\n"
                "FORMAT RULES:\n"
                "- Write in plain paragraphs, NO numbered lists or bullets\n"
                "- NO symbols (no >, -, *, bullets, special characters)\n"
                "- Simple section headers in ALL CAPS\n"
                "- Natural, flowing text that is easy to read\n\n"
                "ORGANIZE INTO TWO PARTS:\n\n"
                "PART 1 - PAPER BY PAPER\n"
                "For each paper write a clear paragraph covering: what the paper ACTUALLY DOES (its approach, methods, focus), what it FOUND or PROPOSED (results, frameworks, insights), and its contributions. Use EXACT terms from analysis. Focus on POSITIVE descriptions of what exists, not what's missing.\n\n"
                "PART 2 - THEMATIC OVERVIEW\n"
                "Group papers by common themes. Write paragraphs describing:\n"
                "What approaches papers USE and HOW (empirical methods with details, conceptual frameworks with components, review methodologies with scope)\n"
                "What contexts papers ADDRESS and WHY (populations studied, settings examined, applications developed)\n"
                "What tools or instruments papers EMPLOY and HOW (exact names, versions, purposes)\n"
                "What techniques papers APPLY and RESULTS (statistical tests with outcomes, analytical methods with findings)\n"
                "What papers FOUND, CONCLUDED, or PROPOSED (specific results, frameworks, insights, recommendations)\n\n"
                "QUALITY GUIDELINES:\n"
                "- POSITIVE FOCUS: Describe what papers DO and ACCOMPLISH - their scope, coverage, and contributions\n"
                "- DETAILED: Include specific methods, results, and contributions\n"
                "- CONSTRUCTIVE SCOPE: Present research boundaries positively (focuses on X, addresses Y context, pilot/theoretical stage)\n"
                "- SYNTHESIS: Identify connections, patterns, and relationships between papers\n\n"
                "CRITICAL: Use EXACT wording from analyses. Focus on WHAT EXISTS and HOW IT WAS USED. Write clearly and simply. NO complex formatting."
            )
            history.add_user_message(f"Synthesize these research findings:\n\n{analysis}")

            settings = self.chat_service.get_prompt_execution_settings_class()(
                service_id=self.service_id, max_tokens=2000, temperature=0.3,
            )
            result = await self.chat_service.get_chat_message_contents(
                chat_history=history, settings=settings, kernel=self.kernel,
            )
            synthesis = result[0].content if result else "Synthesis failed"
            print("Synthesizer: Synthesis complete")
            return synthesis

        print("   Analysis is large, processing in chunks...")
        chunks = self._chunk_analysis(analysis, max_chars=8000)
        print(f"   Split into {len(chunks)} chunks")

        async def synthesize_chunk(chunk: str, n: int, total: int) -> str:
            print(f"   Synthesizing chunk {n}/{total}...")
            history = ChatHistory()
            history.add_system_message(
                "You are a research synthesizer. Organize paper analyses preserving ALL details.\n"
                "Keep specific methods, results, metrics, technologies for each paper.\n"
                "This is part of a larger analysis - focus on the papers in this section.\n\n"
                "CRITICAL: Write ONLY the synthesis content - NO metadata, chunk markers, or organizational notes.\n"
                "DO NOT include phrases like 'Chunk X/Y', 'END OF CHUNK', emoji, or special symbols."
            )
            history.add_user_message(f"Organize these paper analyses:\n\n{chunk[:8000]}")

            settings = self.chat_service.get_prompt_execution_settings_class()(
                service_id=self.service_id, max_tokens=1500, temperature=0.3,
            )
            try:
                result = await self.chat_service.get_chat_message_contents(
                    chat_history=history, settings=settings, kernel=self.kernel,
                )
                return result[0].content if result else f"Chunk {n} failed"
            except Exception as e:
                print(f"   WARNING: Chunk {n} failed: {str(e)[:100]}")
                return f"Chunk {n}: Synthesis failed"

        results: List[str] = []
        for i, c in enumerate(chunks, 1):
            try:
                results.append(await synthesize_chunk(c, i, len(chunks)))
            except Exception as e:
                print(f"   ERROR: Chunk {i} failed: {str(e)[:100]}")
                results.append(f"Chunk {i}: Failed")

        if not results:
            print("Synthesizer: ERROR - No valid synthesis results")
            return None

        synthesis = "\n\n".join(results)
        results.clear()
        print(f"Synthesizer: Sequential synthesis complete ({len(chunks)} chunks combined in order)")
        return synthesis
