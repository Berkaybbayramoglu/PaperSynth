"""Writer agent: drafts and revises the final synthesis report.

For small inputs the report is written in a single LLM call. For larger
synthesis text the agent splits by ``PAPER N:`` boundaries, writes each
chunk with cumulative context, and concatenates the parts. When called
with ``feedback`` it does an incremental fix-pass instead of rewriting from
scratch.
"""
from typing import List, Optional

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory

from research_synthesis.utils.text_cleaning import clean_markdown, quick_clean


CONTEXT_TOKEN_LIMIT = 14200
SAFETY_MARGIN_TOKENS = 500


class WriterAgent:
    """Produces the final research synthesis report."""

    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)

    def _chunk_synthesis(self, synthesis: str, max_chars: int = 10000) -> List[str]:
        import re

        parts = re.split(r"(PAPER \d+:)", synthesis)
        chunks: List[str] = []
        current = ""
        for p in parts:
            if not p.strip():
                continue
            if len(current) + len(p) > max_chars and current:
                chunks.append(current)
                current = p
            else:
                current += p
        if current:
            chunks.append(current)

        if len(chunks) <= 1 and len(synthesis) > max_chars:
            chunks = [synthesis[i : i + max_chars] for i in range(0, len(synthesis), max_chars)]
        return chunks if chunks else [synthesis]

    async def _write_chunk(
        self,
        chunk: str,
        chunk_num: int,
        total_chunks: int,
        is_first: bool,
        is_last: bool,
        previous_content: Optional[str] = None,
        all_chunks_summary: Optional[str] = None,
    ) -> str:
        history = ChatHistory()

        if is_first and is_last:
            section_instruction = (
                "Write a COMPLETE report with ALL sections:\n"
                "1. OVERVIEW\n"
                "2. DETAILED PAPER-BY-PAPER ANALYSIS\n"
                "3. INTEGRATED SYNTHESIS AND COMPARISON\n"
                "4. CONCLUSIONS"
            )
        elif is_first:
            section_instruction = (
                "Write the FIRST PART of the report:\n"
                "1. OVERVIEW - Brief introduction to ALL papers in the study\n"
                "2. DETAILED PAPER-BY-PAPER ANALYSIS - Write detailed analysis for papers in this chunk\n"
                "(More papers will be analyzed in subsequent parts)"
            )
        elif is_last:
            section_instruction = (
                "Write the FINAL PART of the report:\n"
                "1. DETAILED PAPER-BY-PAPER ANALYSIS (CONTINUED) - Complete analysis for remaining papers\n"
                "2. INTEGRATED SYNTHESIS AND COMPARISON - Comprehensive synthesis comparing ALL papers from all chunks\n"
                "3. CONCLUSIONS - Final insights considering ALL papers"
            )
        else:
            section_instruction = (
                "Write the MIDDLE PART of the report:\n"
                "DETAILED PAPER-BY-PAPER ANALYSIS (CONTINUED) - Write detailed analysis for papers in this chunk\n"
                "(Maintain consistency with previous papers)"
            )

        context_info = ""
        truncated_prev = ""
        if previous_content:
            max_prev_chars = 11000
            truncated_prev = (
                previous_content[-max_prev_chars:]
                if len(previous_content) > max_prev_chars
                else previous_content
            )
            context_info = (
                f"\n\nPREVIOUSLY WRITTEN REPORT SECTIONS (maintain consistency and continue):\n"
                f"{'...[earlier content]...' + chr(10) + chr(10) if len(previous_content) > max_prev_chars else ''}"
                f"{truncated_prev}\n"
                f"[Continue from here with same style and detail level]"
            )

        synthesis_context = ""
        if is_last and all_chunks_summary:
            max_synth_chars = 10000
            truncated_synth = (
                all_chunks_summary[:max_synth_chars]
                if len(all_chunks_summary) > max_synth_chars
                else all_chunks_summary
            )
            synthesis_context = (
                f"\n\nFULL SYNTHESIS DATA (for comprehensive integrated synthesis section):\n"
                f"{truncated_synth}\n"
                f"{'...[more papers]...' + chr(10) if len(all_chunks_summary) > max_synth_chars else ''}"
                f"[Use ALL papers from above to create integrated comparison]"
            )

        history.add_system_message(self._chunk_system_prompt(chunk_num, total_chunks, section_instruction))

        conclusions_warning = ""
        if is_last:
            conclusions_warning = (
                "\n\nCONCLUSIONS SECTION MANDATORY RULES:\n"
                "This is the LAST CHUNK. When you write CONCLUSIONS:\n"
                "- Write EXACTLY 2-3 paragraphs ONLY (each 3-5 sentences)\n"
                "- STOP immediately after 2-3 paragraphs\n"
                "- DO NOT repeat sentences or ideas\n"
                "- Keep it brief and impactful\n\n"
            )

        history.add_user_message(
            f"Synthesis data for this section:\n\n{chunk}"
            f"{context_info}"
            f"{synthesis_context}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Extract content from synthesis data and write as YOUR OWN report\n"
            "- DO NOT copy chunk markers, metadata, or organizational notes from synthesis\n"
            "- DO NOT include phrases like 'Chunk 1/4', 'END OF CHUNK', 'Continued in next chunk'\n"
            "- DO NOT include emoji or special symbols\n"
            "- DO NOT REPEAT the same sentence or paragraph multiple times\n"
            "- VARY your language - avoid repeating 'DOES NOT' or 'THE PAPER DOES NOT' excessively\n"
            "- Write ONLY the final report content - professional, clean, complete\n"
            "- Use EXACT terminology from synthesis but present it as integrated analysis\n"
            f"{conclusions_warning}"
            f"Write exhaustively detailed analysis focusing on what papers DID and ACCOMPLISHED. Include EVERY detail from the synthesis."
        )

        total_input_chars = len(chunk) + len(truncated_prev) + len(synthesis_context) + 1000
        estimated_input = total_input_chars // 4
        max_output = min(6500, CONTEXT_TOKEN_LIMIT - estimated_input - SAFETY_MARGIN_TOKENS)
        max_output = max(3000, max_output)

        settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id, max_tokens=max_output, temperature=0.4,
        )
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history, settings=settings, kernel=self.kernel,
        )
        raw = result[0].content if result else f"[Chunk {chunk_num} writing failed]"
        return quick_clean(raw)

    def _chunk_system_prompt(self, chunk_num: int, total_chunks: int, section_instruction: str) -> str:
        return (
            f"You are writing part {chunk_num}/{total_chunks} of a research synthesis report.\n\n"
            f"{section_instruction}\n\n"
            "CRITICAL FORMAT RULES - MUST FOLLOW:\n"
            "- Write ONLY in plain paragraphs\n"
            "- Section headings in ALL CAPS with blank line before and after\n"
            "- NEVER use these symbols: ** __ * | - > ### ` ~ = bullets dashes\n"
            "- NO numbered lists (1. 2. 3.)\n"
            "- NO bullet points or dashes at line start\n"
            "- NO tables or special formatting\n"
            "- Write like a simple text document - clean and readable\n\n"
            "CONTENT FORMATTING REQUIREMENTS:\n\n"
            "FOR PAPER-BY-PAPER ANALYSIS SECTIONS:\n"
            "Number each paper: PAPER 1, PAPER 2, PAPER 3, etc. (use ALL CAPS heading for each)\n"
            "Write 5-7 paragraphs per paper. USE EXACT TERMINOLOGY FROM PAPER TEXT.\n"
            "Cover: paper title, authors, year, publication, what paper ACCOMPLISHES, approach and methods USED (describe actual work), findings MADE (describe actual results/frameworks), scope COVERED.\n"
            "CRITICAL: FOCUS ON WHAT PAPERS DID. Describe actual methods used, actual results found, actual frameworks proposed with full details. For conceptual/review papers, describe frameworks or synthesis components in depth. NEVER start with 'no X' or 'lacks Y' - begin with what EXISTS.\n\n"
            "FOR INTEGRATED SYNTHESIS (if last chunk):\n"
            "Organize THEMATICALLY, NOT paper-by-paper. USE EXACT TERMS FROM PAPERS.\n"
            "IMPORTANT: Each subsection (A, B, C, D, E, F) must start on a NEW LINE with blank line before it.\n\n"
            "A. APPROACHES AND METHODS\n"
            "What papers ACTUALLY DID with details.\n\n"
            "B. CONTEXTS AND SETTINGS\n"
            "EXACT populations/settings/applications.\n\n"
            "C. INSTRUMENTS AND TOOLS\n"
            "EXACT tools/measures with usage.\n\n"
            "D. TECHNIQUES AND ANALYSES\n"
            "What papers APPLIED with outcomes.\n\n"
            "E. FINDINGS AND CONTRIBUTIONS\n"
            "EXACT results/frameworks.\n\n"
            "F. SCOPE AND FOCUS\n"
            "What papers COVERED. Frame positively.\n\n"
            "Use EXACT terms. Write in flowing paragraphs under each theme heading.\n\n"
            "QUALITY GUIDELINES:\n"
            "- POSITIVE FOCUS FIRST: Describe what papers DID, methods USED, results FOUND, frameworks PROPOSED.\n"
            "- Be CONCISE: Don't repeat the same points multiple times. State it once clearly.\n"
            "- BALANCED TONE: Highlight contributions and value prominently.\n"
            "- DEEPER COMPARISON: Identify shared themes, complementary approaches, divergent perspectives.\n"
            "- VALUE PROPOSITION: Explain WHY each paper matters.\n"
            "- STRUCTURED SYNTHESIS: Organize comparison as similarities, differences, then implications.\n\n"
            "CRITICAL: Write detailed analysis describing what papers ACCOMPLISHED. Be thorough but not repetitive."
        )

    async def write_report(
        self,
        synthesis: str,
        feedback: Optional[str] = None,
        feedback_sources: Optional[dict] = None,
        previous_report: Optional[str] = None,
    ) -> str:
        """Write the report from synthesis text.

        If ``feedback`` and ``previous_report`` are given, performs an
        incremental fix-pass instead of generating from scratch.
        """
        synthesis_size = len(synthesis)
        needs_chunking = synthesis_size > 10000

        if feedback:
            sources = []
            if feedback_sources:
                if feedback_sources.get("reviewer"):
                    sources.append("Reviewer")
                if feedback_sources.get("chief"):
                    sources.append("Chief")
                if feedback_sources.get("validator"):
                    sources.append("Validator")
            source_str = f" ({', '.join(sources)})" if sources else ""
            print(f"\nWriter: Fixing report incrementally based on feedback{source_str}...")
        elif needs_chunking:
            print(f"\nWriter: Writing initial report (chunked mode: {synthesis_size} chars)...")
            chunks = self._chunk_synthesis(synthesis, max_chars=10000)
            print(f"Writer: Split into {len(chunks)} chunks for detailed writing...")

            full_synthesis = synthesis
            report_parts: List[str] = []
            accumulated = ""

            for i, chunk in enumerate(chunks, 1):
                is_first = i == 1
                is_last = i == len(chunks)
                print(f"  Writing chunk {i}/{len(chunks)}...")
                part = await self._write_chunk(
                    chunk, i, len(chunks), is_first, is_last,
                    previous_content=accumulated or None,
                    all_chunks_summary=full_synthesis if is_last else None,
                )
                report_parts.append(part)
                accumulated = accumulated + "\n\n" + part if accumulated else part

            full_report = "\n\n".join(report_parts)
            print(f"Writer: Chunked report completed ({len(chunks)} parts combined)")
            return full_report
        else:
            print(f"\nWriter: Writing initial report (single pass: {synthesis_size} chars)...")

        history = ChatHistory()
        history.add_system_message(self._single_pass_system_prompt())

        if feedback and previous_report:
            prompt = self._incremental_fix_prompt(previous_report, synthesis, feedback)
        elif feedback:
            prompt = self._feedback_prompt(synthesis, feedback)
        else:
            prompt = self._fresh_prompt(synthesis)

        history.add_user_message(prompt)

        system_msg = history.messages[0].content if history.messages else ""
        estimated_input_tokens = (len(system_msg) + len(prompt)) // 4
        max_output_tokens = min(8000, CONTEXT_TOKEN_LIMIT - estimated_input_tokens - SAFETY_MARGIN_TOKENS)
        max_output_tokens = max(4000, max_output_tokens)

        print(f"Writer: Estimated input ~{estimated_input_tokens} tokens, max output {max_output_tokens} tokens")

        settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id, max_tokens=max_output_tokens, temperature=0.4,
        )
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history, settings=settings, kernel=self.kernel,
        )
        raw = result[0].content if result else "Report writing failed"
        report = clean_markdown(raw)
        print(f"Writer: Report {'rewritten' if feedback else 'completed'} (markdown cleaned)")
        return report

    def _single_pass_system_prompt(self) -> str:
        return (
            "You are a professional academic report writer creating a detailed research synthesis.\n\n"
            "REQUIRED STRUCTURE (use plain text ALL CAPS headings):\n\n"
            "1. OVERVIEW\n"
            "   Brief introduction to all papers and their domains.\n\n"
            "2. DETAILED PAPER-BY-PAPER ANALYSIS\n"
            "   Write a SEPARATE comprehensive subsection for EACH paper covering title, authors, year, "
            "research focus, what the paper does, approach/methods, findings, and scope.\n"
            "   (Each paper gets 3-5 paragraphs)\n\n"
            "3. INTEGRATED SYNTHESIS AND COMPARISON\n"
            "   Synthesize THEMATICALLY across ALL papers. Each subsection (A, B, C, D, E, F) on a NEW LINE:\n"
            "   A. APPROACHES\n   B. CONTEXTS\n   C. INSTRUMENTS\n"
            "   D. TECHNIQUES\n   E. FINDINGS\n   F. FUTURE DIRECTIONS\n\n"
            "4. CONCLUSIONS\n   Final insights and implications.\n\n"
            "CRITICAL FORMATTING RULES:\n"
            "- Plain text only, no markdown\n"
            "- No ASCII tables, no asterisks, no pipes, no triple-dash separators\n"
            "- Section 2 has individual analysis for each paper\n"
            "- Section 3 integrates and compares all papers together\n"
            "- Be SPECIFIC with methods, results, technologies\n"
        )

    def _incremental_fix_prompt(self, previous_report: str, synthesis: str, feedback: str) -> str:
        return (
            f"TASK: Fix specific issues in this research report based on feedback.\n\n"
            f"CURRENT REPORT (first 8000 chars):\n{previous_report[:8000]}\n\n"
            f"SYNTHESIS (reference for details):\n{synthesis[:6000]}\n\n"
            f"FEEDBACK:\n{feedback}\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"- Extract improvements from feedback and write as YOUR OWN corrections\n"
            f"- DO NOT copy feedback section markers\n"
            f"- DO NOT include chunk markers or emoji\n"
            f"- DO NOT REPEAT the same sentence or paragraph multiple times\n"
            f"- Write ONLY the final corrected report - professional, clean, complete\n\n"
            f"FIX INSTRUCTIONS:\n"
            f"1. Read feedback - it tells you exactly what to add or fix\n"
            f"2. Add missing details where specified\n"
            f"3. Fix vague descriptions with specific details from synthesis\n"
            f"4. Keep all well-written sections unchanged\n"
            f"5. Maintain report structure and section order\n\n"
            f"FORMAT REQUIREMENTS:\n"
            f"- Plain text only, ALL CAPS headings, clean paragraph spacing\n"
            f"- NO symbols: **, __, *, |, ---, ###, `, ~~~\n\n"
            f"OUTPUT: Complete corrected report addressing all feedback points."
        )

    def _feedback_prompt(self, synthesis: str, feedback: str) -> str:
        return (
            f"Write a complete research synthesis report addressing these feedback points:\n\n"
            f"FEEDBACK:\n{feedback}\n\n"
            f"SYNTHESIS DATA:\n{synthesis[:8000]}\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"- Extract guidance from feedback and write as YOUR OWN report\n"
            f"- DO NOT copy feedback section markers or chunk markers\n"
            f"- DO NOT include emoji or special symbols\n"
            f"- DO NOT REPEAT the same sentence or paragraph multiple times\n"
            f"- Write ONLY the final report - professional, clean, complete\n\n"
            f"STRUCTURE:\n"
            f"1. OVERVIEW\n"
            f"2. DETAILED PAPER-BY-PAPER ANALYSIS (number papers: PAPER 1, PAPER 2, etc.)\n"
            f"3. INTEGRATED SYNTHESIS (A. B. C. on separate lines)\n"
            f"4. CONCLUSIONS\n\n"
            f"FORMAT: plain text only, ALL CAPS headings, no markdown symbols.\n"
            f"Include EVERY detail from synthesis. Address all feedback points."
        )

    def _fresh_prompt(self, synthesis: str) -> str:
        return (
            f"Write an EXHAUSTIVELY DETAILED research synthesis report in PLAIN TEXT.\n\n"
            f"ABSOLUTE REQUIREMENT: Include EVERY SINGLE detail from the synthesis. Use EXACT terminology.\n\n"
            f"STRUCTURE:\n"
            f"1. OVERVIEW - Brief introduction to all papers\n"
            f"2. DETAILED PAPER-BY-PAPER ANALYSIS - Number each paper (PAPER 1, PAPER 2, ...). 5-7 paragraphs each.\n"
            f"3. INTEGRATED SYNTHESIS AND COMPARISON - Thematic, with A. B. C. D. E. F. subsections on separate lines.\n"
            f"4. CONCLUSIONS\n\n"
            f"FORMAT RULES:\n"
            f"- Plain paragraphs, ALL CAPS section headings\n"
            f"- No symbols ** __ * | - > ### ` ~ =, no bullets, no numbered lists\n"
            f"- Clean text document style\n\n"
            f"QUALITY GUIDELINES:\n"
            f"- CONCISE: State points once, avoid repetition\n"
            f"- BALANCED: Highlight contributions and limitations constructively\n"
            f"- DEEP COMPARISON: Identify shared themes, complementary approaches\n"
            f"- VALUE PROPOSITION: Explain WHY each paper matters\n"
            f"- STRUCTURED: Organize comparison as similarities, differences, implications\n\n"
            f"Synthesis data (extract content, ignore metadata):\n{synthesis[:20000]}"
        )
