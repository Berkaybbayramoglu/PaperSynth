"""Reader agent: extracts text from papers and summarizes each one with the LLM."""
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory

try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    try:
        import PyPDF2
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        print("WARNING: PDF support not available. Install: pip install pypdf")


class ReaderAgent:
    """Reads PDF / TXT / MD papers and produces a detail-rich LLM extraction."""

    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)

    def _read_pdf(self, pdf_path: str) -> str:
        try:
            try:
                import pypdf
                with open(pdf_path, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    return "".join(page.extract_text() + "\n" for page in reader.pages)
            except ImportError:
                import PyPDF2
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    return "".join(page.extract_text() + "\n" for page in reader.pages)
        except Exception as e:
            print(f"ERROR: PDF reading error: {e}")
            return ""

    def _read_text_file(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"ERROR: Text file reading error: {e}")
            return ""

    def _chunk_text(self, text: str, max_chars: int = 8000) -> List[str]:
        if not text:
            return [""]
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    async def read_paper(self, paper_path: str) -> Dict[str, str]:
        """Read one paper file and return a dict with the LLM-extracted summary."""
        print(f"\nReader: Reading {Path(paper_path).name}...")

        ext = Path(paper_path).suffix.lower()
        if ext == ".pdf":
            if not PDF_AVAILABLE:
                return {"title": Path(paper_path).name, "content": "", "error": "PDF library not installed"}
            content = self._read_pdf(paper_path)
        elif ext in (".txt", ".md", ".markdown"):
            content = self._read_text_file(paper_path)
        else:
            print(f"WARNING: Unsupported file type: {ext}. Trying as text...")
            content = self._read_text_file(paper_path)

        if not content:
            return {"title": Path(paper_path).name, "content": "", "error": "Empty or unreadable file"}

        chunks = self._chunk_text(content, max_chars=5000)
        if len(chunks) > 1:
            print(f"   Paper is large ({len(content)} chars), processing in {len(chunks)} chunks...")

        extracted_parts: List[str] = []
        for i, chunk in enumerate(chunks, 1):
            if len(chunks) > 1:
                print(f"   Processing chunk {i}/{len(chunks)}...")

            history = ChatHistory()
            history.add_system_message(
                "You are a comprehensive research paper analyzer. Extract EVERYTHING - leave NO detail out.\n\n"
                "Extract in COMPLETE DETAIL:\n"
                "1. Title, authors, year, publication venue\n"
                "2. Abstract and research objectives\n"
                "3. Core concepts, theories, frameworks - ALL of them\n"
                "4. EVERY AI/technology application mentioned (name each one)\n"
                "5. EVERY methodology, tool, technique, algorithm (list all by name)\n"
                "6. ALL datasets: size, source, characteristics, collection methods\n"
                "7. ALL experiments: what was tested, how, with what parameters\n"
                "8. ALL results: metrics, percentages, improvements, comparisons\n"
                "9. ALL findings, innovations, contributions\n"
                "10. ALL conclusions and practical implications\n\n"
                "CRITICAL: Extract EVERY detail mentioned. If paper says 'Random Forest with 100 trees', "
                "write that exact detail. If it says '95.3% accuracy', write exact number. "
                "DO NOT summarize or skip anything. Extract EVERYTHING."
            )
            prompt = f"Extract key information from this part of a research paper:\n\n{chunk}"
            if len(chunks) > 1:
                prompt = f"Part {i}/{len(chunks)} of research paper:\n\n{chunk}"
            history.add_user_message(prompt)

            settings = self.chat_service.get_prompt_execution_settings_class()(
                service_id=self.service_id, max_tokens=1500, temperature=0.2,
            )
            try:
                result = await self.chat_service.get_chat_message_contents(
                    chat_history=history, settings=settings, kernel=self.kernel,
                )
                extracted_parts.append(result[0].content if result else f"Chunk {i} failed")
            except Exception as e:
                print(f"   WARNING: Chunk {i} failed: {str(e)[:100]}")
                extracted_parts.append(f"Chunk {i}: Extraction failed")

        if len(chunks) > 1:
            extracted = "\n\n".join(f"Section {i}:\n{p}" for i, p in enumerate(extracted_parts, 1))
        else:
            extracted = extracted_parts[0]
        extracted_parts.clear()

        print(f"Reader: Extracted from {Path(paper_path).name}")
        return {
            "title": Path(paper_path).name,
            "raw_content": content[:5000],
            "extracted": extracted,
        }

    async def read_multiple_papers(
        self, paper_paths: List[str], parallel: bool = True, batch_size: int = 6,
    ) -> Optional[List[Dict[str, str]]]:
        """Read a list of papers; either sequentially or in order-preserving parallel batches."""
        if not parallel or len(paper_paths) <= 1:
            print(f"\nReader: Processing {len(paper_paths)} papers sequentially...")
            papers = []
            for i, path in enumerate(paper_paths, 1):
                print(f"   [{i}/{len(paper_paths)}] Processing {Path(path).name}...")
                papers.append(await self.read_paper(path))
            return papers

        print(f"\nReader: Processing {len(paper_paths)} papers in parallel (batch_size={batch_size})...")
        all_papers: List[Optional[Dict]] = [None] * len(paper_paths)

        for batch_start in range(0, len(paper_paths), batch_size):
            batch_paths = paper_paths[batch_start : batch_start + batch_size]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(paper_paths) + batch_size - 1) // batch_size

            if total_batches > 1:
                print(f"   Processing batch {batch_num}/{total_batches} ({len(batch_paths)} papers)...")

            tasks = [self.read_paper(path) for path in batch_paths]
            papers = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(papers):
                global_idx = batch_start + i
                if isinstance(result, Exception):
                    print(f"   WARNING: Paper {global_idx + 1} failed: {str(result)[:100]}")
                    all_papers[global_idx] = {
                        "title": Path(paper_paths[global_idx]).name,
                        "content": "",
                        "error": str(result),
                    }
                else:
                    all_papers[global_idx] = result

        valid = [p for p in all_papers if p and isinstance(p, dict) and p.get("extracted")]
        if len(valid) < len(all_papers):
            print(f"   WARNING: {len(all_papers) - len(valid)} papers failed to load")
        print(f"Reader: Completed reading {len(valid)} valid papers")
        return valid if valid else None
