"""
Multi-Agent Research Paper Synthesis System
Uses Semantic Kernel + Local LLM + Hierarchical Review

Architecture:
Reader → Analyzer → Synthesizer → Writer → Reviewer → Chief Reviewer
                                      ↑                      ↓
                                      └──────Feedback────────┘

Supports: PDF, TXT, MD files
Outputs: TXT and DOCX (Word) formats
"""
import asyncio
from pathlib import Path
from typing import List, Dict
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory

# PDF reading libraries
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

# Word document generation
try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("WARNING: Word export not available. Install: pip install python-docx")


class ReaderAgent:
    """Reads and extracts content from research papers (PDF, TXT, MD)"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
    
    def _read_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            # Try pypdf first
            try:
                import pypdf
                with open(pdf_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                # Fallback to PyPDF2
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
        except Exception as e:
            print(f"ERROR: PDF reading error: {e}")
            return ""
    
    def _read_text_file(self, file_path: str) -> str:
        """Read plain text or markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"ERROR: Text file reading error: {e}")
            return ""
    
    def _chunk_text(self, text: str, max_chars: int = 8000) -> List[str]:
        """Split text into chunks that fit within token limits"""
        # Rough estimate: 1 token ~= 4 characters
        # For 12k context, use ~8k chars per chunk to be safe
        chunks = []
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i+max_chars])
        return chunks
    
    async def read_paper(self, paper_path: str) -> Dict[str, str]:
        """Read a single paper and extract key information"""
        print(f"\nReader: Reading {Path(paper_path).name}...")
        
        # Determine file type and read content
        file_extension = Path(paper_path).suffix.lower()
        
        if file_extension == '.pdf':
            if not PDF_AVAILABLE:
                print(f"ERROR: PDF support not available. Install: pip install pypdf")
                return {"title": Path(paper_path).name, "content": "", "error": "PDF library not installed"}
            content = self._read_pdf(paper_path)
        elif file_extension in ['.txt', '.md', '.markdown']:
            content = self._read_text_file(paper_path)
        else:
            print(f"WARNING: Unsupported file type: {file_extension}. Trying as text...")
            content = self._read_text_file(paper_path)
        
        if not content:
            print(f"ERROR: Failed to read {paper_path}")
            return {"title": Path(paper_path).name, "content": "", "error": "Empty or unreadable file"}
        
        # Check content length and chunk if needed
        chunks = self._chunk_text(content, max_chars=5000)
        
        if len(chunks) > 1:
            print(f"   Paper is large ({len(content)} chars), processing in {len(chunks)} chunks...")
        
        # Extract from each chunk
        extracted_parts = []
        for i, chunk in enumerate(chunks, 1):
            if len(chunks) > 1:
                print(f"   Processing chunk {i}/{len(chunks)}...")
            
            history = ChatHistory()
            history.add_system_message(
                "You are a comprehensive research paper analyzer. Extract ALL key information in DETAIL:\\n"
                "1. Title, authors, year\\n"
                "2. Abstract and main objectives\\n"
                "3. Core concepts, theories, frameworks\\n"
                "4. AI/Technology role and applications (if mentioned)\\n"
                "5. Methodology, tools, techniques, technologies\\n"
                "6. Datasets, experiments, use cases\\n"
                "7. Key findings, results, innovations\\n"
                "8. Conclusions and implications\\n\\n"
                "Be DETAILED and THOROUGH. Extract as much information as possible."
            )
            
            chunk_prompt = f"Extract key information from this part of a research paper:\n\n{chunk}"
            if len(chunks) > 1:
                chunk_prompt = f"Part {i}/{len(chunks)} of research paper:\n\n{chunk}"
            
            history.add_user_message(chunk_prompt)
            
            execution_settings = self.chat_service.get_prompt_execution_settings_class()(
                service_id=self.service_id,
                max_tokens=1500,  # More tokens for detailed extraction
                temperature=0.2,
            )
            
            try:
                result = await self.chat_service.get_chat_message_contents(
                    chat_history=history,
                    settings=execution_settings,
                    kernel=self.kernel,
                )
                extracted_parts.append(result[0].content if result else f"Chunk {i} failed")
            except Exception as e:
                print(f"   WARNING: Chunk {i} failed: {str(e)[:100]}")
                extracted_parts.append(f"Chunk {i}: Extraction failed")
        
        # Combine extracted information
        if len(chunks) > 1:
            extracted = "\n\n".join([f"Section {i}:\n{part}" for i, part in enumerate(extracted_parts, 1)])
        else:
            extracted = extracted_parts[0]
        
        print(f"Reader: Extracted from {Path(paper_path).name}")
        
        return {
            "title": Path(paper_path).name,
            "raw_content": content[:5000],  # Store only preview to save memory
            "extracted": extracted
        }
    
    async def read_multiple_papers(self, paper_paths: List[str]) -> List[Dict[str, str]]:
        """Read multiple papers - processes each separately to stay within token limits"""
        print(f"\n📚 Reader: Processing {len(paper_paths)} papers...")
        papers = []
        for i, path in enumerate(paper_paths, 1):
            print(f"   [{i}/{len(paper_paths)}] Processing {Path(path).name}...")
            paper_data = await self.read_paper(path)
            papers.append(paper_data)
        return papers


class AnalyzerAgent:
    """Analyzes findings across multiple papers"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
    
    async def analyze_papers(self, papers: List[Dict[str, str]]) -> str:
        """Analyze findings from multiple papers - uses batch processing for large sets"""
        print(f"\nAnalyzer: Analyzing {len(papers)} papers...")
        
        # Process in batches of 3 papers (with 14k context we can handle this)
        batch_size = 3
        batch_analyses = []
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(papers) + batch_size - 1) // batch_size
            
            print(f"   Analyzing batch {batch_num}/{total_batches} ({len(batch)} papers)...")
            
            # Combine extracted information for this batch - use more content for thorough analysis
            combined_info = "\n\n".join([
                f"Paper: {p['title']}\n{p['extracted'][:4000]}"
                for p in batch
            ])
            
            history = ChatHistory()
            history.add_system_message(
                "You are a comprehensive research analyst. For EACH paper, extract in DETAIL:\n"
                "1. Core concepts and theoretical frameworks\n"
                "2. AI/Technology role and applications (if any)\n"
                "3. Methods, tools, techniques, and technologies used\n"
                "4. Datasets, experiments, and use cases\n"
                "5. Key findings, results, and innovations\n"
                "6. Practical applications and implications\n\n"
                "Focus especially on: AI applications, technological innovations, practical use cases, "
                "and how technology/AI is used in the domain. Be thorough and detailed."
            )
            history.add_user_message(
                f"Analyze these research papers:\n\n{combined_info}"
            )
            
            execution_settings = self.chat_service.get_prompt_execution_settings_class()(
                service_id=self.service_id,
                max_tokens=2000,
                temperature=0.3,
            )
            
            result = await self.chat_service.get_chat_message_contents(
                chat_history=history,
                settings=execution_settings,
                kernel=self.kernel,
            )
            
            batch_analysis = result[0].content if result else f"Batch {batch_num} analysis failed"
            batch_analyses.append(f"Batch {batch_num}:\n{batch_analysis}")
        
        # Combine all batch analyses
        full_analysis = "\n\n" + "="*50 + "\n\n".join(batch_analyses)
        print(f"Analyzer: Completed analysis of all batches")
        return full_analysis


class SynthesizerAgent:
    """Synthesizes insights from analysis"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
    
    async def synthesize(self, analysis: str) -> str:
        """Create synthesis of key insights"""
        print(f"\nSynthesizer: Creating synthesis...")
        
        history = ChatHistory()
        history.add_system_message(
            "You are a comprehensive research synthesizer. Create a DETAILED synthesis covering:\n"
            "1. Key concepts and theoretical frameworks across ALL papers\n"
            "2. AI and technology applications - how AI/tech is used in the domain\n"
            "3. Methodologies, tools, techniques, and technologies\n"
            "4. Use cases, experiments, datasets, and practical applications\n"
            "5. Innovations and findings across papers\n"
            "6. Patterns, trends, and differences in approaches\n\n"
            "Be COMPREHENSIVE. Include specific details from EVERY paper. "
            "Focus especially on technological and AI applications."
        )
        history.add_user_message(
            f"Synthesize these research findings:\n\n{analysis}"
        )
        
        execution_settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id,
            max_tokens=3500,  # More capacity for comprehensive synthesis
            temperature=0.4,
        )
        
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history,
            settings=execution_settings,
            kernel=self.kernel,
        )
        
        synthesis = result[0].content if result else "Synthesis failed"
        print(f"Synthesizer: Synthesis complete")
        return synthesis


class WriterAgent:
    """Writes and rewrites the research report"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
    
    async def write_report(self, synthesis: str, feedback: str = None) -> str:
        """Write or rewrite report based on synthesis and optional feedback"""
        if feedback:
            print(f"\nWriter: Rewriting report based on feedback...")
        else:
            print(f"\nWriter: Writing initial report...")
        
        history = ChatHistory()
        history.add_system_message(
            "You are a professional academic report writer creating a formal, comprehensive research synthesis.\n\n"
            "REQUIRED STRUCTURE (use these as plain text section headings):\n"
            "1. OVERVIEW - Brief description of all papers and their focus areas\n"
            "2. KEY CONCEPTS AND FRAMEWORKS - Core ideas and theories from all papers\n"
            "3. AI AND TECHNOLOGY APPLICATIONS - How AI/technology is used across papers\n"
            "4. METHODOLOGIES AND TECHNIQUES - Methods, tools, technologies used\n"
            "5. USE CASES AND PRACTICAL APPLICATIONS - Real-world applications, experiments\n"
            "6. FINDINGS AND INNOVATIONS - Key results and discoveries\n"
            "7. COMPARATIVE ANALYSIS - How approaches differ and compare\n"
            "8. SUMMARY AND SYNTHESIS - Overall insights and conclusions\n\n"
            "CRITICAL FORMATTING RULES - YOU MUST FOLLOW THESE:\n"
            "✗ FORBIDDEN: Do NOT use **, **, |, ---, ###, ===, `, ~~~, or any markdown\n"
            "✗ FORBIDDEN: Do NOT create ASCII tables with pipes and dashes\n"
            "✗ FORBIDDEN: Do NOT use asterisks for bold or emphasis\n\n"
            "✓ REQUIRED: Write section headings in plain ALL CAPS text\n"
            "✓ REQUIRED: Write in continuous prose paragraphs\n"
            "✓ REQUIRED: For comparisons, write: 'Lockhart uses rhetorical methods while Dewey uses experiential approaches.'\n"
            "✓ REQUIRED: For lists, write natural sentences: 'The main methods include first, second, and third...'\n\n"
            "EXAMPLE OF CORRECT FORMAT:\n"
            "OVERVIEW\n\n"
            "This report synthesizes three papers on education theory. Lockhart presents a philosophical critique. Dewey explores experiential learning. Freire examines critical pedagogy.\n\n"
            "METHODOLOGIES USED\n\n"
            "Lockhart employs rhetorical and analogical methods using narrative devices to expose systemic flaws. His approach is normative analysis prescribing what education should be.\n\n"
            "Write like this - plain professional text with no special characters."
        )
        
        if feedback:
            prompt = (
                f"Previous synthesis:\n{synthesis}\n\n"
                f"Reviewer feedback:\n{feedback}\n\n"
                f"Rewrite the COMPLETE, COMPREHENSIVE report in PLAIN TEXT. Remove ALL markdown symbols (**, |, ---, etc). "
                f"Use plain prose with ALL CAPS section headings. Include DETAILED information from EVERY paper. "
                f"Focus on: concepts, AI/technology applications, use cases, techniques, findings. Be THOROUGH - aim for 20+ pages."
            )
        else:
            prompt = (
                f"Write a COMPREHENSIVE, DETAILED research synthesis report in PLAIN TEXT (no markdown symbols).\n"
                f"Use ALL CAPS section headings and plain prose paragraphs.\n\n"
                f"CRITICAL: Include DETAILED information from ALL papers. Cover:\n"
                f"- All concepts and frameworks\n"
                f"- All AI/technology applications and use cases\n"
                f"- All methodologies and techniques\n"
                f"- All findings and innovations\n"
                f"- Comparative analysis across papers\n\n"
                f"Be THOROUGH and COMPREHENSIVE. Aim for 20-25 pages of detailed content.\n\n"
                f"Synthesis:\n{synthesis}"
            )
        
        history.add_user_message(prompt)
        
        execution_settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id,
            max_tokens=8000,  # ~20-25 pages for comprehensive detailed report
            temperature=0.5,
        )
        
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history,
            settings=execution_settings,
            kernel=self.kernel,
        )
        
        report = result[0].content if result else "Report writing failed"
        print(f"Writer: Report {'rewritten' if feedback else 'completed'}")
        return report


class ReviewerAgent:
    """First-level review: checks technical quality"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
    
    async def review(self, report: str) -> Dict[str, any]:
        """Review report for quality and completeness"""
        print(f"\nReviewer: Reviewing report...")
        
        history = ChatHistory()
        history.add_system_message(
            "You are a technical reviewer checking COMPREHENSIVENESS and QUALITY. Verify:\n"
            "1. ALL papers are included with DETAILED information (not just brief mentions)\n"
            "2. AI/Technology applications are thoroughly covered\n"
            "3. Concepts, use cases, techniques are explained in detail\n"
            "4. Each paper's methodology and findings are described comprehensively\n"
            "5. Comparisons and differences are clear and detailed\n"
            "6. Report is LONG and COMPREHENSIVE (should be 15+ pages worth of content)\n"
            "7. Written in PLAIN TEXT (no **, |, ---, or markdown symbols)\n"
            "8. Format is professional and readable\n\n"
            "REJECT if: report is too short, missing paper details, lacks AI/tech focus, has markdown symbols.\n"
            "Respond with: APPROVE or REJECT followed by specific feedback on what's missing."
        )
        history.add_user_message(
            f"Review this research report:\n\n{report}"
        )
        
        execution_settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id,
            max_tokens=1000,
            temperature=0.2,
        )
        
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history,
            settings=execution_settings,
            kernel=self.kernel,
        )
        
        review_text = result[0].content if result else "Review failed"
        approved = "APPROVE" in review_text.upper()
        
        print(f"Reviewer: {('APPROVED' if approved else 'REJECTED')}")
        
        return {
            "approved": approved,
            "feedback": review_text,
            "level": "reviewer"
        }


class ChiefReviewerAgent:
    """Final approval: checks academic rigor and quality"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
    
    async def final_review(self, report: str) -> Dict[str, any]:
        """Final review for academic quality"""
        print(f"\nChief Reviewer: Final review...")
        
        history = ChatHistory()
        history.add_system_message(
            "You are the final quality reviewer ensuring COMPREHENSIVE COVERAGE. Verify:\n"
            "1. ALL papers are covered with substantial detail (not superficial)\n"
            "2. AI/Technology role is thoroughly explained across papers\n"
            "3. Concepts, use cases, techniques are comprehensively described\n"
            "4. Report is LONG enough (15-25 pages worth) with rich detail\n"
            "5. Methodologies and findings are accurately and fully described\n"
            "6. Report is practical, useful, and insightful\n"
            "7. Synthesis integrates findings across papers effectively\n\n"
            "REJECT if report is too brief or lacks comprehensive detail.\n"
            "Respond with: APPROVE or REJECT followed by high-level feedback."
        )
        history.add_user_message(
            f"Final review of this research report:\n\n{report}"
        )
        
        execution_settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id,
            max_tokens=1000,
            temperature=0.2,
        )
        
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history,
            settings=execution_settings,
            kernel=self.kernel,
        )
        
        review_text = result[0].content if result else "Final review failed"
        approved = "APPROVE" in review_text.upper()
        
        print(f"Chief Reviewer: {('FINAL APPROVAL' if approved else 'REJECTED')}")
        
        return {
            "approved": approved,
            "feedback": review_text,
            "level": "chief_reviewer"
        }


def save_report_as_word(report_text: str, output_path: Path):
    """Save report as formatted Word document - detects sections intelligently"""
    if not DOCX_AVAILABLE:
        return
    
    doc = Document()
    
    # Add title
    title = doc.add_heading('Research Paper Synthesis Report', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.font.color.rgb = RGBColor(0, 51, 102)  # Dark blue
    
    doc.add_paragraph()  # Spacing
    
    # Set default font
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)
    
    # Split into paragraphs
    paragraphs = report_text.split('\n\n')
    
    # Section keywords to detect headings
    section_keywords = [
        'OVERVIEW', 'INTRODUCTION', 'METHODOLOGIES USED', 'METHODOLOGY', 
        'KEY DIFFERENCES', 'DIFFERENCES', 'RESULTS COMPARISON', 'RESULTS',
        'SUMMARY', 'CONCLUSION', 'CONCLUSIONS', 'FINDINGS'
    ]
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Check if this is a section heading
        first_line = para.split('\n')[0].strip()
        is_heading = False
        
        # Detect if first line is a section heading
        for keyword in section_keywords:
            if keyword in first_line.upper() and len(first_line) < 60:
                # This is likely a heading
                h = doc.add_heading(first_line, level=1)
                h.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in h.runs:
                    run.font.size = Pt(14)
                    run.font.color.rgb = RGBColor(0, 51, 102)
                
                # Add rest of paragraph if exists
                rest = '\n'.join(para.split('\n')[1:]).strip()
                if rest:
                    p = doc.add_paragraph(rest)
                    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    p.paragraph_format.line_spacing = 1.15
                    p.paragraph_format.space_after = Pt(6)
                
                is_heading = True
                break
        
        if not is_heading:
            # Regular paragraph - handle bullet points
            if para.startswith('•') or para.startswith('-'):
                # Bullet list
                for line in para.split('\n'):
                    line = line.strip()
                    if line.startswith('•'):
                        doc.add_paragraph(line[1:].strip(), style='List Bullet')
                    elif line.startswith('-'):
                        doc.add_paragraph(line[1:].strip(), style='List Bullet')
                    elif line:
                        doc.add_paragraph(line, style='List Bullet')
            else:
                # Normal paragraph with full justification
                p = doc.add_paragraph(para)
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.first_line_indent = Inches(0.25)
    
    # Save document
    doc.save(str(output_path))


class ResearchSynthesisBehaviorTree:
    """Orchestrates the entire multi-agent workflow"""
    
    def __init__(self, reader, analyzer, synthesizer, writer, reviewer, chief_reviewer):
        self.reader = reader
        self.analyzer = analyzer
        self.synthesizer = synthesizer
        self.writer = writer
        self.reviewer = reviewer
        self.chief_reviewer = chief_reviewer
    
    async def run(self, paper_paths: List[str], max_iterations: int = 3) -> Dict:
        """Execute the full research synthesis pipeline"""
        print("\n" + "="*70)
        print("RESEARCH SYNTHESIS BEHAVIOR TREE")
        print("="*70)
        
        # Phase 1: Read papers
        papers = await self.reader.read_multiple_papers(paper_paths)
        if not papers:
            return {"success": False, "error": "No papers read"}
        
        # Phase 2: Analyze
        analysis = await self.analyzer.analyze_papers(papers)
        
        # Phase 3: Synthesize
        synthesis = await self.synthesizer.synthesize(analysis)
        
        # Phase 4-6: Write, Review, Rewrite loop
        report = None
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'─'*70}")
            print(f"Writing Iteration {iteration}/{max_iterations}")
            print(f"{'─'*70}")
            
            # Phase 4: Write/Rewrite
            if iteration == 1:
                report = await self.writer.write_report(synthesis)
            else:
                # Use combined feedback from both reviewers
                combined_feedback = f"{reviewer_result['feedback']}"
                if 'chief_feedback' in locals():
                    combined_feedback += f"\n\nChief Reviewer: {chief_feedback}"
                report = await self.writer.write_report(synthesis, combined_feedback)
            
            # Phase 5: First review
            reviewer_result = await self.reviewer.review(report)
            
            if not reviewer_result["approved"]:
                print(f"WARNING: Iteration {iteration}: Reviewer rejected, retrying...")
                continue
            
            # Phase 6: Chief review (only if first review passed)
            chief_result = await self.chief_reviewer.final_review(report)
            
            if chief_result["approved"]:
                print(f"\n{'='*70}")
                print(f"SUCCESS! Report approved on iteration {iteration}")
                print(f"{'='*70}")
                return {
                    "success": True,
                    "iterations": iteration,
                    "report": report,
                    "final_feedback": chief_result["feedback"]
                }
            else:
                print(f"WARNING: Iteration {iteration}: Chief Reviewer rejected, retrying...")
                chief_feedback = chief_result["feedback"]
        
        # Max iterations reached
        print(f"\n{'='*70}")
        print(f"WARNING: Max iterations reached. Using last version.")
        print(f"{'='*70}")
        return {
            "success": False,
            "iterations": max_iterations,
            "report": report,
            "final_feedback": "Maximum iterations reached"
        }


async def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Multi-Agent Research Paper Synthesis System                 ║
║  Powered by: Semantic Kernel + Local LLM                     ║
║  Agents: Reader → Analyzer → Synthesizer → Writer            ║
║          → Reviewer → Chief Reviewer                         ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize Semantic Kernel
    print("Initializing Semantic Kernel...")
    kernel = Kernel()
    
    # Connect to Local LLM (LM Studio)
    service_id = "local-gpt"
    chat_service = OpenAIChatCompletion(
        service_id=service_id,
        ai_model_id="qwen3-vl-8b",
        api_key="NA",
    )
    chat_service.client.base_url = "http://localhost:1234/v1"
    kernel.add_service(chat_service)
    print("Connected to Local LLM")
    
    # Create all agents
    print("Creating agents...")
    reader = ReaderAgent(kernel, service_id)
    analyzer = AnalyzerAgent(kernel, service_id)
    synthesizer = SynthesizerAgent(kernel, service_id)
    writer = WriterAgent(kernel, service_id)
    reviewer = ReviewerAgent(kernel, service_id)
    chief_reviewer = ChiefReviewerAgent(kernel, service_id)
    print("All agents created")
    
    # Create behavior tree
    tree = ResearchSynthesisBehaviorTree(
        reader, analyzer, synthesizer, writer, reviewer, chief_reviewer
    )
    
    # Check for existing papers in sample_papers directory
    sample_dir = Path("sample_papers")
    
    # Look for PDF, TXT, MD files
    paper_paths = []
    if sample_dir.exists():
        for ext in ['*.pdf', '*.txt', '*.md']:
            paper_paths.extend([str(p) for p in sample_dir.glob(ext)])
    
    if not paper_paths:
        # Create sample papers for demonstration
        print("\n📝 No papers found. Creating sample papers for demonstration...")
        sample_dir.mkdir(exist_ok=True)
        
        sample_papers = [
            {
                "filename": "lockhart_lament.txt",
                "content": """
Title: A Mathematician's Lament - How School Cheats Us Out of Our Most Fascinating and Imaginative Art Form
Author: Paul Lockhart
Year: 2002

Abstract:
This essay presents a philosophical critique of contemporary mathematics education in American schools. Lockhart argues that mathematics has been transformed from a creative, artistic discipline into a mechanical, procedural subject focused on memorization and algorithmic manipulation.

Methodology:
The author employs rhetorical analysis, metaphorical reasoning, and narrative critique. Using extended analogies (comparing mathematics education to teaching music by only reading sheet music without ever playing), Lockhart exposes what he views as systemic failures. His approach is normative and philosophical rather than empirical.

Key Tools and Techniques:
- Conceptual thought experiments
- Analogical reasoning (math as art/music)
- Narrative storytelling and personal observation
- Critical pedagogy framework
- Normative analysis prescribing ideal education

Findings:
Mathematics is misrepresented as procedures rather than creative exploration. Creativity is suppressed through rigid, standardized instruction. Assessment focuses on correct answers rather than mathematical thinking. The role of mathematician as explorer and creator is completely absent from K-12 education.

Dataset/Evidence:
No quantitative data. Evidence consists of:
- Personal teaching experience
- Observations of curriculum standards
- Anecdotal examples from classrooms
- Conceptual analysis of textbooks and syllabi

Results:
Lockhart identifies four systemic problems: proceduralization of mathematics, suppression of creativity, dominance of standardization, and erasure of mathematical identity. He proposes mathematics should be taught as an art form with emphasis on exploration, discovery, and beauty.

Conclusion:
Current mathematics education is fundamentally broken. Reform requires reconceptualizing mathematics as creative exploration rather than skill acquisition. Teachers should act as guides in mathematical discovery, not deliverers of procedures.
                """
            },
            {
                "filename": "dewey_experiential_learning.txt",
                "content": """
Title: Experience and Education - The Theory of Experiential Learning
Author: John Dewey
Year: 1938

Abstract:
Dewey presents a systematic theory of experiential education as an alternative to traditional didactic instruction. He argues that genuine education comes through experience, but not all experiences are equally educative. The work establishes criteria for evaluating educational experiences.

Methodology:
Philosophical and theoretical analysis combined with observational evidence from progressive schools. Dewey uses dialectical reasoning to contrast traditional and progressive education, then synthesizes a balanced approach. His method is pragmatic philosophy applied to educational practice.

Key Tools and Techniques:
- Philosophical inquiry and conceptual analysis
- Comparative method (traditional vs progressive)
- Dialectical reasoning and synthesis
- Pragmatic epistemology
- Case analysis from laboratory schools

Findings:
Education must be grounded in student experience, but experiences must be carefully structured. Continuity and interaction are essential principles. Teachers must understand how experiences lead to growth versus miseducation. Freedom without guidance leads to chaos; structure without meaning leads to passive learning.

Dataset/Evidence:
- Observational data from University of Chicago Laboratory Schools
- Comparative analysis of traditional and progressive classrooms
- Longitudinal observations of student development
- Qualitative case studies of teaching methods

Results:
Dewey establishes that effective experiential learning requires: connection to student interests, progression from simple to complex, social interaction and collaboration, reflective thinking integrated with action, and teacher as facilitator rather than authority.

Key Differences from Traditional Methods:
Traditional education emphasizes transmission of knowledge, teacher authority, passive reception, and isolated skills. Experiential education emphasizes construction of knowledge, democratic participation, active engagement, and integrated understanding.

Conclusion:
Neither pure traditional nor pure progressive education is adequate. Effective education requires structured experiences that connect to student interests, promote growth, and develop reflective thinking. Teachers must be skilled at designing educative experiences, not just delivering content.
                """
            },
            {
                "filename": "piaget_cognitive_development.txt",
                "content": """
Title: The Psychology of Intelligence and Cognitive Development Theory
Author: Jean Piaget
Year: 1947-1950

Abstract:
Piaget presents a comprehensive theory of cognitive development based on decades of empirical observation. He proposes that children progress through distinct stages of intellectual development, each characterized by qualitatively different ways of thinking and understanding the world.

Methodology:
Clinical interview method combined with systematic observation. Piaget used semi-structured interviews with children, presenting them with cognitive tasks and carefully analyzing their responses and reasoning processes. His approach is constructivist and developmental.

Key Tools and Techniques:
- Clinical interview method (méthode clinique)
- Systematic observation of children solving problems
- Longitudinal developmental studies
- Task-based cognitive assessment
- Qualitative analysis of reasoning patterns
- Cross-sectional comparisons across age groups

Findings:
Children progress through four major stages: sensorimotor (0-2 years), preoperational (2-7 years), concrete operational (7-11 years), and formal operational (11+ years). Each stage represents qualitatively different cognitive structures. Development occurs through assimilation and accommodation.

Dataset/Evidence:
- Over 20 years of observations with hundreds of children
- Detailed case studies of Piaget's own three children
- Systematic tasks: conservation, classification, seriation
- Cross-cultural replications confirming stage sequences
- Longitudinal tracking of individual children

Results:
Key findings include: conservation of quantity emerges around age 7, abstract reasoning develops in adolescence, children are active constructors of knowledge, cognitive development cannot be rushed through direct instruction, social interaction facilitates cognitive growth, and stages are universal across cultures though timing may vary.

Experimental Methodology:
Conservation tasks: showing children equal quantities then transforming one and asking if they remain equal. Classification tasks: sorting objects by multiple attributes. Seriation tasks: ordering objects by size. Three mountains task: testing perspective-taking abilities.

Key Differences from Behaviorist Approaches:
While behaviorists view learning as stimulus-response associations shaped by reinforcement, Piaget emphasizes active construction, qualitative stages, and intrinsic motivation. He focuses on understanding rather than performance, mental structures rather than behaviors.

Conclusion:
Cognitive development is a constructive process where children actively build increasingly sophisticated mental structures. Education must respect developmental stages and provide experiences that promote disequilibrium and subsequent accommodation. Teachers should facilitate discovery rather than transmit knowledge.
                """
            },
            {
                "filename": "freire_critical_pedagogy.txt",
                "content": """
Title: Pedagogy of the Oppressed - Critical Consciousness and Liberatory Education
Author: Paulo Freire
Year: 1968

Abstract:
Freire presents a revolutionary theory of education as a practice of freedom. He critiques traditional "banking" education where teachers deposit knowledge into passive students, proposing instead a dialogical, problem-posing approach that develops critical consciousness and transforms oppressive social structures.

Methodology:
Critical pedagogy combining Marxist analysis, phenomenology, and participatory action research. Freire's approach is praxis-oriented: reflection and action are inseparable. His methodology emerged from literacy programs with oppressed Brazilian peasants.

Key Tools and Techniques:
- Dialogical education (teacher-student dialogue)
- Problem-posing methodology
- Generative themes from learner experience
- Conscientization (development of critical consciousness)
- Participatory action research
- Sociocultural analysis of power structures

Findings:
Traditional education serves to maintain oppression by treating students as passive receptacles. Banking education domesticates students, preventing critical thinking about social conditions. Liberation requires dialogical education where teachers and students are co-investigators. Education must begin with learner experience and move toward critical analysis of social reality.

Dataset/Evidence:
- Participatory research with Brazilian literacy programs
- Qualitative case studies of adult education
- Analysis of curriculum and teaching practices
- Observations from Chilean and African education projects
- Testimonials from peasant learners

Results:
Literacy rates improved dramatically when education connected to lived experience. Students who participated in dialogical education developed critical consciousness and became agents of social change. Banking education perpetuated passivity; problem-posing education cultivated agency. Cultural invasion through education maintained oppression; cultural synthesis promoted liberation.

Methodological Innovation:
Freire's culture circles replaced traditional classrooms. Instead of teacher lectures, participants engaged in dialogue about generative themes from their lives. Literacy was taught through words with existential meaning to learners. Education became consciousness-raising rather than skill training.

Key Differences from Traditional and Progressive Approaches:
While traditional education emphasizes transmission and progressive education emphasizes experience, Freire emphasizes transformation. His focus is explicitly political: education either domesticates or liberates. Unlike Dewey's focus on democratic experience, Freire focuses on class struggle and oppression.

Conclusion:
Education is never neutral. It either maintains oppression or promotes liberation. Effective liberatory education must be dialogical, grounded in learner experience, focused on critical consciousness, connected to action for social transformation, and conducted through authentic teacher-student partnership. The goal is not just learning but humanization and social justice.
                """
            },
            {
                "filename": "vygotsky_sociocultural_theory.txt",
                "content": """
Title: Mind in Society - The Development of Higher Psychological Processes
Author: Lev Vygotsky
Year: 1978 (posthumous compilation)

Abstract:
Vygotsky presents a sociocultural theory of cognitive development emphasizing the fundamental role of social interaction and cultural tools in intellectual growth. He proposes that higher mental functions develop through internalization of social processes, challenging purely individual constructivist theories.

Methodology:
Genetic method (studying development through transformation), cross-cultural comparative analysis, and experimental-developmental approach. Vygotsky combined psychological experiments with anthropological and historical analysis of tool use and cultural practices.

Key Tools and Techniques:
- Zone of Proximal Development (ZPD) analysis
- Scaffolding and guided participation
- Cultural-historical analysis
- Microgenetic method (intensive study of change)
- Experimental tasks with tool mediation
- Cross-cultural comparative studies

Findings:
Higher mental functions are socially constructed through interaction with more knowledgeable others. Learning leads development (not vice versa). The Zone of Proximal Development defines the range between independent and assisted performance. Cultural tools (especially language) mediate thinking. Social speech becomes inner speech becomes thought.

Dataset/Evidence:
- Experimental studies with Soviet children
- Cross-cultural research in Central Asia
- Studies of deaf children and language development
- Memory experiments with and without mnemonic tools
- Problem-solving tasks with adult guidance
- Analysis of play and imagination development

Results:
Children can solve problems with assistance before they can solve them independently. The ZPD varies by domain and individual. Effective instruction targets the ZPD, not current developmental level. Language transforms thinking from spontaneous to systematic. Cultural tools amplify cognitive capabilities. Collaborative learning accelerates development.

Experimental Methodology:
Double stimulation method: giving children problems plus potential tools and observing how they use tools to solve problems. Studied memory with and without visual aids. Analyzed problem-solving with adult hints versus independent work. Compared development in different cultural contexts.

Key Differences from Piaget:
While Piaget emphasized individual construction and stages emerging from action on objects, Vygotsky emphasized social construction and continuous development through cultural participation. Piaget saw language following thought; Vygotsky saw language transforming thought. Piaget focused on universal stages; Vygotsky emphasized cultural variation.

Instructional Implications:
Teaching should aim slightly ahead of development, working in the ZPD. Peer collaboration facilitates learning. Language is essential for intellectual development. Apprenticeship and guided participation are natural learning contexts. Assessment should measure potential for learning, not just current achievement.

Conclusion:
Cognitive development is fundamentally social and cultural. Higher mental functions emerge from interpersonal processes that become intrapersonal. Education's role is to create zones of proximal development through collaboration, scaffolding, and cultural tool use. Effective teaching requires understanding both current development and developmental potential.
                """
            }
        ]
        
        for paper in sample_papers:
            paper_path = sample_dir / paper["filename"]
            paper_path.write_text(paper["content"])
        
        paper_paths = [str(sample_dir / p["filename"]) for p in sample_papers]
    
    print(f"Found/Created {len(paper_paths)} papers:")
    for path in paper_paths:
        print(f"   - {Path(path).name}")
    
    # Run the synthesis pipeline
    result = await tree.run(paper_paths, max_iterations=2)
    
    # Save final report with timestamp
    if result["report"]:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save TXT format
        output_txt = Path(f"research_report_{timestamp}.txt")
        output_txt.write_text(result["report"])
        print(f"\nText report saved to: {output_txt}")
        
        # Save Word format
        if DOCX_AVAILABLE:
            output_docx = Path(f"research_report_{timestamp}.docx")
            save_report_as_word(result["report"], output_docx)
            print(f"Word report saved to: {output_docx}")
        else:
            print("WARNING: Word export skipped (install python-docx to enable)")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SYNTHESIS SUMMARY")
    print(f"{'='*70}")
    print(f"Success: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nFinal Report Preview:")
    print("─"*70)
    print(result['report'][:500] + "..." if len(result['report']) > 500 else result['report'])
    print("─"*70)


if __name__ == "__main__":
    asyncio.run(main())
