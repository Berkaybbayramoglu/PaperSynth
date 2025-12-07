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
from typing import List, Dict, Any
import re  # For regex operations
import httpx  # For HTTP client timeout configuration
from openai import AsyncOpenAI  # For custom client with timeout
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

# JSON for progress tracking and caching
import json
from datetime import datetime


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
        if not text:
            return [""]  # Return list with empty string to avoid index errors
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
        
        # Free memory from extraction chunks
        extracted_parts.clear()
        
        print(f"Reader: Extracted from {Path(paper_path).name}")
        
        return {
            "title": Path(paper_path).name,
            "raw_content": content[:5000],  # Store only preview to save memory
            "extracted": extracted
        }
    
    async def read_multiple_papers(self, paper_paths: List[str], parallel: bool = True, batch_size: int = 6) -> List[Dict[str, str]] | None:
        """Read multiple papers with optional parallel processing
        
        Args:
            paper_paths: List of paths to papers
            parallel: If True, read in parallel. If False, sequential.
            batch_size: Max papers to process in parallel at once
        """
        if not parallel or len(paper_paths) <= 1:
            # Sequential processing
            print(f"\n📚 Reader: Processing {len(paper_paths)} papers sequentially...")
            papers = []
            for i, path in enumerate(paper_paths, 1):
                print(f"   [{i}/{len(paper_paths)}] Processing {Path(path).name}...")
                result = await self.read_paper(path)
                papers.append(result)
            return papers
        
        # Parallel processing with batch control (order-preserving)
        print(f"\n📚 Reader: Processing {len(paper_paths)} papers in parallel (batch_size={batch_size})...")
        
        # Pre-allocate list to maintain original order
        all_papers = [None] * len(paper_paths)
        
        for batch_start in range(0, len(paper_paths), batch_size):
            batch_paths = paper_paths[batch_start:batch_start + batch_size]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(paper_paths) + batch_size - 1) // batch_size
            
            if total_batches > 1:
                print(f"   Processing batch {batch_num}/{total_batches} ({len(batch_paths)} papers)...")
            
            # Create tasks for this batch (will complete in any order)
            tasks = [self.read_paper(path) for path in batch_paths]
            
            # Execute batch in parallel
            papers = await asyncio.gather(*tasks, return_exceptions=True)
        
            # Store results in correct positions to maintain original order
            for i, result in enumerate(papers):
                global_idx = batch_start + i
                if isinstance(result, Exception):
                    print(f"   WARNING: Paper {global_idx + 1} failed: {str(result)[:100]}")
                    all_papers[global_idx] = {
                        "title": Path(paper_paths[global_idx]).name,
                        "content": "",
                        "error": str(result)
                    }
                else:
                    all_papers[global_idx] = result
        
        # Filter out None/empty papers
        valid_papers = [p for p in all_papers if p and isinstance(p, dict) and p.get('extracted')]
        
        if len(valid_papers) < len(all_papers):
            print(f"   WARNING: {len(all_papers) - len(valid_papers)} papers failed to load")
        
        print(f"Reader: Completed reading {len(valid_papers)} valid papers")
        return valid_papers if valid_papers else None


class AnalyzerAgent:
    """Analyzes findings across multiple papers"""
    
    def __init__(self, kernel: Kernel, service_id: str, memory_agent=None):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
        self.memory_agent = memory_agent  # Enable caching
    
    async def _analyze_single_paper(self, paper: Dict[str, str], paper_num: int, total_papers: int) -> str:
        """Analyze a single paper with full detail extraction (with caching)"""
        # Validate paper structure
        if not paper or not isinstance(paper, dict) or not paper.get('extracted') or not paper.get('title'):
            return f"PAPER {paper_num}: ERROR - Invalid paper structure\n"
        
        # Check cache first if memory agent available
        if self.memory_agent:
            paper_hash = MemoryAgent.compute_hash(paper['extracted'])
            cached = await self.memory_agent.get_analysis(paper['title'], paper_hash)
            if cached and isinstance(cached, dict) and cached.get('analysis'):
                print(f"   ✓ Using cached analysis for paper {paper_num}/{total_papers}: {paper['title'][:50]}")
                return f"PAPER {paper_num}: {paper['title']}\n{'='*80}\n{cached['analysis']}\n"
        
        print(f"   Analyzing paper {paper_num}/{total_papers}: {paper['title'][:50]}...")
        
        history = ChatHistory()
        history.add_system_message(
                "You are a research analyst. Write an EXHAUSTIVE, DETAILED analysis in clear paragraphs.\n\n"
                "FORMAT RULES:\n"
                "- Write in plain, flowing paragraphs\n"
                "- NO numbered lists, NO bullets, NO symbols (-, *, >, ⚠)\n"
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
        
        execution_settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id,
            max_tokens=2500,  # Increased for more detailed analysis
            temperature=0.2,   # Lower temperature for accuracy
        )
        
        try:
            result = await self.chat_service.get_chat_message_contents(
                chat_history=history,
                settings=execution_settings,
                kernel=self.kernel,
            )
            
            analysis = result[0].content if result else f"Analysis failed for {paper['title']}"
            
            # Cache analysis if memory agent available
            if self.memory_agent and analysis and isinstance(analysis, str) and len(analysis) > 50:
                try:
                    paper_hash = MemoryAgent.compute_hash(paper['extracted'])
                    await self.memory_agent.store_analysis(paper['title'], paper_hash, analysis)
                except Exception as cache_error:
                    print(f"   WARNING: Cache storage failed: {str(cache_error)[:100]}")
            
            return f"PAPER {paper_num}: {paper['title']}\n{'='*80}\n{analysis}\n"
            
        except Exception as e:
            error_msg = str(e)[:200]
            print(f"   WARNING: Analysis failed for paper {paper_num}: {error_msg}")
            # Return structured error for better tracking
            return f"PAPER {paper_num}: {paper['title']}\n{'='*80}\nERROR: Analysis failed - {error_msg}\n"
    
    async def analyze_papers(self, papers: List[Dict[str, str]], parallel: bool = True, batch_size: int = 4) -> str | None:
        """Analyze papers with optional parallel processing
        
        Args:
            papers: List of paper dictionaries
            parallel: If True, analyze in parallel. If False, sequential.
            batch_size: Max papers to analyze in parallel at once (lower for LLM stability)
        """
        if not parallel or len(papers) <= 1:
            # Sequential processing
            print(f"\nAnalyzer: Analyzing {len(papers)} papers sequentially...")
            paper_analyses = []
            for i, paper in enumerate(papers, 1):
                analysis = await self._analyze_single_paper(paper, i, len(papers))
                paper_analyses.append(analysis)
        else:
            # Parallel processing with batch control
            print(f"\nAnalyzer: Analyzing {len(papers)} papers in parallel (batch_size={batch_size})...")
            
            # Pre-allocate list to maintain order
            paper_analyses = [None] * len(papers)
            
            for batch_start in range(0, len(papers), batch_size):
                batch_papers = papers[batch_start:batch_start + batch_size]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (len(papers) + batch_size - 1) // batch_size
                
                if total_batches > 1:
                    print(f"   Analyzing batch {batch_num}/{total_batches} ({len(batch_papers)} papers)...")
                
                # Create tasks for this batch with index tracking
                # Each task includes paper metadata for traceability
                tasks = [
                    self._analyze_single_paper(paper, batch_start + i + 1, len(papers)) 
                    for i, paper in enumerate(batch_papers)
                ]
                
                # Execute batch in parallel (order preserved by pre-allocated list)
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Verify batch results maintain paper order
                print(f"      Batch {batch_num} completed: {len(batch_results)} papers processed")
                
                # Store results in correct positions to maintain order
                for i, result in enumerate(batch_results):
                    paper_analyses[batch_start + i] = result
        
        # Handle exceptions and combine results
        valid_analyses = []
        for i, result in enumerate(paper_analyses, 1):
            if isinstance(result, Exception):
                print(f"   ERROR: Paper {i} analysis failed: {str(result)[:100]}")
                valid_analyses.append(f"PAPER {i}: Analysis failed - {str(result)}\n")
            elif result and isinstance(result, str) and len(result.strip()) > 0:
                valid_analyses.append(result)
            else:
                print(f"   WARNING: Paper {i} returned empty analysis")
                valid_analyses.append(f"PAPER {i}: Empty analysis result\n")
        
        if not valid_analyses:
            return None
        
        full_analysis = "\n\n".join(valid_analyses)
        # Clear intermediate results to free memory
        valid_analyses.clear()
        mode = "parallel" if parallel else "sequential"
        print(f"Analyzer: Completed {mode} analysis of {len(papers)} papers")
        return full_analysis


class SynthesizerAgent:
    """Synthesizes insights from analysis"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
    
    def _chunk_analysis(self, analysis: str, max_chars: int = 8000) -> List[str]:
        """Split analysis into manageable chunks to avoid context overflow
        
        Returns chunks with metadata to preserve context across boundaries.
        """
        # Try to split by paper sections first
        papers = analysis.split("PAPER ")
        chunks = []
        current_chunk = ""
        paper_count = 0
        
        for i, paper in enumerate(papers):
            if not paper.strip():
                continue
            
            paper_text = f"PAPER {paper}" if i > 0 else paper
            if i > 0:
                paper_count += 1
            
            # If adding this paper would exceed limit, save current chunk and start new
            if len(current_chunk) + len(paper_text) > max_chars and current_chunk:
                # Add metadata to chunk end
                chunk_end_marker = f"\n[END OF CHUNK - Continued in next chunk]\n"
                chunks.append(current_chunk + chunk_end_marker)
                # Add metadata to next chunk start
                chunk_start_marker = f"[CONTINUED FROM PREVIOUS CHUNK - Paper {paper_count}]\n\n"
                current_chunk = chunk_start_marker + paper_text
            else:
                current_chunk += "\n\n" + paper_text if current_chunk else paper_text
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [analysis[:max_chars]]
    
    async def synthesize(self, analysis: str) -> str | None:
        """Create synthesis of key insights - handles large analysis by chunking
        
        Returns:
            Synthesized string or None if synthesis fails
        """
        if not analysis or len(analysis.strip()) == 0:
            print(f"\nSynthesizer: ERROR - Empty analysis provided")
            return None
        
        print(f"\nSynthesizer: Creating synthesis...")
        
        # Check if analysis is too large for context
        analysis_length = len(analysis)
        print(f"   Analysis size: {analysis_length} characters")
        
        # If analysis is small enough, process directly
        if analysis_length < 8000:
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
            history.add_user_message(
                f"Synthesize these research findings:\n\n{analysis}"
            )
            
            execution_settings = self.chat_service.get_prompt_execution_settings_class()(
                service_id=self.service_id,
                max_tokens=2000,  # Reduced for faster synthesis
                temperature=0.3,
            )
            
            result = await self.chat_service.get_chat_message_contents(
                chat_history=history,
                settings=execution_settings,
                kernel=self.kernel,
            )
            
            synthesis = result[0].content if result else "Synthesis failed"
            print(f"Synthesizer: Synthesis complete")
            return synthesis
        
        # Analysis is large - process in chunks
        print(f"   Analysis is large, processing in chunks...")
        chunks = self._chunk_analysis(analysis, max_chars=8000)
        print(f"   Split into {len(chunks)} chunks")
        
        # Create tasks for parallel chunk synthesis
        async def synthesize_chunk(chunk: str, chunk_num: int, total_chunks: int) -> str:
            """Synthesize a single chunk"""
            print(f"   Synthesizing chunk {chunk_num}/{total_chunks}...")
            
            history = ChatHistory()
            history.add_system_message(
                "You are a research synthesizer. Organize paper analyses preserving ALL details.\n"
                "Keep specific methods, results, metrics, technologies for each paper.\n"
                "This is part of a larger analysis - focus on the papers in this section.\n\n"
                "CRITICAL: Write ONLY the synthesis content - NO metadata, chunk markers, or organizational notes.\n"
                "DO NOT include phrases like 'Chunk X/Y', 'END OF CHUNK', emoji, or special symbols."
            )
            history.add_user_message(
                f"Organize these paper analyses:\n\n{chunk[:8000]}"
            )
            
            execution_settings = self.chat_service.get_prompt_execution_settings_class()(
                service_id=self.service_id,
                max_tokens=1500,  # Reduced for faster chunk processing
                temperature=0.3,
            )
            
            try:
                result = await self.chat_service.get_chat_message_contents(
                    chat_history=history,
                    settings=execution_settings,
                    kernel=self.kernel,
                )
                return result[0].content if result else f"Chunk {chunk_num} failed"
            except Exception as e:
                print(f"   WARNING: Chunk {chunk_num} failed: {str(e)[:100]}")
                return f"Chunk {chunk_num}: Synthesis failed"
        
        # Process chunks sequentially (LLM can only handle one request at a time anyway)
        # Sequential processing prevents request queuing and gives better control
        valid_syntheses = []
        for i, chunk in enumerate(chunks, 1):
            try:
                result = await synthesize_chunk(chunk, i, len(chunks))
                valid_syntheses.append(result)
            except Exception as e:
                print(f"   ERROR: Chunk {i} failed: {str(e)[:100]}")
                valid_syntheses.append(f"Chunk {i}: Failed")
        
        # Combine all chunk syntheses in original order
        if not valid_syntheses:
            print(f"Synthesizer: ERROR - No valid synthesis results")
            return None
        
        synthesis = "\n\n".join(valid_syntheses)
        # Clear intermediate data to free memory
        valid_syntheses.clear()
        
        print(f"Synthesizer: Sequential synthesis complete ({len(chunks)} chunks combined in order)")
        return synthesis


class WriterAgent:
    """Writes and rewrites the research report with chunked processing for large synthesis"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
    
    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting from text to ensure plain text output"""
        import re
        
        # Remove bold/italic markers FIRST (before table processing)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # *italic*
        text = re.sub(r'__(.+?)__', r'\1', text)      # __bold__
        text = re.sub(r'_(.+?)_', r'\1', text)        # _italic_
        
        # Remove markdown headings but keep the text
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # ## Heading
        
        # Clean up table formatting - convert to structured plain text
        lines = text.split('\n')
        cleaned_lines = []
        in_table = False
        table_header = None
        
        for line in lines:
            # Skip table separator lines (---|----|---)
            if re.match(r'^[\s\|\-:]+$', line):
                continue
            
            # Detect table rows (lines with multiple pipes)
            if '|' in line and line.count('|') >= 2:
                # Extract cells, remove empty ones
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                
                if not cells:
                    continue
                
                # First row with cells is the header
                if not in_table:
                    in_table = True
                    table_header = cells
                    # Don't write header, we'll use it for context
                    continue
                
                # Subsequent rows: format as "Header: Value"
                if table_header and len(cells) <= len(table_header):
                    for i, cell in enumerate(cells):
                        if i < len(table_header):
                            cleaned_lines.append(f"  • {table_header[i]}: {cell}")
                    cleaned_lines.append("")  # Blank line between rows
                else:
                    # Fallback: just list the cells
                    cleaned_lines.append("  • " + ", ".join(cells))
            else:
                # Not a table line
                in_table = False
                table_header = None
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up any remaining special characters and formatting symbols
        text = re.sub(r'[`~]+', '', text)  # Remove backticks and tildes
        text = re.sub(r'={3,}', '', text)  # Remove === lines
        text = re.sub(r'-{3,}(?=\s|$)', '', text)  # Remove --- lines (not in words like "state-of-the-art")
        text = text.replace('—', '')  # Remove em-dash
        text = text.replace('–', '')  # Remove en-dash
        # Remove list markers at line start (A. B. C. or 1. 2. 3.)
        text = re.sub(r'^\s*[A-Z]\.\s+', '', text, flags=re.MULTILINE)  # A. B. C.
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # 1. 2. 3.
        
        # Remove non-breaking spaces and other invisible characters
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        text = text.replace('\u200b', '')   # Zero-width space
        text = text.replace('\u2028', '\n') # Line separator
        text = text.replace('\u2029', '\n\n') # Paragraph separator
        
        # Normalize whitespace: replace multiple spaces with single space (except at line start for indentation)
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # Keep leading spaces (indentation), normalize the rest
            leading_spaces = len(line) - len(line.lstrip())
            content = line.strip()
            if content:
                # Replace multiple spaces in content with single space
                content = re.sub(r' {2,}', ' ', content)
                normalized_lines.append(' ' * leading_spaces + content)
            else:
                normalized_lines.append('')
        
        text = '\n'.join(normalized_lines)
        
        # Final cleanup: remove multiple blank lines again
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix ALL CAPS text while preserving section headings
        lines = text.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append(line)
                continue
            
            # Skip lines with bullets or list markers (these are NOT to be converted)
            if stripped.startswith('-') or stripped.startswith('•') or re.match(r'^[A-Z]\.\s', stripped) or re.match(r'^\d+\.\s', stripped):
                fixed_lines.append(line)
                continue
            
            # Check if line is ALL CAPS
            if not stripped.isupper():
                fixed_lines.append(line)
                continue
            
            # Preserve short section headings (< 60 chars, no sentence punctuation)
            # More aggressive: only headings, convert everything else
            if len(stripped) < 60 and not any(char in stripped for char in '.,:;?!'):
                # This is a heading - keep ALL CAPS
                fixed_lines.append(line)
            else:
                # Long ALL CAPS or contains punctuation - convert to normal case
                # Preserve leading whitespace
                leading_space = len(line) - len(line.lstrip())
                
                # Convert to proper sentence case
                # Split by sentence endings
                sentences = re.split(r'([.!?]\s+)', stripped)
                fixed_sentences = []
                for i, part in enumerate(sentences):
                    if i % 2 == 0:  # Actual sentence content (not delimiter)
                        if part:
                            # Capitalize first letter, lowercase rest
                            fixed = part[0].upper() + part[1:].lower() if len(part) > 1 else part.upper()
                            fixed_sentences.append(fixed)
                    else:  # Delimiter (. ! ?)
                        fixed_sentences.append(part)
                
                fixed_line = ''.join(fixed_sentences)
                fixed_lines.append(' ' * leading_space + fixed_line)
        
        text = '\n'.join(fixed_lines)
        
        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        return text.strip()


class WriterAgent:
    """Writes structured synthesis reports from organized synthesis data"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
    
    def _clean_markdown(self, text: str) -> str:
        """Remove ALL markdown/formatting - output is plain text paragraphs only"""
        # First pass already done in main _clean_markdown above
        # This is for double-checking report output
        
        # Remove any remaining markdown
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'__(.+?)__', r'\1', text)      # __bold__
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # *italic*
        text = re.sub(r'_(.+?)_', r'\1', text)        # _italic_
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = text.replace('—', '').replace('–', '-')
        text = re.sub(r'^\s*[->•]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    async def _write_chunk(self, chunk: str, chunk_num: int, total_chunks: int, 
                          previous_content: str = None, is_last: bool = False,
                          all_chunks_summary: str = None) -> str:
        """Write one chunk of the report with cumulative context"""
        history = ChatHistory()
        
        # Determine section based on chunk position
        if chunk_num == 1:
            section_instruction = (
                "Write the OVERVIEW and begin DETAILED PAPER-BY-PAPER ANALYSIS.\n"
                "Overview should be 2-3 paragraphs summarizing what papers are covered.\n"
                "Then write COMPLETE analysis for each paper in this chunk (5-7 paragraphs per paper)."
            )
        elif is_last:
            section_instruction = (
                "Complete DETAILED PAPER-BY-PAPER ANALYSIS for remaining papers (5-7 paragraphs each).\n"
                "Then write INTEGRATED SYNTHESIS AND COMPARISON section (organize thematically).\n"
                "End with CONCLUSIONS (2-3 paragraphs summarizing key takeaways and future directions)."
            )
        else:
            section_instruction = (
                "Continue DETAILED PAPER-BY-PAPER ANALYSIS for papers in this chunk.\n"
                "Write COMPLETE analysis for each paper (5-7 paragraphs per paper).\n"
                "Maintain consistent style and detail level with previous sections."
            )
        
        # Build context from previous chunks
        context_info = ""
        if previous_content and len(previous_content) > 100:
            # Give as much previous content as possible without overflow
            # Reserve space: system(800) + current chunk(2500) + output(8000) = 11300
            # Available for previous: 14200 - 11300 = 2900 tokens ≈ 11600 chars
            max_prev_chars = 11000
            truncated_prev = previous_content[-max_prev_chars:] if len(previous_content) > max_prev_chars else previous_content
            context_info = (
                f"\n\nPREVIOUSLY WRITTEN REPORT SECTIONS (maintain consistency and continue):\n"
                f"{'...[earlier content]...\n\n' if len(previous_content) > max_prev_chars else ''}"
                f"{truncated_prev}\n"
                f"[Continue from here with same style and detail level]"
            )
        
        synthesis_context = ""
        if is_last and all_chunks_summary:
            # For final synthesis, provide synthesis data (truncated to 10k for token safety)
            # This chunk will do final synthesis, so it needs paper data for comparison
            max_synth_chars = 10000  # Balanced: enough detail, safe for context limit
            truncated_synth = all_chunks_summary[:max_synth_chars] if len(all_chunks_summary) > max_synth_chars else all_chunks_summary
            synthesis_context = (
                f"\n\nFULL SYNTHESIS DATA (for comprehensive integrated synthesis section):\n"
                f"{truncated_synth}\n"
                f"{'...[more papers]...\n' if len(all_chunks_summary) > max_synth_chars else ''}"
                f"[Use ALL papers from above to create integrated comparison]"
            )
        
        history.add_system_message(
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
            "Write 5-7 paragraphs per paper. USE EXACT TERMINOLOGY FROM PAPER TEXT.\n"
            "Cover: paper title, authors, year, publication, what paper ACCOMPLISHES, approach and methods USED (describe actual work done), findings MADE (describe actual results/frameworks), scope COVERED.\n"
            "CRITICAL: FOCUS ON WHAT PAPERS DID AND ACCOMPLISHED. Describe actual methods used, actual results found, actual frameworks proposed with full details. For conceptual/review papers, describe frameworks or synthesis in depth. NEVER start with 'no X' or 'lacks Y' - begin with what EXISTS and was DONE.\n\n"
            "FOR INTEGRATED SYNTHESIS (if last chunk):\n"
            "Organize THEMATICALLY, NOT paper-by-paper. USE EXACT TERMS FROM PAPERS. Write subsections on:\n\n"
            "APPROACHES AND METHODS: What papers ACTUALLY DID with full details (e.g., Smith conducted double-blind RCT with 200 subjects using randomized block design, Lee performed 25 grounded theory interviews with iterative thematic coding, Garcia proposed integrated framework synthesizing three theoretical perspectives)\n"
            "CONTEXTS AND SETTINGS: EXACT populations/settings/applications (e.g., Brown studied 200 ICU nurses across 5 hospitals over 18 months, Khan examined 15 dairy farms in Jammu Kashmir tracking production)\n"
            "INSTRUMENTS AND TOOLS: EXACT tools/measures with usage (e.g., Jones used BDI-II 21-item scale with Cronbach alpha 0.89, Zhang implemented TensorFlow 2.5 CNN with ResNet-50 trained on 10k images, Ahmed reviewed 87 sources 2015-2023 using PRISMA)\n"
            "TECHNIQUES AND ANALYSES: What papers APPLIED with outcomes (e.g., Smith applied ANOVA with Tukey post-hoc finding F(2,197)=12.4 p<0.001, Ahmed proposed 4-stage framework with implementation guidelines)\n"
            "FINDINGS AND CONTRIBUTIONS: EXACT results/frameworks (e.g., Smith found significant improvement with large effect d=0.87 and 95% CI [0.62, 1.12], Ahmed proposed comprehensive AI-agriculture framework across 4 stages)\n"
            "SCOPE AND FOCUS: What papers COVERED (e.g., Smith focused on urban teaching hospitals examining nurse interventions over 6 months, Ahmed targeted livestock management emphasizing resource efficiency). Frame positively.\n\n"
            "Use EXACT terms. Write in flowing paragraphs under each theme heading.\n\n"
            "QUALITY GUIDELINES:\n"
            "- POSITIVE FOCUS FIRST: Describe what papers DID, methods USED, results FOUND, frameworks PROPOSED. Lead with accomplishments, not gaps.\n"
            "- Be CONCISE: Don't repeat the same points multiple times. State it once clearly.\n"
            "- BALANCED TONE: Highlight contributions and value prominently. Scope boundaries mentioned constructively at end.\n"
            "- DEEPER COMPARISON: Identify shared themes, complementary approaches, divergent perspectives, synthesis opportunities.\n"
            "- VALUE PROPOSITION: Explain WHY each paper matters - contributions to field, who benefits, practical applications.\n"
            "- STRUCTURED SYNTHESIS: In comparison, organize as: what was done first, similarities, differences, then implications.\n\n"
            "CRITICAL: Write detailed analysis describing what papers ACCOMPLISHED. Be thorough but not repetitive. START with what EXISTS and was DONE, not what's absent."
        )
        
        history.add_user_message(
            f"Synthesis data for this section:\n\n{chunk}"
            f"{context_info}"
            f"{synthesis_context}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Extract content from synthesis data and write as YOUR OWN report\n"
            "- DO NOT copy chunk markers, metadata, or organizational notes from synthesis\n"
            "- DO NOT include phrases like 'Chunk 1/4', 'END OF CHUNK', 'Continued in next chunk'\n"
            "- DO NOT include emoji or special symbols (🧭, ✅, 📌, →, ---)\n"
            "- DO NOT REPEAT the same sentence or paragraph multiple times\n"
            "- VARY your language - avoid repeating 'DOES NOT' or 'THE PAPER DOES NOT' excessively\n"
            "- Write ONLY the final report content - professional, clean, complete\n"
            "- Use EXACT terminology from synthesis but present it as integrated analysis\n\n"
            f"Write exhaustively detailed analysis focusing on what papers DID and ACCOMPLISHED. Include EVERY detail from the synthesis."
        )
        
        # Calculate safe token limit - account for ALL input (chunk + previous + synthesis)
        total_input_chars = len(chunk) + len(truncated_prev if previous_content else "") + len(synthesis_context) + 1000
        estimated_input = total_input_chars // 4
        max_output = min(5000, 14200 - estimated_input - 500)  # Increased to 5000 for detailed reports
        max_output = max(2500, max_output)  # Minimum 2500 tokens
        
        execution_settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id,
            max_tokens=max_output,
            temperature=0.4,
        )
        
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history,
            settings=execution_settings,
            kernel=self.kernel,
        )
        
        chunk_content = result[0].content if result else ""
        return self._clean_markdown(chunk_content)
    
    def _chunk_synthesis(self, synthesis: str, max_chars: int = 10000) -> List[str]:
        """Split synthesis by papers to maintain semantic boundaries
        
        Args:
            synthesis: Full synthesis text
            max_chars: Maximum characters per chunk
            
        Returns:
            List of synthesis chunks, each containing complete paper sections
        """
        # Try to split by "PAPER X:" markers to keep papers together
        import re
        paper_pattern = r'(PAPER \d+:)'
        parts = re.split(paper_pattern, synthesis)
        
        chunks = []
        current_chunk = ""
        
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            
            # If adding this part exceeds limit and we have content, start new chunk
            if len(current_chunk) + len(part) > max_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += part
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # If no PAPER markers found, fall back to simple chunking
        if len(chunks) <= 1 and len(synthesis) > max_chars:
            chunks = []
            for i in range(0, len(synthesis), max_chars):
                chunks.append(synthesis[i:i+max_chars])
        
        return chunks if chunks else [synthesis]
    
    async def _write_chunk(self, chunk: str, chunk_num: int, total_chunks: int, 
                          is_first: bool, is_last: bool, previous_content: str = None,
                          all_chunks_summary: str = None) -> str:
        """Write report section for a synthesis chunk with cumulative context
        
        Args:
            chunk: Synthesis chunk to write about
            chunk_num: Current chunk number (1-indexed)
            total_chunks: Total number of chunks
            is_first: Whether this is the first chunk (include OVERVIEW)
            is_last: Whether this is the last chunk (include SYNTHESIS and CONCLUSIONS)
            previous_content: Summary of previously written content for context
            all_chunks_summary: Summary of all synthesis chunks for final synthesis
        """
        history = ChatHistory()
        
        if is_first and is_last:
            # Single chunk - full report
            section_instruction = (
                "Write a COMPLETE report with ALL sections:\n"
                "1. OVERVIEW\n"
                "2. DETAILED PAPER-BY-PAPER ANALYSIS\n"
                "3. INTEGRATED SYNTHESIS AND COMPARISON\n"
                "4. CONCLUSIONS"
            )
        elif is_first:
            # First chunk - overview and first papers
            section_instruction = (
                "Write the FIRST PART of the report:\n"
                "1. OVERVIEW - Brief introduction to ALL papers in the study\n"
                "2. DETAILED PAPER-BY-PAPER ANALYSIS - Write detailed analysis for papers in this chunk\n"
                "(More papers will be analyzed in subsequent parts)"
            )
        elif is_last:
            # Last chunk - final papers, synthesis, conclusions
            section_instruction = (
                "Write the FINAL PART of the report:\n"
                "1. DETAILED PAPER-BY-PAPER ANALYSIS (CONTINUED) - Complete analysis for remaining papers\n"
                "2. INTEGRATED SYNTHESIS AND COMPARISON - Comprehensive synthesis comparing ALL papers from all chunks\n"
                "3. CONCLUSIONS - Final insights considering ALL papers"
            )
        else:
            # Middle chunk - just more papers
            section_instruction = (
                "Write the MIDDLE PART of the report:\n"
                "DETAILED PAPER-BY-PAPER ANALYSIS (CONTINUED) - Write detailed analysis for papers in this chunk\n"
                "(Maintain consistency with previous papers)"
            )
        
        context_info = ""
        if previous_content:
            # Give as much previous content as possible without overflow
            # Reserve space: system(800) + current chunk(2500) + output(8000) = 11300
            # Available for previous: 14200 - 11300 = 2900 tokens ≈ 11600 chars
            max_prev_chars = 11000
            truncated_prev = previous_content[-max_prev_chars:] if len(previous_content) > max_prev_chars else previous_content
            context_info = (
                f"\n\nPREVIOUSLY WRITTEN REPORT SECTIONS (maintain consistency and continue):\n"
                f"{'...[earlier content]...\n\n' if len(previous_content) > max_prev_chars else ''}"
                f"{truncated_prev}\n"
                f"[Continue from here with same style and detail level]"
            )
        
        synthesis_context = ""
        if is_last and all_chunks_summary:
            # For final synthesis, provide synthesis data (truncated to 10k for token safety)
            # This chunk will do final synthesis, so it needs paper data for comparison
            max_synth_chars = 10000  # Balanced: enough detail, safe for context limit
            truncated_synth = all_chunks_summary[:max_synth_chars] if len(all_chunks_summary) > max_synth_chars else all_chunks_summary
            synthesis_context = (
                f"\n\nFULL SYNTHESIS DATA (for comprehensive integrated synthesis section):\n"
                f"{truncated_synth}\n"
                f"{'...[more papers]...\n' if len(all_chunks_summary) > max_synth_chars else ''}"
                f"[Use ALL papers from above to create integrated comparison]"
            )
        
        history.add_system_message(
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
            "Write 5-7 paragraphs per paper. USE EXACT TERMINOLOGY FROM PAPER TEXT.\n"
            "Cover: paper title, authors, year, publication, what paper ACCOMPLISHES, approach and methods USED (describe actual work), findings MADE (describe actual results/frameworks), scope COVERED.\n"
            "CRITICAL: FOCUS ON WHAT PAPERS DID. Describe actual methods used, actual results found, actual frameworks proposed with full details. For conceptual/review papers, describe frameworks or synthesis components in depth. NEVER start with 'no X' or 'lacks Y' - begin with what EXISTS.\n\n"
            "FOR INTEGRATED SYNTHESIS (if last chunk):\n"
            "Organize THEMATICALLY, NOT paper-by-paper. USE EXACT TERMS FROM PAPERS. Write subsections on:\n\n"
            "APPROACHES AND METHODS: What papers ACTUALLY DID with details (e.g., Smith conducted double-blind RCT with 200 subjects using randomized block design, Lee performed 25 grounded theory interviews with iterative coding, Garcia proposed integrated framework synthesizing three theories)\n"
            "CONTEXTS AND SETTINGS: EXACT populations/settings/applications (e.g., Brown studied 200 ICU nurses across 5 hospitals over 18 months, Khan examined 15 dairy farms in Jammu Kashmir tracking production)\n"
            "INSTRUMENTS AND TOOLS: EXACT tools/measures with usage (e.g., Jones used BDI-II 21-item scale with 0.89 reliability, Zhang implemented TensorFlow 2.5 CNN with ResNet-50 trained on 10k images, Ahmed reviewed 87 sources 2015-2023)\n"
            "TECHNIQUES AND ANALYSES: What papers APPLIED with outcomes (e.g., Smith applied ANOVA with Tukey post-hoc finding F(2,197)=12.4 p<0.001, Ahmed proposed 4-stage framework with implementation guidelines)\n"
            "FINDINGS AND CONTRIBUTIONS: EXACT results/frameworks (e.g., Smith found significant improvement with d=0.87 and 95% CI [0.62, 1.12], Ahmed proposed comprehensive AI-agriculture framework across 4 stages)\n"
            "SCOPE AND FOCUS: What papers COVERED (e.g., Smith focused on urban teaching hospitals examining nurse interventions, Ahmed targeted livestock management emphasizing resource efficiency). Frame positively.\n\n"
            "Use EXACT terms. Write in flowing paragraphs under each theme heading.\n\n"
            "QUALITY GUIDELINES:\n"
            "- POSITIVE FOCUS FIRST: Describe what papers DID, methods USED, results FOUND, frameworks PROPOSED. Lead with accomplishments.\n"
            "- Be CONCISE: Don't repeat the same points multiple times. State it once clearly.\n"
            "- BALANCED TONE: Highlight contributions and value prominently. Scope boundaries mentioned constructively at end.\n"
            "- DEEPER COMPARISON: Identify shared themes, complementary approaches, divergent perspectives, synthesis opportunities.\n"
            "- VALUE PROPOSITION: Explain WHY each paper matters - contributions to field, who benefits, practical applications.\n"
            "- STRUCTURED SYNTHESIS: In comparison, organize as: what was done first, similarities, differences, then implications.\n\n"
            "CRITICAL: Write detailed analysis describing what papers ACCOMPLISHED. Be thorough but not repetitive. START with what EXISTS and was DONE."
        )
        
        history.add_user_message(
            f"Synthesis data for this section:\n\n{chunk}"
            f"{context_info}"
            f"{synthesis_context}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Extract content from synthesis data and write as YOUR OWN report\n"
            "- DO NOT copy chunk markers, metadata, or organizational notes from synthesis\n"
            "- DO NOT include phrases like 'Chunk 1/4', 'END OF CHUNK', 'Continued in next chunk'\n"
            "- DO NOT include emoji or special symbols (🧭, ✅, 📌, →, ---)\n"
            "- DO NOT REPEAT the same sentence or paragraph multiple times\n"
            "- VARY your language - avoid repeating 'DOES NOT' or 'THE PAPER DOES NOT' excessively\n"
            "- Write ONLY the final report content - professional, clean, complete\n"
            "- Use EXACT terminology from synthesis but present it as integrated analysis\n\n"
            f"Write exhaustively detailed analysis focusing on what papers DID and ACCOMPLISHED. Include EVERY detail from the synthesis."
        )
        
        # Calculate safe token limit - account for ALL input (chunk + previous + synthesis)
        total_input_chars = len(chunk) + len(truncated_prev if previous_content else "") + len(synthesis_context) + 1000
        estimated_input = total_input_chars // 4
        max_output = min(5000, 14200 - estimated_input - 500)  # Increased to 5000 for detailed reports
        max_output = max(2500, max_output)  # Minimum 2500 tokens
        
        execution_settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id,
            max_tokens=max_output,
            temperature=0.4,
        )
        
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history,
            settings=execution_settings,
            kernel=self.kernel,
        )
        
        raw_content = result[0].content if result else f"[Chunk {chunk_num} writing failed]"
        return self._clean_markdown(raw_content)
    
    async def write_report(self, synthesis: str, feedback: str = None, feedback_sources: dict = None, 
                          previous_report: str = None) -> str:
        """Write or rewrite report based on synthesis and optional feedback
        
        Args:
            synthesis: Synthesized analysis to write about
            feedback: Combined feedback string
            feedback_sources: Dict with keys 'reviewer', 'chief', 'validator' for tracking
            previous_report: Previous report version (for incremental fixes)
        """
        # Check if synthesis is too large for single pass
        synthesis_size = len(synthesis)
        needs_chunking = synthesis_size > 10000  # > 10k chars needs chunking (reduced from 15k)
        
        if feedback:
            sources = []
            if feedback_sources:
                if feedback_sources.get('reviewer'): sources.append('Reviewer')
                if feedback_sources.get('chief'): sources.append('Chief')
                if feedback_sources.get('validator'): sources.append('Validator')
            source_str = f" ({', '.join(sources)})" if sources else ""
            print(f"\nWriter: Fixing report incrementally based on feedback{source_str}...")
        elif needs_chunking:
            print(f"\nWriter: Writing initial report (chunked mode: {synthesis_size} chars)...")
            # Use chunked writing for large synthesis
            chunks = self._chunk_synthesis(synthesis, max_chars=10000)
            print(f"Writer: Split into {len(chunks)} chunks for detailed writing...")
            
            # Keep all synthesis chunks available (full detail, no truncation)
            full_synthesis = synthesis
            
            report_parts = []
            accumulated_report = ""  # Full text of all previous parts
            
            for i, chunk in enumerate(chunks, 1):
                is_first = (i == 1)
                is_last = (i == len(chunks))
                print(f"  Writing chunk {i}/{len(chunks)}...")
                
                # Pass accumulated report (all previous parts) and full synthesis
                part = await self._write_chunk(
                    chunk, i, len(chunks), is_first, is_last,
                    previous_content=accumulated_report if accumulated_report else None,
                    all_chunks_summary=full_synthesis if is_last else None
                )
                report_parts.append(part)
                
                # Accumulate full report text for next chunk
                if accumulated_report:
                    accumulated_report += "\n\n" + part
                else:
                    accumulated_report = part
            
            # Combine all parts
            full_report = "\n\n".join(report_parts)
            print(f"Writer: Chunked report completed ({len(chunks)} parts combined)")
            return full_report
        else:
            print(f"\nWriter: Writing initial report (single pass: {synthesis_size} chars)...")
        
        history = ChatHistory()
        history.add_system_message(
            "You are a professional academic report writer creating a detailed research synthesis.\n\n"
            "REQUIRED STRUCTURE (use plain text ALL CAPS headings):\n\n"
            "1. OVERVIEW\n"
            "   Brief introduction to all papers and their domains.\n\n"
            "2. DETAILED PAPER-BY-PAPER ANALYSIS\n"
            "   Write a SEPARATE comprehensive subsection for EACH paper:\n"
            "   - Paper title, authors, year\n"
            "   - Research focus and objectives\n"
            "   - What paper does (USE EXACT TERMS describing paper's work - empirical study, conceptual proposal, review, critique)\n"
            "   - Approach/methods (EXACT names/descriptions - IF empirical: sample, measures, tests | IF conceptual: describes framework, proposes model | IF review: reviews X studies)\n"
            "   - Findings/contributions (EXACT results/arguments/conclusions - statistics, framework names, insights, recommendations)\n"
            "   - Scope & Boundaries (describe coverage: e.g., 'theoretical framework without empirical validation', 'pilot study with 50 participants', 'proof-of-concept stage')\n"
            "   (Repeat for ALL papers - each gets 3-5 paragraphs)\n\n"
            "3. INTEGRATED SYNTHESIS AND COMPARISON\n"
            "   Synthesize THEMATICALLY across ALL papers. EXTRACT EXACT TERMINOLOGY. Write 6 subsections:\n\n"
            "   A. Approaches: What each paper does (e.g., Brown: quasi-experimental 300 subjects, Lee: phenomenological interviews, Khan: proposes conceptual model)\n"
            "   B. Contexts: EXACT populations/settings/applications (e.g., Smith: 150 diabetes patients clinic, Ahmed: livestock sector Kashmir)\n"
            "   C. Instruments: EXACT tools/measures (e.g., Chen: PyTorch 1.9, Jones: STAI Form Y, Garcia: theoretical analysis)\n"
            "   D. Techniques: What each paper uses/proposes (e.g., Zhang: k-fold cross-validation, Ahmed: circular economy integration, Lee: reflexive thematic analysis)\n"
            "   E. Findings: EXACT results/conclusions/contributions (e.g., Zhang: 94.2% F1-score, Ahmed: framework combining AI with mixed farming, Lee: 4 emergent themes)\n"
            "   F. Future Directions: Research scope and opportunities (e.g., Zhang: lab validation stage - ready for field testing, Ahmed: theoretical framework - awaiting empirical validation, Smith: pilot study with 50 participants - suitable for larger replication)\n\n"
            "   Format: ALWAYS describe what paper does. Use EXACT wording. Even if conceptual, describe the work done. Organize by THEMES.\n\n"
            "4. CONCLUSIONS\n"
            "   Final insights and implications.\n\n"
            "CRITICAL FORMATTING RULES - YOU MUST FOLLOW THESE:\n"
            "✗ FORBIDDEN: Do NOT use **, **, |, ---, ###, ===, `, ~~~, or any markdown\n"
            "✗ FORBIDDEN: Do NOT create ASCII tables with pipes and dashes\n"
            "✗ FORBIDDEN: Do NOT use asterisks for bold or emphasis\n\n"
            "✓ REQUIRED: Write in plain text (no markdown)\n"
            "✓ REQUIRED: Section 2 has individual analysis for each paper\n"
            "✓ REQUIRED: Section 3 integrates and compares all papers together\n"
            "✓ REQUIRED: Be SPECIFIC with methods, results, technologies\n\n"
            "EXAMPLE (POSITIVE LANGUAGE): Zhang 2023 used Random Forest (100 trees) for feed optimization, achieving 15% cost reduction on 50 farms. "
            "Biswas 2023 employed ResNet-50 CNN for disease detection, reaching 92% accuracy on 10k images.\n\n"
            "SYNTHESIS EXAMPLE: Four papers used CNNs (Biswas: ResNet-50 92%, Li: custom 95.3%, Patel: EfficientNet 91.7%). "
            "Healthcare dominated with 3 papers (Zhang, Li, Patel) achieving >90% accuracy.\n\n"
        )
        
        if feedback and previous_report:
            # Incremental fix mode - focused improvements
            prompt = (
                f"TASK: Fix specific issues in this research report based on feedback.\n\n"
                f"CURRENT REPORT (first 8000 chars):\n{previous_report[:8000]}\n\n"
                f"SYNTHESIS (reference for details):\n{synthesis[:6000]}\n\n"
                f"FEEDBACK:\n{feedback}\n\n"
                f"CRITICAL INSTRUCTIONS:\n"
                f"- Extract improvements from feedback and write as YOUR OWN corrections\n"
                f"- DO NOT copy feedback section markers (=== REVIEWER FEEDBACK ===)\n"
                f"- DO NOT include phrases like 'Chunk 1/4', 'END OF CHUNK', 'Continued in next chunk'\n"
                f"- DO NOT include emoji or special symbols (🧭, ✅, 📌, →, ---)\n"
                f"- DO NOT REPEAT the same sentence or paragraph multiple times\n"
                f"- VARY your language - avoid repeating 'DOES NOT' or 'THE PAPER DOES NOT' excessively\n"
                f"- Write ONLY the final corrected report - professional, clean, complete\n\n"
                f"FIX INSTRUCTIONS:\n"
                f"1. Read feedback carefully - it tells you exactly what to add or fix\n"
                f"2. Add missing details where specified (e.g., platform names, framework versions)\n"
                f"3. Fix vague descriptions with specific details from synthesis\n"
                f"4. Keep all well-written sections unchanged\n"
                f"5. Maintain report structure and section order\n\n"
                f"FORMAT REQUIREMENTS:\n"
                f"- Write in PLAIN TEXT ONLY (no markdown)\n"
                f"- Use ALL CAPS for section headings\n"
                f"- NO symbols: **, __, *, |, ---, ###, `, ~~~\n"
                f"- Clean paragraph spacing (double newline between sections)\n"
                f"- NO special characters in whitespace\n\n"
                f"OUTPUT: Complete corrected report addressing all feedback points."
            )
        elif feedback:
            # Fallback: no previous report, write fresh with feedback in mind
            prompt = (
                f"Write a complete research synthesis report addressing these feedback points:\n\n"
                f"FEEDBACK:\n{feedback}\n\n"
                f"SYNTHESIS DATA:\n{synthesis[:8000]}\n\n"
                f"CRITICAL INSTRUCTIONS:\n"
                f"- Extract guidance from feedback and write as YOUR OWN report\n"
                f"- DO NOT copy feedback section markers (=== REVIEWER FEEDBACK ===)\n"
                f"- DO NOT include phrases like 'Chunk 1/4', 'END OF CHUNK'\n"
                f"- DO NOT include emoji or special symbols (🧭, ✅, 📌, →, ---)\n"
                f"- DO NOT REPEAT the same sentence or paragraph multiple times\n"
                f"- VARY your language - avoid repeating 'DOES NOT' or 'THE PAPER DOES NOT' excessively\n"
                f"- Write ONLY the final report - professional, clean, complete\n\n"
                f"STRUCTURE:\n"
                f"1. OVERVIEW\n"
                f"2. DETAILED PAPER-BY-PAPER ANALYSIS\n"
                f"   For EACH paper include:\n"
                f"   - Methods with exact parameters (Random Forest 100 trees, ResNet-50)\n"
                f"   - Technologies and tools with versions (TensorFlow 2.x, PyTorch 1.x)\n"
                f"   - Platforms (AWS SageMaker, Google Colab, Azure ML)\n"
                f"   - Datasets with sizes and sources\n"
                f"   - Results with exact metrics (95.3% accuracy, 18% improvement)\n"
                f"3. INTEGRATED SYNTHESIS (thematic organization)\n"
                f"4. CONCLUSIONS\n\n"
                f"FORMAT REQUIREMENTS:\n"
                f"- PLAIN TEXT ONLY (no markdown symbols)\n"
                f"- ALL CAPS for section headings\n"
                f"- Clean paragraph spacing\n"
                f"- NO **, __, *, |, ---, ### symbols\n\n"
                f"Include EVERY detail from synthesis. Address all feedback points."
            )
        else:
            prompt = (
                f"Write an EXHAUSTIVELY DETAILED research synthesis report in PLAIN TEXT.\n\n"
                f"ABSOLUTE REQUIREMENT: Include EVERY SINGLE detail from the synthesis. Use EXACT terminology from papers. Do NOT skip or paraphrase.\n\n"
                f"STRUCTURE:\n"
                f"1. OVERVIEW - Brief introduction to all papers\n\n"
                f"2. DETAILED PAPER-BY-PAPER ANALYSIS\n"
                f"   For EACH paper, write comprehensive analysis (5-7 paragraphs) using EXACT TERMS from synthesis:\n"
                f"   - Paper title (exact), ALL authors (exact names), year, publication\n"
                f"   - Paper type and objectives\n"
                f"   - What paper does (USE EXACT TERMS describing paper's work - empirical study, conceptual proposal, review, critique)\n"
                f"   - Approach/methods (EXACT names/descriptions - IF empirical: sample, measures, tests | IF conceptual: describes framework, proposes model | IF review: reviews X studies)\n"
                f"   - Findings/contributions (EXACT results/arguments/conclusions - statistics, framework names, insights, recommendations)\n"
                f"   - Limitations (specific about what's missing - e.g., 'no empirical testing', 'small sample', 'lacks deployment')\n"
                f"   (Repeat for ALL papers - each gets 3-5 paragraphs)\n\n"
                f"3. INTEGRATED SYNTHESIS AND COMPARISON\n"
                f"   Synthesize THEMATICALLY across ALL papers. EXTRACT EXACT TERMINOLOGY. Write subsections on:\n\n"
                f"   APPROACHES: What each paper does (e.g., Brown conducted quasi-experimental study with 300 subjects, Khan proposed conceptual model)\n"
                f"   CONTEXTS: EXACT populations/settings/applications (e.g., Smith studied 150 diabetes patients in clinic setting, Ahmed examined livestock sector in Kashmir)\n"
                f"   INSTRUMENTS: EXACT tools/measures (e.g., Chen used PyTorch version 1.9, Jones used STAI Form Y, Garcia performed theoretical analysis)\n"
                f"   TECHNIQUES: What each paper uses or proposes (e.g., Zhang applied k-fold cross-validation, Ahmed proposed circular economy integration, Lee used reflexive thematic analysis)\n"
                f"   FINDINGS: EXACT results/conclusions/contributions (e.g., Zhang achieved 94.2% F1-score, Ahmed developed framework combining AI with mixed farming, Lee identified 4 emergent themes)\n"
                f"   GAPS: Specific limitations (e.g., Zhang lacked real-world validation, Ahmed was purely conceptual with no empirical data, Smith used small homogeneous sample)\n\n"
                f"   ALWAYS describe what paper does. Use EXACT wording. Write in flowing paragraphs under theme headings.\n\n"
                f"4. CONCLUSIONS - Comprehensive final insights\n\n"
                f"FORMAT RULES - MUST FOLLOW:\n"
                f"- Write ONLY in plain paragraphs\n"
                f"- Section headings in ALL CAPS with blank line before and after\n"
                f"- NEVER use symbols: ** __ * | - > ### ` ~ = bullets dashes\n"
                f"- NO numbered lists (1. 2. 3.)\n"
                f"- NO bullet points or dashes at line start\n"
                f"- Write like a simple, clean text document\n\n"
                f"QUALITY GUIDELINES:\n"
                f"- Be CONCISE: State points once clearly, avoid repetition\n"
                f"- BALANCED TONE: Highlight contributions AND limitations constructively\n"
                f"- DEEPER COMPARISON: Identify shared themes, complementary approaches, divergent perspectives\n"
                f"- VALUE PROPOSITION: Explain WHY each paper matters and its practical value\n"
                f"- STRUCTURED SYNTHESIS: Organize comparison as similarities, differences, then implications\n\n"
                f"CRITICAL INSTRUCTIONS:\n"
                f"- Extract content from synthesis data and write as YOUR OWN report\n"
                f"- DO NOT copy chunk markers, metadata, or organizational notes from synthesis\n"
                f"- DO NOT include phrases like 'Chunk 1/4', 'END OF CHUNK', 'Continued in next chunk'\n"
                f"- DO NOT include emoji or special symbols (🧭, ✅, 📌, →, ---)\n"
                f"- DO NOT REPEAT the same sentence or paragraph multiple times\n"
                f"- VARY your language - avoid repeating 'DOES NOT' or 'THE PAPER DOES NOT' excessively\n"
                f"- Write ONLY the final report content - professional, clean, complete\n"
                f"- Use EXACT terminology from synthesis but present it as integrated analysis\n\n"
                f"Synthesis data (extract content, ignore metadata):\n{synthesis[:20000]}"
            )
        
        history.add_user_message(prompt)
        
        # Dynamic token calculation to prevent context overflow
        # Context limit: 14200 tokens (~4 chars per token)
        # Estimate input tokens from prompt length
        system_msg = history.messages[0].content if history.messages else ""
        user_msg = prompt
        estimated_input_chars = len(system_msg) + len(user_msg)
        estimated_input_tokens = estimated_input_chars // 4  # Rough estimate
        
        # Leave safety margin and calculate max output tokens
        context_limit = 14200
        safety_margin = 500
        max_output_tokens = min(8000, context_limit - estimated_input_tokens - safety_margin)
        max_output_tokens = max(4000, max_output_tokens)  # At least 4000 tokens
        
        print(f"Writer: Estimated input ~{estimated_input_tokens} tokens, max output {max_output_tokens} tokens")
        
        execution_settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id,
            max_tokens=max_output_tokens,  # Dynamic based on input size
            temperature=0.4,   # Lower temp for accuracy
        )
        
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history,
            settings=execution_settings,
            kernel=self.kernel,
        )
        
        raw_report = result[0].content if result else "Report writing failed"
        report = self._clean_markdown(raw_report)
        print(f"Writer: Report {'rewritten' if feedback else 'completed'} (markdown cleaned)")
        return report


class ReviewerAgent:
    """First-level review: checks technical quality"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
    
    async def review(self, report: str) -> Dict[str, Any]:
        """Review report for quality and completeness"""
        print(f"\nReviewer: Reviewing report...")
        
        history = ChatHistory()
        history.add_system_message(
            "You are an EXTREMELY STRICT reviewer checking for COMPLETE DETAIL. Your job is to catch ANY missing information.\n\n"
            "Check EVERY paper section individually:\n\n"
            "FOR EACH PAPER CHECK:\n"
            "1. Are ALL methods listed by EXACT name? (Random Forest, XGBoost, ResNet-50, etc)\n"
            "2. Are method parameters included? (number of trees, learning rate, layers, etc)\n"
            "3. Are ALL results listed with EXACT metrics? (95.3% accuracy, not 'high accuracy')\n"
            "4. Are ALL technologies/tools/platforms mentioned by name?\n"
            "5. Are dataset details complete? (exact size, source, features)\n"
            "6. Are experiment details complete? (what was tested, how)\n"
            "7. Are ALL comparisons with baselines included?\n"
            "8. Can a reader fully understand this paper WITHOUT reading the original?\n\n"
            "GENERAL CHECKS:\n"
            "9. Each paper has 5+ paragraphs of detailed content?\n"
            "10. Integrated synthesis section compares ALL papers with specific details?\n"
            "11. No vague language like 'various methods', 'good results', 'machine learning'?\n"
            "12. No markdown symbols (**, |, ---, ###)?\n"
            "13. Report is 20+ pages of dense, detailed content?\n"
            "14. NO EXCESSIVE REPETITION - same sentence/paragraph repeated multiple times?\n"
            "15. VARIED LANGUAGE - not overusing 'DOES NOT', 'THE PAPER DOES NOT', 'LACKS'?\n\n"
            "REJECT if you find:\n"
            "- ANY generic/vague descriptions\n"
            "- Missing specific metrics or method names\n"
            "- Papers described in less than 4-5 paragraphs\n"
            "- Any detail that seems incomplete\n"
            "- Same sentences or paragraphs repeated multiple times\n"
            "- Excessive use of negative phrases like 'DOES NOT' (more than 5-10 times)\n\n"
            "Respond: APPROVE or REJECT with LIST of specific missing details."
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
    
    async def final_review(self, report: str) -> Dict[str, Any]:
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
            "7. Synthesis integrates findings across papers effectively\n"
            "8. NO excessive repetition - varied language throughout\n"
            "9. Balanced tone - not overusing negative phrases like 'DOES NOT'\n\n"
            "REJECT if report is too brief, lacks comprehensive detail, or has excessive repetition.\n"
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


class ValidationAgent:
    """Validates report against synthesis to prevent hallucinations"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
    
    async def validate(self, report: str, synthesis: str) -> Dict[str, Any]:
        """Verify report content matches synthesis - catch hallucinations"""
        print(f"\nValidation Agent: Checking for fabricated information...")
        
        history = ChatHistory()
        history.add_system_message(
            "You are a fact-checking validator. Compare the REPORT against the SYNTHESIS.\n\n"
            "Your ONLY job: Find any information in the report that is NOT in the synthesis.\n\n"
            "CHECK FOR:\n"
            "1. Methods/algorithms named in report but not in synthesis\n"
            "2. Metrics/numbers in report but not in synthesis\n"
            "3. Technologies/tools in report but not in synthesis\n"
            "4. Paper details in report but not in synthesis\n"
            "5. Any claims, results, or findings not supported by synthesis\n\n"
            "CRITICAL: Flag ANY fabricated or hallucinated content.\n\n"
            "If report only contains information from synthesis: VALID\n"
            "If report contains ANY unsupported information: INVALID\n\n"
            "Respond: VALID or INVALID with list of fabricated items."
        )
        history.add_user_message(
            f"SYNTHESIS (source of truth):\n{synthesis[:8000]}\n\n"
            f"REPORT (check against synthesis):\n{report[:8000]}\n\n"
            f"Does report contain ANY information not in synthesis?"
        )
        
        execution_settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id,
            max_tokens=800,
            temperature=0.1,
        )
        
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history,
            settings=execution_settings,
            kernel=self.kernel,
        )
        
        validation = result[0].content if result else "Validation failed"
        is_valid = "VALID" in validation.upper() and "INVALID" not in validation.upper()
        
        print(f"Validation Agent: {'VALID - No hallucinations detected' if is_valid else 'INVALID - Hallucinations found'}")
        
        return {
            "valid": is_valid,
            "feedback": validation,
            "issues_found": [] if is_valid else validation.split('\n')
        }


class MetricsExtractorAgent:
    """Extracts all numerical metrics and creates comparison table"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)
    
    async def extract_metrics(self, analysis: str) -> str:
        """Extract all metrics from papers and create comparison summary"""
        print(f"\nMetrics Extractor: Extracting numerical data...")
        
        history = ChatHistory()
        history.add_system_message(
            "You are a metrics extraction specialist. Extract ALL numerical data from papers.\n\n"
            "For EACH paper, extract:\n"
            "1. All accuracy/precision/recall/F1 scores\n"
            "2. All percentage improvements\n"
            "3. All cost reductions or efficiency gains\n"
            "4. Dataset sizes and characteristics\n"
            "5. Model parameters (trees, layers, learning rates, etc)\n"
            "6. Processing times, computational costs\n"
            "7. Any other numerical results\n\n"
            "Format as plain text summary (NO markdown, NO tables with pipes):\n\n"
            "METRICS SUMMARY\n\n"
            "Paper 1 - Title\n"
            "- Accuracy: 95.3%\n"
            "- Precision: 92.1%\n"
            "- Dataset: 10,000 samples\n"
            "- Model: Random Forest (100 trees, max_depth=10)\n\n"
            "Paper 2 - Title\n"
            "...\n\n"
            "Write plain text format. Be thorough - include EVERY metric."
        )
        history.add_user_message(
            f"Extract all metrics from these paper analyses:\n\n{analysis[:10000]}"
        )
        
        execution_settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id,
            max_tokens=2000,
            temperature=0.1,
        )
        
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history,
            settings=execution_settings,
            kernel=self.kernel,
        )
        
        metrics = result[0].content if result else "Metrics extraction failed"
        print(f"Metrics Extractor: Extraction complete")
        return metrics


class OrderVerificationAgent:
    """Verifies paper order preservation across pipeline stages"""
    
    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
    
    async def verify_order(self, original_paths: List[str], processed_data: List, 
                          stage_name: str) -> Dict:
        """Verify that processed data maintains original paper order
        
        Args:
            original_paths: Original list of paper file paths
            processed_data: Processed data (papers/analysis) to verify
            stage_name: Name of the pipeline stage for logging
        
        Returns:
            Dict with 'verified' (bool), 'mismatches' (list), 'details' (str)
        """
        print(f"\n🔍 OrderVerification: Checking {stage_name}...")
        
        if len(original_paths) != len(processed_data):
            return {
                "verified": False,
                "mismatches": ["Length mismatch"],
                "details": f"Expected {len(original_paths)} items, got {len(processed_data)}"
            }
        
        mismatches = []
        for i, (path, data) in enumerate(zip(original_paths, processed_data)):
            # Extract paper identifier from original path
            expected_name = Path(path).stem.lower()
            
            # Check if data contains the expected paper reference
            data_str = str(data).lower() if data else ""
            
            # Robust verification: check multiple markers
            # 1. Filename appears in data
            # 2. Paper position marker appears (PAPER X:)
            filename_found = expected_name in data_str
            position_marker = f"paper {i+1}" in data_str or f"paper_{i+1}" in data_str
            
            if not (filename_found or position_marker):
                mismatches.append(f"Position {i}: Expected '{Path(path).name}' (paper {i+1}) not found in data")
        
        if mismatches:
            details = f"Found {len(mismatches)} order mismatches:\n" + "\n".join(mismatches[:5])
            print(f"   ❌ Order verification FAILED: {len(mismatches)} mismatches")
            return {
                "verified": False,
                "mismatches": mismatches,
                "details": details
            }
        
        print(f"   ✅ Order verified: All {len(original_paths)} items in correct order")
        return {
            "verified": True,
            "mismatches": [],
            "details": f"All {len(original_paths)} items verified in correct order"
        }


class MemoryAgent:
    """Caches paper analyses to avoid reprocessing"""
    
    def __init__(self, cache_file: str = "analysis_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        self.lock = None  # Will be initialized when first async method is called
    
    def __del__(self):
        """Cleanup on destruction - ensure cache is saved"""
        try:
            if hasattr(self, 'cache') and self.cache:
                self._save_cache()
        except:
            pass  # Ignore errors during cleanup
    
    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    print(f"\nMemory Agent: Loaded cache from {self.cache_file}")
                    return json.load(f)
            except Exception as e:
                print(f"\nMemory Agent: Cache load error: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk (with atomic write)"""
        try:
            # Write to temporary file first (atomic operation)
            temp_file = self.cache_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
            # Atomic rename
            temp_file.replace(self.cache_file)
            print(f"Memory Agent: Cache saved to {self.cache_file}")
        except Exception as e:
            print(f"Memory Agent: Cache save error: {e}")
            # Clean up temp file if it exists
            try:
                if 'temp_file' in locals() and temp_file.exists():
                    temp_file.unlink()
            except:
                pass
    
    async def get_analysis(self, paper_title: str, paper_hash: str) -> Dict:
        """Get cached analysis for a paper (thread-safe)"""
        # Lazy initialization of lock (must be created in async context)
        if self.lock is None:
            self.lock = asyncio.Lock()
        
        async with self.lock:
            key = f"{paper_title}_{paper_hash}"
            if key in self.cache:
                print(f"Memory Agent: Using cached analysis for {paper_title[:50]}")
                return self.cache[key]
            return None
    
    async def store_analysis(self, paper_title: str, paper_hash: str, analysis: str):
        """Store analysis in cache (thread-safe)"""
        # Lazy initialization of lock (must be created in async context)
        if self.lock is None:
            self.lock = asyncio.Lock()
        
        async with self.lock:
            key = f"{paper_title}_{paper_hash}"
            self.cache[key] = {
                "paper_title": paper_title,
                "paper_hash": paper_hash,
                "analysis": analysis,
                "cached_at": datetime.now().isoformat()
            }
            self._save_cache()
            print(f"Memory Agent: Cached analysis for {paper_title[:50]}")
    
    async def clear_cache(self):
        """Clear all cached data (thread-safe)"""
        # Lazy initialization of lock (must be created in async context)
        if self.lock is None:
            self.lock = asyncio.Lock()
        
        async with self.lock:
            self.cache = {}
            self._save_cache()
            print("Memory Agent: Cache cleared")
    
    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute hash of paper content for cache validation"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()


class ProgressTracker:
    """Tracks detailed progress through the synthesis pipeline (thread-safe)"""
    
    def __init__(self, log_file: str = "synthesis_progress.json"):
        self.log_file = Path(log_file)
        self.lock = asyncio.Lock()  # Thread-safe logging
        self.progress = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now().isoformat(),
            "stages": [],
            "papers": [],
            "metrics": {},
            "order_verifications": []
        }
    
    def __del__(self):
        """Cleanup on destruction - ensure progress is saved"""
        try:
            if hasattr(self, 'progress') and self.progress:
                self._save_progress()
        except:
            pass  # Ignore errors during cleanup
    
    async def log_stage(self, stage_name: str, status: str, details: Dict = None):
        """Log progress of a pipeline stage (thread-safe)"""
        async with self.lock:
            stage_entry = {
                "stage": stage_name,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "details": details or {}
            }
            self.progress["stages"].append(stage_entry)
            self._save_progress()
            print(f"Progress Tracker: {stage_name} - {status}")
    
    async def log_paper(self, paper_title: str, status: str, details: Dict = None):
        """Log progress of individual paper processing (thread-safe)"""
        async with self.lock:
            paper_entry = {
                "paper": paper_title,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "details": details or {}
            }
            self.progress["papers"].append(paper_entry)
            self._save_progress()
    
    async def log_metric(self, metric_name: str, value: any):
        """Log a metric (token counts, timing, etc) (thread-safe)"""
        async with self.lock:
            self.progress["metrics"][metric_name] = value
            self._save_progress()
    
    async def log_order_verification(self, stage_name: str, verification_result: dict):
        """Log order verification result (thread-safe)"""
        async with self.lock:
            entry = {
                "stage": stage_name,
                "verified": verification_result["verified"],
                "timestamp": datetime.now().isoformat(),
                "details": verification_result["details"]
            }
            if verification_result["mismatches"]:
                entry["mismatches"] = verification_result["mismatches"][:10]  # First 10
            self.progress["order_verifications"].append(entry)
            self._save_progress()
    
    def _save_progress(self):
        """Save progress to disk (atomic write to prevent corruption)"""
        try:
            # Write to temp file first
            temp_file = self.log_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, indent=2)
            # Atomic rename
            temp_file.replace(self.log_file)
        except Exception as e:
            print(f"Progress Tracker: Save error: {e}")
            # Clean up temp file
            try:
                if 'temp_file' in locals() and temp_file.exists():
                    temp_file.unlink()
            except:
                pass
    
    def get_summary(self) -> str:
        """Get human-readable progress summary"""
        completed_stages = sum(1 for s in self.progress["stages"] if s["status"] == "completed")
        failed_stages = sum(1 for s in self.progress["stages"] if s["status"] == "failed")
        total_papers = len(self.progress["papers"])
        order_verified = all(v["verified"] for v in self.progress["order_verifications"]) if self.progress["order_verifications"] else True
        
        return (
            f"Progress Summary:\n"
            f"- Session: {self.progress['session_id']}\n"
            f"- Stages: {completed_stages} completed, {failed_stages} failed\n"
            f"- Papers: {total_papers} processed\n"
            f"- Metrics: {len(self.progress['metrics'])} tracked\n"
            f"- Order Verifications: {len(self.progress['order_verifications'])} checks, {'✅ All passed' if order_verified else '❌ Some failed'}\n"
        )


def save_report_as_word(report_text: str, output_path: Path):
    """Save report as formatted Word document with intelligent section detection"""
    if not DOCX_AVAILABLE:
        return
    
    doc = Document()
    
    # Add title
    title = doc.add_heading('Research Paper Synthesis Report', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.font.color.rgb = RGBColor(0, 51, 102)  # Dark blue
        run.font.size = Pt(18)
        run.font.bold = True
    
    # Add timestamp
    timestamp_para = doc.add_paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    timestamp_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in timestamp_para.runs:
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(128, 128, 128)
    
    doc.add_paragraph()  # Spacing
    
    # Set default font
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)
    
    # Process text line by line for better control
    lines = report_text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Detect ALL CAPS headings (main sections and paper titles)
        # ALL CAPS line with < 120 chars and no lowercase letters = heading
        if line.isupper() and len(line) < 120 and not any(c.islower() for c in line):
            # Determine heading level
            if any(keyword in line for keyword in ['OVERVIEW', 'DETAILED PAPER-BY-PAPER ANALYSIS', 
                                                    'INTEGRATED SYNTHESIS', 'COMPARISON', 'CONCLUSIONS']):
                # Major section heading (Level 1)
                h = doc.add_heading(line, level=1)
                h.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in h.runs:
                    run.font.size = Pt(16)
                    run.font.color.rgb = RGBColor(0, 51, 102)
                    run.font.bold = True
            elif line.startswith('PAPER ') and ':' in line:
                # Paper heading (Level 2)
                h = doc.add_heading(line, level=2)
                h.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in h.runs:
                    run.font.size = Pt(14)
                    run.font.color.rgb = RGBColor(0, 76, 153)
                    run.font.bold = True
            else:
                # Subsection heading (Level 3)
                h = doc.add_heading(line, level=3)
                h.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in h.runs:
                    run.font.size = Pt(12)
                    run.font.color.rgb = RGBColor(51, 102, 153)
                    run.font.bold = True
            
            i += 1
            continue
        
        # Regular paragraph - collect all consecutive non-heading lines
        para_lines = []
        while i < len(lines):
            current_line = lines[i].strip()
            
            # Stop if we hit an empty line followed by a heading or just a heading
            if not current_line:
                # Check if next line is a heading
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and next_line.isupper() and len(next_line) < 120:
                        break  # Stop before empty line that precedes heading
                i += 1
                continue
            
            # Stop if this line is a heading
            if current_line.isupper() and len(current_line) < 120 and not any(c.islower() for c in current_line):
                break
            
            para_lines.append(current_line)
            i += 1
        
        # Create paragraph from collected lines
        if para_lines:
            para_text = ' '.join(para_lines)
            
            # Handle bullet points or lists
            if para_text.startswith('•') or para_text.startswith('-') or para_text.startswith('*'):
                # Bullet list item
                cleaned_text = para_text[1:].strip() if para_text[0] in ['•', '-', '*'] else para_text
                p = doc.add_paragraph(cleaned_text, style='List Bullet')
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.space_after = Pt(4)
            else:
                # Normal paragraph
                p = doc.add_paragraph(para_text)
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.line_spacing = 1.2
                p.paragraph_format.space_after = Pt(8)
                p.paragraph_format.space_before = Pt(2)
                # Only indent if paragraph is long enough (not short headings/titles)
                if len(para_text) > 80:
                    p.paragraph_format.first_line_indent = Inches(0.0)  # No indent for academic style
                
                # Make text more readable
                for run in p.runs:
                    run.font.size = Pt(11)
                    run.font.name = 'Calibri'
    
    # Add footer with page numbers
    section = doc.sections[0]
    footer = section.footer
    footer_para = footer.paragraphs[0]
    footer_para.text = "Page "
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Save document
    doc.save(str(output_path))


class ResearchSynthesisBehaviorTree:
    """Orchestrates the entire multi-agent workflow with enhanced validation and tracking"""
    
    def __init__(self, reader, analyzer, synthesizer, writer, reviewer, chief_reviewer,
                 validator, metrics_extractor, memory_agent, progress_tracker, order_verifier):
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
    
    async def run(self, paper_paths: List[str], max_iterations: int = 3, 
                  parallel_processing: bool = False, read_batch_size: int = 6, 
                  analyze_batch_size: int = 4) -> Dict:
        """Execute the full research synthesis pipeline with enhanced features
        
        Args:
            paper_paths: List of paths to research papers
            max_iterations: Maximum writing iterations before giving up
            parallel_processing: Enable parallel processing (set False for single-threaded LLMs like LM Studio)
            read_batch_size: Papers to read in parallel at once (only used if parallel_processing=True)
            analyze_batch_size: Papers to analyze in parallel at once (only used if parallel_processing=True)
        
        Note: LM Studio and most local LLMs process requests sequentially (queue them).
              Sequential mode (parallel_processing=False) is recommended for better control.
        """
        print("\n" + "="*70)
        mode = "PARALLEL" if parallel_processing else "SEQUENTIAL"
        print(f"RESEARCH SYNTHESIS BEHAVIOR TREE ({mode} MODE)")
        print("="*70)
        if parallel_processing:
            print(f"Parallelization: Read batch={read_batch_size}, Analyze batch={analyze_batch_size}")
        
        await self.progress_tracker.log_stage("pipeline_start", "started", 
                                       {"paper_count": len(paper_paths), 
                                        "parallel": parallel_processing})
        
        # Phase 1: Read papers
        await self.progress_tracker.log_stage("reading", "in_progress")
        papers = await self.reader.read_multiple_papers(
            paper_paths, parallel=parallel_processing, batch_size=read_batch_size
        )
        if not papers or len(papers) == 0:
            await self.progress_tracker.log_stage("reading", "failed")
            return {
                "success": False,
                "error": "No papers read successfully",
                "report": None,
                "metrics_summary": None,
                "iterations": 0,
                "progress_summary": self.progress_tracker.get_summary()
            }
        
        # Validate paper data structure
        invalid_papers = [i for i, p in enumerate(papers) if not isinstance(p, dict) or not p.get('extracted')]
        if invalid_papers:
            print(f"WARNING: {len(invalid_papers)} papers have invalid structure")
        
        await self.progress_tracker.log_stage("reading", "completed", {"papers_read": len(papers)})
        
        # Phase 1.5: Verify reading order
        read_verification = await self.order_verifier.verify_order(paper_paths, papers, "after_reading")
        await self.progress_tracker.log_order_verification("reading", read_verification)
        if not read_verification["verified"]:
            print(f"❌ CRITICAL: Paper order corrupted after reading!")
            return {
                "success": False,
                "error": "Order verification failed after reading",
                "details": read_verification["details"],
                "report": None,
                "metrics_summary": None,
                "iterations": 0,
                "progress_summary": self.progress_tracker.get_summary()
            }
        
        # Verify paper count matches expected - with retry for missing papers
        expected_count = len(paper_paths)
        actual_count = len(papers)
        if actual_count != expected_count:
            print(f"⚠️  WARNING: Paper count mismatch! Expected {expected_count} papers but got {actual_count}")
            print(f"Expected papers: {[Path(p).name for p in paper_paths]}")
            print(f"Received papers: {[p.get('filename', 'UNKNOWN') for p in papers]}")
            
            # Find missing papers
            received_filenames = {p.get('filename', '') for p in papers}
            missing_paths = [p for p in paper_paths if Path(p).name not in received_filenames]
            
            if missing_paths:
                print(f"🔄 Attempting to re-read {len(missing_paths)} missing paper(s)...")
                print(f"Missing: {[Path(p).name for p in missing_paths]}")
                
                # Retry reading only the missing papers
                retry_papers = await self.reader.read_papers(missing_paths)
                
                if retry_papers:
                    # Filter valid papers from retry
                    valid_retry = [p for p in retry_papers if isinstance(p, dict) and p.get('extracted')]
                    if valid_retry:
                        papers.extend(valid_retry)
                        actual_count = len(papers)
                        print(f"✅ Successfully recovered {len(valid_retry)} missing paper(s)")
                        print(f"Total papers now: {actual_count}")
                
                # Final check after retry
                if actual_count != expected_count:
                    print(f"❌ CRITICAL: Still missing {expected_count - actual_count} paper(s) after retry!")
                    print(f"Continuing with {actual_count} paper(s) - report quality will be affected")
                    # Don't fail - continue with whatever we have
                else:
                    print(f"✅ All {expected_count} papers recovered successfully")
        
        # Phase 2: Analyze (with caching)
        await self.progress_tracker.log_stage("analyzing", "in_progress")
        analysis = await self.analyzer.analyze_papers(
            papers, parallel=parallel_processing, batch_size=analyze_batch_size
        )
        if not analysis:
            await self.progress_tracker.log_stage("analyzing", "failed")
            return {
                "success": False,
                "error": "Analysis failed - no results returned",
                "report": None,
                "metrics_summary": None,
                "iterations": 0,
                "progress_summary": self.progress_tracker.get_summary()
            }
        await self.progress_tracker.log_stage("analyzing", "completed")
        
        # Note: Analysis order verification skipped - analysis is a combined string.
        # Papers order was already verified after reading phase.
        # Individual paper analyses maintain order through pre-allocated list indexing.
        
        # Phase 2.7: Extract metrics summary
        await self.progress_tracker.log_stage("metrics_extraction", "in_progress")
        metrics_summary = await self.metrics_extractor.extract_metrics(analysis)
        await self.progress_tracker.log_stage("metrics_extraction", "completed")
        
        # Phase 3: Synthesize (include metrics)
        await self.progress_tracker.log_stage("synthesizing", "in_progress")
        synthesis = await self.synthesizer.synthesize(analysis)
        if not synthesis:
            await self.progress_tracker.log_stage("synthesizing", "failed")
            return {
                "success": False,
                "error": "Synthesis failed - no output generated",
                "report": None,
                "metrics_summary": None,
                "iterations": 0,
                "progress_summary": self.progress_tracker.get_summary()
            }
        synthesis_with_metrics = f"{synthesis}\n\n{metrics_summary}"
        # Clear large intermediate variables
        analysis = None  # Free analysis string memory
        await self.progress_tracker.log_stage("synthesizing", "completed")
        
        # Phase 4-7: Write, Validate, Review, Rewrite loop
        report = None
        chief_feedback = None
        validation_feedback = None
        reviewer_feedback = None
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'─'*70}")
            print(f"Writing Iteration {iteration}/{max_iterations}")
            print(f"{'─'*70}")
            
            await self.progress_tracker.log_stage(f"writing_iteration_{iteration}", "in_progress")
            
            # Phase 4: Write/Rewrite
            if iteration == 1:
                report = await self.writer.write_report(synthesis_with_metrics)
            else:
                # Build clean feedback without section markers that could leak into report
                feedback_parts = []
                feedback_sources = {}
                
                if reviewer_feedback:
                    # Extract content without headers
                    clean_reviewer = reviewer_feedback.replace("=== REVIEWER FEEDBACK ===", "").strip()
                    feedback_parts.append(f"Reviewer notes: {clean_reviewer}")
                    feedback_sources['reviewer'] = True
                if chief_feedback:
                    clean_chief = chief_feedback.replace("=== CHIEF REVIEWER FEEDBACK ===", "").strip()
                    feedback_parts.append(f"Chief reviewer notes: {clean_chief}")
                    feedback_sources['chief'] = True
                if validation_feedback:
                    clean_validation = validation_feedback.replace("=== VALIDATION FEEDBACK ===", "").strip()
                    feedback_parts.append(f"Validation notes: {clean_validation}")
                    feedback_sources['validator'] = True
                
                combined_feedback = "\n\n".join(feedback_parts) if feedback_parts else None
                # Pass previous report for incremental fixes (not full rewrite)
                report = await self.writer.write_report(synthesis_with_metrics, combined_feedback, 
                                                       feedback_sources, previous_report=report)
            
            # Phase 4.5: Validate against synthesis (prevent hallucinations)
            await self.progress_tracker.log_stage(f"validation_iteration_{iteration}", "in_progress")
            validation_result = await self.validator.validate(report, synthesis_with_metrics)
            
            if not validation_result["valid"]:
                print(f"WARNING: Iteration {iteration}: Validation failed - hallucinations detected!")
                validation_feedback = validation_result["feedback"]  # Store for next iteration
                await self.progress_tracker.log_stage(f"validation_iteration_{iteration}", "failed",
                                              {"issues": validation_result["issues_found"]})
                continue
            
            validation_feedback = None  # Reset if validation passed
            await self.progress_tracker.log_stage(f"validation_iteration_{iteration}", "completed")
            
            # Phase 5: First review
            await self.progress_tracker.log_stage(f"review_iteration_{iteration}", "in_progress")
            reviewer_result = await self.reviewer.review(report)
            
            if not reviewer_result["approved"]:
                print(f"WARNING: Iteration {iteration}: Reviewer rejected, retrying...")
                reviewer_feedback = reviewer_result["feedback"]  # Store for next iteration
                await self.progress_tracker.log_stage(f"review_iteration_{iteration}", "rejected")
                continue
            
            # Reset reviewer feedback if approved
            reviewer_feedback = None
            await self.progress_tracker.log_stage(f"review_iteration_{iteration}", "approved")
            
            # Phase 6: Chief review (only if first review passed)
            await self.progress_tracker.log_stage(f"chief_review_iteration_{iteration}", "in_progress")
            chief_result = await self.chief_reviewer.final_review(report)
            
            if chief_result["approved"]:
                print(f"\n{'='*70}")
                print(f"SUCCESS! Report approved on iteration {iteration}")
                print(f"{'='*70}")
                await self.progress_tracker.log_stage(f"chief_review_iteration_{iteration}", "approved")
                await self.progress_tracker.log_stage("pipeline_complete", "success",
                                              {"iterations": iteration})
                return {
                    "success": True,
                    "iterations": iteration,
                    "report": report,
                    "metrics_summary": metrics_summary,
                    "final_feedback": chief_result["feedback"],
                    "progress_summary": self.progress_tracker.get_summary()
                }
            else:
                print(f"WARNING: Iteration {iteration}: Chief Reviewer rejected, retrying...")
                chief_feedback = chief_result["feedback"]  # Store for next iteration
                await self.progress_tracker.log_stage(f"chief_review_iteration_{iteration}", "rejected")
            
            # Clear iteration-specific variables to free memory
            try:
                del validation_result
            except NameError:
                pass
            try:
                del reviewer_result
            except NameError:
                pass
            try:
                del chief_result
            except NameError:
                pass
        
        # Max iterations reached
        print(f"\n{'='*70}")
        print(f"WARNING: Max iterations reached. Using last version.")
        print(f"{'='*70}")
        await self.progress_tracker.log_stage("pipeline_complete", "max_iterations_reached")
        return {
            "success": False,
            "iterations": max_iterations,
            "report": report if report else "Report generation incomplete",
            "metrics_summary": metrics_summary if metrics_summary else "No metrics",
            "final_feedback": "Maximum iterations reached",
            "progress_summary": self.progress_tracker.get_summary()
        }


async def main(papers_directory: str = None):
    """
    Main entry point
    
    Args:
        papers_directory: Directory containing papers to analyze (PDF, TXT, MD files).
                         If None, will look for 'papers' directory, then exit if not found.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Multi-Agent Research Paper Synthesis System (ENHANCED)      ║
║  Powered by: Semantic Kernel + Local LLM                     ║
║  Core Agents: Reader → Analyzer → Synthesizer → Writer       ║
║               → Reviewer → Chief Reviewer                    ║
║  Enhanced:    Validator + MetricsExtractor + Memory          ║
║               + ProgressTracker + OrderVerifier              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize Semantic Kernel
    print("Initializing Semantic Kernel...")
    kernel = Kernel()
    
    # Connect to Local LLM (LM Studio) with no timeout limit
    service_id = "local-gpt"
    
    # Create custom HTTP client with unlimited timeout (local LLM can take as long as needed)
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=30.0),  # No timeout, 30s connect only
    )
    
    # Create AsyncOpenAI client with custom httpx client
    openai_client = AsyncOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="NA",
        http_client=http_client,
    )
    
    chat_service = OpenAIChatCompletion(
        service_id=service_id,
        ai_model_id="qwen3-vl-8b",
        async_client=openai_client,
    )
    kernel.add_service(chat_service)
    print("Connected to Local LLM (no timeout - unlimited generation time)")
    
    # Create all agents
    print("Creating agents...")
    
    # Create memory agent first (needed by analyzer)
    print("  - Memory subsystem...")
    memory_agent = MemoryAgent("analysis_cache.json")
    
    print("  - Core pipeline agents (6)...")
    reader = ReaderAgent(kernel, service_id)
    analyzer = AnalyzerAgent(kernel, service_id, memory_agent)  # Pass memory agent
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
    
    # Create behavior tree
    tree = ResearchSynthesisBehaviorTree(
        reader, analyzer, synthesizer, writer, reviewer, chief_reviewer,
        validator, metrics_extractor, memory_agent, progress_tracker, order_verifier
    )
    
    # Determine papers directory
    if papers_directory:
        papers_dir = Path(papers_directory)
    else:
        # Default to 'papers' directory in current working directory
        papers_dir = Path("sample_papers")
    
    print(f"\nLooking for papers in: {papers_dir.absolute()}")
    
    # Look for PDF, TXT, MD files
    paper_paths = []
    if papers_dir.exists():
        for ext in ['*.pdf', '*.txt', '*.md']:
            paper_paths.extend([str(p) for p in papers_dir.glob(ext)])
    
    if not paper_paths or len(paper_paths) == 0:
        print(f"\n{'='*70}")
        print(f"❌ ERROR: No papers found in {papers_dir.absolute()}")
        print(f"{'='*70}")
        print(f"\nPlease ensure your papers directory contains PDF, TXT, or MD files.")
        print(f"\nUsage:")
        print(f"  1. Create a directory with your papers")
        print(f"  2. Add PDF, TXT, or MD research papers")
        print(f"  3. Run: python research_synthesis_system.py <directory_name>")
        print(f"\nExample:")
        print(f"  python research_synthesis_system.py sample_papers")
        print(f"\nDefault: Looks for 'sample_papers' directory if no argument given")
        return
    
    print(f"\nFound {len(paper_paths)} papers:")
    for path in paper_paths:
        print(f"   - {Path(path).name}")
    
    # Run the synthesis pipeline
    # Note: parallel_processing=False is default since LM Studio queues requests anyway
    result = await tree.run(paper_paths, max_iterations=3, parallel_processing=False)
    
    # Validate result
    if not result or not isinstance(result, dict):
        print("\n❌ ERROR: Pipeline returned invalid result")
        return
    
    # Check for errors - but still save report if available
    if not result.get("success"):
        print(f"\n⚠️  Pipeline completed with issues: {result.get('error', 'Unknown error')}")
        if result.get('details'):
            print(f"   Details: {result.get('details')}")
        print(f"\nProgress summary:")
        print(result.get('progress_summary', 'No progress data'))
        print(f"\nℹ️  Report will still be saved (last version available)...")
        # Don't return - continue to save report
    
    # Save final report with timestamp
    if result.get("report") and isinstance(result["report"], str) and len(result["report"]) > 0:
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
    print(f"Success: {result.get('success', False)}")
    print(f"Iterations: {result.get('iterations', 0)}")
    
    if result.get('progress_summary'):
        print(f"\n{result['progress_summary']}")
    
    print(f"\nFinal Report Preview:")
    print("─"*70)
    report_text = result.get('report', '')
    if report_text and len(report_text) > 0:
        print(report_text[:500] + "..." if len(report_text) > 500 else report_text)
    else:
        print("(No report generated)")
    print("─"*70)


if __name__ == "__main__":
    import sys
    # Check for command line argument
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(directory))
