
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import logging
from dotenv import load_dotenv

from llama_parse import LlamaParse
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContractProcessor:
    """Main class for processing legal contracts using LlamaParse and Gemini LLM"""
    
    def __init__(self):
        """Initialize the contract processor with API keys and models"""
        self.setup_apis()
        self.setup_models()
        self.setup_embeddings()
        
    def setup_apis(self):
        """Setup API keys and configurations"""
        self.llama_parse_api_key = os.getenv("LLAMA_PARSE_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.llama_parse_api_key:
            raise ValueError("LLAMA_PARSE_API_KEY not found in environment variables")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=self.gemini_api_key)
        
        self.parser = LlamaParse(
            api_key=self.llama_parse_api_key,
            parse_mode="parse_page_with_llm",
            high_res_ocr=True,
            adaptive_long_table=True,
            outlined_table_extraction=True,
            output_tables_as_HTML=True,
        )
        
    def setup_models(self):
        """Setup Gemini model for text analysis"""
        self.gemini_model = genai.GenerativeModel('gemini-flash-latest')
        
    def setup_embeddings(self):
        """Setup embedding model for semantic search (bonus feature)"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings_index = None
            self.clause_texts = []
        except Exception as e:
            logger.warning(f"Could not initialize embedding model: {e}")
            self.embedding_model = None
    
    def extract_text_from_pdf(self, pdf_path: str, save_txt: bool = False) -> str:
        """Extract text from PDF using LlamaParse and optionally save as TXT"""
        try:
            logger.info(f"Extracting text from {pdf_path}")
            documents = self.parser.load_data(pdf_path)
            
            full_text = ""
            for doc in documents:
                full_text += doc.text + "\n\n"
                
            logger.info(f"Successfully extracted {len(full_text)} characters from {pdf_path}")
            
            if save_txt and full_text.strip():
                txt_path = str(Path(pdf_path).with_suffix('.txt'))
                try:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(full_text)
                    logger.info(f"Saved extracted text to: {txt_path}")
                except Exception as e:
                    logger.warning(f"Could not save TXT file: {e}")
            
            return full_text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def create_clause_extraction_prompt(self, contract_text: str) -> str:
        """Create prompt for clause extraction using few-shot examples"""
        prompt = """You are a legal expert analyzing contracts. Extract the following specific clauses from the contract text:

1. TERMINATION CONDITIONS: Any clauses related to how/when the contract can be terminated
2. CONFIDENTIALITY CLAUSES: Any clauses related to confidentiality, non-disclosure, or information protection
3. LIABILITY CLAUSES: Any clauses related to liability, indemnification, or limitation of damages

For each clause type, provide:
- The exact text of the clause (if found)
- "NOT FOUND" if no relevant clause exists

Format your response as JSON:
{
    "termination_clause": "exact text or NOT FOUND",
    "confidentiality_clause": "exact text or NOT FOUND", 
    "liability_clause": "exact text or NOT FOUND"
}

EXAMPLE:
For a contract containing "Either party may terminate this agreement with 30 days written notice", you would extract:
{
    "termination_clause": "Either party may terminate this agreement with 30 days written notice",
    "confidentiality_clause": "NOT FOUND",
    "liability_clause": "NOT FOUND"
}

CONTRACT TEXT TO ANALYZE:
""" + contract_text

        return prompt
    
    def create_summary_prompt(self, contract_text: str) -> str:
        """Create prompt for contract summarization"""
        prompt = """You are a legal expert. Create a concise 100-150 word summary of this contract highlighting:

1. Purpose of the agreement
2. Key obligations of each party  
3. Notable risks or penalties

Keep the summary professional and factual. Focus on the most important aspects that a business executive would need to know.

CONTRACT TEXT:
""" + contract_text

        return prompt
    
    def extract_clauses_with_gemini(self, contract_text: str, use_chunking: bool = True) -> Dict[str, str]:
        """Extract specific clauses using Gemini LLM with optional chunking"""
        try:
            if use_chunking and len(contract_text) > 700000:
                logger.info(f"Document is large ({len(contract_text)} chars), using chunking approach")
                return self.extract_clauses_with_chunking(contract_text)
            
            prompt = self.create_clause_extraction_prompt(contract_text)
            
            logger.info(f"🚀 Sending {len(contract_text)} characters to Gemini for clause extraction")
            logger.info(f"📊 Total prompt size: {len(prompt)} characters")
            
            response = self.gemini_model.generate_content(prompt)
            logger.info(f"📥 Received Gemini response: {len(response.text)} characters")
            
            try:
                result = json.loads(response.text)
                logger.info("✅ Successfully parsed JSON response from Gemini")
                return {
                    "termination_clause": result.get("termination_clause", "NOT FOUND"),
                    "confidentiality_clause": result.get("confidentiality_clause", "NOT FOUND"),
                    "liability_clause": result.get("liability_clause", "NOT FOUND")
                }
            except json.JSONDecodeError:
                logger.warning("⚠️ Could not parse JSON response, using fallback extraction")
                logger.debug(f"Raw Gemini response (first 500 chars): {response.text[:500]}...")
                return self.fallback_clause_extraction(response.text)
                
        except Exception as e:
            logger.error(f"Error extracting clauses: {e}")
            return {
                "termination_clause": "ERROR",
                "confidentiality_clause": "ERROR", 
                "liability_clause": "ERROR"
            }
    
    def fallback_clause_extraction(self, gemini_response: str) -> Dict[str, str]:
        """Fallback method for clause extraction if JSON parsing fails"""
        logger.info("🔄 Using fallback extraction from Gemini response")
        
        result = {
            "termination_clause": "NOT FOUND",
            "confidentiality_clause": "NOT FOUND",
            "liability_clause": "NOT FOUND"
        }
        
        cleaned_response = self.clean_gemini_response(gemini_response)
        
        lines = cleaned_response.split('\n')
        
        current_clause = None
        clause_text = []
        
        for line in lines:
            line_stripped = line.strip()
            line_lower = line.lower()
            
            if not line_stripped or line_stripped in ['{', '}', ',']:
                continue
            
            if '"termination_clause"' in line_lower:
                if current_clause and clause_text:
                    result[current_clause] = self.extract_clause_content('\n'.join(clause_text))
                current_clause = "termination_clause"
                clause_text = [line_stripped]
            elif '"confidentiality_clause"' in line_lower:
                if current_clause and clause_text:
                    result[current_clause] = self.extract_clause_content('\n'.join(clause_text))
                current_clause = "confidentiality_clause"
                clause_text = [line_stripped]
            elif '"liability_clause"' in line_lower:
                if current_clause and clause_text:
                    result[current_clause] = self.extract_clause_content('\n'.join(clause_text))
                current_clause = "liability_clause"
                clause_text = [line_stripped]
            elif current_clause and line_stripped:
                clause_text.append(line_stripped)
        
        if current_clause and clause_text:
            result[current_clause] = self.extract_clause_content('\n'.join(clause_text))
        
        for clause_type, clause_content in result.items():
            if clause_content != "NOT FOUND":
                logger.info(f"✅ Fallback extracted {clause_type}: {len(clause_content)} characters")
        
        return result
    
    def clean_gemini_response(self, response: str) -> str:
        """Clean up Gemini response text"""
        cleaned = response.replace('```json', '').replace('```', '')
        return cleaned.strip()
    
    def extract_clause_content(self, raw_clause_text: str) -> str:
        """Extract clean clause content from raw text"""
        content = raw_clause_text
        
        import re
        content = re.sub(r'"[^"]*_clause"\s*:\s*"?', '', content)
        
       
        content = re.sub(r'[",}]+$', '', content)
        
        content = re.sub(r'^"', '', content)
        
       
        content = content.replace('\\"', '"')
        
        content = content.strip()
        
        if not content or content.upper() == "NOT FOUND":
            return "NOT FOUND"
        
        return content
    
    def extract_clauses_with_chunking(self, contract_text: str) -> Dict[str, str]:
        """Extract clauses from large documents using chunking approach"""
        chunks = self.chunk_text_for_processing(contract_text, max_chunk_size=25000)
        
        all_clauses = {
            "termination_clause": "NOT FOUND",
            "confidentiality_clause": "NOT FOUND",
            "liability_clause": "NOT FOUND"
        }
        
        for i, chunk in enumerate(chunks):
            logger.info(f"🔍 Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            prompt = self.create_clause_extraction_prompt(chunk)
            
            try:
                response = self.gemini_model.generate_content(prompt)
                
                try:
                    result = json.loads(response.text)
                except json.JSONDecodeError:
                    result = self.fallback_clause_extraction(response.text)
                
                # Update results if clauses found
                for clause_type in all_clauses.keys():
                    if result.get(clause_type, "NOT FOUND") != "NOT FOUND":
                        all_clauses[clause_type] = result[clause_type]
                        logger.info(f"✅ Found {clause_type} in chunk {i+1}")
                        
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                continue
        
        return all_clauses
    
    def summarize_contract_with_gemini(self, contract_text: str, use_chunking: bool = True) -> str:
        """Generate contract summary using Gemini LLM with optional chunking"""
        try:
            # For very large documents, use chunking
            if use_chunking and len(contract_text) > 700000:
                logger.info(f"Document is large ({len(contract_text)} chars), using chunking for summary")
                return self.summarize_with_chunking(contract_text)
            
            # Standard approach for smaller documents
            prompt = self.create_summary_prompt(contract_text)
            
            # Log the size of text being sent to Gemini
            logger.info(f"🚀 Sending {len(contract_text)} characters to Gemini for summarization")
            logger.info(f"📊 Total prompt size: {len(prompt)} characters")
            
            response = self.gemini_model.generate_content(prompt)
            summary = response.text.strip()
            
            # Ensure summary is within word limit
            words = summary.split()
            if len(words) > 150:
                summary = ' '.join(words[:150]) + "..."
                
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "ERROR: Could not generate summary"
    
    def summarize_with_chunking(self, contract_text: str) -> str:
        """Generate summary from large documents using chunking approach"""
        chunks = self.chunk_text_for_processing(contract_text, max_chunk_size=25000)
        
        chunk_summaries = []
        
        # Summarize each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"📝 Summarizing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            prompt = self.create_summary_prompt(chunk)
            
            try:
                response = self.gemini_model.generate_content(prompt)
                chunk_summary = response.text.strip()
                
                # Limit chunk summary to ~50 words
                words = chunk_summary.split()
                if len(words) > 50:
                    chunk_summary = ' '.join(words[:50]) + "..."
                
                chunk_summaries.append(chunk_summary)
                
            except Exception as e:
                logger.error(f"Error summarizing chunk {i+1}: {e}")
                continue
        
        # Combine chunk summaries into final summary
        if chunk_summaries:
            combined_summary = " ".join(chunk_summaries)
            
            # Final summary should be 100-150 words
            words = combined_summary.split()
            if len(words) > 150:
                combined_summary = ' '.join(words[:150]) + "..."
            
            logger.info(f"✅ Generated combined summary from {len(chunks)} chunks")
            return combined_summary
        else:
            return "ERROR: Could not generate summary from chunks"
    
    def process_contract(self, pdf_path: str, contract_id: str) -> Dict:
        """Process a single contract using intelligent semantic search + parallel processing"""
        logger.info(f"Processing contract {contract_id}: {pdf_path}")
        
        # Extract text from PDF
        contract_text = self.extract_text_from_pdf(pdf_path)
        
        if not contract_text:
            return {
                "contract_id": contract_id,
                "summary": "ERROR: Could not extract text",
                "termination_clause": "ERROR",
                "confidentiality_clause": "ERROR", 
                "liability_clause": "ERROR"
            }
        
        # Use intelligent semantic search + parallel processing
        logger.info(f"🧠 Starting intelligent processing for {contract_id}")
        clauses, summary = self.process_contract_intelligent(contract_text, contract_id)
        logger.info(f"✅ Completed intelligent processing for {contract_id}")
        
        # Store for semantic search (bonus feature)
        if self.embedding_model:
            self.clause_texts.extend([
                f"{contract_id}_termination: {clauses['termination_clause']}",
                f"{contract_id}_confidentiality: {clauses['confidentiality_clause']}",
                f"{contract_id}_liability: {clauses['liability_clause']}"
            ])
        
        return {
            "contract_id": contract_id,
            "summary": summary,
            "termination_clause": clauses["termination_clause"],
            "confidentiality_clause": clauses["confidentiality_clause"],
            "liability_clause": clauses["liability_clause"]
        }
    
    def process_contract_intelligent(self, contract_text: str, contract_id: str) -> Tuple[Dict[str, str], str]:
        """
        Intelligent contract processing using semantic search + parallel processing
        
        Returns:
            Tuple of (clauses_dict, summary_string)
        """
        import concurrent.futures
        
        # Step 1: Find relevant sections using semantic search
        logger.info(f"🔍 Finding relevant clause sections using semantic search")
        relevant_sections = self.find_relevant_clause_sections(contract_text)
        
        # Step 2: Parallel processing
        logger.info(f"⚡ Starting parallel processing: clauses + summary")
        
        clauses_result = {}
        summary_result = ""
        
        def extract_clauses_from_sections():
            nonlocal clauses_result
            try:
                logger.info(f"🎯 Processing {sum(len(sections) for sections in relevant_sections.values())} relevant sections for clauses")
                clauses_result = self.extract_clauses_from_relevant_sections(relevant_sections)
            except Exception as e:
                logger.error(f"Error in clause extraction: {e}")
                clauses_result = {
                    "termination_clause": "ERROR",
                    "confidentiality_clause": "ERROR",
                    "liability_clause": "ERROR"
                }
        
        def generate_summary_from_full_text():
            nonlocal summary_result
            try:
                logger.info(f"📝 Generating summary from full text ({len(contract_text)} chars)")
                summary_result = self.summarize_contract_with_gemini(contract_text, use_chunking=False)
            except Exception as e:
                logger.error(f"Error in summary generation: {e}")
                summary_result = "ERROR: Could not generate summary"
        
        # Run both processes in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            clause_future = executor.submit(extract_clauses_from_sections)
            summary_future = executor.submit(generate_summary_from_full_text)
            
            # Wait for both to complete
            concurrent.futures.wait([clause_future, summary_future])
        
        logger.info(f"✅ Parallel processing completed for {contract_id}")
        return clauses_result, summary_result
    
    def find_relevant_clause_sections(self, contract_text: str, section_size: int = 2000) -> Dict[str, List[str]]:
        """
        Find relevant sections for each clause type using semantic search
        
        Args:
            contract_text: Full contract text
            section_size: Size of each text section for embedding
            
        Returns:
            Dict with clause types as keys and relevant text sections as values
        """
        if not self.embedding_model:
            logger.warning("No embedding model available, using full text")
            return {
                "termination": [contract_text],
                "confidentiality": [contract_text], 
                "liability": [contract_text]
            }
        
        # Split text into sections
        sections = self.split_text_into_sections(contract_text, section_size)
        logger.info(f"📄 Split contract into {len(sections)} sections for analysis")
        
        # Define search queries for each clause type
        search_queries = {
            "termination": [
                "termination of agreement contract end",
                "terminate this agreement notice period",
                "expiration end of contract term"
            ],
            "confidentiality": [
                "confidential information non-disclosure",
                "proprietary information confidentiality",
                "trade secrets confidential data"
            ],
            "liability": [
                "liability limitation damages indemnification",
                "limitation of liability damages",
                "indemnify hold harmless liability"
            ]
        }
        
        relevant_sections = {}
        
        # Find relevant sections for each clause type
        for clause_type, queries in search_queries.items():
            logger.info(f"🔎 Searching for {clause_type} related sections")
            
            all_scores = []
            
            # Get embeddings for all queries
            for query in queries:
                query_embedding = self.embedding_model.encode([query])
                section_embeddings = self.embedding_model.encode(sections)
                
                # Calculate similarities using numpy (avoiding sklearn dependency)
                import numpy as np
                
                # Normalize embeddings
                query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
                section_norm = section_embeddings / np.linalg.norm(section_embeddings, axis=1, keepdims=True)
                
                # Calculate cosine similarity
                similarities = np.dot(query_norm, section_norm.T)[0]
                all_scores.append(similarities)
            
            # Average scores across all queries for this clause type
            avg_scores = np.mean(all_scores, axis=0)
            
            # Get top 3 most relevant sections
            top_indices = np.argsort(avg_scores)[::-1][:3]
            relevant_sections[clause_type] = [sections[i] for i in top_indices if avg_scores[i] > 0.3]
            
            logger.info(f"✅ Found {len(relevant_sections[clause_type])} relevant sections for {clause_type}")
        
        return relevant_sections
    
    def split_text_into_sections(self, text: str, section_size: int = 2000) -> List[str]:
        """Split text into overlapping sections for better context preservation"""
        sections = []
        overlap = section_size // 4  # 25% overlap
        
        for i in range(0, len(text), section_size - overlap):
            section = text[i:i + section_size]
            if len(section.strip()) > 100:  # Only include substantial sections
                sections.append(section.strip())
        
        return sections
    
    def extract_clauses_from_relevant_sections(self, relevant_sections: Dict[str, List[str]]) -> Dict[str, str]:
        """Extract all clauses from semantically relevant sections using single Gemini call"""
        clauses = {
            "termination_clause": "NOT FOUND",
            "confidentiality_clause": "NOT FOUND",
            "liability_clause": "NOT FOUND"
        }
        
        # Combine all relevant sections into one text
        all_relevant_text = ""
        section_info = []
        
        for section_type, sections in relevant_sections.items():
            if sections:
                combined_sections = "\n\n".join(sections)
                all_relevant_text += f"\n\n=== {section_type.upper()} RELEVANT SECTIONS ===\n{combined_sections}\n"
                section_info.append(f"{section_type}: {len(sections)} sections ({len(combined_sections)} chars)")
        
        if not all_relevant_text.strip():
            logger.warning("No relevant sections found for any clause type")
            return clauses
        
        logger.info(f"🤖 Single Gemini call for ALL clauses: {len(all_relevant_text)} chars total")
        logger.info(f"📋 Section breakdown: {', '.join(section_info)}")
        
        # Create comprehensive prompt for all clause types
        prompt = self.create_comprehensive_clause_prompt(all_relevant_text)
        
        try:
            response = self.gemini_model.generate_content(prompt)
            logger.info(f"📥 Received comprehensive response: {len(response.text)} characters")
            
            # Parse JSON response
            try:
                result = json.loads(response.text)
                logger.info("✅ Successfully parsed JSON response from Gemini")
                
                clauses["termination_clause"] = result.get("termination_clause", "NOT FOUND")
                clauses["confidentiality_clause"] = result.get("confidentiality_clause", "NOT FOUND")
                clauses["liability_clause"] = result.get("liability_clause", "NOT FOUND")
                
                # Log extraction results
                for clause_type, clause_content in clauses.items():
                    if clause_content != "NOT FOUND":
                        logger.info(f"✅ Extracted {clause_type}: {len(clause_content)} chars")
                    else:
                        logger.info(f"⚠️ No {clause_type} found")
                        
            except json.JSONDecodeError:
                logger.warning("⚠️ Could not parse JSON response, using fallback extraction")
                clauses = self.fallback_clause_extraction(response.text)
                
        except Exception as e:
            logger.error(f"Error extracting clauses: {e}")
            # Return error clauses
            for key in clauses.keys():
                clauses[key] = "ERROR"
        
        return clauses
    
    def create_focused_clause_prompt(self, text: str, clause_type: str) -> str:
        """Create a focused prompt for extracting a specific clause type"""
        
        clause_descriptions = {
            "termination": "termination conditions, contract end procedures, notice requirements, or expiration terms",
            "confidentiality": "confidentiality obligations, non-disclosure requirements, or proprietary information protection",
            "liability": "liability limitations, indemnification clauses, or damage limitation provisions"
        }
        
        prompt = f"""You are a legal expert. Extract and summarize the {clause_descriptions[clause_type]} from the following contract text.

INSTRUCTIONS:
1. Find the specific clause related to {clause_type}
2. Extract the exact relevant text
3. If multiple related clauses exist, combine them
4. If no relevant clause exists, respond with "NOT FOUND"
5. Keep the response concise but complete

CONTRACT TEXT:
{text}

EXTRACTED {clause_type.upper()} CLAUSE:"""

        return prompt
    
    def create_comprehensive_clause_prompt(self, all_relevant_text: str) -> str:
        """Create a comprehensive prompt for extracting all clause types in one call"""
        
        prompt = """You are a legal expert analyzing contract sections. Extract ALL THREE types of clauses from the provided relevant sections:

1. TERMINATION CONDITIONS: Any clauses related to how/when the contract can be terminated
2. CONFIDENTIALITY CLAUSES: Any clauses related to confidentiality, non-disclosure, or information protection  
3. LIABILITY CLAUSES: Any clauses related to liability, indemnification, or limitation of damages

INSTRUCTIONS:
- Analyze ALL the provided sections below
- For each clause type, extract the exact relevant text if found
- If a clause type is not found, use "NOT FOUND"
- Combine multiple related clauses of the same type if they exist
- Keep responses concise but complete

FORMAT your response as JSON:
{
    "termination_clause": "exact text or NOT FOUND",
    "confidentiality_clause": "exact text or NOT FOUND", 
    "liability_clause": "exact text or NOT FOUND"
}

EXAMPLE:
{
    "termination_clause": "Either party may terminate this agreement with 30 days written notice",
    "confidentiality_clause": "All information disclosed shall remain confidential for 5 years",
    "liability_clause": "NOT FOUND"
}

RELEVANT CONTRACT SECTIONS TO ANALYZE:
""" + all_relevant_text

        return prompt
    
    def build_semantic_search_index(self):
        """Build FAISS index for semantic search over clauses (bonus feature)"""
        if not self.embedding_model or not self.clause_texts:
            logger.warning("Cannot build semantic search index: missing embeddings or clause texts")
            return
            
        try:
            logger.info("Building semantic search index...")
            
            # Generate embeddings for all clauses
            embeddings = self.embedding_model.encode(self.clause_texts)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.embeddings_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.embeddings_index.add(embeddings.astype('float32'))
            
            logger.info(f"Built semantic search index with {len(self.clause_texts)} clauses")
            
        except Exception as e:
            logger.error(f"Error building semantic search index: {e}")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Perform semantic search over extracted clauses (bonus feature)"""
        if not self.embedding_model or not self.embeddings_index:
            return []
            
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.embeddings_index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.clause_texts):
                    results.append((self.clause_texts[idx], float(score)))
                    
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def process_input(self, input_path: str, output_file: str = "contract_analysis.csv"):
        """
        Flexible processing method that handles both single PDF files and directories
        
        Args:
            input_path: Path to a single PDF file or directory containing PDFs
            output_file: Output file name for results
            
        Returns:
            List of processing results
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error(f"Path {input_path} does not exist")
            return []
        
        pdf_files = []
        
        if input_path.is_file():
            # Single PDF file
            if input_path.suffix.lower() == '.pdf':
                pdf_files = [input_path]
                logger.info(f"Processing single PDF: {input_path}")
            else:
                logger.error(f"File {input_path} is not a PDF")
                return []
        elif input_path.is_dir():
            # Directory containing PDFs
            pdf_files = list(input_path.glob("*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDF files found in {input_path}")
                return []
            logger.info(f"Found {len(pdf_files)} PDF files in directory: {input_path}")
        else:
            logger.error(f"Invalid path: {input_path}")
            return []
        
        results = []
        
        # Process each PDF
        for i, pdf_file in enumerate(tqdm(pdf_files, desc="Processing contracts")):
            if len(pdf_files) == 1:
                # Single file - use filename as contract ID
                contract_id = pdf_file.stem
            else:
                # Multiple files - use numbered IDs
                contract_id = f"contract_{i+1:03d}_{pdf_file.stem}"
            
            result = self.process_contract(str(pdf_file), contract_id)
            results.append(result)
        
        # Build semantic search index (bonus feature)
        self.build_semantic_search_index()
        
        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Also save as JSON for better readability
        json_file = output_file.replace('.csv', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results also saved to {json_file}")
        
        return results

    def chunk_text_for_processing(self, text: str, max_chunk_size: int = 30000) -> List[str]:
        """
        Split large text into chunks for processing (optional optimization)
        
        Args:
            text: Full contract text
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        # Split by paragraphs first to maintain context
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed limit, start new chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split text into {len(chunks)} chunks for processing")
        return chunks

    def process_contracts_batch(self, pdf_directory: str, output_file: str = "contract_analysis.csv"):
        """Legacy method - redirects to process_input for backward compatibility"""
        return self.process_input(pdf_directory, output_file)


def main():
    """Main function to run the contract processing pipeline"""
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "contract_analysis.csv"
        
        print(f"🚀 Processing: {input_path}")
        print(f"📄 Output: {output_file}")
        
        try:
            processor = ContractProcessor()
            results = processor.process_input(input_path, output_file)
            
            if results:
                print(f"\n✅ Successfully processed {len(results)} contract(s)")
                print(f"📊 Results saved to: {output_file}")
            else:
                print("❌ No contracts were processed")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        
        return
    
    # Default demo mode
    processor = ContractProcessor()
    
    # Process contracts from current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # For demo, we'll process the example text file as if it were extracted from PDF
    demo_text_file = os.path.join(current_dir, "pdf.txt")
    
    if os.path.exists(demo_text_file):
        logger.info("Processing demo contract from pdf.txt")
        
        # Read the demo text
        with open(demo_text_file, 'r', encoding='utf-8') as f:
            contract_text = f.read()
        
        # Process the demo contract
        clauses = processor.extract_clauses_with_gemini(contract_text)
        summary = processor.summarize_contract_with_gemini(contract_text)
        
        result = {
            "contract_id": "demo_contract",
            "summary": summary,
            "termination_clause": clauses["termination_clause"],
            "confidentiality_clause": clauses["confidentiality_clause"],
            "liability_clause": clauses["liability_clause"]
        }
        
        # Save demo result
        df = pd.DataFrame([result])
        df.to_csv("demo_contract_analysis.csv", index=False)
        
        # Pretty print results
        print("\n" + "="*80)
        print("DEMO CONTRACT ANALYSIS RESULTS")
        print("="*80)
        print(f"Contract ID: {result['contract_id']}")
        print(f"\nSummary:\n{result['summary']}")
        print(f"\nTermination Clause:\n{result['termination_clause']}")
        print(f"\nConfidentiality Clause:\n{result['confidentiality_clause']}")
        print(f"\nLiability Clause:\n{result['liability_clause']}")
        print("="*80)
        
        # Demo semantic search if available
        if processor.embedding_model:
            processor.clause_texts = [
                f"demo_termination: {clauses['termination_clause']}",
                f"demo_confidentiality: {clauses['confidentiality_clause']}",
                f"demo_liability: {clauses['liability_clause']}"
            ]
            processor.build_semantic_search_index()
            
            # Test semantic search
            search_results = processor.semantic_search("contract termination conditions", top_k=3)
            if search_results:
                print("\nSEMANTIC SEARCH DEMO - Query: 'contract termination conditions'")
                print("-" * 60)
                for text, score in search_results:
                    print(f"Score: {score:.3f} | {text}")
    
    else:
        logger.info("No demo file found. Usage examples:")
        logger.info("  Single PDF: processor.process_input('path/to/contract.pdf')")
        logger.info("  Directory:  processor.process_input('path/to/pdf/directory')")
        logger.info("  Command line: python parse.py <pdf_file_or_directory>")


if __name__ == "__main__":
    main()