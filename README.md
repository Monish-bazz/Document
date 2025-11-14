# AI Contract Processing Pipeline with Intelligent Analysis

A comprehensive, intelligent pipeline for processing legal contracts using LlamaParse for PDF text extraction, Google Gemini LLM for clause extraction and summarization, with advanced semantic search and parallel processing capabilities.

##  Project Overview

This project implements an **intelligent automated contract analysis system** that:
- **Extracts text** from PDF contracts using LlamaParse with advanced OCR
- **Intelligently identifies** key legal clauses using semantic search (termination, confidentiality, liability)
- **Generates concise summaries** (100-150 words) with parallel processing
- **Provides semantic search** capabilities over extracted clauses
- **Uses focused prompting** for improved clause extraction accuracy
- **Supports both single PDFs and batch processing** with incremental database building
- **Offers web interface** for easy file uploads and processing
- **Optimized performance** with parallel processing and single Gemini API calls

##  Intelligent System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Files     │───▶│   LlamaParse     │───▶│  Extracted Text │
│  Single/Batch   │    │   - High-res OCR │    │   (36KB avg)    │
│                 │    │   - Table Extract│    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐                            ┌─────────────────┐
│  Web Interface  │                            │ Semantic Search │
│  - Upload UI    │                            │ - Split into 2KB│
│  - Results View │                            │ - Find Relevant │
│  - Search API   │                            │ - 3 Sections/Type│
└─────────────────┘                            └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Output Files  │◀───│ Parallel Process │◀───│ Intelligent     │
│   - CSV/JSON    │    │ Thread 1: Clauses│    │ Processing      │
│   - Incremental │    │ Thread 2: Summary│    │                 │
│   - Database    │    │ (Single API Call)│    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Requirements

### Dependencies
Install required packages using:
```bash
pip install -r requirements.txt
```

### API Keys Required
1. **LlamaParse API Key**: Get from [LlamaIndex Cloud](https://developers.llamaindex.ai/python/cloud/general/api_key/)
2. **Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

## ⚙️ Setup Instructions

### 1. Clone/Download the Project
```bash
# If using git
git clone <repository-url>
cd pdf-contract-processor

# Or download and extract the files
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Edit the `.env` file and add your API keys:
```env
LLAMA_PARSE_API_KEY=your_llama_parse_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Prepare Your Data
- Place PDF contract files in a directory


###  Web Interface (Recommended)
The easiest way to use the intelligent contract processor:

```bash
# Start the web interface with automatic browser opening
python start_web_interface.py
```

**Features:**
- **Drag-and-drop PDF uploads** (single or multiple files)
- **Real-time intelligent processing** with progress indicators
- **Interactive results display** with clause highlighting
- **Built-in semantic search** over extracted clauses
- **Download results** in CSV and JSON formats
- **System status monitoring** for API key configuration

###  Incremental Batch Processing
Build a comprehensive contract database over time:

```bash
# Process all PDFs in a directory (incremental)
python batch_process.py path/to/contracts/

# Custom output file
python batch_process.py path/to/contracts/ my_database.csv
```

**Smart Features:**
- **Resumes processing** from where it left off
- **Skips already processed** contracts automatically
- **Builds cumulative database** (never overwrites previous results)
- **Handles interruptions** gracefully (Ctrl+C safe)
- **Real-time progress** with detailed logging

###  Command Line Usage

#### Single PDF or Directory
```bash
# Process a single PDF with intelligent analysis
python parse.py path/to/contract.pdf

# Process all PDFs in a directory
python parse.py path/to/pdf/directory

# Specify custom output file
python parse.py path/to/input intelligent_results.csv
```

#### Demo with Provided Text
```bash
python parse.py
```
Processes the demo contract from `pdf.txt` with full intelligent analysis.

###  Programmatic Usage with Intelligent Processing
```python
from parse import ContractProcessor

# Initialize processor with semantic search capabilities
processor = ContractProcessor()

# Intelligent processing (semantic search + parallel processing)
results = processor.process_input("path/to/pdf/file_or_directory")

# Process single contract with full intelligence
result = processor.process_contract("contract.pdf", "contract_001")
```

###  Advanced Semantic Search
```python
# After processing contracts, perform intelligent semantic search
search_results = processor.semantic_search("termination conditions", top_k=5)
for text, score in search_results:
    print(f"Relevance: {score:.3f} | {text[:100]}...")

# Search specific clause types
termination_results = processor.semantic_search("contract end notice period", top_k=3)
confidentiality_results = processor.semantic_search("non-disclosure information", top_k=3)
```

##  Output Format

The system generates two output files:

### CSV Output (`contract_analysis.csv`)
| contract_id | summary | termination_clause | confidentiality_clause | liability_clause |
|-------------|---------|-------------------|----------------------|------------------|
| contract_001 | Brief summary... | Termination text... | Confidentiality text... | Liability text... |

### JSON Output (`contract_analysis.json`)
```json
[
  {
    "contract_id": "contract_001",
    "summary": "This Chase Affiliate Agreement establishes terms for...",
    "termination_clause": "Either Affiliate or Chase may terminate this Agreement...",
    "confidentiality_clause": "Each party agrees that all information shall remain strictly confidential...",
    "liability_clause": "Chase shall have no liability for any indirect, incidental..."
  }
]
```

##  Configuration Options

### LlamaParse Configuration
The system uses the following LlamaParse settings:

### Gemini Model Configuration
- Model: `gemini-flash-latest` (optimized for speed and accuracy)
- Temperature: Default (balanced creativity/accuracy)
- Comprehensive prompting with structured JSON output
- Clean fallback extraction for malformed responses

##  Intelligent Processing Features

###  Semantic Search-Driven Clause Detection
- **Smart Section Identification**: Splits contracts into 2KB sections with 25% overlap
- **Targeted Query Matching**: Uses predefined semantic queries for each clause type
- **Relevance Scoring**: Cosine similarity with 0.3 threshold for quality filtering
- **Focused Processing**: Sends only relevant sections to Gemini (not full 36KB text)

###  Performance Optimizations
- **Single API Call**: Extracts all 3 clause types in one Gemini request (3→1 calls)
- **Parallel Processing**: Clause extraction and summarization run simultaneously
- **Speed Improvement**: ~55% faster processing (22s → 10s average)
- **Cost Efficiency**: 67% fewer API calls, reduced token usage

###  Processing Pipeline
```
PDF (36KB) → Semantic Search → Relevant Sections (6KB) → Single Gemini Call → All Clauses
           ↓
Full Text (36KB) → Parallel Thread → Gemini → Summary
```

##  Advanced Features Implemented

### 1. Intelligent Semantic Search
- **SentenceTransformer**: `all-MiniLM-L6-v2` embeddings
- **FAISS Vector Index**: Efficient similarity search with cosine similarity
- **Query Optimization**: Predefined queries for each clause type
- **Section-Level Analysis**: 2KB chunks with context preservation

### 2. Comprehensive Prompting System
- **Structured JSON Output**: Clean, parsable responses
- **Comprehensive Instructions**: Single prompt for all clause types
- **Fallback Extraction**: Regex-based cleaning for malformed JSON
- **Content Sanitization**: Removes JSON artifacts from clause text

### 3. Incremental Database Building
- **Resume Capability**: Continues from where processing stopped
- **Duplicate Prevention**: Skips already processed contracts
- **Cumulative Results**: Builds comprehensive contract database over time
- **Interrupt Safety**: Graceful handling of Ctrl+C interruptions



##  Project Structure

```
pdf-contract-processor/
├── parse.py                     # Main processing pipeline
├── app.py                      # Flask web application
├── start_web_interface.py      # Web interface startup script
├── requirements.txt            # Python dependencies
├── .env                       # API keys (create this)
├── README.md                  # This file
├── templates/                 # HTML templates for web interface
│   ├── index.html            # Main upload page
│   └── results.html          # Results display page
├── uploads/                   # Uploaded files (created automatically)
├── results/                   # Processing results (created automatically)
├── batch_process.py          # Batch processing utility
```

##  Key Features

### Text Extraction
- **LlamaParse Integration**: Advanced PDF parsing with OCR
- **Table Handling**: Extracts tables as HTML for better structure
- **Multi-page Support**: Handles complex legal documents

### Clause Extraction
- **Targeted Analysis**: Focuses on termination, confidentiality, and liability clauses
- **Few-shot Learning**: Uses examples to improve extraction accuracy
- **Robust Parsing**: Fallback mechanisms for reliable extraction

### Summarization
- **Word Limit Control**: Ensures 100-150 word summaries
- **Executive Focus**: Highlights purpose, obligations, and risks
- **Professional Tone**: Maintains legal document formality

### Bonus Features
- **Semantic Search**: Find similar clauses across contracts
- **Vector Embeddings**: Efficient similarity computation
- **Interactive Queries**: Search for specific legal concepts

##  Error Handling

The system includes comprehensive error handling:
- API key validation
- PDF processing errors
- LLM response parsing failures
- Network connectivity issues
- File I/O errors

##  Performance Considerations

- **Batch Processing**: Efficient handling of multiple contracts
- **Progress Tracking**: Visual progress bars for long operations
- **Memory Management**: Optimized for large document processing
- **Caching**: Embedding model caching for repeated use

##  Security Best Practices

- **Environment Variables**: API keys stored securely in `.env`
- **Input Validation**: Sanitized file paths and content
- **Error Logging**: Detailed logs without exposing sensitive data

