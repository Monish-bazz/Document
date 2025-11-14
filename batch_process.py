
import sys
import os
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from parse import ContractProcessor

def load_existing_results(output_file):
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            existing_contracts = set(df['contract_id'].tolist()) if 'contract_id' in df.columns else set()
            print(f"📄 Found existing results file with {len(df)} contracts")
            return df, existing_contracts
        except Exception as e:
            print(f"⚠️ Error reading existing file: {e}")
            return pd.DataFrame(), set()
    else:
        print(f"📄 Creating new results file: {output_file}")
        return pd.DataFrame(), set()

def append_result_to_csv(result, output_file):
    try:
        new_df = pd.DataFrame([result])
        
        if os.path.exists(output_file):
            new_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            new_df.to_csv(output_file, mode='w', header=True, index=False)
        
        return True
    except Exception as e:
        print(f"❌ Error appending to CSV: {e}")
        return False

def append_result_to_json(result, json_file):
    try:
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(result)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"❌ Error appending to JSON: {e}")
        return False

def generate_contract_id(pdf_file, existing_contracts):
    base_id = pdf_file.stem
    contract_id = base_id
    counter = 1
    
    while contract_id in existing_contracts:
        contract_id = f"{base_id}_{counter:03d}"
        counter += 1
    
    return contract_id

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_process.py <pdf_directory> [output_file]")
        print("\nExample:")
        print("  python batch_process.py ./contracts/")
        print("  python batch_process.py ./contracts/ cumulative_results.csv")
        print("\nFeatures:")
        print("  • Processes PDFs one by one")
        print("  • Appends results to single CSV file")
        print("  • Skips already processed contracts")
        print("  • Resumes from where it left off")
        sys.exit(1)
    
    pdf_directory = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "cumulative_contract_analysis.csv"
    json_file = output_file.replace('.csv', '.json')
    
    if not os.path.exists(pdf_directory):
        print(f"❌ Error: Directory '{pdf_directory}' does not exist")
        sys.exit(1)
    
    if not os.path.isdir(pdf_directory):
        print(f"❌ Error: '{pdf_directory}' is not a directory")
        sys.exit(1)
    
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Error: No PDF files found in '{pdf_directory}'")
        sys.exit(1)
    
    print(f"🚀 Incremental Batch Processing Started")
    print("=" * 60)
    print(f"📁 Directory: {pdf_directory}")
    print(f"📄 Output CSV: {output_file}")
    print(f"📋 Output JSON: {json_file}")
    print(f"📊 Found {len(pdf_files)} PDF files")
    
    existing_df, existing_contracts = load_existing_results(output_file)
    
    pdf_files_to_process = []
    for pdf_file in pdf_files:
        potential_id = pdf_file.stem
        already_processed = any(existing_id.startswith(potential_id) for existing_id in existing_contracts)
        if not already_processed:
            pdf_files_to_process.append(pdf_file)
        else:
            print(f"⏭️ Skipping already processed: {pdf_file.name}")
    
    if not pdf_files_to_process:
        print(f"\n✅ All {len(pdf_files)} PDF files have already been processed!")
        print(f"📊 Total contracts in database: {len(existing_contracts)}")
        sys.exit(0)
    
    print(f"\n🔄 Processing {len(pdf_files_to_process)} new PDF files...")
    print(f"📈 Will add to existing {len(existing_contracts)} contracts")
    
    try:
        processor = ContractProcessor()
        
        successful_count = 0
        failed_count = 0
        
        for pdf_file in tqdm(pdf_files_to_process, desc="Processing PDFs"):
            try:
                contract_id = generate_contract_id(pdf_file, existing_contracts)
                
                print(f"\n📄 Processing: {pdf_file.name}")
                print(f"🆔 Contract ID: {contract_id}")
                
                result = processor.process_contract(str(pdf_file), contract_id)
                
                if result and result.get('summary') != "ERROR: Could not extract text":
                    result['source_file'] = pdf_file.name
                    
                    if append_result_to_csv(result, output_file):
                        print(f"✅ Added to CSV: {contract_id}")
                    
                    if append_result_to_json(result, json_file):
                        print(f"✅ Added to JSON: {contract_id}")
                    
                    existing_contracts.add(contract_id)
                    successful_count += 1
                    
                    summary_preview = result['summary'][:100] + "..." if len(result['summary']) > 100 else result['summary']
                    print(f"📝 Summary: {summary_preview}")
                    
                else:
                    print(f"❌ Failed to process: {pdf_file.name}")
                    failed_count += 1
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {e}")
                failed_count += 1
                continue
        
        print(f"\n" + "=" * 60)
        print(f"📊 BATCH PROCESSING COMPLETE")
        print(f"✅ Successfully processed: {successful_count} contracts")
        print(f"❌ Failed: {failed_count} contracts")
        print(f"📈 Total contracts in database: {len(existing_contracts)}")
        print(f"📄 Results saved to: {output_file}")
        print(f"📋 JSON saved to: {json_file}")
        
        if os.path.exists(output_file):
            final_df = pd.read_csv(output_file)
            print(f"📊 Final CSV contains: {len(final_df)} contracts")
        
        if successful_count > 0 and processor.embedding_model:
            print(f"\n🔍 Semantic Search Demo (across all contracts):")
            print("Query: 'contract termination'")
            search_results = processor.semantic_search("contract termination", top_k=5)
            for text, score in search_results:
                print(f"  Score: {score:.3f} | {text[:80]}...")
                
    except KeyboardInterrupt:
        print(f"\n⚠️ Processing interrupted by user")
        print(f"📊 Processed so far: {successful_count} contracts")
        print(f"💾 Results saved to: {output_file}")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
