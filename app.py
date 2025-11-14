
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from parse import ContractProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'contract_processor_secret_key_2024'  
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            flash('No files selected', 'error')
            return redirect(url_for('index'))
        
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            flash('No files selected', 'error')
            return redirect(url_for('index'))
        
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_upload_dir, exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(session_upload_dir, filename)
                file.save(file_path)
                uploaded_files.append(file_path)
            else:
                flash(f'Invalid file type: {file.filename}. Only PDF files are allowed.', 'warning')
        
        if not uploaded_files:
            flash('No valid PDF files uploaded', 'error')
            return redirect(url_for('index'))
        
        processor = ContractProcessor()
        
        output_file = os.path.join(RESULTS_FOLDER, f'analysis_{session_id}.csv')
        if len(uploaded_files) == 1:
            results = processor.process_input(uploaded_files[0], output_file)
        else:
            results = processor.process_input(session_upload_dir, output_file)
        
        if results:
            flash(f'Successfully processed {len(results)} contract(s)', 'success')
            return redirect(url_for('results', session_id=session_id))
        else:
            flash('Failed to process contracts', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        flash(f'Error processing files: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/results/<session_id>')
def results(session_id):
    try:
        csv_file = os.path.join(RESULTS_FOLDER, f'analysis_{session_id}.csv')
        json_file = os.path.join(RESULTS_FOLDER, f'analysis_{session_id}.json')
        
        if not os.path.exists(json_file):
            flash('Results not found', 'error')
            return redirect(url_for('index'))
        
        with open(json_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        return render_template('results.html', 
                             results=results_data, 
                             session_id=session_id,
                             csv_file=f'analysis_{session_id}.csv',
                             json_file=f'analysis_{session_id}.json')
        
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        flash(f'Error loading results: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/download/<session_id>/<file_type>')
def download_results(session_id, file_type):
    try:
        if file_type == 'csv':
            filename = f'analysis_{session_id}.csv'
        elif file_type == 'json':
            filename = f'analysis_{session_id}.json'
        else:
            flash('Invalid file type', 'error')
            return redirect(url_for('index'))
        
        file_path = os.path.join(RESULTS_FOLDER, filename)
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            flash('File not found', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/search', methods=['POST'])
def semantic_search():
    """API endpoint for semantic search"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        session_id = data.get('session_id', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        processor = ContractProcessor()
        
        json_file = os.path.join(RESULTS_FOLDER, f'analysis_{session_id}.json')
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            processor.clause_texts = []
            for result in results_data:
                contract_id = result['contract_id']
                processor.clause_texts.extend([
                    f"{contract_id}_termination: {result['termination_clause']}",
                    f"{contract_id}_confidentiality: {result['confidentiality_clause']}",
                    f"{contract_id}_liability: {result['liability_clause']}"
                ])
            
            processor.build_semantic_search_index()
            
            search_results = processor.semantic_search(query, top_k=10)
            
            return jsonify({
                'results': [{'text': text, 'score': score} for text, score in search_results]
            })
        else:
            return jsonify({'error': 'No results found for this session'}), 404
            
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        llama_key = os.getenv("LLAMA_PARSE_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        status_info = {
            'llama_parse_configured': bool(llama_key and llama_key != "your_llama_parse_api_key_here"),
            'gemini_configured': bool(gemini_key and gemini_key != "your_gemini_api_key_here"),
            'upload_folder': os.path.exists(UPLOAD_FOLDER),
            'results_folder': os.path.exists(RESULTS_FOLDER)
        }
        
        return jsonify(status_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    llama_key = os.getenv("LLAMA_PARSE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not llama_key or llama_key == "your_llama_parse_api_key_here":
        print("⚠️  Warning: LLAMA_PARSE_API_KEY not configured in .env file")
    
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        print("⚠️  Warning: GEMINI_API_KEY not configured in .env file")
    
    print("🚀 Starting Contract Processing Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
