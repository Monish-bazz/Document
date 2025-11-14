"""
Simple startup script for the Contract Processing Web Interface
"""

import os
import sys
import webbrowser
import time
from threading import Timer

def open_browser():
    """Open browser after a short delay"""
    webbrowser.open('http://localhost:5000')

def main():
    print("🚀 Starting Contract Processing Web Interface...")
    print("=" * 60)
    
    # Check if API keys are configured
    from dotenv import load_dotenv
    load_dotenv()
    
    llama_key = os.getenv("LLAMA_PARSE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    print("📋 System Check:")
    
    if not llama_key or llama_key == "your_llama_parse_api_key_here":
        print("❌ LLAMA_PARSE_API_KEY not configured")
        print("   Please update your .env file with your LlamaParse API key")
        print("   Get it from: https://developers.llamaindex.ai/python/cloud/general/api_key/")
    else:
        print("✅ LLAMA_PARSE_API_KEY configured")
    
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        print("❌ GEMINI_API_KEY not configured")
        print("   Please update your .env file with your Gemini API key")
        print("   Get it from: https://makersuite.google.com/app/apikey")
    else:
        print("✅ GEMINI_API_KEY configured")
    
    print("\n🌐 Web Interface Features:")
    print("   • Upload single PDF files or multiple files")
    print("   • Automatic clause extraction (termination, confidentiality, liability)")
    print("   • AI-powered contract summarization")
    print("   • Semantic search across extracted clauses")
    print("   • Export results in CSV and JSON formats")
    
    print("\n📱 Opening web interface...")
    print("   URL: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Open browser after 2 seconds
    Timer(2.0, open_browser).start()
    
    # Start Flask app
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down web interface...")
        print("Thank you for using the Contract Processing Pipeline!")
    except Exception as e:
        print(f"\n❌ Error starting web interface: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that your API keys are configured in .env file")
        print("3. Ensure no other application is using port 5000")
        sys.exit(1)

if __name__ == "__main__":
    main()
