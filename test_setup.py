#!/usr/bin/env python3
"""Quick test script to verify RAG model setup"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

try:
    print("Testing RAG setup...")
    print("=" * 50)
    
    # Test imports
    print("✓ Importing required modules...")
    import google.genai as genai
    import pandas as pd
    import numpy as np
    from dotenv import load_dotenv
    print("  - google-genai: OK")
    print("  - pandas: OK")
    print("  - numpy: OK")
    print("  - python-dotenv: OK")
    
    # Test environment
    print("\n✓ Checking environment...")
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "your_api_key_here":
        print(f"  - GEMINI_API_KEY: Found (length: {len(api_key)})")
    else:
        print("  - GEMINI_API_KEY: NOT SET (you need to set this in .env)")
    
    # Test CSV data
    print("\n✓ Checking data file...")
    csv_path = os.path.join(os.path.dirname(__file__), "data", "one_min_price.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  - Data file found: {csv_path}")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
    else:
        print(f"  - Data file NOT found: {csv_path}")
    
    print("\n" + "=" * 50)
    print("Setup verification complete!")
    print("\nNext steps:")
    print("1. Add your GEMINI_API_KEY to the .env file")
    print("2. Run: python rag.py")
    print("3. Start querying your data!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease run: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
