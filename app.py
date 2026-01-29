"""Entry point for Hugging Face Spaces deployment."""
import sys
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).parent))

# Import and run the main app
from app.assistant_v2 import main

if __name__ == "__main__":
    main()
