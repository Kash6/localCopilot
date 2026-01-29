"""Entry point for Streamlit Cloud deployment."""
import sys
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).parent))

# Import and run the main app
if __name__ == "__main__":
    from app.assistant_v2 import main
    main()
