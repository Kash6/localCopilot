# Hugging Face Spaces Deployment Guide

## Quick Deploy

1. Create a new Space on Hugging Face:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Streamlit" as SDK
   - Select "Public" or "Private"

2. Clone and push your repository:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy all files from localCopilot
cp -r /path/to/localCopilot/* .

# Rename README_HF.md to README.md
mv README_HF.md README.md

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

3. Configure secrets (optional):
   - Go to your Space settings
   - Add secrets for Qdrant Cloud:
     - QDRANT_URL
     - QDRANT_API_KEY

## Files for Deployment

Required files:
- `app.py` - Entry point for HF Spaces
- `requirements.txt` - Python dependencies
- `packages.txt` - System dependencies
- `README.md` - Space description (use README_HF.md)
- `.streamlit/config.toml` - Streamlit theme
- `app/` - Application code
- `data/` - Sample code files

## GPU Upgrade (Optional)

Free tier runs on CPU. For GPU:
1. Go to Space settings
2. Select "Hardware" tab
3. Choose GPU tier (paid)
4. Restart Space

With GPU, the LoRA features and quantization will work.

## Testing Locally Before Deploy

```bash
# Test with CPU (simulates free HF Spaces)
streamlit run app.py
```

## Troubleshooting

If deployment fails:
1. Check build logs in HF Spaces
2. Verify all dependencies in requirements.txt
3. Ensure app.py is in root directory
4. Check that data folder has sample files

## Performance Notes

- CPU inference: ~10-30s per response
- GPU inference: ~2-5s per response
- First load: ~2-3 minutes (model download)
- Subsequent loads: ~30s (cached)
