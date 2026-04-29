# FaceSwap Pro Local

A local web application for high-quality face-head swapping. Everything runs locally on your CPU — no cloud APIs, no GPU required.

## Features
- **Robust Detection:** Uses UniFace (YOLOv8 + 106pt Landmark Refinement).
- **Identity Transfer:** Uses Inswapper 128 for natural face replacement.
- **Seamless Blending:** XSeg masking, LAB color correction, and Poisson blending.
- **Face Restoration:** Optional GFPGAN v1.4 enhancement for crisp results.
- **Full Resolution:** Processes images at up to 1024px while maintaining quality.

## Project Structure
```
faceswap-app/
├── backend/            # FastAPI application logic
├── frontend/           # Vanilla JS/HTML/CSS UI
├── models/             # ONNX model files (download required)
├── templates/          # Pre-loaded target scenes
└── requirements.txt
```

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 2. Download Required Models
Only two models need manual download. The UniFace models will auto-download on first run.

**Inswapper 128 (Identity Transfer):**
- [Download inswapper_128.onnx](https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx)
- Place at: `models/inswapper_128.onnx`

**GFPGAN v1.4 (Restoration):**
- [Download gfpgan_1.4.onnx](https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx)
- Place at: `models/gfpgan_1.4.onnx`

### 3. Add Templates
Place your favorite target images (JPG/PNG) into the `templates/` directory. They will appear in the grid when you start the app.

### 4. Run the Application
```bash
# From the faceswap-app/ directory
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open in Browser
Visit [http://localhost:8000](http://localhost:8000)

## Implementation Details
- **Backend:** FastAPI
- **Inference:** ONNX Runtime (CPUExecutionProvider)
- **Face Analysis:** UniFace (YOLOv8, ArcFace, XSeg, LandmarkDetector)
- **Image Processing:** OpenCV, NumPy, SciPy

## Error Handling
- The app handles extreme side-profile angles (>65°) by rejecting them for better quality.
- If a face is found but angled moderately (40-65°), a warning is displayed.
- All errors (no face detected, file too small, etc.) surface clearly in the UI.

## License
MIT
