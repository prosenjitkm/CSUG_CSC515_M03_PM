# Interactive Face Swap Project (CSC515 Module 03)

Single-file Python project that blends two images by swapping one detected face from a source image into one or more detected faces in a target image.

This implementation is designed for coursework usability:
- It auto-checks/install runtime dependencies on first run.
- It shows indexed face previews so the user chooses exact face indexes.
- It supports GPU when available and safely falls back to CPU.
- It keeps all input/output artifacts in a local `resources/` folder.

---

## 1) Project Goal

Given:
- `swapFrom.jpg` (source image: the face to copy)
- `swapTo.jpg` (target image: where to place it)

The script:
1. Detects faces in both images.
2. Displays indexed visual previews for source and target faces.
3. Prompts user to choose:
   - One source face index
   - One target face index, or all targets
4. Performs face swap blending with InsightFace `inswapper_128` model.
5. Saves final result as `swapped.jpg`.

---

## 2) Repository Architecture

```text
CSUG_CSC515_M03_PM/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── resources/
    ├── swapFrom.jpg                  # input source image
    ├── swapTo.jpg                    # input target image
    ├── swapped.jpg                   # final output
    ├── swapFrom_faces_indexed.jpg    # source: bounding boxes + indexes
    ├── swapFrom_faces_crops.jpg      # source: indexed face crop grid
    ├── swapTo_faces_indexed.jpg      # target: bounding boxes + indexes
    └── swapTo_faces_crops.jpg        # target: indexed face crop grid
```

### Main design layers in `main.py`

1. **Bootstrap / dependency layer**
   - `ensure_runtime_dependencies()`
   - Verifies and installs missing packages from `RUNTIME_IMPORT_TO_PIP`.
   - Removes `opencv-python-headless` if present so GUI windows (`cv2.imshow`) work.

2. **Environment and model setup layer**
   - `ensure_resources_dir()`
   - `ensure_inswapper_model()`
   - `get_provider_order()`
   - `init_face_analysis()` and `init_swapper()`

3. **Image/face processing layer**
   - `load_image()`
   - `detect_faces()`
   - `sort_faces_left_to_right()`
   - `write_face_preview()`
   - `write_face_crops_grid()`

4. **Interactive UX layer**
   - `show_image_preview()`
   - `prompt_face_selection()`
   - `prompt_and_copy_if_missing()`

5. **Swap execution layer**
   - `swap_faces()`
   - `main()` orchestrates end-to-end flow.

---

## 3) End-to-End Execution Flow

1. **Startup dependency check**
   - Script checks imports and installs missing packages.
   - If environment changed, script exits with message: run again.

2. **Resources validation**
   - Ensures `resources/` exists.
   - Expects:
     - `resources/swapFrom.jpg`
     - `resources/swapTo.jpg`
   - If either is missing, prompts for full file path and copies selected file into `resources/` using expected names.

3. **Provider selection (GPU first)**
   - Uses ONNX Runtime providers.
   - Tries `CUDAExecutionProvider` + CPU fallback chain.
   - If GPU initialization fails, automatically retries CPU-only.

4. **Face detection**
   - Detects faces in source and target.
   - Fails fast if either has zero faces.

5. **Preview generation + display**
   - Creates indexed bounding-box preview image.
   - Creates tiled face-crops gallery with face indexes.
   - Opens preview windows for user confirmation (auto-continues after close/key/timeout).

6. **Interactive index selection**
   - User picks source face index (required if multiple).
   - User picks target face index, or presses Enter to swap all detected target faces.

7. **Face swap + output**
   - Downloads `inswapper_128.onnx` if missing.
   - Applies swap and writes `resources/swapped.jpg`.
   - Shows final swapped preview window.

---

## 4) Inputs, Outputs, and Naming Convention

### Required logical inputs
- **Source face image**: `swapFrom.jpg`
- **Target image**: `swapTo.jpg`

If you provide differently named files, that is fine: the script asks for paths and copies them into `resources/` with canonical names.

### Generated outputs
- `swapped.jpg` (final)
- `swapFrom_faces_indexed.jpg`
- `swapFrom_faces_crops.jpg`
- `swapTo_faces_indexed.jpg`
- `swapTo_faces_crops.jpg`

These JPG artifacts are intentionally git-ignored in `.gitignore`.

---

## 5) Installation and Run

## Option A: recommended (virtual environment)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

## Option B: run directly (script auto-installs what is missing)

```powershell
python main.py
```

Notes:
- On first run, auto-install may modify packages and ask you to run `python main.py` again.
- On Windows, close other Python apps if package install fails due to file lock/permission issues.

---

## 6) Runtime Behavior Details

### Face index stability
Detected faces are sorted left-to-right (`sort_faces_left_to_right`) before indexing so indexes are more stable and understandable.

### Preview window behavior
`show_image_preview()` opens a window and proceeds when:
- user closes the window, or
- user presses any key, or
- timeout is reached (default 5000 ms).

### GPU/CPU fallback
- Preferred provider order is GPU then CPU.
- If GPU cannot initialize, warnings are logged and processing continues on CPU.

### Logging
The script logs key events at INFO level:
- detected face counts
- provider in use
- generated preview file paths
- final output status

---

## 7) Common Troubleshooting (Windows)

### `ModuleNotFoundError: onnxruntime` or other missing package
Run again so bootstrap can install dependencies:

```powershell
python main.py
```

If still failing, create a clean venv and install from requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

### Preview windows do not open
Cause is often `opencv-python-headless` being installed. This project attempts to remove it automatically and install `opencv-python`.

Manual fix if needed:

```powershell
python -m pip uninstall -y opencv-python-headless
python -m pip install --upgrade opencv-python
```

### Permission denied / file lock (example: `cv2.pyd`, WinError 5)
1. Close PyCharm Python console, Jupyter, and any process using the interpreter.
2. Retry installation/run.
3. Prefer a fresh virtual environment.

### GPU not used even though available
- Ensure CUDA-compatible ONNX Runtime package and matching NVIDIA/CUDA setup.
- The script will continue on CPU if GPU provider is unavailable or fails to initialize.

---

## 8) Assignment Mapping

This project satisfies the assignment intent of blending images using OpenCV + InsightFace by:
- combining two images into a synthetic output,
- enabling user-controlled face replacement,
- producing one executable Python entry file (`main.py`).

---

## 9) Limitations

- Swap quality depends on face angle, lighting, occlusion, and resolution.
- Extreme pose mismatch can produce artifacts.
- Group photos with small faces may reduce detection/swap quality.
- Model download (`inswapper_128.onnx`) requires internet once on first use.

---

## 10) Ethics and Usage Note

Use this project only for coursework, learning, or consent-based image editing. Avoid deceptive or harmful use of face-swapped media.

