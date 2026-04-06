from __future__ import annotations

from pathlib import Path
import importlib.metadata
import importlib.util
import logging
import shutil
import subprocess
import sys
import time
import urllib.request


RUNTIME_IMPORT_TO_PIP = [
    ("numpy", "numpy"),
    ("onnxruntime", "onnxruntime"),
    ("onnx", "onnx"),
    ("insightface", "insightface"),
    ("cv2", "opencv-python"),
]


def is_import_available(module_name: str) -> bool:
    """Return True if a module can be imported without importing it now."""
    return importlib.util.find_spec(module_name) is not None


def is_pip_package_installed(package_name: str) -> bool:
    """Return True if a pip package exists in the current environment."""
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def run_pip(args: list[str]) -> bool:
    """Execute pip and return False if installation fails."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", *args])
        return True
    except subprocess.CalledProcessError as exc:
        print("Dependency installation failed.")
        print(f"Command: {sys.executable} -m pip {' '.join(args)}")
        print(f"Pip exit code: {exc.returncode}")
        print("Tip: close any running Python app (PyCharm console/Jupyter), then run again.")
        return False


def is_running_in_virtualenv() -> bool:
    """Return True when executing inside a virtual environment."""
    return hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix


def ensure_runtime_dependencies() -> int:
    """Install missing dependencies before importing heavy runtime modules.

    If any package is installed/removed, the script exits so the user can re-run
    in a clean interpreter (important on Windows to avoid file lock conflicts).
    """
    missing = [pip_name for module_name, pip_name in RUNTIME_IMPORT_TO_PIP if not is_import_available(module_name)]
    environment_changed = False
    install_prefix = ["install"] + ([] if is_running_in_virtualenv() else ["--user"])

    if missing:
        print(f"Missing packages detected: {', '.join(missing)}")
        print("Installing missing packages...")
        if not run_pip([*install_prefix, "--upgrade", "--force-reinstall", *missing]):
            return 1
        still_missing = [
            pip_name
            for module_name, pip_name in RUNTIME_IMPORT_TO_PIP
            if pip_name in missing and not is_import_available(module_name)
        ]
        if still_missing:
            print(f"Installation completed but imports still missing: {', '.join(still_missing)}")
            print("Try using Python 3.12/3.13 with a clean virtual environment, then run again.")
            return 1
        environment_changed = True

    # Keep OpenCV GUI-capable for preview windows.
    if is_pip_package_installed("opencv-python-headless"):
        print("Removing 'opencv-python-headless' to keep GUI preview support...")
        if not run_pip(["uninstall", "-y", "opencv-python-headless"]):
            return 1
        environment_changed = True
        if not run_pip([*install_prefix, "opencv-python"]):
            return 1

    if environment_changed:
        print("Dependencies updated. Please run the script again.")
        return 2

    return 0


print("Startup check: on first run, dependencies may be installed and you may be asked to run this script one more time.")
bootstrap_status = ensure_runtime_dependencies()
if bootstrap_status != 0:
    raise SystemExit(0 if bootstrap_status == 2 else 1)

import cv2
import insightface
import numpy as np
import onnxruntime as ort
from insightface import model_zoo
from insightface.app import FaceAnalysis


INSWAPPER_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
LOG_FORMAT = "%(levelname)s: %(message)s"


def configure_logging() -> logging.Logger:
    """Configure console logging for the interactive workflow."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logging.getLogger("face_swap")


def sort_faces_left_to_right(faces):
    """Keep face indexes stable by sorting detections left-to-right."""
    return sorted(faces, key=lambda face: face.bbox[0])


def ensure_resources_dir(base_dir: Path, logger: logging.Logger) -> Path:
    """Create the resources folder if it does not already exist."""
    resources_dir = base_dir / "resources"
    if not resources_dir.exists():
        resources_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created missing folder: %s", resources_dir)
    return resources_dir


def ensure_inswapper_model(logger: logging.Logger) -> Path:
    """Ensure the face-swap model exists locally and return its path."""
    model_path = Path.home() / ".insightface" / "models" / "inswapper_128.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        logger.info("Downloading model to %s", model_path)
        urllib.request.urlretrieve(INSWAPPER_URL, str(model_path))

    return model_path


def get_provider_order():
    """Prefer GPU if ONNX Runtime exposes it; otherwise use CPU."""
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def init_face_analysis(provider_order, logger: logging.Logger):
    """Initialize FaceAnalysis, retrying on CPU if GPU init fails."""
    try:
        app = FaceAnalysis(name="buffalo_l", providers=provider_order)
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app, provider_order
    except Exception as exc:
        logger.warning("GPU initialization failed (%s). Falling back to CPU.", exc)
        cpu_only = ["CPUExecutionProvider"]
        app = FaceAnalysis(name="buffalo_l", providers=cpu_only)
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app, cpu_only


def init_swapper(model_path: Path, provider_order, logger: logging.Logger):
    """Initialize the swapper model, retrying on CPU if GPU init fails."""
    try:
        return model_zoo.get_model(str(model_path), providers=provider_order), provider_order
    except Exception as exc:
        logger.warning("Swapper GPU init failed (%s). Falling back to CPU.", exc)
        cpu_only = ["CPUExecutionProvider"]
        return model_zoo.get_model(str(model_path), providers=cpu_only), cpu_only


def prompt_and_copy_if_missing(image_path: Path, label: str, logger: logging.Logger) -> Path:
    """Prompt for a missing image path and copy the selected file into resources."""
    if image_path.exists():
        return image_path

    while True:
        user_input = input(
            f"Missing {label} at '{image_path}'. Enter full path to the {label} image (or press Enter to quit): "
        ).strip().strip('"')
        if not user_input:
            raise RuntimeError(f"{label} is required to continue.")

        candidate = Path(user_input)
        if not candidate.exists():
            logger.error("Path does not exist. Try again.")
            continue

        img = cv2.imread(str(candidate))
        if img is None:
            logger.error("File could not be read as an image. Try again.")
            continue

        shutil.copy2(str(candidate), str(image_path))
        logger.info("Copied %s to %s", label, image_path)
        return image_path


def load_image(path: Path, logger: logging.Logger):
    """Load an image from disk and return the numpy array or None."""
    img = cv2.imread(str(path))
    if img is None:
        logger.error("Failed to read %s", path)
    return img


def detect_faces(app: FaceAnalysis, source_img, target_img, logger: logging.Logger):
    """Detect faces in both images and fail fast if either image has none."""
    source_faces = app.get(source_img)
    target_faces = app.get(target_img)

    if not source_faces:
        logger.error("No face found in source image.")
        return None, None
    if not target_faces:
        logger.error("No face found in target image.")
        return None, None

    logger.info("Detected %d face(s) in source and %d face(s) in target.", len(source_faces), len(target_faces))
    return source_faces, target_faces


def write_face_preview(image, faces_sorted, preview_path: Path, logger: logging.Logger):
    """Draw face boxes and index labels on a full image."""
    preview = image.copy()
    for i, face in enumerate(faces_sorted):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(preview, str(i), (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(str(preview_path), preview)
    logger.info("Preview saved: %s", preview_path)
    return preview


def write_face_crops_grid(image, faces_sorted, crops_path: Path, logger: logging.Logger, thumb_size=180, cols=6):
    """Create a tiled gallery of face crops to help the user confirm detection."""
    if not faces_sorted:
        return None

    tiles = []
    height, width = image.shape[:2]

    for i, face in enumerate(faces_sorted):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop = image[y1:y2, x1:x2]

        # Some detections can sit on the edge of the image; use a placeholder if crop is empty.
        if crop.size == 0:
            crop = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)

        cv2.putText(crop, f"idx {i}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        tiles.append(crop)

    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)

    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        y0 = r * thumb_size
        x0 = c * thumb_size
        canvas[y0 : y0 + thumb_size, x0 : x0 + thumb_size] = tile

    cv2.imwrite(str(crops_path), canvas)
    logger.info("Face crops grid saved: %s", crops_path)
    return canvas


def show_image_preview(window_title: str, img, logger: logging.Logger, timeout_ms: int = 5000):
    """Show an image automatically and continue after close or timeout.

    The preview window opens by default so the user can see the detected faces
    immediately. The app proceeds when the user closes the window, presses a
    key, or the timeout expires.
    """
    if img is None:
        return

    try:
        cv2.imshow(window_title, img)
        start = time.time()
        while True:
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(100) != -1:
                break
            if (time.time() - start) * 1000 >= timeout_ms:
                break
        cv2.destroyWindow(window_title)
    except cv2.error as exc:
        logger.warning("Could not open window '%s' (%s).", window_title, exc)


def prompt_face_selection(face_count: int, label: str, allow_all: bool, logger: logging.Logger):
    """Prompt for a face index.

    If allow_all is True, pressing Enter means "select all faces".
    If allow_all is False, pressing Enter means "use the only available face".
    """
    if face_count == 1:
        logger.info("Only one %s face found; using index 0.", label)
        return 0

    if allow_all:
        prompt = f"Enter {label} face index (0 to {face_count - 1}), or press Enter to swap ALL faces: "
    else:
        prompt = f"Enter {label} face index (0 to {face_count - 1}): "

    while True:
        text = input(prompt).strip()
        if text == "":
            if allow_all:
                logger.info("User selected ALL %s faces.", label)
                return None
            logger.error("A %s face index is required.", label)
            continue

        try:
            idx = int(text)
        except ValueError:
            logger.error("Please enter an integer index.")
            continue

        if idx < 0 or idx >= face_count:
            logger.error("Index out of range. Use 0 to %d.", face_count - 1)
            continue

        logger.info("User selected %s face index %d.", label, idx)
        return idx


def swap_faces(target_img, source_face, faces_to_swap, swapper):
    """Apply the source face to one or more target faces."""
    result = target_img.copy()
    for face in faces_to_swap:
        result = swapper.get(result, face, source_face, paste_back=True)
    return result


def main() -> int:
    logger = configure_logging()
    base_dir = Path(__file__).resolve().parent
    resources_dir = ensure_resources_dir(base_dir, logger)

    source_path = resources_dir / "swapFrom.jpg"
    target_path = resources_dir / "swapTo.jpg"
    output_path = resources_dir / "swapped.jpg"
    source_preview_path = resources_dir / "swapFrom_faces_indexed.jpg"
    source_crops_path = resources_dir / "swapFrom_faces_crops.jpg"
    target_preview_path = resources_dir / "swapTo_faces_indexed.jpg"
    target_crops_path = resources_dir / "swapTo_faces_crops.jpg"

    try:
        prompt_and_copy_if_missing(source_path, "swapFrom.jpg", logger)
        prompt_and_copy_if_missing(target_path, "swapTo.jpg", logger)
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    logger.info("insightface %s", insightface.__version__)
    logger.info("numpy %s", np.__version__)

    source_img = load_image(source_path, logger)
    target_img = load_image(target_path, logger)
    if source_img is None or target_img is None:
        return 1

    provider_order = get_provider_order()
    logger.info("Preferred providers: %s", provider_order)

    app, app_providers = init_face_analysis(provider_order, logger)
    logger.info("FaceAnalysis providers in use: %s", app_providers)

    source_faces, target_faces = detect_faces(app, source_img, target_img, logger)
    if not source_faces or not target_faces:
        return 1

    source_faces_sorted = sort_faces_left_to_right(source_faces)
    source_preview_img = write_face_preview(source_img, source_faces_sorted, source_preview_path, logger)
    source_crops_img = write_face_crops_grid(source_img, source_faces_sorted, source_crops_path, logger)

    logger.info("Detected %d source face(s).", len(source_faces_sorted))
    show_image_preview("Source Faces - Indexed", source_preview_img, logger)
    show_image_preview("Source Faces - Crops", source_crops_img, logger)

    source_index = prompt_face_selection(len(source_faces_sorted), "source", allow_all=False, logger=logger)
    source_face = source_faces_sorted[source_index]

    target_faces_sorted = sort_faces_left_to_right(target_faces)
    target_preview_img = write_face_preview(target_img, target_faces_sorted, target_preview_path, logger)
    target_crops_img = write_face_crops_grid(target_img, target_faces_sorted, target_crops_path, logger)

    logger.info("Detected %d target face(s).", len(target_faces_sorted))
    show_image_preview("Target Faces - Indexed", target_preview_img, logger)
    show_image_preview("Target Faces - Crops", target_crops_img, logger)

    target_selection = prompt_face_selection(len(target_faces_sorted), "target", allow_all=True, logger=logger)
    if target_selection is None:
        faces_to_swap = target_faces_sorted
        logger.info("Swapping source face into ALL detected target faces.")
    else:
        faces_to_swap = [target_faces_sorted[target_selection]]
        logger.info("Swapping source face into target index: %d", target_selection)

    swapper_model_path = ensure_inswapper_model(logger)
    swapper, swapper_providers = init_swapper(swapper_model_path, provider_order, logger)
    logger.info("Swapper providers in use: %s", swapper_providers)

    swapped = swap_faces(target_img, source_face, faces_to_swap, swapper)
    show_image_preview("Swapped Result", swapped, logger)

    if not cv2.imwrite(str(output_path), swapped):
        logger.error("Failed to write %s", output_path)
        return 1

    cv2.destroyAllWindows()
    logger.info("Success: saved %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
