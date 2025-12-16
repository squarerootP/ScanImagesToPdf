# PDF Generator

A document scanning tool that uses YOLO-based segmentation to detect, crop, and straighten document images, then compiles them into a PDF.

## Features

- Automatic document detection and segmentation using YOLO
- Oriented bounding box (OBB) cropping to straighten tilted documents
- Automatic upscaling to a uniform page size (1000x1400)
- Batch processing of image directories
- Generates both processed and original comparison PDFs
- Simple Tkinter GUI for easy operation

## Requirements

- Python 3.x (see `.python-version`)
- [uv](https://github.com/astral-sh/uv) for project management

## Installation

### Install uv

**Option 1: Standalone installer (recommended)**

```sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Option 2: Install via pip**

```sh
pip install uv
```

### Install project dependencies

```sh
uv sync
```

This will install all dependencies defined in `pyproject.toml`.

## Usage

Run the application:

```sh
uv run full_run.py
```

A GUI window will appear with the following options:

1. **Image Directory** - Select the folder containing your document images
2. **Output Directory** - Select where to save the generated PDF
3. **Output Filename** - Enter the name for the output PDF (default: `output.pdf`)

Click **Generate PDF** to process the images.

## Input Requirements

- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif`
- Images should be named so that alphabetical sorting produces the correct page order (e.g., `001.jpg`, `002.jpg`, ... or `page_1.jpg`, `page_2.jpg`, ...)

## Output

- `<filename>.pdf` - Processed PDF with cropped and straightened documents
- `<filename>_original.pdf` - Comparison PDF with original images resized to the same dimensions

## Model

The application uses a YOLO model (`best (4).pt`) for document segmentation. Ensure this file is present in the project root.

## Testing

Several test directories with sample images are included for testing:

- `test_doc1/`
- `test_doc2/`
- `test_doc3/`
- `sparse_singular_test1/`

Use these folders as input when running the application to verify functionality.

## Project Structure

```
.
├── full_run.py          # Main application with GUI
├── main.py              # Alternative entry point
├── best (4).pt          # YOLO model weights
├── pyproject.toml       # Project dependencies
├── .python-version      # Python version specification
└── temp_processed/      # Temporary directory for processed images (created at runtime)
```