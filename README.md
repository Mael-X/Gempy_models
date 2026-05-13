# Sandbox Geological Visualization Setup

This project visualizes GemPy geological models in a physical sandbox using a Kinect v1 for depth sensing and a projector for display.

## Dependencies

### Python Packages
- numpy
- matplotlib
- pandas
- pyvista
- gempy
- gempy_viewer
- opencv-python
- freenect (for Kinect)

### System Dependencies (Linux)
- libfreenect-dev
- python3-dev
- pytorch (for GemPy)

## Installation

1. Install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install libfreenect-dev python3-dev
   ```

2. Install Python packages:
   ```bash
   pip install numpy matplotlib pandas pyvista gempy gempy_viewer opencv-python freenect
   ```

3. Install PyTorch (refer to https://pytorch.org/get-started/locally/)

## Usage

1. Run the model extraction to generate the lithology data:
   ```bash
   python model_extraction.py
   ```

2. Calibrate the Kinect depth if needed (edit sandbox_mapper.py).

3. Run the main visualization:
   ```bash
   python main.py
   ```

## Files

- `model_extraction.py`: Extracts lithology voxels from GemPy model
- `kinect_capture.py`: Handles Kinect depth and RGB capture
- `sandbox_mapper.py`: Maps depth data to lithology colors
- `main.py`: Main loop for real-time visualization
- `monoclinal_model.ipynb`: Original GemPy model notebook

## Calibration

The depth mapping needs calibration. In `sandbox_mapper.py`, adjust `depth_offset` and `depth_scale` to match Kinect depth (mm) to model z-coordinates.

For example, measure the depth at known z-heights in the sandbox.</content>
<parameter name="filePath">c:\Users\mael\Desktop\FabLab\README.md