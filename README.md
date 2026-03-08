# UAV Mapper

Stitch drone video footage into a single overhead map using OpenCV and PySide6.

Features:
- Frame extraction with duplicate filtering and optional blur rejection
- ORB feature matching with greedy frame selection
- Exposure normalisation (CLAHE) across frames before stitching
- Multi-band blending for smooth seams
- Auto-rotation and content cropping of the final map
- CUDA acceleration on NVIDIA GPUs, with automatic CPU fallback
- Interactive map viewer with zoom, pan, and marker placement

---

## Requirements

- Python 3.10
- Windows 10/11
- NVIDIA GPU with CUDA support (recommended) **or** any CPU (integrated graphics works, just slower)

---

## Setup — CPU / integrated graphics

1. Create a Python 3.10 virtual environment:
   ```
   py -3.10 -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   python -m pip install -r requirements.txt
   ```

3. Run:
   ```
   python src/main.py
   ```

---

## Setup — CUDA (NVIDIA GPU, recommended for best performance)

This requires building OpenCV from source with CUDA support. Do this once on your CUDA-capable machine.

### Prerequisites
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (12.x or newer)
- [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/) with the "Desktop development with C++" workload
- [CMake](https://cmake.org/download/) added to system PATH
- [Git](https://git-scm.com/)

### Build steps

1. Clone OpenCV source:
   ```
   cd C:\
   mkdir opencv-build && cd opencv-build
   git clone https://github.com/opencv/opencv.git
   git clone https://github.com/opencv/opencv_contrib.git
   mkdir build && cd build
   ```

2. Install numpy for the system Python 3.10:
   ```
   C:\Users\<you>\AppData\Local\Programs\Python\Python310\python.exe -m pip install "numpy<2"
   ```

3. Run CMake (replace the Python path and `CUDA_ARCH_BIN` with your GPU's compute capability — see table below):
   ```
   cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DOPENCV_DNN_CUDA=OFF -DCUDA_ARCH_BIN=7.5 -DWITH_OPENCL=OFF -DOPENCV_EXTRA_MODULES_PATH="C:/opencv-build/opencv_contrib/modules" -DBUILD_opencv_python3=ON -DINSTALL_PYTHON_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF -DPYTHON3_EXECUTABLE="C:/Users/<you>/AppData/Local/Programs/Python/Python310/python.exe" "C:/opencv-build/opencv"
   ```

4. Build (takes 1–3 hours):
   ```
   cmake --build C:\opencv-build\build --config Release --parallel 8
   ```

5. Install to a clean location:
   ```
   cmake --install C:\opencv-build\build --config Release --prefix C:\opencv
   ```

6. Add OpenCV to your system PATH (run PowerShell as administrator, then restart your terminal):
   ```
   [System.Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\opencv\x64\vc17\bin", "Machine")
   ```

7. Remove the `opencv-python` line from `requirements.txt`, then install the rest:
   ```
   python -m pip install "numpy<2" PySide6
   ```

8. Copy the built `.pyd` into your venv:
   ```
   copy C:\opencv-build\build\lib\python3\Release\cv2.cp310-win_amd64.pyd ".venv\Lib\site-packages\cv2\"
   ```

9. Verify CUDA is working:
   ```python
   import cv2
   print(cv2.cuda.getCudaEnabledDeviceCount())  # should print 1
   ```

10. Run:
    ```
    python src/main.py
    ```

### CUDA compute capability by GPU

| GPU             | CUDA_ARCH_BIN |
|-----------------|---------------|
| RTX 2070 Super  | 7.5           |
| RTX 3080        | 8.6           |
| RTX 4090        | 8.9           |

Check your GPU at: https://developer.nvidia.com/cuda-gpus

---

## Usage

1. Click **Select Video** and choose a drone footage file (`.mp4`, `.mov`, `.avi`)
2. Adjust parameters in the sidebar if needed
3. Click **Generate Map**
4. The stitched map saves automatically as `stitched_map.png` in the working directory
5. Use **Save As** to save to a custom location or format

---

## Parameters

Default values are automatically chosen based on whether a CUDA GPU is detected.

### Frame Extraction

| Parameter       | GPU default | CPU default | Description |
|-----------------|-------------|-------------|-------------|
| Seconds step    | 0.33        | 0.5         | Time between sampled frames. Lower = denser sampling and better coverage. |
| Max frames      | 120         | 60          | Hard cap on extracted frames. |
| Extract MP      | 4.0         | 2.0         | Megapixels to downscale frames to at extraction time. |
| Similarity thr  | 10.0        | 10.0        | Frames with a mean pixel difference below this are dropped as near-duplicates. |
| Blur threshold  | 0 (off)     | 0 (off)     | Laplacian variance below this rejects a frame as too blurry. 0 disables the check. |

### Stitching

| Parameter       | GPU default | CPU default | Description |
|-----------------|-------------|-------------|-------------|
| Mode            | panorama    | panorama    | Spherical warp (panorama) handles UAV parallax better than flat-scan mode. |
| Work MP         | 3.0         | 1.5         | Internal stitching resolution. Higher = sharper output but much slower on CPU. |
| ORB features    | 8000        | 3000        | Keypoints detected per frame. More = more robust matching. |
| Min keypoints   | 150         | 100         | Frames with fewer keypoints than this are discarded before stitching. |

### Frame Selection

| Parameter        | Default | Description |
|------------------|---------|-------------|
| Min motion px    | 5.0     | Frame pairs with less motion than this are rejected (avoids near-identical pairs). |
| Target motion px | 25.0    | Ideal inter-frame motion. Pairs close to this value are scored higher. |
| Max stitch frames | 80 / 40 | How many frames are passed to the final stitcher after selection. |

---

## Tips

- **Stitching fails or produces few frames** — lower `Seconds step` to get more frames, or lower `Min motion px` if the drone was moving slowly.
- **Output is blurry or has seams** — raise `Work MP` and `ORB features` if on GPU. Make sure lighting was consistent during the flight.
- **Too slow on CPU** — lower `Work MP` to 1.0 and `ORB features` to 2000. Reduce `Max stitch frames` to 30.
- **Blur threshold** — leave at 0 for typical drone footage which is naturally soft. Raise to 20–50 only if you have very sharp footage and want to filter out motion-blurred frames.
