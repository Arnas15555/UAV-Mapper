# UAV Mapper

A desktop application that stitches drone video footage into a single overhead map image. Frames are extracted, filtered for duplicates and blur, matched with ORB feature detection, and blended into a seamless panorama displayed in an interactive viewer.

**Features:**
- Time-based frame extraction with duplicate and blur filtering
- Greedy frame selection scored by overlap quality and inter-frame motion
- Exposure normalisation (CLAHE) and multi-band blending for smooth seams
- Auto-rotation and content cropping of the final map
- CUDA acceleration on NVIDIA GPUs with automatic CPU fallback
- Interactive map viewer with zoom and pan

---

## Project Structure

```
src/
├── main.py                  # Application entry point
├── gui/
│   ├── app_window.py        # Main window, toolbar, sidebar, and pipeline wiring
│   └── map_view.py          # Zoomable/pannable map viewer
├── mapping/
│   ├── video_extractor.py   # Frame extraction, deduplication, and blur filtering
│   ├── stitcher.py          # ORB feature matching, frame selection, and panorama stitching
│   └── postprocess.py       # Black-border cropping, auto-rotation, and content cropping
└── utils/
    ├── config.py            # Default parameter values (GPU/CPU adaptive)
    └── pipeline_worker.py   # Background threads for the pipeline and file saving
```

---

## Requirements

- Python 3.10
- Windows 10/11
- NVIDIA GPU with CUDA support (recommended) **or** any CPU

---

## Setup — CPU / integrated graphics

1. Create a virtual environment:
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

## Setup — CUDA (NVIDIA GPU)

This requires building OpenCV from source with CUDA support.

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

3. Run CMake — replace `<you>` with your username and set `CUDA_ARCH_BIN` to your GPU's compute capability (find yours at https://developer.nvidia.com/cuda-gpus):
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

---

## Usage

1. Click **Select Video** and choose a drone footage file (`.mp4`, `.mov`, `.avi`)
2. Adjust parameters in the sidebar if needed (defaults work well for most footage)
3. Click **Generate Map**
4. The map saves automatically as `stitched_map.png` in the working directory
5. Use **Save As** to export to a custom location or format

---

## Parameters

Defaults are automatically tuned based on whether a CUDA GPU is detected.

### Frame Extraction

| Parameter       | GPU default | CPU default | Description |
|-----------------|-------------|-------------|-------------|
| Seconds step    | 0.33        | 0.5         | Time between sampled frames. Lower = denser sampling. |
| Max frames      | 120         | 60          | Hard cap on extracted frames. |
| Extract MP      | 4.0         | 2.0         | Resolution (megapixels) to downscale frames to at extraction time. |
| Similarity thr  | 10.0        | 10.0        | Frames with mean pixel difference below this are dropped as near-duplicates. |
| Blur threshold  | 0 (off)     | 0 (off)     | Frames with Laplacian variance below this are rejected as blurry. 0 disables the check. |

### Stitching

| Parameter       | GPU default | CPU default | Description |
|-----------------|-------------|-------------|-------------|
| Mode            | panorama    | panorama    | Spherical warp handles UAV parallax better than flat-scan mode. |
| Work MP         | 3.0         | 1.5         | Internal stitching resolution. Higher = sharper but slower. |
| ORB features    | 8000        | 3000        | Keypoints detected per frame. More = more robust matching. |
| Min keypoints   | 150         | 100         | Frames with fewer keypoints than this are discarded before stitching. |

### Frame Selection

| Parameter         | GPU default | CPU default | Description |
|-------------------|-------------|-------------|-------------|
| Min motion px     | 5.0         | 5.0         | Frame pairs with less motion than this are rejected. |
| Target motion px  | 25.0        | 25.0        | Ideal inter-frame motion. Pairs close to this are scored higher. |
| Max stitch frames | 80          | 40          | How many frames are passed to the stitcher after selection. |

---

## Troubleshooting

- **Stitching fails or produces too few frames** — lower `Seconds step`, or lower `Min motion px` if the drone was moving slowly.
- **Output has visible seams** — raise `Work MP` and `ORB features`. Consistent lighting during the flight also helps.
- **Too slow on CPU** — lower `Work MP` to 1.0, `ORB features` to 2000, and `Max stitch frames` to 30.
- **Blur threshold** — leave at 0 for typical drone footage. Raise to 20–50 to filter motion-blurred frames.
