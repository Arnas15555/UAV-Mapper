# UAV Mapper

Stitch drone video footage into a single overhead map using OpenCV and PySide6.

---

## Requirements

- Python 3.10
- NVIDIA GPU with CUDA support (recommended) or any machine with integrated graphics (CPU fallback)
- Windows 10/11

---

## Setup — Standard (CPU / integrated graphics)

1. Create a Python 3.10 virtual environment:
   ```
   py -3.10 -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install "numpy<2" PySide6 opencv-python==4.10.0.84
   ```

3. Run:
   ```
   python src/main.py
   ```

---

## Setup — CUDA (NVIDIA GPU, recommended for best performance)

This requires building OpenCV from source. Do this once on your CUDA-capable machine.
cuDNN is not required — CUDA alone is sufficient for the GPU acceleration used by this project.

### Prerequisites
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (13.x recommended)
- [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/) with "Desktop development with C++" workload
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

3. Run CMake (replace Python path with yours, CUDA_ARCH_BIN=7.5 is for RTX 2070 Super — check yours at https://developer.nvidia.com/cuda-gpus):
   ```
   cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DOPENCV_DNN_CUDA=OFF "-DCUDA_ARCH_BIN=7.5" -DWITH_OPENCL=OFF -DOPENCV_EXTRA_MODULES_PATH="C:/opencv-build/opencv_contrib/modules" -DBUILD_opencv_python3=ON -DINSTALL_PYTHON_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF -DPYTHON3_EXECUTABLE="C:/Users/<you>/AppData/Local/Programs/Python/Python310/python.exe" "C:/opencv-build/opencv"
   ```

4. Build (takes 1-3 hours):
   ```
   cmake --build . --config Release --jobs 8
   ```

5. Copy the built `.pyd` into your venv:
   ```
   copy C:\opencv-build\build\lib\python3\Release\cv2.cp310-win_amd64.pyd ".venv\Lib\site-packages\cv2\"
   ```

6. Install remaining dependencies:
   ```
   pip install "numpy<2" PySide6
   ```

7. Verify CUDA is working:
   ```python
   import cv2
   print(cv2.cuda.getCudaEnabledDeviceCount())  # should print 1
   ```

8. Run:
   ```
   python src/main.py
   ```

### CUDA compute capability by GPU
- RTX 2070 Super: 7.5
- RTX 3080: 8.6
- RTX 4090: 8.9

Check your GPU at: https://developer.nvidia.com/cuda-gpus

---

## Usage

1. Click **Select Video** and choose a drone footage file (.mp4, .mov, .avi)
2. Adjust parameters if needed (see recommended settings below)
3. Click **Generate Map**
4. The stitched map is saved automatically as `stitched_map.png` in the working directory
5. Use **Save As** to save to a custom location

### Recommended settings for 4K 25fps shaky drone footage

| Parameter         | Value |
|-------------------|-------|
| Seconds step      | 0.25  |
| Max frames        | 70    |
| Extract MP        | 1.5   |
| Similarity thr    | 12.0  |
| Mode              | scans |
| Work MP           | 1.2   |
| Min keypoints     | 90    |
| ORB features      | 5000  |
