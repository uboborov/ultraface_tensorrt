# ultraface_tensorrt
### TensorRT implementation of the UltraFace detector 
https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface
**33 FPS** on jetson nano with **```version-RFB-640.onnx```**
### requires:
CUDA >= 10.2
TensorRT >= 8.0
OpenCV >= 4.4.x
### prepare:
Copy file **version-RFB-320.onnx** or **version-RFB-640.onnx** to the project_dir/**ultraface.onnx**
Uncomment **```#define USE_RFB_640```** or **```#define USE_RFB_320```** in **ultraface.cpp**
### build on target:
```
git clone git@github.com:uboborov/ultraface_tensorrt.git
cd ultraface_tensorrt
mkdir build && cd build
cmake ..
make
```
### run:
To convert from ONNX to TensorRT engine
```
./ultraface -s
```
To infer with TensorRT engine
```
./ultraface -d
```
