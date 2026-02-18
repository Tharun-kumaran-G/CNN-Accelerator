# Real-Time CNN Acceleration on PYNQ-Z2 (Zynq-7000 SoC)

## Overview
This project implements a hardware-accelerated convolutional neural network (CNN) inference system on the PYNQ-Z2 board, which integrates an ARM Cortex-A9 processor with FPGA fabric. Compute-intensive CNN layers are offloaded to the FPGA to achieve real-time or near real-time performance, while the ARM processor handles image preprocessing, control logic, and post-processing. The system demonstrates measurable performance improvements compared to a CPU-only implementation.

## Objectives
- Design and implement a CNN inference pipeline on PYNQ-Z2  
- Accelerate convolutional layers using FPGA fabric via HLS  
- Compare CPU-only vs FPGA-accelerated performance  
- Achieve real-time or near real-time inference on embedded hardware  

## Hardware Platform
- Board: PYNQ-Z2 (Zynq-7000 XC7Z020)  
- Processor: Dual-core ARM Cortex-A9  
- FPGA Fabric: CNN accelerator  
- Interfaces: AXI Lite, AXI DMA / AXI Stream  
- Input: Camera / image dataset  
- Output: Display / console  

## CNN Model
- Model Type: (Custom CNN / MobileNet / Tiny CNN)  
- Task: (Image classification / Object detection)  
- Input Resolution: (e.g., 32×32 / 224×224)  
- Quantization: (e.g., INT8 / Fixed-point)  

## System Architecture
The system follows a hardware–software co-design approach:
- ARM Processor:
  - Image capture / dataset loading  
  - Preprocessing  
  - Control and post-processing  
- FPGA Fabric:
  - Convolution  
  - Activation  
  - Pooling  

High-level data flow:  
Input → ARM → FPGA Accelerator → ARM → Output  

Detailed architecture: `docs/system_architecture.md`

## Performance Summary

| Metric       | CPU Only | FPGA Accelerated | Speedup |
|--------------|----------|------------------|---------|
| Latency (ms) | XX       | XX               | XX×     |
| Throughput   | XX FPS   | XX FPS           | XX×     |
| Power (W)    | XX       | XX               | XX      |

## FPGA Resource Utilization

| Resource | Usage |
|----------|-------|
| LUTs     | XX %  |
| BRAM     | XX %  |
| DSPs     | XX %  |

## Demo
Demo video / screenshots:  
`demo/demo_video_link.txt`

## How to Run
1. Flash PYNQ image to SD card  
2. Program FPGA bitstream  
3. Run Python notebook / ARM application  
4. Observe inference output  

Detailed steps: `docs/software_design.md`

## Results
Sample outputs and accuracy results are provided in `docs/results.md`.

## Conclusion
This project demonstrates the effectiveness of FPGA-based CNN acceleration on embedded platforms, achieving significant speedup and improved efficiency compared to CPU-only inference.

## Team
- Name 1  
- Name 2  
- Name 3  

## License
MIT License
