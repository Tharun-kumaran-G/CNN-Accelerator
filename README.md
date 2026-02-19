# Real-Time CNN Acceleration on PYNQ-Z2 (Zynq-7000 SoC)

## Overview  
This project presents a **hardware-accelerated Convolutional Neural Network (CNN) inference system** deployed on the **PYNQ-Z2 (Zynq-7000 SoC)** platform. The design follows a **hardware–software co-design approach**, where compute-intensive CNN layers are offloaded to the FPGA fabric, while the ARM Cortex-A9 processor manages image preprocessing, system control, and post-processing.  

The primary goal is to demonstrate **real-time or near real-time inference** for a **2-class image classification task** on an embedded platform, with clear performance gains over a CPU-only implementation. The system highlights the practicality of deploying machine learning inference on resource-constrained edge devices using FPGA acceleration.

---

## Objectives  
- Design and implement an end-to-end CNN inference pipeline on the PYNQ-Z2 platform  
- Accelerate compute-intensive CNN layers using FPGA fabric via High-Level Synthesis (HLS)  
- Integrate ARM–FPGA communication using AXI interfaces and DMA  
- Compare performance between CPU-only and FPGA-accelerated inference  
- Achieve real-time or near real-time inference suitable for embedded edge applications  

---

## Hardware Platform  
- **Board:** PYNQ-Z2 (Zynq-7000 XC7Z020)  
- **Processor:** Dual-core ARM Cortex-A9  
- **FPGA Fabric:** Custom CNN accelerator synthesized using HLS  
- **Interfaces:** AXI-Lite (control), AXI DMA / AXI-Stream (data transfer)  
- **Input:** Image dataset (or live camera feed, if applicable)  
- **Output:** Classification result displayed on console / output visualization  

---

## CNN Model  
- **Model Type:** Lightweight custom CNN optimized for FPGA deployment  
- **Task:** 2-class image classification  
- **Input Resolution:** As defined by the trained model (e.g., 32×32 or 64×64)  
- **Quantization:** Fixed-point / INT8 quantization for hardware efficiency  
- **Deployment:** Trained offline and weights exported for FPGA inference  

---

## System Architecture  
The system follows a **hardware–software co-design paradigm**:

**ARM Processor (Processing System – PS):**  
- Image loading and preprocessing  
- Control of FPGA accelerator  
- Post-processing and result interpretation  

**FPGA Fabric (Programmable Logic – PL):**  
- Convolution operations  
- Activation functions  
- Pooling layers  

**High-Level Data Flow:**  

Detailed architecture diagrams and explanations are provided in:  
`docs/system_architecture.md`

---

## Performance Summary  

| Metric        | CPU Only | FPGA Accelerated | Speedup |
|---------------|----------|------------------|---------|
| Latency (ms)  | XX       | XX               | XX×     |
| Throughput    | XX FPS   | XX FPS           | XX×     |
| Power (W)     | XX       | XX               | XX      |

> Replace XX values with your measured results.

---

## FPGA Resource Utilization  

| Resource | Utilization |
|----------|-------------|
| LUTs     | XX %        |
| BRAM     | XX %        |
| DSPs     | XX %        |

---

## Demo  
Demo video and screenshots showcasing the working system are provided in:  
`demo/demo_video_link.txt`

---

## How to Run  

1. Flash the PYNQ image to the SD card and boot the PYNQ-Z2 board  
2. Program the FPGA with the provided bitstream  
3. Run the Python inference script or notebook on the ARM processor  
4. Observe classification output and performance metrics  

Detailed setup and execution steps are available in:  
`docs/software_design.md`

---

## Results  
Sample outputs, prediction results, and accuracy metrics are documented in:  
`docs/results.md`

---

## Conclusion  
This project demonstrates the **effectiveness of FPGA-based CNN acceleration on embedded platforms**, achieving significant improvements in inference latency and throughput compared to CPU-only execution. The design validates the feasibility of deploying efficient machine learning inference pipelines on edge devices using hardware acceleration.

---

## Team  
- Name 1  
- Name 2  
- Name 3  

---

## License  
MIT License  

