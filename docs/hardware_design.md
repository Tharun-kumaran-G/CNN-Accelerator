## Hardware Design

### Accelerator Overview  
The CNN accelerator is implemented as a custom HLS-generated IP core deployed on the FPGA fabric of the PYNQ-Z2 (Zynq-7000 SoC). The accelerator performs end-to-end inference for a lightweight **3-layer CNN** designed for **2-class image classification (e.g., Person vs Object)** on **64×64 grayscale input images**.

The following CNN layers are fully implemented in hardware:

- Convolution Layer 1:  
  - Input: 64×64×1  
  - Kernel: 3×3  
  - Filters: 8  
  - Activation: ReLU  
  - Max Pooling: 2×2  

- Convolution Layer 2:  
  - Input: 31×31×8  
  - Kernel: 3×3  
  - Filters: 16  
  - Activation: ReLU  
  - Max Pooling: 2×2  

- Convolution Layer 3:  
  - Input: 14×14×16  
  - Kernel: 3×3  
  - Filters: 32  
  - Activation: ReLU  
  - Max Pooling: 2×2  

- Fully Connected Layer:  
  - Input: 6×6×32 (flattened)  
  - Output: 64 neurons  

- Output Layer:  
  - 2-class classification (Class 0 / Class 1)  
  - ArgMax decision logic implemented in hardware  

In addition to classification, the accelerator performs **lightweight region localization** by identifying the spatial region of maximum activation in the final pooled feature map. This enables simple bounding-box estimation directly in hardware without requiring a separate detection network.

---

### HLS Implementation  
The accelerator is implemented using **Vitis HLS**, allowing the CNN to be described in C++ and synthesized into RTL.

- **HLS Tool:** Vitis HLS  
- **Numeric Representation:** INT8 weights and biases with fixed-point accumulation  
- **Intermediate Precision:** 16-bit and 32-bit fixed-point accumulators  
- **Interfaces:**  
  - AXI4 Master for image, weights, and result buffers  
  - AXI-Lite for control and configuration  

The design uses quantized arithmetic to significantly reduce memory footprint and improve computational efficiency while maintaining sufficient accuracy for 2-class classification.

---

### Vivado Block Design  
The accelerator is integrated into the Zynq system using Vivado IP Integrator with the following components:

- **Zynq Processing System (PS):**  
  - Executes host-side control software  
  - Handles image preprocessing and post-processing  

- **AXI DMA:**  
  - Transfers image data and result buffers between PS and PL  

- **CNN Accelerator IP:**  
  - Custom HLS-generated IP core performing CNN inference  

Multiple AXI memory-mapped interfaces are used to stream input images and model parameters from DDR memory to the accelerator and to return inference results back to the ARM processor.

---

### Resource Utilization  
The CNN accelerator utilizes on-chip FPGA resources to store intermediate feature maps and accelerate compute-intensive operations:

- **BRAM:** Used for buffering intermediate feature maps (conv and pooling outputs)  
- **DSP Blocks:** Used for multiply–accumulate (MAC) operations in convolution and fully connected layers  
- **LUTs and FFs:** Used for control logic, activation functions, and reduction operations  

Final utilization values can be obtained from the Vivado post-implementation report and should be documented here.

---

### Optimization Techniques  

The following HLS optimization techniques are applied to improve throughput and performance:

- **Loop Pipelining:**  
  Inner loops in convolution, pooling, and fully connected layers are pipelined to achieve an initiation interval (II) of 1 for key operations.

- **Parallelism:**  
  Independent filter and channel computations are structured to exploit spatial parallelism across feature maps.

- **On-Chip Buffering:**  
  Intermediate feature maps are stored in BRAM using dual-port memory bindings to reduce external memory accesses.

- **Array Partitioning:**  
  Fully connected layer outputs and classification scores are partitioned to enable parallel accumulation.

- **Fixed-Point Quantization:**  
  INT8 weights with fixed-point accumulation significantly reduce hardware complexity and improve performance compared to floating-point arithmetic.

These optimizations enable the accelerator to achieve near real-time inference performance on the PYNQ-Z2 platform while staying within the FPGA resource constraints.
