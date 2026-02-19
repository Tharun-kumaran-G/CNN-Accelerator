## Software Design

### Software Stack  
The software stack is built around the PYNQ framework and provides a Python-based control layer on the ARM processor for managing data movement and accelerator execution.

- **PYNQ Framework:** FPGA overlay management and register-level control  
- **Python:** High-level application logic and orchestration of inference  
- **C/C++ (HLS):** Hardware accelerator implementation  
- **OpenCV:** Image loading, resizing, and basic preprocessing  
- **NumPy:** Data formatting and optional CPU-side baseline inference  

This stack enables rapid prototyping and direct interaction with custom HLS IP cores deployed on the PYNQ-Z2 platform.

---

### ARM Control Flow  
The ARM Cortex-A9 processor acts as the system controller and communicates with the CNN accelerator using **AXI memory-mapped interfaces through AXI Interconnect and SmartConnect**. Since the design does not use AXI DMA, all data transfers are performed through memory-mapped transactions.

The ARM-side control flow includes:

1. Loading the FPGA bitstream (overlay) at runtime  
2. Allocating input and output buffers in shared DDR memory  
3. Writing image data and model parameters to memory locations accessible by the accelerator  
4. Configuring accelerator control registers via AXI-Lite  
5. Triggering accelerator execution  
6. Polling the status registers to detect inference completion  
7. Reading back classification results and metadata from shared memory  

This approach simplifies system integration at the cost of higher data transfer overhead compared to DMA-based designs, which is acceptable for lightweight CNN workloads on embedded platforms.

---

### Preprocessing  
Input images are preprocessed on the ARM processor before being passed to the FPGA accelerator:

- Resize images to **64×64 grayscale**  
- Convert to unsigned 8-bit integer format  
- Normalize or scale pixel values according to the trained model  
- Flatten the image into a contiguous buffer in DDR memory  

Preprocessing overhead is kept minimal so that the majority of computation is handled by the FPGA accelerator.

---

### Inference Pipeline  
The complete software-driven inference pipeline is:

1. Load input image from dataset or camera  
2. Preprocess image on ARM processor  
3. Write input data to shared DDR memory  
4. Trigger CNN accelerator execution via AXI-Lite control interface  
5. Accelerator reads input and weights through AXI Interconnect / SmartConnect  
6. Accelerator performs CNN inference in hardware  
7. ARM reads back classification results and auxiliary outputs  
8. Post-process and display predicted class and confidence  

---

### Deployment (PYNQ-Z2)  
To deploy and execute the system on the PYNQ-Z2 board:

1. Flash the PYNQ image to the SD card and boot the board  
2. Copy the FPGA bitstream (`.bit`) and hardware description (`.hwh`) to the board  
3. Transfer Python inference scripts and trained weight files to the board  
4. Load the FPGA overlay using the PYNQ framework  
5. Run the Python application to perform CNN inference  
6. Observe classification results and timing information via the terminal  

This deployment approach enables rapid testing and evaluation of the FPGA-accelerated CNN system on embedded hardware without requiring complex driver development.
