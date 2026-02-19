# ============================================================
# run_hls.tcl — Vitis HLS Synthesis Script
# CNN Accelerator: Cat vs Dog Classifier
# Target: PYNQ-Z2 (XC7Z020CLG400-1)
# Clock:  10 ns (100 MHz)
# ============================================================

# Create and open project
open_project cnn_detect_hls

# Set top-level function
set_top cnn_detect

# Add design files
add_files cnn_detect.cpp
add_files cnn_detect.h

# Add testbench (excluded from synthesis)
add_files -tb cnn_detect_tb.cpp

# Open solution targeting Vivado IP export
open_solution "solution1" -flow_target vivado

# Set target FPGA part
set_part {xc7z020clg400-1}

# Set clock constraint: 10 ns = 100 MHz
create_clock -period 10 -name default

# ============================================================
# Compiler Settings
# ============================================================
config_compile -name_max_length 80
config_interface -m_axi_latency 64

# ============================================================
# Step 1: C Simulation
#   Validates functional correctness before synthesis.
#   Runs cnn_detect_tb.cpp test cases.
# ============================================================
puts "========================================="
puts "Running C Simulation..."
puts "========================================="
csim_design -clean

# ============================================================
# Step 2: C Synthesis
#   Generates RTL (Verilog/VHDL) from HLS C++.
#   Produces timing, latency, and resource estimates.
# ============================================================
puts "========================================="
puts "Running HLS Synthesis..."
puts "========================================="
csynth_design

# ============================================================
# Step 3: Export IP
#   Packages synthesized design as Vivado IP catalog entry.
#   Output: cnn_detect_hls/solution1/impl/ip/
# ============================================================
puts "========================================="
puts "Exporting IP Catalog..."
puts "========================================="
export_design \
    -format      ip_catalog \
    -description "Cat/Dog CNN Accelerator for PYNQ-Z2" \
    -vendor      "user" \
    -library     "hls" \
    -version     "1.0"

puts "========================================="
puts "Done. IP exported to:"
puts "  cnn_detect_hls/solution1/impl/ip/"
puts ""
puts "Synthesis report:"
puts "  cnn_detect_hls/solution1/syn/report/cnn_detect_csynth.rpt"
puts "========================================="

exit
