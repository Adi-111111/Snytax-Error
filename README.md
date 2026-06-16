# Snytax Error — Hardware Accelerated Neural Network Decision Visualiser

EE2 Design Project, Imperial College London, Summer 2026

A fully pipelined MLP on a PYNQ-Z1 FPGA that renders a classifier's decision boundary as a live 640×480 colour heatmap on HDMI, faster and more energy-efficient than an equivalent CPU implementation.

## Team

Shivam Sangani, Aadhith Nair, Aryan Jain, Eesha Walji, Herbie Skidmore, Rhys Lloyd

## How it works

Upload a CSV dataset via a web UI on your laptop. The system trains a 16→64→32→3 MLP and quantises the weights to Q4.4, then loads them onto the FPGA over AXI-Lite. The FPGA renders the decision boundary in real time — pick any two features as screen axes and sweep the rest with sliders, with the full heatmap updating within one frame of each change.
