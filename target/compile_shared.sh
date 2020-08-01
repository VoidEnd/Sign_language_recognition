#!/bin/bash

aarch64-xilinx-linux-gcc -fPIC  \
  -shared ./dpu_SignLanguageMNISTnet_0.elf -o ./dpuv2_rundir/libdpumodelSignLanguageMNISTnet.so