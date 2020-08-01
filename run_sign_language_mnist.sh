#!/bin/bash

rm -r train
rm -r keras2tf
rm -r freeze
rm -r quantize
rm -r deploy

mkdir train
mkdir keras2tf
mkdir freeze
mkdir quantize
mkdir deploy
mkdir deploy/images
mkdir deploy/custom_images

python3 custom_test_image.py

python3 main.py 

freeze_graph --input_graph=./train/tf_complete_model.pb \
    --input_checkpoint=./train/tfchkpt.ckpt \
    --input_binary=true \
    --output_graph=./freeze/frozen_graph.pb \
    --output_node_names=activation_4_1/Softmax

python3 evaluate_accuracy.py \
   --graph=./freeze/frozen_graph.pb \
   --input_node=input_1_1 \
   --output_node=activation_4_1/Softmax \
   --batchsize=32

vai_q_tensorflow --version

vai_q_tensorflow quantize \
        --input_frozen_graph=./freeze/frozen_graph.pb \
        --input_nodes=input_1_1 \
        --input_shapes=?,28,28,1 \
        --output_nodes=activation_4_1/Softmax  \
        --input_fn=image_input_fn.calib_input \
        --output_dir=quantize \
        --calib_iter=100

python3 evaluate_accuracy.py \
   --graph=./quantize/quantize_eval_model.pb \
   --input_node=input_1_1 \
   --output_node=activation_4_1/Softmax \
   --batchsize=32

BOARD=ZCU104
ARCH=/opt/vitis_ai/compiler/arch/dpuv2/${BOARD}/${BOARD}.json

vai_c_tensorflow \
       --frozen_pb=./quantize/deploy_model.pb \
       --arch=${ARCH} \
       --output_dir=launchmodel \
       --net_name=SignLanguageMNISTnet \
       --options    "{'mode':'normal'}" 

cp launchmodel/*.elf deploy/.
cp -r target/* deploy/.