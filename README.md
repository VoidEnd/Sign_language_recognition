# 手语识别
## 概要
  2020年新⼯科联盟-Xilinx暑期学校（Summer School）项⽬，Ultra96-V2平台上基于TensorFlow和Keras的手语识别系统，数据集为Kaggle的MNIST手语数据集。目前仅是学习所用，原作者源码https://github.com/beetleboxorg/sign_language_mnist

## 实验环境
* OS: Ubuntu 20.04
* Vitis AI version: V1.1
* FPGA used: ZYNQ ultrascale+ AMP(Ultra96-v2)
* Tensorflow version: 1.15.0
* Keras version: 2.2.5

## 数据集
  已上传至仓库，文件名mnist，也可从Kaggle查看下载[手语MNIST数据集](https://www.kaggle.com/datamunge/sign-language-mnist)

## 安装指南
### 安装Vitis-AI
1） 安装Git  
```bash
sudo apt-get install git
```  
2） 启动一个终端，找一个你喜欢的目录（注意不要有中文和空格），从Github克隆vitis ai的仓库，当然要注意使用的版本，选择正确的分支，默认分支是最新版  
```bash
git clone https://github.com/Xilinx/Vitis-AI.git
```
如果速度比较慢的话可以使用Gitee的镜像源
```bash
git clone https://gitee.com/xiaobolin/Vitis-AI.git
```

3） 进入克隆好的文件夹，使用以下指令拉取最新的docker镜像，这条指令也用于以后启动vitis AI
```bash
./docker_run.sh xilinx/vitis-ai
```
  
## 程序运行指南
程序是在Vitis-AI环境下运行的，可以参考[Vitis-AI](https://www.xilinx.com/html_docs/vitis_ai/1_1/zkj1576857115470.html)的的官方文档
### 创建好需要用到的文件夹

```bash
rm -r train
rm -r keras2tf
rm -r freeze
rm -r quantize
rm -r deploy

mkdir train
mkdir keras2tf
mkdir freeze
mkdir quantize
mkdir launchmodel
mkdir deploy
mkdir deploy/images
```
所创建文件夹的内容
* train存放keras训练好的 .h5模型文件
* keras2tf存放讲 .h5模型文件转化后的TensorFlow的 .pb模型文件和 .ckpt文件
* freeze存放将 .pb模型文件和 .ckpt文件freeze（冻结）后的 .pb文件
* quantize存放将freeze（冻结）后的 .pb文件quantize（量化）后的 .pb模型文件
* launchmodel存放将quantize（量化）后的 .pb模型文件编译后的 .elf文件
* deploy存放上板时所需的文件，拷贝自target文件和launchmodel文件
* deploy/images存放处理后的测试图片

仓库中文件夹内容
* test文件夹中为一组测试图片
* target文件夹中为上板时所需程序

### 训练keras模型
首先通过以下指令进入TensorFlow环境，后续运行指令均在此环境下
```bash
conda activate vitis-ai-tensorflow
```
可以通过以下指令查看TensorFlow和keras的版本
```bash
conda list
```
将仓库内的文件下载下来后，进入其中，通过以下指令训练keras模型
```bash
python3 main.py 
```
main.py文件中为keras模型训练以及转化为TensorFlow模型的Python代码

### 测试图片的处理
我们要将测试所用的图片处理为灰度图，并储存在./deploy/images文件中，保持上一步的路径不变，通过以下指令处理
```bash
python3 custom_test_image.py
```
custom_test_image.py文件中为实现转化的Python代码

### 模型冻结
保持上一步的路径不变，运行以下指令
```bash
freeze_graph --input_graph=./train/tf_complete_model.pb \
    --input_checkpoint=./train/tfchkpt.ckpt \
    --input_binary=true \
    --output_graph=./freeze/frozen_graph.pb \
    --output_node_names=activation_4_1/Softmax
```

### 模型量化
保持上一步的路径不变，运行以下指令
```bash
vai_q_tensorflow quantize \
        --input_frozen_graph=./freeze/frozen_graph.pb \
        --input_nodes=input_1_1 \
        --input_shapes=?,28,28,1 \
        --output_nodes=activation_4_1/Softmax  \
        --input_fn=image_input_fn.calib_input \
        --output_dir=quantize \
        --calib_iter=100
```

### 模型编译
根据所使用的FPGA板选择对应的JSON格式的DPU体系结构配置文件，例如Vitis-AI环境中已有的ZCU102和ZCU104，以ZCU104为例，保持上一步的路径不变，运行以下指令
```bash
BOARD=ZCU104
ARCH=/opt/vitis_ai/compiler/arch/dpuv2/${BOARD}/${BOARD}.json

vai_c_tensorflow \
       --frozen_pb=./quantize/deploy_model.pb \
       --arch=${ARCH} \
       --output_dir=launchmodel \
       --net_name=SignLanguageMNISTnet \
       --options    "{'mode':'normal'}" 
```
这里所使用的是Ultra96-V2，所需的json文件（u96pynq.json），以及json文件指定路径的 .dcf编译中间文件（dpuPynq.dcf），均已上传至仓库，指令为
```bash
ARCH=./u96pynq.json

vai_c_tensorflow \
       --frozen_pb=./quantize/deploy_model.pb \
       --arch=${ARCH} \
       --output_dir=launchmodel \
       --net_name=SignLanguageMNISTnet \
       --options    "{'mode':'normal'}" 
```

### 模型评估
模型评估每一步都可以进行，保持上一步的路径不变，运行以下指令，注意pb文件的路径记得要换
```bash
pb = ./

python3 evaluate_accuracy.py \
   --graph=${pb} \
   --input_node=input_1_1 \
   --output_node=activation_4_1/Softmax \
   --batchsize=32
```

### 准备上板所需文件
保持上一步的路径不变，运行以下指令
```bash
cp launchmodel/*.elf deploy/.
cp -r target/* deploy/.
```

## 在FPGA上运行
关于如何连接到Ultra96-PYNQ的Jupyter Notebooks，可以参考官方文档https://ultra96-pynq.readthedocs.io/en/latest/getting_started.html#connecting-to-jupyter-notebooks

## 配置Ultra96-V2板的环境
1） 配置环境有两种可选，一种是[PYNQ v2.5](https://github.com/Xilinx/PYNQ/releases)，另一种是[ultra96v2 oob](http://avnet.me/ultra96-v2-oob)，选择其中一种，下载其镜像文件如果选择了PYNQ v2.5，这里有已经配置好DPU驱动的镜像文件
> https://pan.baidu.com/s/1CASOuR8ahbarEBOv_7oNQw  
提取码：s2fe

2） 或者单独下载[DPU驱动文件](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.1_dnndk.tar.gz)  

3） 安装DPU驱动文件，将下好的驱动文件拷贝至板上，在板上运行以下代码
```bash
tar -xzvf vitis-ai_v1.1_dnndk.tar.gz
 cd vitis-ai_v1.1_dnndk 
./install.sh
```

## 第一种环境配置的运行方法（ultra96v2 oob）
1） 连接板子，建议使用MobaXterm连接  
2） 将deploy文件夹整体拷贝至板上
3） 进入deploy文件夹，运行以下指令，将模型共享至 .so文件
```bash
aarch64-xilinx-linux-gcc -fPIC  \
  -shared ./dpu_SignLanguageMNISTnet_0.elf -o ./dpuv2_rundir/libdpumodelSignLanguageMNISTnet.so
```
4） 调用DPU进行测试，保持当前路径不变，运行以下指令
```bash
python3 sign_language_app.py -t 1 -b 1 -j / home / root / deploy / dpuv2_rundir /
```

## 第二种环境配置的运行方法（PYNQ v2.5）
1） 连接板子，建议使用MobaXterm连接  
2） 将deploy文件夹整体拷贝至板上  
3） 登录至板子的Jupyter Notebooks  
4） 进入deploy文件夹，运行以下指令，将模型共享至 .so文件
```bash
aarch64-xilinx-linux-gcc -fPIC  \
  -shared ./dpu_SignLanguageMNISTnet_0.elf -o ./dpuv2_rundir/libdpumodelSignLanguageMNISTnet.so
```
5） 运行测试文件（这里没有用到DPU），测试文件目前缺失，会补
