基于CUDA实现的高斯滤波
====================

该实现参考了OpenCV中“**cv::gpu::GaussianBlur**”的实现：

1. 支持1～4字节的各种通道数据类型；

2. 支持5种边界类型，即  
   REPLICATE:    aaaaaa|abcdefgh|hhhhhhh  
   REFLECT:      fedcba|abcdefgh|hgfedcb  
   REFLECT\_101: gfedcb|abcdefgh|gfedcba  
   WRAP:         cdefgh|abcdefgh|abcdefg  
   CONSTANT:     000000|abcdefgh|0000000

3. **放宽通道数限制，可取“大于4”的通道数，最大取值范围取决于共享内存的大小**；

4. **放宽高斯核大小限制，可取“大于31的核直径”，最大取值范围取决于共享内存的大小**；

以下是使用本人实现对fruits.jpg进行高斯滤波的处理结果（使用15x15的高斯核）：

- 原始图像  
![原始图像](./picture/fruits.jpg)
- 高斯滤波图像  
![高斯滤波图像](./picture/fruits_blured.jpg)

在放宽高斯核大小的限制后，滤波函数的实现将会需要额外的数组边界指示器变量，并且循环不能展开（unroll）。当核的尺寸较小时，可以使用指示器限制无用的访存，而当核的尺寸较大时，这点访存效率会被循环和条件分支带来的负面影响所淹没掉！

基本上，当高斯核尺寸小于等于13时，本人实现的效率略优（OpenCV的CPU实现使用了SSE加速，在3x3的情况下并不输于GPU实现）。以下是使用不同分辨率的标准图像（3通道单字节）进行的效率测试：  

- CPU: Intel i7 3820QM （2.7~3.1GHz）；
- GPU: Nvidia GT 650M （950MHz / 2 Multiprocessors / 384 CUDA Cores）；
- OpenCV: v2.4.9

Info: Test on config(640x480x3, 3x3, reflect101).  
Info: OpenCV CPU implement use time 0.898168ms.  
Info: OpenCV GPU implement use time 41.960202ms.  
Info: Custom GPU implement use time 1.674206ms.  

Info: Test on config(640x480x3, 5x5, reflect101).  
Info: OpenCV CPU implement use time 1.337350ms.  
Info: OpenCV GPU implement use time 2.106013ms.  
Info: Custom GPU implement use time 1.633534ms.  

Info: Test on config(640x480x3, 7x7, reflect101).  
Info: OpenCV CPU implement use time 2.600882ms.  
Info: OpenCV GPU implement use time 2.131443ms.  
Info: Custom GPU implement use time 1.853467ms.  

Info: Test on config(640x480x3, 9x9, reflect101).  
Info: OpenCV CPU implement use time 3.141212ms.  
Info: OpenCV GPU implement use time 2.209974ms.  
Info: Custom GPU implement use time 1.783488ms.  

Info: Test on config(640x480x3, 11x11, reflect101).  
Info: OpenCV CPU implement use time 3.746085ms.  
Info: OpenCV GPU implement use time 2.153147ms.  
Info: Custom GPU implement use time 2.012354ms.  

Info: Test on config(640x480x3, 13x13, reflect101).  
Info: OpenCV CPU implement use time 4.359853ms.  
Info: OpenCV GPU implement use time 2.338186ms.  
Info: Custom GPU implement use time 2.014292ms.  

Info: Test on config(640x480x3, 15x15, reflect101).  
Info: OpenCV CPU implement use time 4.737570ms.  
Info: OpenCV GPU implement use time 2.470085ms.  
Info: Custom GPU implement use time 2.124601ms.  

Info: Test on config(640x480x3, 31x31, reflect101).  
Info: OpenCV CPU implement use time 9.910712ms.  
Info: OpenCV GPU implement use time 3.215714ms.  
Info: Custom GPU implement use time 2.917326ms.  

Info: Test on config(1920x1080x3, 3x3, reflect101).  
Info: OpenCV CPU implement use time 6.698495ms.  
Info: OpenCV GPU implement use time 8.395621ms.  
Info: Custom GPU implement use time 7.809184ms.  

Info: Test on config(1920x1080x3, 5x5, reflect101).  
Info: OpenCV CPU implement use time 8.863159ms.  
Info: OpenCV GPU implement use time 8.781549ms.  
Info: Custom GPU implement use time 7.833587ms.  

Info: Test on config(1920x1080x3, 7x7, reflect101).  
Info: OpenCV CPU implement use time 17.232629ms.  
Info: OpenCV GPU implement use time 8.386650ms.  
Info: Custom GPU implement use time 7.980348ms.  

Info: Test on config(1920x1080x3, 9x9, reflect101).  
Info: OpenCV CPU implement use time 21.317470ms.  
Info: OpenCV GPU implement use time 8.682302ms.  
Info: Custom GPU implement use time 7.977535ms.  

Info: Test on config(1920x1080x3, 11x11, reflect101).  
Info: OpenCV CPU implement use time 25.792192ms.  
Info: OpenCV GPU implement use time 9.105633ms.  
Info: Custom GPU implement use time 9.043333ms.  

Info: Test on config(1920x1080x3, 13x13, reflect101).  
Info: OpenCV CPU implement use time 30.144215ms.  
Info: OpenCV GPU implement use time 9.282348ms.  
Info: Custom GPU implement use time 8.820777ms.  

Info: Test on config(1920x1080x3, 15x15, reflect101).  
Info: OpenCV CPU implement use time 32.988939ms.  
Info: OpenCV GPU implement use time 9.544625ms.  
Info: Custom GPU implement use time 9.780751ms.  

Info: Test on config(1920x1080x3, 31x31, reflect101).  
Info: OpenCV CPU implement use time 66.574160ms.  
Info: OpenCV GPU implement use time 13.813289ms.  
Info: Custom GPU implement use time 14.183214ms.  

Info: Test on config(5616x3744x3, 3x3, reflect101).  
Info: OpenCV CPU implement use time 70.485138ms.  
Info: OpenCV GPU implement use time 69.205489ms.  
Info: Custom GPU implement use time 67.251634ms.  

Info: Test on config(5616x3744x3, 5x5, reflect101).  
Info: OpenCV CPU implement use time 92.967880ms.  
Info: OpenCV GPU implement use time 73.350236ms.  
Info: Custom GPU implement use time 67.196404ms.  

Info: Test on config(5616x3744x3, 7x7, reflect101).  
Info: OpenCV CPU implement use time 178.376349ms.  
Info: OpenCV GPU implement use time 76.004143ms.  
Info: Custom GPU implement use time 75.316900ms.  

Info: Test on config(5616x3744x3, 9x9, reflect101).  
Info: OpenCV CPU implement use time 219.983199ms.  
Info: OpenCV GPU implement use time 78.958796ms.  
Info: Custom GPU implement use time 75.277254ms.  

Info: Test on config(5616x3744x3, 11x11, reflect101).  
Info: OpenCV CPU implement use time 261.863159ms.  
Info: OpenCV GPU implement use time 82.856013ms.  
Info: Custom GPU implement use time 83.588452ms.  

Info: Test on config(5616x3744x3, 13x13, reflect101).  
Info: OpenCV CPU implement use time 302.283108ms.  
Info: OpenCV GPU implement use time 85.653718ms.  
Info: Custom GPU implement use time 84.138817ms.  

Info: Test on config(5616x3744x3, 15x15, reflect101).  
Info: OpenCV CPU implement use time 347.109510ms.  
Info: OpenCV GPU implement use time 88.793219ms.  
Info: Custom GPU implement use time 92.668048ms.  

Info: Test on config(5616x3744x3, 31x31, reflect101).  
Info: OpenCV CPU implement use time 682.140528ms.  
Info: OpenCV GPU implement use time 132.053482ms.  
Info: Custom GPU implement use time 137.693895ms.

注意：在编译本代码时，请记得为nvcc加上“-rdc=true”编译选项，对应的“Visual Studio”选项卡是“CUDA C/C++”-->“Common”-->“Generate Relocatable Device Code”。