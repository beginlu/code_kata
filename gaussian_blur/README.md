基于CUDA实现的高斯滤波
====================

该实现参考了OpenCV中“**cv::gpu::GaussianBlur**”的实现，并进一步扩展其功能：

1. 支持1～4字节的各种通道数据类型；

2. 支持5种边界类型，即  
   REPLICATE:    aaaaaa|abcdefgh|hhhhhhh  
   REFLECT:      fedcba|abcdefgh|hgfedcb  
   REFLECT\_101: gfedcb|abcdefgh|gfedcba  
   WRAP:         cdefgh|abcdefgh|abcdefg  
   CONSTANT:     000000|abcdefgh|0000000

3. **放宽通道数限制，可取“大于4”的通道数，最大取值取决于共享内存的大小**；

4. **放宽高斯核大小限制，可取“大于31的核直径”，最大取值取决于共享内存的大小**；

以下是使用本人实现对fruits.jpg进行高斯滤波的处理结果（使用15x15的高斯核）：

- 原始图像  
![原始图像](./picture/fruits.jpg)
- 高斯滤波图像  
![高斯滤波图像](./picture/fruits_blured.jpg)

在放宽高斯核大小的限制后，滤波函数的实现将需要额外的数组边界指示器变量，并且循环不能展开（unroll）。当核的尺寸较小时，可以使用指示器避免无用的访存，而当核的尺寸较大时，这点访存效率会被循环和条件分支带来的负面影响所淹没！

基本上，当高斯核尺寸小于等于13时，本人实现的效率略优（OpenCV的CPU实现使用了SSE加速，在3x3的情况下并不输于GPU实现）。以下是使用不同分辨率的标准图像（3通道单字节）进行的效率测试，每种测试均为执行20次后的平均结果：  

- CPU: Intel i7 3820QM （2.7~3.1GHz）；
- GPU: Nvidia GT 650M （950MHz / 2 Multiprocessors / 384 CUDA Cores）；
- OpenCV: v2.4.9

Info: Test on config(640x480x3, 3x3, reflect101).  
Info: OpenCV CPU implement use time 0.950000ms.  
Info: OpenCV GPU implement use time 22.000000ms.  
Info: Custom GPU implement use time 1.600000ms.  

Info: Test on config(640x480x3, 5x5, reflect101).  
Info: OpenCV CPU implement use time 1.350000ms.  
Info: OpenCV GPU implement use time 1.850000ms.  
Info: Custom GPU implement use time 1.500000ms.  

Info: Test on config(640x480x3, 7x7, reflect101).  
Info: OpenCV CPU implement use time 2.550000ms.  
Info: OpenCV GPU implement use time 1.850000ms.  
Info: Custom GPU implement use time 1.700000ms.  

Info: Test on config(640x480x3, 9x9, reflect101).  
Info: OpenCV CPU implement use time 3.200000ms.  
Info: OpenCV GPU implement use time 1.950000ms.  
Info: Custom GPU implement use time 1.650000ms.  

Info: Test on config(640x480x3, 11x11, reflect101).  
Info: OpenCV CPU implement use time 3.750000ms.  
Info: OpenCV GPU implement use time 1.950000ms.  
Info: Custom GPU implement use time 1.800000ms.  

Info: Test on config(640x480x3, 13x13, reflect101).  
Info: OpenCV CPU implement use time 4.300000ms.  
Info: OpenCV GPU implement use time 2.050000ms.  
Info: Custom GPU implement use time 1.800000ms.  

Info: Test on config(640x480x3, 15x15, reflect101).  
Info: OpenCV CPU implement use time 5.100000ms.  
Info: OpenCV GPU implement use time 2.050000ms.  
Info: Custom GPU implement use time 1.950000ms.  

Info: Test on config(640x480x3, 31x31, reflect101).  
Info: OpenCV CPU implement use time 9.850000ms.  
Info: OpenCV GPU implement use time 2.850000ms.  
Info: Custom GPU implement use time 2.700000ms.  

Info: Test on config(1920x1080x3, 3x3, reflect101).  
Info: OpenCV CPU implement use time 6.650000ms.  
Info: OpenCV GPU implement use time 7.750000ms.  
Info: Custom GPU implement use time 7.200000ms.  

Info: Test on config(1920x1080x3, 5x5, reflect101).  
Info: OpenCV CPU implement use time 8.800000ms.  
Info: OpenCV GPU implement use time 7.800000ms.  
Info: Custom GPU implement use time 7.000000ms.  

Info: Test on config(1920x1080x3, 7x7, reflect101).  
Info: OpenCV CPU implement use time 17.300000ms.  
Info: OpenCV GPU implement use time 8.150000ms.  
Info: Custom GPU implement use time 7.800000ms.  

Info: Test on config(1920x1080x3, 9x9, reflect101).  
Info: OpenCV CPU implement use time 21.200000ms.  
Info: OpenCV GPU implement use time 8.450000ms.  
Info: Custom GPU implement use time 7.800000ms.  

Info: Test on config(1920x1080x3, 11x11, reflect101).  
Info: OpenCV CPU implement use time 25.100000ms.  
Info: OpenCV GPU implement use time 8.800000ms.  
Info: Custom GPU implement use time 8.800000ms.  

Info: Test on config(1920x1080x3, 13x13, reflect101).  
Info: OpenCV CPU implement use time 29.300000ms.  
Info: OpenCV GPU implement use time 9.050000ms.  
Info: Custom GPU implement use time 8.700000ms.  

Info: Test on config(1920x1080x3, 15x15, reflect101).  
Info: OpenCV CPU implement use time 33.550000ms.  
Info: OpenCV GPU implement use time 9.450000ms.  
Info: Custom GPU implement use time 9.550000ms.  

Info: Test on config(1920x1080x3, 31x31, reflect101).  
Info: OpenCV CPU implement use time 65.650000ms.  
Info: OpenCV GPU implement use time 13.600000ms.  
Info: Custom GPU implement use time 13.900000ms.  

Info: Test on config(5616x3744x3, 3x3, reflect101).  
Info: OpenCV CPU implement use time 63.350000ms.  
Info: OpenCV GPU implement use time 68.950000ms.  
Info: Custom GPU implement use time 67.300000ms.  

Info: Test on config(5616x3744x3, 5x5, reflect101).  
Info: OpenCV CPU implement use time 91.550000ms.  
Info: OpenCV GPU implement use time 73.100000ms.  
Info: Custom GPU implement use time 66.650000ms.  

Info: Test on config(5616x3744x3, 7x7, reflect101).  
Info: OpenCV CPU implement use time 177.600000ms.  
Info: OpenCV GPU implement use time 75.800000ms.  
Info: Custom GPU implement use time 74.850000ms.  

Info: Test on config(5616x3744x3, 9x9, reflect101).  
Info: OpenCV CPU implement use time 219.000000ms.  
Info: OpenCV GPU implement use time 79.650000ms.  
Info: Custom GPU implement use time 74.250000ms.  

Info: Test on config(5616x3744x3, 11x11, reflect101).  
Info: OpenCV CPU implement use time 256.950000ms.  
Info: OpenCV GPU implement use time 82.450000ms.  
Info: Custom GPU implement use time 83.200000ms.  

Info: Test on config(5616x3744x3, 13x13, reflect101).  
Info: OpenCV CPU implement use time 298.050000ms.  
Info: OpenCV GPU implement use time 85.350000ms.  
Info: Custom GPU implement use time 82.900000ms.  

Info: Test on config(5616x3744x3, 15x15, reflect101).  
Info: OpenCV CPU implement use time 348.750000ms.  
Info: OpenCV GPU implement use time 87.750000ms.  
Info: Custom GPU implement use time 91.900000ms.  

Info: Test on config(5616x3744x3, 31x31, reflect101).  
Info: OpenCV CPU implement use time 673.250000ms.  
Info: OpenCV GPU implement use time 130.400000ms.  
Info: Custom GPU implement use time 135.900000ms.

**注意：在编译本代码时，请记得为nvcc加上“-rdc=true”编译选项，对应的“Visual Studio”选项卡是“CUDA C/C++”-->“Common”-->“Generate Relocatable Device Code”。**