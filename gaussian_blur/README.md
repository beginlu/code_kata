基于CUDA实现的高斯滤波
====================

该实现参考了OpenCV中“**cv::gpu::GaussianBlur**”的实现：

1. 支持1~4通道（可根据需要扩展）；

2. 支持1～4字节的各种通道数据类型；

3. 支持5种边界类型，即  
   REPLICATE:    aaaaaa|abcdefgh|hhhhhhh  
   REFLECT:      fedcba|abcdefgh|hgfedcb  
   REFLECT\_101: gfedcb|abcdefgh|gfedcba  
   WRAP:         cdefgh|abcdefgh|abcdefg  
   CONSTANT:     000000|abcdefgh|0000000

4. 放宽高斯核大小限制（小于32），取决于共享内存的大小；

以下是使用本人实现对fruits.jpg进行高斯滤波的处理结果（使用15x15的高斯核）：

- 原始图像  
![原始图像](./pic/fruits.jpg)
- 高斯滤波图像  
![高斯滤波图像](./pic/fruits_blured.jpg)

基本上，在放宽高斯核大小的限制后，滤波函数的实现将会需要额外的数组边界指示器变量，并且循环不能展开（unroll）。当核的尺寸较小时，可以避免无用的访存，而当核的尺寸较大时，这点访存效率会被循环和条件分支带来的负面影响所淹没掉！

基本上，当高斯核尺寸小于等于11时，本人实现的效率占优（OpenCV的CPU实现使用了SSE加速，在3x3的情况下并不输于GPU实现）。以下是使用不同分辨率的标准图像（3通道单字节）进行的效率测试：  

- CPU: Intel i7 3820QM （2.7~3.1GHz）；
- GPU: Nvidia GT 650M （950MHz / 2 Multiprocessors / 384 CUDA Cores）；
- OpenCV: v2.4.9
- 具体的测试代码及一般函数调用请参考文件“impl\_custom\_test.cu”中的函数“ImplCustomTest2”和“ImplCustomTest3”

Test on image size (640 x 480), kernel size (3 x 3).  
OpenCV CPU implement use time 0.891868ms.  
OpenCV GPU implement use time 40.679177ms.  
Custom implement use time 1.474777ms.  

Test on image size (640 x 480), kernel size (7 x 7).  
OpenCV CPU implement use time 2.496911ms.  
OpenCV GPU implement use time 1.957981ms.  
Custom implement use time 1.654344ms.  

Test on image size (640 x 480), kernel size (9 x 9).  
OpenCV CPU implement use time 3.058913ms.  
OpenCV GPU implement use time 2.044496ms.  
Custom implement use time 1.760016ms.  

Test on image size (640 x 480), kernel size (11 x 11).  
OpenCV CPU implement use time 3.846743ms.  
OpenCV GPU implement use time 2.052098ms.  
Custom implement use time 1.786815ms.  

Test on image size (640 x 480), kernel size (13 x 13).  
OpenCV CPU implement use time 4.480093ms.  
OpenCV GPU implement use time 2.286820ms.  
Custom implement use time 1.919513ms.  

Test on image size (640 x 480), kernel size (15 x 15).  
OpenCV CPU implement use time 5.177456ms.  
OpenCV GPU implement use time 2.296057ms.  
Custom implement use time 2.082621ms.  

Test on image size (640 x 480), kernel size (31 x 31).  
OpenCV CPU implement use time 9.871821ms.  
OpenCV GPU implement use time 3.001820ms.  
Custom implement use time 3.050627ms.  

Test on image size (1920 x 1080), kernel size (3 x 3).  
OpenCV CPU implement use time 6.772995ms.  
OpenCV GPU implement use time 8.227930ms.  
Custom implement use time 6.906378ms.  

Test on image size (1920 x 1080), kernel size (7 x 7).  
OpenCV CPU implement use time 17.850775ms.  
OpenCV GPU implement use time 8.357968ms.  
Custom implement use time 7.665585ms.  

Test on image size (1920 x 1080), kernel size (9 x 9).  
OpenCV CPU implement use time 21.634760ms.  
OpenCV GPU implement use time 8.651532ms.  
Custom implement use time 7.931287ms.  

Test on image size (1920 x 1080), kernel size (11 x 11).  
OpenCV CPU implement use time 25.310109ms.  
OpenCV GPU implement use time 8.967181ms.  
Custom implement use time 8.430228ms.  

Test on image size (1920 x 1080), kernel size (13 x 13).  
OpenCV CPU implement use time 28.988308ms.  
OpenCV GPU implement use time 9.209544ms.  
Custom implement use time 9.244400ms.  

Test on image size (1920 x 1080), kernel size (15 x 15).  
OpenCV CPU implement use time 33.095887ms.  
OpenCV GPU implement use time 9.489614ms.  
Custom implement use time 9.850192ms.  

Test on image size (1920 x 1080), kernel size (31 x 31).  
OpenCV CPU implement use time 66.525403ms.  
OpenCV GPU implement use time 13.708415ms.  
Custom implement use time 15.702545ms.  

Test on image size (5616 x 3744), kernel size (3 x 3).  
OpenCV CPU implement use time 62.498902ms.  
OpenCV GPU implement use time 68.211334ms.  
Custom implement use time 59.880507ms.  

Test on image size (5616 x 3744), kernel size (7 x 7).  
OpenCV CPU implement use time 172.641784ms.  
OpenCV GPU implement use time 75.116876ms.  
Custom implement use time 71.764437ms.  

Test on image size (5616 x 3744), kernel size (9 x 9).  
OpenCV CPU implement use time 218.662210ms.  
OpenCV GPU implement use time 79.779464ms.  
Custom implement use time 74.409669ms.  

Test on image size (5616 x 3744), kernel size (11 x 11).  
OpenCV CPU implement use time 253.462236ms.  
OpenCV GPU implement use time 81.586159ms.  
Custom implement use time 80.491460ms.  

Test on image size (5616 x 3744), kernel size (13 x 13).  
OpenCV CPU implement use time 290.957290ms.  
OpenCV GPU implement use time 84.831595ms.  
Custom implement use time 88.978744ms.  

Test on image size (5616 x 3744), kernel size (15 x 15).  
OpenCV CPU implement use time 322.821865ms.  
OpenCV GPU implement use time 87.664947ms.  
Custom implement use time 95.241015ms.  

Test on image size (5616 x 3744), kernel size (31 x 31).  
OpenCV CPU implement use time 650.212162ms.  
OpenCV GPU implement use time 128.814671ms.  
Custom implement use time 153.414797ms.
