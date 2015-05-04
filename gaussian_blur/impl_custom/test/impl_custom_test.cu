#include "test_gaussian_blur/impl_custom/test/impl_custom_test.hpp"

#include <cstdlib>
#include <vector>
#include <Windows.h>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "test_gaussian_blur/impl_custom/filter/filter.hpp"

namespace test_gaussian_blur {

  __global__ void dummy(void) {}

  int ImplCustomTest1(void) {
    typedef Vec3<float> ST;
    typedef ST          DT;
    //typedef unsigned short ST;
    //typedef ST             DT;
    
    typedef Vec4<int> Dimension;
    std::vector<Dimension> dim_array;
    dim_array.push_back(Dimension(3, 3, 3, 3));
    dim_array.push_back(Dimension(32, 32, 9, 9));
    dim_array.push_back(Dimension(39, 39, 9, 9));
    dim_array.push_back(Dimension(80, 80, 17, 17));
    dim_array.push_back(Dimension(148, 148, 17, 17));
    dim_array.push_back(Dimension(148, 148, 51, 51));
    dim_array.push_back(Dimension(40, 40, 71, 71));
    //dim_array.push_back(Dimension(4000, 3000, 7, 7));

    const int num_dims = static_cast<int>(dim_array.size());
    for (int idim = 0; idim < num_dims; ++idim) {
      const Dimension &dim = dim_array[idim];
      const int width  = dim(0);
      const int height = dim(1);
      const int kernel_x_size = dim(2);
      const int kernel_y_size = dim(3);
      const int image_size = width*height;
      const BorderType border_type = BorderType::kReplicate;
      std::vector<ST> src_array(image_size);
      std::vector<DT> dst_array(image_size);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          src_array[y*width+x] = ST(rand()%255);
        }
      }

      fprintf(
        stdout, "Info: Test on (%d, %d, %d, %d).\n",
        width, height, kernel_x_size, kernel_y_size);

      GaussianBlurWithHostBuffer(
        &src_array[0], &dst_array[0],
        width*sizeof(ST), width*sizeof(DT),
        width, height, kernel_x_size, kernel_y_size,
        0.0, 0.0, border_type);

      typedef cv::Vec<typename ChannelType<ST>::value_type, ChannelCount<ST>::count> ocv_src_t;
      typedef cv::Vec<typename ChannelType<DT>::value_type, ChannelCount<DT>::count> ocv_dst_t;
      cv::Mat_<ocv_src_t> ocv_src(height, width, reinterpret_cast<ocv_src_t*>(&src_array[0]));
      cv::Mat_<ocv_dst_t> ocv_dst1(height, width, reinterpret_cast<ocv_dst_t*>(&dst_array[0]));
      cv::Mat_<ocv_dst_t> ocv_dst2(height, width);

      int ocv_border_type = cv::BORDER_DEFAULT;
      switch (border_type) {
      case BorderType::kReplicate:
        ocv_border_type = cv::BORDER_REPLICATE;
        break;
      case BorderType::kReflect:
        ocv_border_type = cv::BORDER_REFLECT;
        break;
      case BorderType::kReflect101:
        ocv_border_type = cv::BORDER_REFLECT101;
        break;
      case BorderType::kWrap:
        ocv_border_type = cv::BORDER_WRAP;
        break;
      case BorderType::kConstant:
        ocv_border_type = cv::BORDER_CONSTANT;
        break;
      default:
        assert(0);
        break;
      }

      double sum = 0.0;
      cv::GaussianBlur(
        ocv_src, ocv_dst2, cv::Size(kernel_x_size, kernel_y_size),
        0.0, 0.0, ocv_border_type);
      for (int y = 0; y < ocv_dst1.rows; ++y) {
        for (int x = 0; x < ocv_dst1.cols; ++x) {
          ocv_dst_t val1 = ocv_dst1(y, x);
          ocv_dst_t val2 = ocv_dst2(y, x);
          sum += cv::norm(val1-val2);
          if (val1 != val2) {
            //TGB_LOG_ERROR("Filter's diff %i and %i.\n", x, y);
          }
        }
      }
      sum = std::sqrt(sum/ocv_src.total());
      TGB_LOG_INFO("Filter's avg error: %.6lf.\n", sum);

      NOOP();
    }

    return 0;
  }

  int ImplCustomTest2(void) {
    // compare with opencv cuda implement
    typedef Vec3<unsigned char> ST;
    typedef ST                  DT;

    typedef Vec4<int> Dimension;
    std::vector<Dimension> dim_array;
    dim_array.push_back(Dimension(640, 480, 3, 3));
    dim_array.push_back(Dimension(640, 480, 7, 7));
    dim_array.push_back(Dimension(640, 480, 9, 9));
    dim_array.push_back(Dimension(640, 480, 11, 11));
    dim_array.push_back(Dimension(640, 480, 13, 13));
    dim_array.push_back(Dimension(640, 480, 15, 15));
    dim_array.push_back(Dimension(640, 480, 31, 31));
    dim_array.push_back(Dimension(1920, 1080, 3, 3));
    dim_array.push_back(Dimension(1920, 1080, 7, 7));
    dim_array.push_back(Dimension(1920, 1080, 9, 9));
    dim_array.push_back(Dimension(1920, 1080, 11, 11));
    dim_array.push_back(Dimension(1920, 1080, 13, 13));
    dim_array.push_back(Dimension(1920, 1080, 15, 15));
    dim_array.push_back(Dimension(1920, 1080, 31, 31));
    dim_array.push_back(Dimension(5616, 3744, 3, 3));
    dim_array.push_back(Dimension(5616, 3744, 7, 7));
    dim_array.push_back(Dimension(5616, 3744, 9, 9));
    dim_array.push_back(Dimension(5616, 3744, 11, 11));
    dim_array.push_back(Dimension(5616, 3744, 13, 13));
    dim_array.push_back(Dimension(5616, 3744, 15, 15));
    dim_array.push_back(Dimension(5616, 3744, 31, 31));

    LARGE_INTEGER tfreq, tbegin, tend;
    QueryPerformanceFrequency(&tfreq);

    cudaError_t cuda_error = cudaSuccess;
    ST *d_src_data_ptr      = nullptr;
    ST *d_dst_cust_data_ptr = nullptr;
    ST *d_dst_ocv_data_ptr  = nullptr;

    const int num_dims = static_cast<int>(dim_array.size());
    for (int idim = 0; idim < num_dims; ++idim) {
      const Dimension &dim = dim_array[idim];
      const int width  = dim(0);
      const int height = dim(1);
      const int kernel_x_size = dim(2);
      const int kernel_y_size = dim(3);
      const int image_size = width*height;
      const BorderType border_type = BorderType::kReflect101;
      std::vector<ST> src_array(image_size);
      std::vector<DT> dst_cust_array(image_size);
      std::vector<DT> dst_ocv_array(image_size);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          src_array[y*width+x] = ST(rand()%255);
        }
      }
      
      fprintf(
        stdout, "Test on image size (%d x %d), kernel size (%d x %d).\n",
        width, height, kernel_x_size, kernel_y_size);

      ST *h_src_data_ptr      = &src_array[0];
      ST *h_dst_ocv_data_ptr  = &dst_ocv_array[0];
      ST *h_dst_cust_data_ptr = &dst_cust_array[0];

      size_t d_src_stride      = 0;
      size_t d_dst_ocv_stride  = 0;
      size_t d_dst_cust_stride = 0;
      if (cudaSuccess != (cuda_error = cudaMallocPitch(
          &d_src_data_ptr, &d_src_stride, sizeof(ST)*width, height)) ||
          cudaSuccess != (cuda_error = cudaMallocPitch(
          &d_dst_ocv_data_ptr, &d_dst_ocv_stride, sizeof(DT)*width, height)) ||
          cudaSuccess != (cuda_error = cudaMallocPitch(
          &d_dst_cust_data_ptr, &d_dst_cust_stride, sizeof(DT)*width, height))) {
        TGB_LOG_ERROR(
          "Allocate device memory failed(%s).\n",
          cudaGetErrorString(cuda_error));
        break;
      }
      //if (cudaSuccess != (cuda_error = cudaMalloc(
      //    &d_src_data_ptr, sizeof(ST)*image_size)) ||
      //    cudaSuccess != (cuda_error = cudaMalloc(
      //    &d_dst_ocv_data_ptr, sizeof(DT)*image_size)) ||
      //    cudaSuccess != (cuda_error = cudaMalloc(
      //    &d_dst_cust_data_ptr, sizeof(DT)*image_size))) {
      //  TGB_LOG_ERROR(
      //    "Allocate device memory failed(%s).\n",
      //    cudaGetErrorString(cuda_error));
      //  break;
      //}
      if (cudaSuccess != (cuda_error = cudaMemcpy2D(
          d_src_data_ptr, sizeof(ST)*width,
          h_src_data_ptr, sizeof(ST)*width,
          sizeof(ST)*width, height, cudaMemcpyHostToDevice))) {
        TGB_LOG_ERROR(
          "Copy source data to device memory failed(%s).\n",
          cudaGetErrorString(cuda_error));
        break;
      }

      const int LOOP = 10;

      const int ocv_src_type = CV_MAKE_TYPE(
        cv::DataType<typename ChannelType<ST>::value_type>::type,
        ChannelCount<ST>::count);
      const int ocv_dst_type = CV_MAKE_TYPE(
        cv::DataType<typename ChannelType<DT>::value_type>::type,
        ChannelCount<ST>::count);

      int ocv_border_type = cv::BORDER_DEFAULT;
      switch (border_type) {
      case BorderType::kReplicate:
        ocv_border_type = cv::BORDER_REPLICATE;
        break;
      case BorderType::kReflect:
        ocv_border_type = cv::BORDER_REFLECT;
        break;
      case BorderType::kReflect101:
        ocv_border_type = cv::BORDER_REFLECT101;
        break;
      case BorderType::kWrap:
        ocv_border_type = cv::BORDER_WRAP;
        break;
      case BorderType::kConstant:
        ocv_border_type = cv::BORDER_CONSTANT;
        break;
      default:
        assert(0);
        break;
      }

      ////////////////////////////////////////////////////////////////////
      // 1. test opencv cpu implement
      {
        cv::Mat ocv_src_mat(
          height, width, ocv_src_type,
          reinterpret_cast<void*>(h_src_data_ptr), sizeof(ST)*width);
        cv::Mat ocv_dst_mat(
          height, width, ocv_dst_type,
          reinterpret_cast<void*>(h_dst_ocv_data_ptr), sizeof(DT)*width);

        // loop multiple times to get average run time
        QueryPerformanceCounter(&tbegin);
        for (int l = 0; l < LOOP; ++l) {
          cv::GaussianBlur(
            ocv_src_mat, ocv_dst_mat,
            cv::Size(kernel_x_size, kernel_y_size),
            0.0, 0.0, ocv_border_type);
        }
        QueryPerformanceCounter(&tend);
        fprintf(
          stdout, "OpenCV CPU implement use time %.6lfms.\n",
          (tend.QuadPart-tbegin.QuadPart)*1000.0/tfreq.QuadPart/LOOP);
      }
      ////////////////////////////////////////////////////////////////////

      ////////////////////////////////////////////////////////////////////
      // 2. test opencv gpu implement
      {
        cv::gpu::GpuMat ocv_src_mat(
          height, width, ocv_src_type,
          reinterpret_cast<void*>(d_src_data_ptr), d_src_stride);
        cv::gpu::GpuMat ocv_dst_mat(
          height, width, ocv_dst_type,
          reinterpret_cast<void*>(d_dst_ocv_data_ptr), d_dst_ocv_stride);

        // warm-up
        dummy<<<1, 1>>>();

        // loop multiple times to get average run time
        QueryPerformanceCounter(&tbegin);
        for (int l = 0; l < LOOP; ++l) {
          cv::gpu::GaussianBlur(
            ocv_src_mat, ocv_dst_mat,
            cv::Size(kernel_x_size, kernel_y_size),
            0.0, 0.0, ocv_border_type);
          cudaDeviceSynchronize();
        }
        QueryPerformanceCounter(&tend);
        fprintf(
          stdout, "OpenCV GPU implement use time %.6lfms.\n",
          (tend.QuadPart-tbegin.QuadPart)*1000.0/tfreq.QuadPart/LOOP);
      }
      ////////////////////////////////////////////////////////////////////

      ////////////////////////////////////////////////////////////////////
      // 3. test my implement
      {
        // warm-up
        dummy<<<1, 1>>>();

        // loop multiple times to get average run time
        QueryPerformanceCounter(&tbegin);
        for (int l = 0; l < LOOP; ++l) {
          GaussianBlurWithDeviceBuffer(
            d_src_data_ptr, d_dst_cust_data_ptr,
            (int)d_src_stride, (int)d_dst_cust_stride,
            width, height, kernel_x_size, kernel_y_size,
            0.0, 0.0, border_type);
          cudaDeviceSynchronize();
        }
        QueryPerformanceCounter(&tend);
        fprintf(
          stdout, "Custom implement use time %.6lfms.\n",
          (tend.QuadPart-tbegin.QuadPart)*1000.0/tfreq.QuadPart/LOOP);
      }
      ////////////////////////////////////////////////////////////////////

      fprintf(stdout, "\n");

      cudaFree(d_dst_cust_data_ptr);
      cudaFree(d_dst_ocv_data_ptr);
      cudaFree(d_src_data_ptr);
      d_dst_cust_data_ptr = nullptr;
      d_dst_ocv_data_ptr = nullptr;
      d_src_data_ptr = nullptr;
    }

    if (d_dst_cust_data_ptr) {
      cudaFree(d_dst_cust_data_ptr);
      d_dst_cust_data_ptr = nullptr;
    }
    if (d_dst_ocv_data_ptr) {
      cudaFree(d_dst_ocv_data_ptr);
      d_dst_ocv_data_ptr = nullptr;
    }
    if (d_src_data_ptr) {
      cudaFree(d_src_data_ptr);
      d_src_data_ptr = nullptr;
    }

    return 0;
  }

  int ImplCustomTest3(void) {
    const std::string src_path = "fruits.jpg";
    const std::string dst_path = "fruits_blured.jpg";

    cv::Mat ocv_src_mat = cv::imread(src_path);
    cv::Mat ocv_dst_mat(
      ocv_src_mat.size(), ocv_src_mat.type());

    typedef Vec3<unsigned char> ST;
    typedef ST                  DT;

    GaussianBlurWithHostBuffer(
      (ST*)ocv_src_mat.data, (DT*)ocv_dst_mat.data,
      ocv_src_mat.step, ocv_dst_mat.step,
      ocv_src_mat.cols, ocv_src_mat.rows,
      15, 15, 0.0, 0.0, BorderType::kReflect101);

    std::vector<int> save_params;
    save_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    save_params.push_back(98);
    cv::imwrite(dst_path, ocv_dst_mat, save_params);
    cv::imshow("Original", ocv_src_mat);
    cv::imshow("Blured", ocv_dst_mat);
    cv::waitKey();

    return 0;
  }

}
