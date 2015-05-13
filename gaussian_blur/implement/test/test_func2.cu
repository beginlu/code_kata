#include <string>
#include <vector>
#include <Windows.h>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "filter/gaussian_blur.cuh"
#include "test/test.hpp"

namespace test_gaussian_blur {

  __global__ void dummy(void) {}

  int TestFunc2(void) {
    typedef unsigned char ST;
    typedef ST            DT;

    std::vector<ConfigSet> config_array;
    config_array.push_back(ConfigSet(640, 480, 3, 3, 3, BorderType::kReflect101));
    config_array.push_back(ConfigSet(640, 480, 3, 5, 5, BorderType::kReflect101));
    config_array.push_back(ConfigSet(640, 480, 3, 7, 7, BorderType::kReflect101));
    config_array.push_back(ConfigSet(640, 480, 3, 9, 9, BorderType::kReflect101));
    config_array.push_back(ConfigSet(640, 480, 3, 11, 11, BorderType::kReflect101));
    config_array.push_back(ConfigSet(640, 480, 3, 13, 13, BorderType::kReflect101));
    config_array.push_back(ConfigSet(640, 480, 3, 15, 15, BorderType::kReflect101));
    config_array.push_back(ConfigSet(640, 480, 3, 31, 31, BorderType::kReflect101));
    config_array.push_back(ConfigSet(1920, 1080, 3, 3, 3, BorderType::kReflect101));
    config_array.push_back(ConfigSet(1920, 1080, 3, 5, 5, BorderType::kReflect101));
    config_array.push_back(ConfigSet(1920, 1080, 3, 7, 7, BorderType::kReflect101));
    config_array.push_back(ConfigSet(1920, 1080, 3, 9, 9, BorderType::kReflect101));
    config_array.push_back(ConfigSet(1920, 1080, 3, 11, 11, BorderType::kReflect101));
    config_array.push_back(ConfigSet(1920, 1080, 3, 13, 13, BorderType::kReflect101));
    config_array.push_back(ConfigSet(1920, 1080, 3, 15, 15, BorderType::kReflect101));
    config_array.push_back(ConfigSet(1920, 1080, 3, 31, 31, BorderType::kReflect101));
    config_array.push_back(ConfigSet(5616, 3744, 3, 3, 3, BorderType::kReflect101));
    config_array.push_back(ConfigSet(5616, 3744, 3, 5, 5, BorderType::kReflect101));
    config_array.push_back(ConfigSet(5616, 3744, 3, 7, 7, BorderType::kReflect101));
    config_array.push_back(ConfigSet(5616, 3744, 3, 9, 9, BorderType::kReflect101));
    config_array.push_back(ConfigSet(5616, 3744, 3, 11, 11, BorderType::kReflect101));
    config_array.push_back(ConfigSet(5616, 3744, 3, 13, 13, BorderType::kReflect101));
    config_array.push_back(ConfigSet(5616, 3744, 3, 15, 15, BorderType::kReflect101));
    config_array.push_back(ConfigSet(5616, 3744, 3, 31, 31, BorderType::kReflect101));

    LARGE_INTEGER tfreq, tbegin, tend;
    QueryPerformanceFrequency(&tfreq);

    cudaError_t cuda_error  = cudaSuccess;
    ST *d_src_data_ptr      = nullptr;
    ST *d_dst_cust_data_ptr = nullptr;
    ST *d_dst_ocv_data_ptr  = nullptr;

    const int num_configs = static_cast<int>(config_array.size());
    for (int iconfig = 0; iconfig < num_configs; ++iconfig) {
      const ConfigSet &config = config_array[iconfig];
      const int image_size = config.height_*config.width_*config.channels_;
      std::vector<ST> src_array(image_size);
      std::vector<DT> dst_ocv_array(image_size);
      std::vector<DT> dst_cust_array(image_size);
      for (int y = 0; y < config.height_; ++y) {
        for (int x = 0; x < config.width_; ++x) {
          for (int c = 0; c < config.channels_; ++c) {
            const int index = (y*config.width_+x)*config.channels_+c;
            src_array[index] = ST(rand()%255);
          }
        }
      }

      ST *h_src_data_ptr      = &src_array[0];
      ST *h_dst_ocv_data_ptr  = &dst_ocv_array[0];
      ST *h_dst_cust_data_ptr = &dst_cust_array[0];
      size_t src_stride = config.width_*config.channels_*sizeof(ST);
      size_t dst_stride = config.width_*config.channels_*sizeof(DT);
      if (cudaSuccess != (cuda_error = cudaMalloc(
          &d_src_data_ptr, image_size*sizeof(ST))) ||
          cudaSuccess != (cuda_error = cudaMalloc(
          &d_dst_ocv_data_ptr, image_size*sizeof(DT))) ||
          cudaSuccess != (cuda_error = cudaMalloc(
          &d_dst_cust_data_ptr, image_size*sizeof(DT)))) {
        TGB_LOG_ERROR(
            "Allocate device memory failed(%s).\n",
            cudaGetErrorString(cuda_error));
        break;
      }
      if (cudaSuccess != (cuda_error = cudaMemcpy2D(
          d_src_data_ptr, src_stride,
          h_src_data_ptr, src_stride,
          src_stride, config.height_,
          cudaMemcpyHostToDevice))) {
        TGB_LOG_ERROR(
          "Copy source data to device memory failed(%s).\n",
          cudaGetErrorString(cuda_error));
        break;
      }

      std::string border_string;
      int ocv_border_type = cv::BORDER_DEFAULT;
      switch (config.border_type_) {
      case BorderType::kReplicate:
        border_string   = "replicate";
        ocv_border_type = cv::BORDER_REPLICATE;
        break;
      case BorderType::kReflect:
        border_string   = "reflect";
        ocv_border_type = cv::BORDER_REFLECT;
        break;
      case BorderType::kReflect101:
        border_string   = "reflect101";
        ocv_border_type = cv::BORDER_REFLECT101;
        break;
      case BorderType::kWrap:
        border_string   = "wrap";
        ocv_border_type = cv::BORDER_WRAP;
        break;
      case BorderType::kConstant:
        border_string   = "constant";
        ocv_border_type = cv::BORDER_CONSTANT;
        break;
      default:
        assert(0);
        break;
      }

      TGB_LOG_INFO(
          "Test on config(%dx%dx%d, %dx%d, %s).\n",
          config.width_, config.height_, config.channels_,
          config.kernel_x_size_, config.kernel_y_size_,
          border_string.c_str());

      const int LOOP = 10;
      const int ocv_src_type =
          CV_MAKE_TYPE(cv::DataDepth<ST>::value, config.channels_);
      const int ocv_dst_type =
          CV_MAKE_TYPE(cv::DataDepth<DT>::value, config.channels_);

      ////////////////////////////////////////////////////////////////////
      // 1. test opencv cpu implement
      if (config.border_type_ != BorderType::kWrap)
      {
        cv::Mat ocv_src_mat(
            config.height_, config.width_, ocv_src_type,
            h_src_data_ptr, src_stride);
        cv::Mat ocv_dst_mat(
            config.height_, config.width_, ocv_dst_type,
            h_dst_ocv_data_ptr, dst_stride);

        // loop multiple times to get average run time
        QueryPerformanceCounter(&tbegin);
        for (int l = 0; l < LOOP; ++l) {
          cv::GaussianBlur(
              ocv_src_mat, ocv_dst_mat,
              cv::Size(config.kernel_x_size_, config.kernel_y_size_),
              0.0, 0.0, ocv_border_type);
        }
        QueryPerformanceCounter(&tend);
        TGB_LOG_INFO(
            "OpenCV CPU implement use time %.6lfms.\n",
            (tend.QuadPart-tbegin.QuadPart)*1000.0/tfreq.QuadPart/LOOP);
      }
      ////////////////////////////////////////////////////////////////////

      ////////////////////////////////////////////////////////////////////
      // 2. test opencv gpu implement
      if (config.border_type_ != BorderType::kWrap &&
          config.channels_ <= 4)
      {
        cv::gpu::GpuMat ocv_src_mat(
            config.height_, config.width_, ocv_src_type,
            d_src_data_ptr, src_stride);
        cv::gpu::GpuMat ocv_dst_mat(
            config.height_, config.width_, ocv_dst_type,
            d_dst_ocv_data_ptr, dst_stride);

        // warm-up
        dummy<<<1, 1>>>();

        // loop multiple times to get average run time
        QueryPerformanceCounter(&tbegin);
        for (int l = 0; l < LOOP; ++l) {
          cv::gpu::GaussianBlur(
              ocv_src_mat, ocv_dst_mat,
              cv::Size(config.kernel_x_size_, config.kernel_y_size_),
              0.0, 0.0, ocv_border_type);
          cudaDeviceSynchronize();
        }
        QueryPerformanceCounter(&tend);
        TGB_LOG_INFO(
            "OpenCV GPU implement use time %.6lfms.\n",
            (tend.QuadPart-tbegin.QuadPart)*1000.0/tfreq.QuadPart/LOOP);
      }
      ////////////////////////////////////////////////////////////////////

      ////////////////////////////////////////////////////////////////////
      // 3. test my implement
      {
        PixelPicker<ST> src_picker(
            config.height_, config.width_, config.channels_,
            src_stride, d_src_data_ptr);
        PixelPicker<DT> dst_picker(
            config.height_, config.width_, config.channels_,
            dst_stride, d_dst_cust_data_ptr);

        // warm-up
        dummy<<<1, 1>>>();
        // loop multiple times to get average run time
        QueryPerformanceCounter(&tbegin);
        for (int l = 0; l < LOOP; ++l) {
          GaussianBlurDevice(
              src_picker, dst_picker,
              config.kernel_x_size_, config.kernel_y_size_,
              0.0, 0.0, config.border_type_);
          cudaDeviceSynchronize();
        }
        QueryPerformanceCounter(&tend);
        TGB_LOG_INFO(
            "Custom GPU implement use time %.6lfms.\n",
            (tend.QuadPart-tbegin.QuadPart)*1000.0/tfreq.QuadPart/LOOP);
      }
      ////////////////////////////////////////////////////////////////////

      TGB_LOG(stdout, "\n");

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

}
