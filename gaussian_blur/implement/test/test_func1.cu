#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "filter/gaussian_blur.cuh"
#include "test/test.hpp"

namespace test_gaussian_blur {

  int TestFunc1(void) {
    typedef unsigned char ST;
    typedef ST            DT;

    std::vector<ConfigSet> config_array;
    //config_array.push_back(ConfigSet(3, 3, 1, 3, 3, BorderType::kReflect101));
    //config_array.push_back(ConfigSet(3, 3, 3, 3, 3, BorderType::kReflect101));
    config_array.push_back(ConfigSet(1379, 939, 3, 15, 17, BorderType::kReflect101));

    const int num_configs = static_cast<int>(config_array.size());
    for (int iconfig = 0; iconfig < num_configs; ++iconfig) {
      const ConfigSet &config = config_array[iconfig];
      const int image_size = config.height_*config.width_*config.channels_;
      std::vector<ST> src_array(image_size);
      std::vector<DT> dst_array(image_size);
      for (int y = 0; y < config.height_; ++y) {
        for (int x = 0; x < config.width_; ++x) {
          for (int c = 0; c < config.channels_; ++c) {
            const int index = (y*config.width_+x)*config.channels_+c;
            src_array[index] = ST((index+1)%255);
          }
        }
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

      cv::Mat ocv_src(
          config.height_, config.width_,
          CV_MAKE_TYPE(cv::DataDepth<ST>::value, config.channels_),
          &src_array[0]);
      cv::Mat ocv_dst1(
          config.height_, config.width_,
          CV_MAKE_TYPE(cv::DataDepth<DT>::value, config.channels_),
          &dst_array[0]);
      cv::Mat ocv_dst2(
          config.height_, config.width_,
          CV_MAKE_TYPE(cv::DataDepth<DT>::value, config.channels_));

      TGB_LOG_INFO(
          "Test on config(%dx%dx%d, %dx%d, %s).\n",
          config.width_, config.height_, config.channels_,
          config.kernel_x_size_, config.kernel_y_size_,
          border_string.c_str());

      PixelPicker<ST> src_picker(
          config.height_, config.width_, config.channels_,
          config.width_*config.channels_*sizeof(ST),
          &src_array[0]);
      PixelPicker<DT> dst_picker(
          config.height_, config.width_, config.channels_,
          config.width_*config.channels_*sizeof(ST),
          &dst_array[0]);
      GaussianBlurHost(
          src_picker, dst_picker,
          config.kernel_x_size_, config.kernel_y_size_,
          0.0, 0.0, config.border_type_);

      double sum = 0.0;
      cv::GaussianBlur(
          ocv_src, ocv_dst2, cv::Size(
          config.kernel_x_size_, config.kernel_y_size_),
          0.0, 0.0, ocv_border_type);
      for (int y = 0; y < config.height_; ++y) {
        const DT *ocv_dst1_data_ptr = ocv_dst1.ptr<DT>(y);
        const DT *ocv_dst2_data_ptr = ocv_dst2.ptr<DT>(y);
        for (int x = 0; x < config.width_; ++x) {
          for (int c = 0; c < config.channels_; ++c) {
            const DT val_dst1 = *(ocv_dst1_data_ptr++);
            const DT val_dst2 = *(ocv_dst2_data_ptr++);
            sum += (val_dst1-val_dst2)*(val_dst1-val_dst2);
            if (val_dst1 != val_dst2) {
              //TGB_LOG_ERROR("Filter's diff (%d, %d, %d).\n", y, x, c);
            }
          }
        }
      }
      sum = std::sqrt(sum/ocv_dst1.total());
      TGB_LOG_INFO("Filter's average error: %.6lf.\n", sum);
    }

    return 0;
  }

}
