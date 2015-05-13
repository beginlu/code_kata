#include "opencv2/highgui/highgui.hpp"
#include "filter/gaussian_blur.cuh"

namespace test_gaussian_blur {

  int TestFunc3(void) {
    const std::string src_path = "fruits.jpg";
    const std::string dst_path = "fruits_blured.jpg";

    cv::Mat ocv_src_mat = cv::imread(src_path);
    cv::Mat ocv_dst_mat(ocv_src_mat.size(), ocv_src_mat.type());

    typedef unsigned char ST;
    typedef ST            DT;
    PixelPicker<ST> src_picker(
        ocv_src_mat.rows, ocv_src_mat.cols, ocv_src_mat.channels(),
        ocv_src_mat.step, reinterpret_cast<ST*>(ocv_src_mat.data));
    PixelPicker<DT> dst_picker(
        ocv_dst_mat.rows, ocv_dst_mat.cols, ocv_dst_mat.channels(),
        ocv_dst_mat.step, reinterpret_cast<DT*>(ocv_dst_mat.data));
    GaussianBlurHost(
        src_picker, dst_picker, 15, 15, 0.0, 0.0, BorderType::kReflect101);

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
