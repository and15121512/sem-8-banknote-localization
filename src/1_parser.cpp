#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;

int main(int argc, char** argv) {
    const float ratio_thresh = 0.7f;
    std::filesystem::path video_loc(argv[1]);
    std::filesystem::path video_name("video.mp4");
    std::filesystem::path input_loc("../input");
    std::filesystem::path output_loc("../output");
    std::filesystem::path frames_loc(output_loc / video_loc / "frames/");
    std::filesystem::path json_loc(output_loc / video_loc / "json/");

    if (2 != argc) {
        cout << "[fatal] One arg expected: path to video file" << std::endl;
    }
    cv::VideoCapture cap((input_loc / video_loc / video_name).string());
    if (!cap.isOpened()) {
        std::cout << "[fatal] Error opening video stream or file: "
            << (input_loc / video_loc / video_name).string()
            << std::endl;
        return -1;
    }

    if (!std::filesystem::exists(frames_loc)) {
        bool success = std::filesystem::create_directories(frames_loc);
    }
    if (!std::filesystem::exists(json_loc)) {
        bool success = std::filesystem::create_directories(json_loc);
    }

    // Algorithm
    size_t all = 0;
    size_t successed = 0;
    cv::Mat frame;
    bool success = cap.read(frame);
    for (; success;) {
        cv::imwrite((frames_loc / ("frame" + std::to_string(all) + ".png")).string(), frame);
        if (success)
            successed += 1;
        all += 1;
        success = cap.read(frame);
    }
    cap.release();
    std::cout << "[info][parser] Written: "
        + std::to_string(successed)
        + "/"
        + std::to_string(all) << std::endl;
    cv::destroyAllWindows();
    return 0;
}
