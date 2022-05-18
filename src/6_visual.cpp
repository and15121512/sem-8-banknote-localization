#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>
#include <regex>

using namespace std;
using json = nlohmann::json;

struct FramePoint {
    std::string id;
    cv::Point2f point;
};

std::vector<cv::Point2f> toCvPointVec(const std::vector<FramePoint>& fps) {
    std::vector<cv::Point2f> cps;
    for (const auto& fp : fps) {
        cps.push_back(fp.point);
    }
    return cps;
}

std::vector<FramePoint> toFramePointVec(const std::vector<cv::Point2f>& cps, const std::vector<FramePoint>& template_fps, size_t frame_cnt) {
    std::regex key_re("^[0-9]+(?=_)");
    std::vector<FramePoint> fps;
    for (const auto& cp : cps) {
        fps.push_back({"", cp});
    }
    for (size_t i = 0; i < template_fps.size(); ++i) {
        fps[i].id = std::regex_replace(template_fps[i].id, key_re, std::to_string(frame_cnt + 1));
    }
    return fps;
}

std::map<int32_t, std::vector<FramePoint>> extractedFromJSON(std::string in_json) {
    std::map<int32_t, std::vector<FramePoint>> res;
    json j = json::parse(in_json);

    for (auto x : j.at("metadata").items()) {
        std::string frame_num_str = x.value().at("vid");
        int32_t frame_num = std::stoi(frame_num_str) - 1;

        int32_t man = x.value().at("man");
        //if (man != 1) {
        //    continue;
        //}

        if (res.find(frame_num) == res.end()) {
            res[frame_num] = std::vector<FramePoint>();
        }
        for (size_t i = 1; i < 9; i += 2) {
            res[frame_num].push_back({
                x.key()
                , { x.value().at("xy").at(i), x.value().at("xy").at(i + 1) }
            });
        }
    }
    return res;
}

int main(int argc, char** argv) {
    const float ratio_thresh = 0.7f;

    if (2 != argc) {
        cout << "[fatal] Two args expected: sample path" << std::endl;
    }
    std::filesystem::path sample_loc(argv[1]);
    std::filesystem::path video_name("video.mp4");
    std::filesystem::path layout_tr_name("3_tracker_output.json");
    std::filesystem::path layout_gt_name("4_ground_truth.json");
    std::filesystem::path input_loc("../input");
    std::filesystem::path output_loc("../output");
    std::filesystem::path layout_loc(output_loc / sample_loc / "json");

    cv::VideoCapture cap((input_loc / sample_loc / video_name).string());
    if (!cap.isOpened()) {
        cout << "[fatal] Error opening video stream or file" << endl;
        return -1;
    }
    cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();

    std::ifstream fin_tr((layout_loc / layout_tr_name).string());
    if (!fin_tr) {
        std::cout << "[fatal] Can't open file with layout" << std::endl; return -1;
    }
    std::string in_json_tr = std::string((std::istreambuf_iterator<char>(fin_tr)), std::istreambuf_iterator<char>());
    std::map<int32_t, std::vector<FramePoint>> corners_tr = extractedFromJSON(in_json_tr);

    std::ifstream fin_gt((layout_loc / layout_gt_name).string());
    std::string in_json_gt;
    std::map<int32_t, std::vector<FramePoint>> corners_gt;
    if (fin_gt) {
        in_json_gt = std::string((std::istreambuf_iterator<char>(fin_gt)), std::istreambuf_iterator<char>());
        corners_gt = extractedFromJSON(in_json_gt);
        //std::cout << "[fatal] Can't open file with layout" << std::endl; return -1;
    }

    // Algorithm
    std::vector<cv::KeyPoint> ref_keypoints;
    cv::Mat ref_descriptors;
    std::vector<size_t> res(2, 0);
    auto font = cv::FONT_HERSHEY_SIMPLEX;
    auto fontScale = 0.5;
    auto thickness = 1;

    for (size_t frame_cnt = 0; ; ++frame_cnt) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        std::vector<cv::Point2f> cor_tr = toCvPointVec(corners_tr[frame_cnt]);
        for (size_t i = 0; i < cor_tr.size(); ++i) {
            cv::circle(frame, cor_tr[i], 2, cv::Scalar(0, 255, 0), -1);
            cv::putText(frame, std::to_string(i), cor_tr[i] + cv::Point2f(-5, -10), font, fontScale, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
        }
        std::vector<cv::Point2f> cor_gt;
        if (!corners_gt.empty()) {
            cor_gt = toCvPointVec(corners_gt[frame_cnt]);
        }
        for (size_t i = 0; i < cor_gt.size(); ++i) {
            cv::circle(frame, cor_gt[i], 2, cv::Scalar(255, 0, 0), -1);
            cv::putText(frame, std::to_string(i), cor_gt[i] + cv::Point2f(-10, 5), font, fontScale, cv::Scalar(255, 0, 0), thickness, cv::LINE_AA);
        }
        
        cv::putText(frame, "Frame: " + std::to_string(frame_cnt), cv::Point2f(10, 25), font, 1, (0, 0, 255), 2, cv::LINE_AA);
        cv::imshow("Frame", frame);
        char c = (char)cv::waitKey(30);
        if (27 == c)
            break;
        if (32 == c) {
            c = 1;
            while (32 != c) {
                c = (char)cv::waitKey(1);
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
