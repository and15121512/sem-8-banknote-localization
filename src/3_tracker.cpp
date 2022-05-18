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
        if (man != 1) {
            continue;
        }

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

void saveToJSON(std::string& out_json, const std::vector<std::vector<FramePoint>>& annotations, const std::vector<size_t>& res) {
    json j = json::parse(out_json);

    json::array_t res_arr = { res[0], res[1] };
    j.at("project").push_back(std::pair<const string, json>("resolution", res_arr));
    //size_t cnt = 0;
    for (auto& x : j.at("metadata").items()) {
        std::string frame_num_str = x.value().at("vid");
        int32_t frame_num = std::stoi(frame_num_str) - 1;

        if (x.value().at("man") == 1) {
            continue;
        }

        auto fps = annotations[frame_num];
        json::number_float_t xx0 = fps[0].point.x;
        json::number_float_t yy0 = fps[0].point.y;
        json::number_float_t xx1 = fps[1].point.x;
        json::number_float_t yy1 = fps[1].point.y;
        json::number_float_t xx2 = fps[2].point.x;
        json::number_float_t yy2 = fps[2].point.y;
        json::number_float_t xx3 = fps[3].point.x;
        json::number_float_t yy3 = fps[3].point.y;
        json::array_t arr = {
            7, 
            xx0, yy0,
            xx1, yy1,
            xx2, yy2,
            xx3, yy3
        };
        x.value().at("xy").swap(arr);
    }
    //std::cout << cnt << std::endl;
    out_json = j.dump();
}

int main(int argc, char** argv) {
    const float ratio_thresh = 0.7f;

    if (2 != argc) {
        cout << "[fatal] Two args expected: sample path" << std::endl;
    }
    std::filesystem::path sample_loc(argv[1]);
    std::filesystem::path video_name("video.mp4");
    std::filesystem::path layout_name("2_first_frame_expanded.json");
    std::filesystem::path input_loc("../input");
    std::filesystem::path output_loc("../output");
    std::filesystem::path layout_loc(output_loc / sample_loc / "json");

    cv::VideoCapture cap((input_loc / sample_loc / video_name).string());
    if (!cap.isOpened()) {
        cout << "[fatal] Error opening video stream or file" << endl;
        return -1;
    }
    cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();

    std::ifstream fin((layout_loc / layout_name).string());
    if (!fin) {
        std::cout << "[fatal] Can't open file with layout" << std::endl; return -1;
    }
    std::string in_json = std::string((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    std::map<int32_t, std::vector<FramePoint>> corners = extractedFromJSON(in_json);

    // Algorithm
    std::vector<std::vector<FramePoint>> annotations;
    std::vector<FramePoint> ref_corners;
    std::vector<cv::KeyPoint> ref_keypoints;
    cv::Mat ref_descriptors;
    std::vector<size_t> res(2, 0);
    for (size_t frame_cnt = 0; ; ++frame_cnt) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        if (0 == frame_cnt) {
            res[0] = frame.size().width;
            res[1] = frame.size().height;
        }

        std::vector<cv::KeyPoint> curr_keypoints;
        cv::Mat curr_descriptors;
        siftPtr->detectAndCompute(frame, cv::noArray(), curr_keypoints, curr_descriptors);
        if (corners.find(frame_cnt) != corners.end()) {
            ref_corners = corners[frame_cnt];
            ref_keypoints = curr_keypoints;
            ref_descriptors = curr_descriptors.clone();
            annotations.push_back(corners[frame_cnt]);
            continue;
        }
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<cv::DMatch> > knn_matches;
        matcher->knnMatch(ref_descriptors, curr_descriptors, knn_matches, 2);

        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        std::vector<cv::Point2f> first;
        std::vector<cv::Point2f> curr;
        for (size_t i = 0; i < good_matches.size(); i++) {
            first.push_back(ref_keypoints[good_matches[i].queryIdx].pt);
            curr.push_back(curr_keypoints[good_matches[i].trainIdx].pt);
        }
        if (first.size() <= 5 || curr.size() <= 5) { continue; }
        cv::Mat H = cv::findHomography(first, curr, cv::RANSAC);
        std::vector<cv::Point2f> new_corners(ref_corners.size());
        cv::perspectiveTransform(toCvPointVec(ref_corners), new_corners, H);
        annotations.push_back(toFramePointVec(new_corners, ref_corners, frame_cnt));

        // Visualization        
        //cv::line(frame, new_corners[0], new_corners[1], cv::Scalar(0, 255, 0), 2);
        //cv::line(frame, new_corners[1], new_corners[2], cv::Scalar(0, 255, 0), 2);
        //cv::line(frame, new_corners[2], new_corners[3], cv::Scalar(0, 255, 0), 2);
        //cv::line(frame, new_corners[3], new_corners[0], cv::Scalar(0, 255, 0), 2);
        //cv::circle(frame, new_corners[0], 5, cv::Scalar(255, 255, 0), -1);
        //cv::circle(frame, new_corners[1], 5, cv::Scalar(0, 255, 255), -1);
        //cv::circle(frame, new_corners[2], 5, cv::Scalar(255, 0, 255), -1);
        //cv::circle(frame, new_corners[3], 5, cv::Scalar(255, 0, 0), -1);
        //cv::Mat out;
        //cv::resize(frame, out, cv::Size(), 2, 2);
        //cv::imshow("Frame", out);

        //char c = (char)cv::waitKey(1);
        //if (c == 27)
        //    break;
        //if (c == 32) {
        //    c = 0;
        //    while (c != 32) {
        //        c = (char)cv::waitKey(1);
        //    }
        //}
    }

    saveToJSON(in_json, annotations, res);
    std::ofstream fout(layout_loc / "3_tracker_output.json");
    if (!fout) {
        std::cout << "Can't write the output." << std::endl;
    }
    fout << in_json;
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
