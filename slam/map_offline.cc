// Offline ORB-SLAM3 (monocular) over a recorded mapping drive (map_recorder.py).
//   map_offline <vocabulary> <settings.yaml> <maps/name>
// Reads <dir>/frames/<unix_ns>.jpg in time order, never faster than they were recorded (so
// local mapping keeps up), then writes into <dir>:
//   poses.txt      per frame: t state map_id  tx ty tz  qx qy qz qw   (camera-to-world)
//   keyframes.txt  per keyframe: t map_id  tx ty tz  qx qy qz qw
//   points.txt     per map point: map_id  x y z  n_observations
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <System.h>

// The atlas (all maps, keyframes, points) is private in System; this offline tool only reads it.
// Explicit template instantiation may name private members, which gives a legal pointer to it.
template <typename Tag, typename Tag::type M>
struct Rob { friend typename Tag::type get(Tag) { return M; } };
struct AtlasTag { typedef ORB_SLAM3::Atlas* ORB_SLAM3::System::*type; friend type get(AtlasTag); };
template struct Rob<AtlasTag, &ORB_SLAM3::System::mpAtlas>;

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s vocabulary settings.yaml maps/<name>\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[3];
    std::vector<std::pair<long long, std::string>> frames;
    for (auto& e : fs::directory_iterator(dir + "/frames"))
        if (e.path().extension() == ".jpg")
            frames.emplace_back(std::stoll(e.path().stem().string()), e.path().string());
    std::sort(frames.begin(), frames.end());
    if (frames.empty()) { std::fprintf(stderr, "no frames in %s/frames\n", dir.c_str()); return 1; }
    std::printf("%zu frames, %.0f s of driving\n", frames.size(),
                (frames.back().first - frames.front().first) * 1e-9);
    std::fflush(stdout);

    ORB_SLAM3::System slam(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, false);
    ORB_SLAM3::Atlas* atlas = slam.*get(AtlasTag());
    FILE* pf = std::fopen((dir + "/poses.txt").c_str(), "w");
    using clk = std::chrono::steady_clock;
    const auto wall0 = clk::now();
    const long long t0 = frames.front().first;
    int tracked = 0;
    for (size_t i = 0; i < frames.size(); i++) {
        const double ts = (frames[i].first - t0) * 1e-9;
        std::this_thread::sleep_until(wall0 + std::chrono::duration_cast<clk::duration>(
                                                  std::chrono::duration<double>(ts)));
        cv::Mat img = cv::imread(frames[i].second, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        Sophus::SE3f Tcw = slam.TrackMonocular(img, ts);
        const int state = slam.GetTrackingState();
        const long map_id = atlas->GetCurrentMap()->GetId();
        if (state == ORB_SLAM3::Tracking::OK) tracked++;
        const Sophus::SE3f Twc = Tcw.inverse();
        const Eigen::Vector3f p = Twc.translation();
        const Eigen::Quaternionf q = Twc.unit_quaternion();
        std::fprintf(pf, "%.9f %d %ld %.5f %.5f %.5f %.6f %.6f %.6f %.6f\n", frames[i].first * 1e-9,
                     state, map_id, p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w());
        if (i % 100 == 0) {
            std::printf("frame %zu/%zu  tracked %d  maps %zu\n", i, frames.size(), tracked,
                        atlas->GetAllMaps().size());
            std::fflush(stdout);
        }
    }
    std::fclose(pf);
    slam.Shutdown();   // finishes local mapping / loop closing first

    FILE* kf = std::fopen((dir + "/keyframes.txt").c_str(), "w");
    FILE* mp = std::fopen((dir + "/points.txt").c_str(), "w");
    size_t nk = 0, np = 0;
    for (ORB_SLAM3::Map* m : atlas->GetAllMaps()) {
        for (ORB_SLAM3::KeyFrame* k : m->GetAllKeyFrames()) {
            if (!k || k->isBad()) continue;
            const Sophus::SE3f Twc = k->GetPoseInverse();
            const Eigen::Vector3f p = Twc.translation();
            const Eigen::Quaternionf q = Twc.unit_quaternion();
            std::fprintf(kf, "%.9f %ld %.5f %.5f %.5f %.6f %.6f %.6f %.6f\n", t0 * 1e-9 + k->mTimeStamp,
                         m->GetId(), p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w());
            nk++;
        }
        for (ORB_SLAM3::MapPoint* pt : m->GetAllMapPoints()) {
            if (!pt || pt->isBad()) continue;
            const Eigen::Vector3f x = pt->GetWorldPos();
            std::fprintf(mp, "%ld %.5f %.5f %.5f %d\n", m->GetId(), x.x(), x.y(), x.z(), pt->Observations());
            np++;
        }
    }
    std::fclose(kf);
    std::fclose(mp);
    std::printf("done: tracked %d of %zu frames, %zu maps, %zu keyframes, %zu points\n", tracked,
                frames.size(), atlas->GetAllMaps().size(), nk, np);
    return 0;
}
