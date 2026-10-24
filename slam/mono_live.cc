// Live monocular ORB-SLAM3 on the robot's USB camera, headless.
// Prints once a second: fps, mean/max tracking time, tracking state, tracked map points, camera position.
// With a view path, also writes an annotated JPEG there ~10x a second for the live web view
// (green = features matched to the map, with any text found in <view>.txt as a caption).
// "tagonly" skips ORB-SLAM entirely (no vocabulary load, far less CPU) - for docking.
//   mono_live <vocabulary> <settings.yaml> [seconds] [view.jpg] [tagonly]
#include <chrono>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "System.h"

// The atlas (all map points) is private in System; explicit template instantiation may name
// private members, which gives a legal pointer to it (read only, for the live mini-map).
template <typename Tag, typename Tag::type M>
struct Rob { friend typename Tag::type get(Tag) { return M; } };
struct AtlasTag { typedef ORB_SLAM3::Atlas* ORB_SLAM3::System::*type; friend type get(AtlasTag); };
template struct Rob<AtlasTag, &ORB_SLAM3::System::mpAtlas>;

// Live mini-map (bottom-left of the view): top-down on the first camera's x/z plane (the camera
// sits level, so that's the floor), no metric scale. Grey = map points, blue = path, red = robot.
static const int MINI = 200;
static cv::Mat g_mini;

static cv::Mat RenderMinimap(ORB_SLAM3::Map* map, const std::vector<cv::Point2f>& path,
                             const Sophus::SE3f& Twc, bool ok) {
    cv::Mat m(MINI, MINI, CV_8UC3, cv::Scalar(35, 30, 30));
    std::vector<cv::Point2f> pts;
    if (map)
        for (auto* p : map->GetAllMapPoints())
            if (p && !p->isBad()) { const Eigen::Vector3f x = p->GetWorldPos(); pts.emplace_back(x.x(), x.z()); }
    std::vector<float> xs, zs;
    for (const auto& q : pts) { xs.push_back(q.x); zs.push_back(q.y); }
    float x0 = 0, x1 = 0, z0 = 0, z1 = 0;
    if (xs.size() > 20) {   // 2-98 % of the points, so a few far outliers don't shrink the room
        auto pct = [](std::vector<float> v, double f) {
            std::nth_element(v.begin(), v.begin() + (size_t)(f * (v.size() - 1)), v.end());
            return v[(size_t)(f * (v.size() - 1))];
        };
        x0 = pct(xs, 0.02); x1 = pct(xs, 0.98); z0 = pct(zs, 0.02); z1 = pct(zs, 0.98);
    }
    for (const auto& q : path) { x0 = std::min(x0, q.x); x1 = std::max(x1, q.x); z0 = std::min(z0, q.y); z1 = std::max(z1, q.y); }
    const float span = std::max({x1 - x0, z1 - z0, 0.2f}) * 1.1f;
    const float s = (MINI - 10) / span, cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
    auto px = [&](float x, float z) { return cv::Point((int)(MINI / 2 + (x - cx) * s), (int)(MINI / 2 - (z - cz) * s)); };
    for (const auto& q : pts) {
        const cv::Point a = px(q.x, q.y);
        if (a.x >= 0 && a.y >= 0 && a.x < MINI && a.y < MINI) m.at<cv::Vec3b>(a) = cv::Vec3b(210, 210, 210);
    }
    for (size_t i = 1; i < path.size(); i++)
        cv::line(m, px(path[i - 1].x, path[i - 1].y), px(path[i].x, path[i].y), cv::Scalar(255, 160, 60), 1, cv::LINE_AA);
    if (ok) {
        const Eigen::Vector3f c = Twc.translation(), f = Twc.rotationMatrix().col(2);
        const float n = std::max(1e-6f, std::hypot(f.x(), f.z()));
        const cv::Point a = px(c.x(), c.z());
        cv::line(m, a, a + cv::Point((int)(14 * f.x() / n), (int)(-14 * f.z() / n)), cv::Scalar(60, 60, 255), 2, cv::LINE_AA);
        cv::circle(m, a, 4, cv::Scalar(60, 60, 255), -1, cv::LINE_AA);
    }
    cv::putText(m, "MAP", {5, 14}, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::rectangle(m, {0, 0}, {MINI - 1, MINI - 1}, cv::Scalar(255, 255, 255), 1);
    return m;
}

static const char* StateName(int s) {
    switch (s) {
        case ORB_SLAM3::Tracking::SYSTEM_NOT_READY: return "NOT_READY";
        case ORB_SLAM3::Tracking::NO_IMAGES_YET: return "NO_IMAGES";
        case ORB_SLAM3::Tracking::NOT_INITIALIZED: return "NOT_INIT";
        case ORB_SLAM3::Tracking::OK: return "OK";
        case ORB_SLAM3::Tracking::RECENTLY_LOST: return "RECENTLY_LOST";
        case ORB_SLAM3::Tracking::LOST: return "LOST";
        default: return "?";
    }
}

static volatile std::sig_atomic_t g_stop = 0;

// Dock AprilTag (tag36h11 ID 0, printed black square measured 79 mm).
constexpr int DOCK_TAG_ID = 0;
constexpr float DOCK_TAG_MM = 79.0f;
const char* TAG_FILE = "/dev/shm/slam_tag.txt";
constexpr double CENTRE_TOL_DEG = 1.5;
// Second tag (ID 1) mounted beside ID 0: centre-to-centre offset along the wall in mm,
// + = to the right of ID 0 as you face the wall. Read from board.txt next to the binary's
// settings; 0 = not fitted (single-tag mode). With both tags in view the pose is solved
// from all 8 corners, which removes the single tag's mirror-image ambiguity.
static double g_board_mm = 0.0;
constexpr int SECOND_TAG_ID = 1;   // "dead centre" band drawn on the view (matches robot_control)

struct TagSeen {
    bool seen = false;
    std::vector<cv::Point2f> corners;
    double dist_cm = 0, x_cm = 0, bearing_deg = 0, yaw_deg = 0;
    // camera in the TAG's frame (floor plane): lateral = sideways from the tag's centre line
    // (+ = tag's +x side), normal = straight out from the wall, heading = where the camera
    // points relative to "straight at the wall" (+ = toward the tag's +x side)
    double lateral_cm = 0, normal_cm = 0, heading_deg = 0;
    // both mirror-image pose solutions, + side first: lateral, normal, heading, fit error
    double alt[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    bool has_alt = false;
    bool board = false;            // pose came from both tags
    std::vector<std::pair<int, std::vector<cv::Point2f>>> all;   // every tag seen (for the overlay)
};

// Finds the dock tag; pose is relative to the camera: dist straight ahead, x to the right,
// bearing = direction to the tag (+ right), yaw = how much the tag faces away (0 = square on).
static TagSeen FindDockTag(const cv::Mat& gray, const cv::aruco::ArucoDetector& det,
                           const cv::Mat& K, const cv::Mat& D) {
    TagSeen t;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    det.detectMarkers(gray, corners, ids);
    for (size_t i = 0; i < ids.size(); i++) t.all.push_back({ids[i], corners[i]});
    int primary = -1;
    float shift = 0.0f;                  // ID 1 alone: its centre is g_board_mm along the wall from ID 0
    for (size_t i = 0; i < ids.size(); i++) if (ids[i] == DOCK_TAG_ID) primary = (int)i;
    if (primary < 0 && g_board_mm != 0.0)
        for (size_t i = 0; i < ids.size(); i++)
            if (ids[i] == SECOND_TAG_ID) { primary = (int)i; shift = (float)g_board_mm; }
    for (size_t i = 0; i < ids.size(); i++) {
        if ((int)i != primary) continue;
        const float h = DOCK_TAG_MM / 2;
        const std::vector<cv::Point3f> obj = {{-h + shift, h, 0}, {h + shift, h, 0}, {h + shift, -h, 0}, {-h + shift, -h, 0}};
        // A small flat tag has two near-equal pose solutions (seen from the left or from the
        // right of its centre line); a single-frame pick flips between them.
        std::vector<cv::Mat> rvecs, tvecs;
        cv::Mat errs;
        const std::vector<cv::Point3f> objc = {{-h, h, 0}, {h, h, 0}, {h, -h, 0}, {-h, -h, 0}};
        if (cv::solvePnPGeneric(objc, corners[i], K, D, rvecs, tvecs, false, cv::SOLVEPNP_IPPE_SQUARE,
                                cv::noArray(), cv::noArray(), errs) < 1) continue;
        if (shift != 0.0f)                    // express the pose in ID 0's frame: tag origin moves by -shift
            for (size_t k = 0; k < rvecs.size(); k++) {
                cv::Mat Rk;
                cv::Rodrigues(rvecs[k], Rk);
                tvecs[k] = tvecs[k] - Rk * (cv::Mat_<double>(3, 1) << shift, 0, 0);
            }
        // Evidence over many frames: each solution puts the camera on one side of the tag's
        // centre line. Average each side's fit error over recent frames and take the side
        // that fits better on average (a single frame's difference is too small to trust).
        static double err_pos = 0.0, err_neg = 0.0;     // running averages per side
        static int n_seen = 0;
        int pick = 0;
        if (rvecs.size() > 1) {
            auto lateral_of = [&](int k) {
                cv::Mat Rk;
                cv::Rodrigues(rvecs[k], Rk);
                const cv::Mat pk = -Rk.t() * tvecs[k];
                return pk.at<double>(0);
            };
            const int pos = lateral_of(0) >= lateral_of(1) ? 0 : 1;   // solution on the + side
            const double a = n_seen < 15 ? 1.0 / (n_seen + 1) : 1.0 / 15;
            err_pos += a * (errs.at<double>(pos) - err_pos);
            err_neg += a * (errs.at<double>(1 - pos) - err_neg);
            n_seen++;
            pick = err_pos <= err_neg ? pos : 1 - pos;
            for (int side = 0; side < 2; side++) {
                const int k = side == 0 ? pos : 1 - pos;
                cv::Mat Rk;
                cv::Rodrigues(rvecs[k], Rk);
                const cv::Mat pk = -Rk.t() * tvecs[k];
                const cv::Mat fk = Rk.t() * (cv::Mat_<double>(3, 1) << 0, 0, 1);
                t.alt[side][0] = pk.at<double>(0) / 10.0;
                t.alt[side][1] = pk.at<double>(2) / 10.0;
                t.alt[side][2] = std::atan2(fk.at<double>(0), -fk.at<double>(2)) * 180.0 / CV_PI;
                t.alt[side][3] = errs.at<double>(k);
            }
            t.has_alt = true;
        }
        cv::Mat rvec = rvecs[pick], tvec = tvecs[pick];
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        const double tx = tvec.at<double>(0), tz = tvec.at<double>(2);
        t.seen = true;
        t.corners = corners[i];
        t.dist_cm = tz / 10.0;
        t.x_cm = tx / 10.0;
        t.bearing_deg = std::atan2(tx, tz) * 180.0 / CV_PI;
        // tag normal (its +z axis) in camera frame; yaw around the camera's vertical axis
        t.yaw_deg = std::atan2(R.at<double>(0, 2), -R.at<double>(2, 2)) * 180.0 / CV_PI;
        const cv::Mat Rt = R.t();
        const cv::Mat p = -Rt * tvec;                                  // camera position in tag frame
        const cv::Mat f = Rt * (cv::Mat_<double>(3, 1) << 0, 0, 1);    // camera forward in tag frame
        t.lateral_cm = p.at<double>(0) / 10.0;
        t.normal_cm = p.at<double>(2) / 10.0;
        t.heading_deg = std::atan2(f.at<double>(0), -f.at<double>(2)) * 180.0 / CV_PI;

        // both tags in view: one unambiguous pose from all 8 corners (frame stays at ID 0)
        if (g_board_mm != 0.0 && shift == 0.0f) {
            for (size_t j = 0; j < ids.size(); j++) {
                if (ids[j] != SECOND_TAG_ID) continue;
                const float cx2 = (float)g_board_mm;
                std::vector<cv::Point3f> obj8 = obj;
                for (const auto& q : obj) obj8.push_back({q.x + cx2, q.y, 0});
                std::vector<cv::Point2f> img8 = corners[i];
                img8.insert(img8.end(), corners[j].begin(), corners[j].end());
                cv::Mat rb, tb;
                if (!cv::solvePnP(obj8, img8, K, D, rb, tb, false, cv::SOLVEPNP_SQPNP)) break;
                cv::solvePnPRefineLM(obj8, img8, K, D, rb, tb);
                cv::Mat Rb;
                cv::Rodrigues(rb, Rb);
                const cv::Mat pb = -Rb.t() * tb;
                const cv::Mat fb = Rb.t() * (cv::Mat_<double>(3, 1) << 0, 0, 1);
                t.lateral_cm = pb.at<double>(0) / 10.0;
                t.normal_cm = pb.at<double>(2) / 10.0;
                t.heading_deg = std::atan2(fb.at<double>(0), -fb.at<double>(2)) * 180.0 / CV_PI;
                t.yaw_deg = std::atan2(Rb.at<double>(0, 2), -Rb.at<double>(2, 2)) * 180.0 / CV_PI;
                for (int side = 0; side < 2; side++) {     // both "answers" = the one real answer
                    t.alt[side][0] = t.lateral_cm;
                    t.alt[side][1] = t.normal_cm;
                    t.alt[side][2] = t.heading_deg;
                    t.alt[side][3] = 0.0;
                }
                t.has_alt = true;
                t.board = true;
                break;
            }
        }
        break;
    }
    return t;
}

static void WriteView(const std::string& path, cv::Mat img, ORB_SLAM3::System* slam, const char* state,
                      double fps, int points, const Eigen::Vector3f& c, const TagSeen& tag, const cv::Mat& K) {
    if (slam) {
        const auto kps = slam->GetTrackedKeyPointsUn();
        const auto mps = slam->GetTrackedMapPoints();
        for (size_t i = 0; i < kps.size(); i++) {
            const bool mapped = i < mps.size() && mps[i] != nullptr;
            cv::circle(img, kps[i].pt, mapped ? 3 : 2, mapped ? cv::Scalar(0, 220, 0) : cv::Scalar(200, 120, 0),
                       mapped ? -1 : 1);
        }
    }
    // grid every 40 px (faint) to compare the picture with the room
    for (int gx = 40; gx < img.cols; gx += 40)
        cv::line(img, {gx, 44}, {gx, img.rows - 27}, cv::Scalar(90, 90, 90), 1, cv::LINE_4);
    for (int gy = 80; gy < img.rows - 27; gy += 40)
        cv::line(img, {0, gy}, {img.cols, gy}, cv::Scalar(90, 90, 90), 1, cv::LINE_4);
    // a box and label on every tag seen
    for (const auto& m : tag.all) {
        const cv::Scalar c = m.first == 0 ? cv::Scalar(255, 0, 255) : cv::Scalar(255, 200, 0);
        for (int k = 0; k < 4; k++) cv::line(img, m.second[k], m.second[(k + 1) % 4], c, 2, cv::LINE_AA);
        cv::Point2f lo = m.second[0];
        for (const auto& q : m.second) { lo.x = std::min(lo.x, q.x); lo.y = std::min(lo.y, q.y); }
        char lbl[16];
        std::snprintf(lbl, sizeof(lbl), "ID %d", m.first);
        cv::putText(img, lbl, {(int)lo.x, std::max(56, (int)lo.y - 6)}, cv::FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv::LINE_AA);
    }
    // centre guide: the camera's optical centre line and the "dead centre" band around it
    const int cx = (int)std::lround(K.at<double>(0, 2));
    const int band = (int)std::lround(K.at<double>(0, 0) * std::tan(CENTRE_TOL_DEG * CV_PI / 180.0));
    const bool centred = tag.seen && std::fabs(tag.bearing_deg) <= CENTRE_TOL_DEG;
    const cv::Scalar guide = centred ? cv::Scalar(80, 255, 80) : cv::Scalar(0, 200, 255);
    cv::line(img, {cx, 44}, {cx, img.rows - 27}, guide, 1, cv::LINE_AA);
    cv::line(img, {cx - band, 44}, {cx - band, img.rows - 27}, guide, 1, cv::LINE_4);
    cv::line(img, {cx + band, 44}, {cx + band, img.rows - 27}, guide, 1, cv::LINE_4);
    if (tag.seen) {
        cv::Point2f m(0, 0);
        for (const auto& p : tag.corners) m += p * 0.25f;
        cv::circle(img, m, 5, guide, -1, cv::LINE_AA);
        for (int k = 0; k < 4; k++)
            cv::line(img, tag.corners[k], tag.corners[(k + 1) % 4], cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
        char tl[120];
        std::snprintf(tl, sizeof(tl), "%s %.0f cm  bearing %+.1f  |  off centre line %+.1f cm  heading %+.1f",
                      tag.board ? "2 TAGS" : "TAG", tag.dist_cm, tag.bearing_deg, tag.lateral_cm, tag.heading_deg);
        cv::rectangle(img, {0, img.rows - 26}, {img.cols, img.rows}, cv::Scalar(0, 0, 0), -1);
        cv::putText(img, tl, {6, img.rows - 8}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 120, 255), 1,
                    cv::LINE_AA);
    }
    std::string caption;
    std::ifstream f(path + ".txt");
    std::getline(f, caption);
    char line[160];
    if (slam)
        std::snprintf(line, sizeof(line), "ORB-SLAM3 %s  %.0f fps  pts %d  pos %+.2f %+.2f %+.2f",
                      state, fps, points, c.x(), c.y(), c.z());
    else
        std::snprintf(line, sizeof(line), "TAG-ONLY (SLAM off)  %.0f fps  %s", fps,
                      centred ? "tag CENTRED" : tag.seen ? "tag off centre" : "no tag");
    cv::rectangle(img, {0, 0}, {img.cols, 44}, cv::Scalar(0, 0, 0), -1);
    const cv::Scalar col = (slam ? std::string(state) == "OK" : centred) ? cv::Scalar(80, 255, 80)
                                                                          : cv::Scalar(60, 160, 255);
    cv::putText(img, line, {6, 18}, cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv::LINE_AA);
    cv::putText(img, caption, {6, 38}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    {   // bottom-right: battery and Pi temperature, written by slam_view.py to <view>.info
        std::string info;
        std::ifstream fi(path + ".info");
        std::getline(fi, info);
        if (!info.empty()) {
            int base = 0;
            const cv::Size sz = cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
            const int x = img.cols - sz.width - 8, y = img.rows - 34;
            cv::rectangle(img, {x - 4, y - sz.height - 4}, {img.cols, y + 6}, cv::Scalar(0, 0, 0), -1);
            cv::putText(img, info, {x, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }
    if (!g_mini.empty()) {   // bottom-left, above the tag bar
        const int y = img.rows - 30 - MINI;
        if (y > 44) g_mini.copyTo(img(cv::Rect(4, y, MINI, MINI)));
    }
    cv::imwrite(path + ".tmp.jpg", img, {cv::IMWRITE_JPEG_QUALITY, 70});
    std::rename((path + ".tmp.jpg").c_str(), path.c_str());   // atomic swap for the web server
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s vocabulary settings.yaml [seconds]\n", argv[0]);
        return 1;
    }
    const double seconds = argc > 3 ? std::stod(argv[3]) : 60.0;
    const std::string view = argc > 4 ? argv[4] : "";
    const bool tag_only = argc > 5 && std::string(argv[5]) == "tagonly";
    {   // two-tag board spacing (mm), kept next to the settings file
        std::string dir = argv[2];
        dir = dir.substr(0, dir.find_last_of('/') + 1);
        std::ifstream bf(dir + "board.txt");
        if (bf >> g_board_mm)
            std::printf("two-tag board: ID 1 at %+.0f mm from ID 0\n", g_board_mm);
    }
    std::signal(SIGINT, [](int) { g_stop = 1; });
    std::signal(SIGTERM, [](int) { g_stop = 1; });

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);   // always track the newest frame
    // Fixed focus: the C920e's continuous autofocus hunts at 30-80 cm and blurs the tag
    // out of detection (2026-10-14: 0/20 frames with autofocus, 20/20 fixed).
    cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
    cap.set(cv::CAP_PROP_FOCUS, 20);
    if (!cap.isOpened()) { std::fprintf(stderr, "camera open failed\n"); return 1; }

    using clk = std::chrono::steady_clock;
    auto t_load = clk::now();
    std::unique_ptr<ORB_SLAM3::System> slam;
    if (!tag_only)
        slam = std::make_unique<ORB_SLAM3::System>(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, false);
    std::printf("loaded in %.1f s%s\n",
                std::chrono::duration<double>(clk::now() - t_load).count(), tag_only ? " (tag only)" : "");
    std::fflush(stdout);

    cv::Mat K, D;
    {
        cv::FileStorage fs(argv[2], cv::FileStorage::READ);
        K = (cv::Mat_<double>(3, 3) << (double)fs["Camera1.fx"], 0, (double)fs["Camera1.cx"],
             0, (double)fs["Camera1.fy"], (double)fs["Camera1.cy"], 0, 0, 1);
        D = (cv::Mat_<double>(1, 4) << (double)fs["Camera1.k1"], (double)fs["Camera1.k2"],
             (double)fs["Camera1.p1"], (double)fs["Camera1.p2"]);
    }
    const cv::aruco::ArucoDetector tag_det(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11));
    TagSeen tag;
    long frame_no = 0;

    cv::Mat frame, gray;
    const auto t0 = clk::now();
    auto t_report = t0, t_view = t0;
    double last_fps = 0;
    ORB_SLAM3::Atlas* atlas = slam ? (*slam).*get(AtlasTag()) : nullptr;
    std::vector<cv::Point2f> path;
    long path_map = -2;
    auto t_mini = t0;
    const char* last_state = "NOT_INIT";
    int last_points = 0;
    int frames = 0;
    double track_sum = 0, track_max = 0;
    while (!g_stop && std::chrono::duration<double>(clk::now() - t0).count() < seconds) {
        if (!cap.read(frame)) continue;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        const double ts = std::chrono::duration<double>(clk::now() - t0).count();
        const auto a = clk::now();
        Sophus::SE3f Tcw;
        if (slam) Tcw = slam->TrackMonocular(gray, ts);
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - a).count();
        frames++; track_sum += ms; track_max = std::max(track_max, ms);
        if (slam) {   // live pose for the lost-tag search: time state map x y z qx qy qz qw (camera-to-world)
            const Sophus::SE3f Twc = Tcw.inverse();
            const Eigen::Quaternionf q = Twc.unit_quaternion();
            FILE* pf = std::fopen("/dev/shm/slam_pose.txt.tmp", "w");
            if (pf) {
                ORB_SLAM3::Map* m = atlas->GetCurrentMap();
                std::fprintf(pf, "%.3f %d %ld %.5f %.5f %.5f %.6f %.6f %.6f %.6f\n",
                             std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(),
                             slam->GetTrackingState(), m ? (long)m->GetId() : -1L,
                             Twc.translation().x(), Twc.translation().y(), Twc.translation().z(),
                             q.x(), q.y(), q.z(), q.w());
                std::fclose(pf);
                std::rename("/dev/shm/slam_pose.txt.tmp", "/dev/shm/slam_pose.txt");
            }
        }
        if (slam) {   // path for the mini-map (restarts when SLAM starts a new map after losing track)
            const bool ok = slam->GetTrackingState() == ORB_SLAM3::Tracking::OK;
            ORB_SLAM3::Map* map = atlas->GetCurrentMap();
            const long id = map ? (long)map->GetId() : -1;
            if (id != path_map) { path.clear(); path_map = id; }
            const Sophus::SE3f Twc = Tcw.inverse();
            const cv::Point2f p(Twc.translation().x(), Twc.translation().z());
            if (ok && (path.empty() || cv::norm(p - path.back()) > 0.005)) path.push_back(p);
            if (std::chrono::duration<double>(clk::now() - t_mini).count() >= 1.0) {
                t_mini = clk::now();
                g_mini = RenderMinimap(map, path, Twc, ok);
            }
        }

        if (++frame_no % 2 == 0 || tag_only) {   // tag check every 2nd frame (every frame without SLAM)
            tag = FindDockTag(gray, tag_det, K, D);
            FILE* tf = std::fopen((std::string(TAG_FILE) + ".tmp").c_str(), "w");
            if (tf) {
                const double now = std::chrono::duration<double>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                if (tag.seen && tag.has_alt)
                    std::fprintf(tf, "%.3f %d %.1f %.1f %.1f %.1f %.1f %.1f %.1f "
                                 "%.2f %.2f %.2f %.4f %.2f %.2f %.2f %.4f\n", now, DOCK_TAG_ID,
                                 tag.dist_cm, tag.x_cm, tag.bearing_deg, tag.yaw_deg,
                                 tag.lateral_cm, tag.normal_cm, tag.heading_deg,
                                 tag.alt[0][0], tag.alt[0][1], tag.alt[0][2], tag.alt[0][3],
                                 tag.alt[1][0], tag.alt[1][1], tag.alt[1][2], tag.alt[1][3]);
                else if (tag.seen)
                    std::fprintf(tf, "%.3f %d %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n", now, DOCK_TAG_ID,
                                 tag.dist_cm, tag.x_cm, tag.bearing_deg, tag.yaw_deg,
                                 tag.lateral_cm, tag.normal_cm, tag.heading_deg);
                else
                    std::fprintf(tf, "%.3f none\n", now);
                std::fclose(tf);
                std::rename((std::string(TAG_FILE) + ".tmp").c_str(), TAG_FILE);
            }
        }

        // clean (unannotated) frame: ~10x a second for dock-view matching; every frame with SLAM on
        // (map_recorder.py saves them for the offline map, and live SLAM runs well under 30 fps)
        if (slam || frame_no % 3 == 0) {
            cv::imwrite("/dev/shm/cam.tmp.jpg", frame, {cv::IMWRITE_JPEG_QUALITY, 90});
            std::rename("/dev/shm/cam.tmp.jpg", "/dev/shm/cam.jpg");
        }

        if (!view.empty() && std::chrono::duration<double>(clk::now() - t_view).count() >= 0.1) {
            t_view = clk::now();
            WriteView(view, frame, slam.get(), last_state, last_fps, last_points, Tcw.inverse().translation(), tag, K);
        }

        if (std::chrono::duration<double>(clk::now() - t_report).count() >= 1.0) {
            const int state = slam ? slam->GetTrackingState() : -1;
            int points = 0;
            if (slam)
                for (auto* p : slam->GetTrackedMapPoints()) points += (p != nullptr);
            const Eigen::Vector3f c = Tcw.inverse().translation();   // camera position in map
            last_state = StateName(state); last_points = points;
            last_fps = frames / std::chrono::duration<double>(clk::now() - t_report).count();
            std::printf("t %5.1f  fps %4.1f  track %5.1f ms (max %5.1f)  %-13s pts %4d  pos %+.2f %+.2f %+.2f\n",
                        ts, frames / std::chrono::duration<double>(clk::now() - t_report).count(),
                        track_sum / frames, track_max, StateName(state), points, c.x(), c.y(), c.z());
            std::fflush(stdout);
            t_report = clk::now(); frames = 0; track_sum = 0; track_max = 0;
        }
    }
    if (slam) slam->Shutdown();
    return 0;
}
