// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <iomanip>
#include <csignal>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <deque>
#include <limits>
#include <Eigen/Geometry>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver2/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
int lidar_type;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr pipe_debug_endcap_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

// Robot pose output with the first valid algorithm pose as origin.
bool robot_pose_origin_enable = true;
bool robot_pose_origin_publish_tf = true;
bool robot_pose_origin_path_enable = true;
int  robot_pose_origin_path_stride = 10;
std::string robot_pose_origin_frame = "robot_origin";
std::string robot_pose_origin_child_frame = "body_in_robot_origin";
M3D robot_origin_R_w = Eye3d;
V3D robot_origin_t_w = Zero3d;
double robot_origin_time = 0.0;
bool robot_origin_inited = false;
nav_msgs::Odometry robotOdomOrigin;
nav_msgs::Path robotPathOrigin;
geometry_msgs::PoseStamped robotPoseOriginStamped;
ofstream robot_pose_csv_log;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

struct RectPipeGeomState
{
    bool valid = false;
    bool axis_reliable = false;
    bool box_reliable = false;
    V3D axis_w = V3D(1.0, 0.0, 0.0);
    V3D u_w = V3D(0.0, 1.0, 0.0);
    V3D v_w = V3D(0.0, 0.0, 1.0);
    V3D axis_point_w = Zero3d;
    V3D centroid_w = Zero3d;
    double t_min = 0.0;
    double t_max = 0.0;
    double u_min = 0.0;
    double u_max = 0.0;
    double v_min = 0.0;
    double v_max = 0.0;
    double length = 0.0;
    double width = 0.0;
    double height = 0.0;
    double global_length = 0.0;
    int u_pos_count = 0;
    int u_neg_count = 0;
    int v_pos_count = 0;
    int v_neg_count = 0;
    bool four_wall_fit_ok = false;
    double normal_planarity = 0.0;
    double straight_confidence = 0.0;
};

struct PipeEndCapState
{
    bool visible = false;
    bool reliable = false;
    bool matched_anchor = false;
    int side = 0;              // -1: low-t slice, +1: high-t slice
    int anchor_id = 0;         // 0: none, 1: entrance, 2: exit
    V3D normal_w = V3D(1.0, 0.0, 0.0);
    V3D center_w = Zero3d;
    double plane_d = 0.0;
    double axis_cos = 0.0;
    double mean_abs_res = 0.0;
    double span_u = 0.0;
    double span_v = 0.0;
    double s_meas = 0.0;
    double s_anchor = 0.0;
    int point_count = 0;
    vector<V3D> points_body;
};


RectPipeGeomState g_pipe_geom;
bool   pipe_prior_enable = true;
bool   pipe_debug_log = false;
bool   pipe_degenerate = false;
bool   pipe_geometry_prior_enable = false;
bool   pipe_endcap_require_lio_healthy = true;
int    pipe_endcap_min_lio_eff = 30;
bool   pipe_endcap_require_2d_span = true;
double pipe_exit_max_forward_dist = 1.50;
int    pipe_map_filter_min_eff = 20;
double pipe_prior_weight = 8.0;
double pipe_axis_align_weight = 3.0;
double pipe_motion_weight = 1.5;
double pipe_min_width = 0.06;
double pipe_max_width = 1.00;
double pipe_min_height = 0.06;
double pipe_max_height = 1.00;
double pipe_axis_conf_threshold = 0.35;
double pipe_degenerate_min_eig = 1e-4;
double pipe_degenerate_ratio = 0.20;
double pipe_last_min_eig = 0.0;
double pipe_last_cond_num = 0.0;
double pipe_last_axial_info = 0.0;
double pipe_last_lateral_info = 0.0;
double pipe_last_axial_ratio = 1.0;
double pipe_last_min_pos_rel = 1.0;
double pipe_last_u_info = 0.0;
double pipe_last_v_info = 0.0;
double pipe_degenerate_cond_thresh = 150.0;
double pipe_degenerate_min_pos_rel = 0.10;
int    pipe_degenerate_hold_frames = 3;
bool   pipe_last_deg_by_abs_eig = false;
bool   pipe_last_deg_by_rel_eig = false;
bool   pipe_last_deg_by_axial_ratio = false;
bool   pipe_last_deg_by_cond = false;
V3D    prev_lidar_pos_world = Zero3d;
bool   prev_lidar_pos_valid = false;
V3D    prev_pipe_u_w = V3D(0.0, 1.0, 0.0);
V3D    prev_pipe_v_w = V3D(0.0, 0.0, 1.0);
bool   prev_pipe_basis_valid = false;
double pipe_mid_section_keep_ratio = 0.60;
double pipe_plane_score_gate = 0.70;
double pipe_plane_abs_pd2_gate = 0.05;
bool   pipe_map_filter_by_residual = true;
bool   pipe_map_use_selected_points = true;
double pipe_map_max_pd2 = 0.04;
double pipe_min_incidence_cos = 0.20;
bool   pipe_apply_incidence_filter_in_lio = true;
bool   pipe_debug_match_stats = false;
bool   pipe_enable_intensity_trim = true;
double pipe_intensity_quantile_low = 0.05;
double pipe_intensity_quantile_high = 0.95;
bool   pipe_debug_print_filter_stats = false;
bool   pipe_last_geom_used_intensity_trim = false;
int    pipe_last_geom_raw_count = 0;
int    pipe_last_geom_filtered_count = 0;
bool   pipe_global_axis_initialized = false;
double pipe_global_t_min = std::numeric_limits<double>::infinity();
double pipe_global_t_max = -std::numeric_limits<double>::infinity();
double pipe_global_length = 0.0;
V3D    pipe_global_axis_w = V3D(1.0, 0.0, 0.0);

PipeEndCapState g_endcap;
bool   pipe_endcap_enable = true;
bool   pipe_endcap_init_origin = false;
bool   pipe_endcap_use_prior = true;
bool   pipe_endcap_learn_exit = true;
double pipe_endcap_weight = 0.2;
int    pipe_endcap_min_points = 25;
int    pipe_endcap_max_prior_points = 80;
double pipe_endcap_axis_cos_thresh = 0.85;
double pipe_endcap_max_plane_res = 0.03;
double pipe_endcap_slice_ratio = 0.18;
double pipe_endcap_slice_min = 0.06;
double pipe_endcap_min_cross_span = 0.08;
double pipe_endcap_anchor_gate = 0.35;
double pipe_endcap_exit_learn_min_s = 0.80;
double pipe_endcap_known_length = -1.0;
bool   pipe_endcap_fallback_enable = false;
double pipe_endcap_fallback_gate = 0.60;
int    pipe_endcap_entry_side = -1;  // legacy only, ignored when exit-only mode is enabled

// Exit-only mode. Entrance end-cap is ignored to avoid false positives from the rear body of a dual-body robot.
bool   pipe_exit_only_mode = true;
bool   pipe_exit_only_front_side = true;
double pipe_exit_min_forward_dist = 0.20;
int    pipe_exit_lock_min_frames = 5;
double pipe_exit_lock_center_gate = 0.10;
double pipe_exit_lock_normal_cos = 0.95;
bool   pipe_exit_prior_requires_lock = true;
bool   pipe_exit_fallback_requires_lock = true;
int    pipe_exit_candidate_frames = 0;
V3D    pipe_exit_candidate_center_w = Zero3d;
V3D    pipe_exit_candidate_normal_w = V3D(1.0, 0.0, 0.0);
bool   pipe_exit_candidate_valid = false;
double pipe_last_distance_to_exit = 0.0;
bool   pipe_last_pipe_s_valid = false;

// After the outlet landmark is locked, the current frame does not need to
// redetect a complete outlet plane every time.  Instead, points that lie close
// to the fixed outlet plane can be collected and used as weak EKF constraints.
bool   pipe_exit_locked_plane_prior_enable = true;
double pipe_exit_prior_point_gate = 0.05;
int    pipe_exit_prior_min_points = 6;
int    pipe_last_exit_locked_plane_points = 0;

// LiDAR-frame self body mask. Points inside this box are considered robot body points
// and are excluded from entrance/exit end-cap detection.
bool   pipe_self_mask_enable = false;
double pipe_self_mask_x_min = -0.70;
double pipe_self_mask_x_max = -0.05;
double pipe_self_mask_y_min = -0.18;
double pipe_self_mask_y_max =  0.18;
double pipe_self_mask_z_min = -0.15;
double pipe_self_mask_z_max =  0.15;
int    pipe_last_self_mask_reject_count = 0;
int    pipe_last_endcap_candidate_count = 0;
int    pipe_last_endcap_prior_rows = 0;
int    pipe_last_endcap_fallback_rows = 0;
bool   pipe_last_endcap_fallback_used = false;
bool   pipe_last_endcap_matched = false;
int    pipe_last_endcap_unmatched_reject = 0;
int    pipe_last_endcap_lock_reject = 0;
int    pipe_last_endcap_gate_reject = 0;
int    pipe_last_endcap_invalid_reject = 0;
int    pipe_last_endcap_reject_code = 0;  // 0 ok, 1 invalid, 2 unmatched, 3 bad weight/cols, 4 wrong anchor, 5 lock missing, 6 no rows after gate
bool   pipe_debug_publish_endcap_points = true;
bool   pipe_origin_initialized = false;
bool   pipe_start_cap_initialized = false;
bool   pipe_exit_cap_initialized = false;
V3D    pipe_origin_w = Zero3d;
V3D    pipe_axis_anchor_w = V3D(1.0, 0.0, 0.0);
V3D    pipe_start_cap_center_w = Zero3d;
V3D    pipe_exit_cap_center_w = Zero3d;
double pipe_start_cap_d = 0.0;
double pipe_exit_cap_d = 0.0;
double pipe_last_pipe_s = 0.0;

// Axial position output for known and unknown pipe length.
// The odom quantity is the LiDAR center axial displacement from the initial frame.
// The scan quantity is the current LiDAR-to-exit distance measured directly in the LiDAR frame.
int    pipe_position_output_mode = 2;  // 0: legacy locked-exit world anchor, 1: scan distance when known length, 2: odom + online length estimate
double pipe_initial_lidar_s = 0.0;
bool   pipe_scan_exit_distance_enable = true;
double pipe_position_axis_lidar_x = 1.0;
double pipe_position_axis_lidar_y = 0.0;
double pipe_position_axis_lidar_z = 0.0;
bool   pipe_axis_odom_initialized = false;
V3D    pipe_lidar_pos0_w = Zero3d;
V3D    pipe_axis0_w = V3D(1.0, 0.0, 0.0);
bool   pipe_last_s_odom_valid = false;
double pipe_last_s_odom = 0.0;
bool   pipe_last_d_exit_scan_valid = false;
double pipe_last_d_exit_scan = 0.0;
double pipe_last_scan_plane_res = 0.0;
bool   pipe_length_estimate_enable = true;
bool   pipe_length_est_valid = false;
double pipe_length_est = 0.0;
double pipe_last_length_meas = 0.0;
int    pipe_length_est_samples = 0;
double pipe_length_update_alpha = 0.05;
double pipe_length_update_gate = 0.20;
double pipe_length_max = 20.0;
double pipe_last_pipe_length_output = 0.0;

// Stable pipe size estimation.
// Entrance/exit end-cap points can pollute single-frame W/H in short sealed pipes.
bool   pipe_size_stabilize_enable = true;
bool   pipe_size_exclude_endcap_points = false;
double pipe_size_endcap_exclusion = 0.15;
double pipe_size_update_s_min_ratio = 0.20;
double pipe_size_update_s_max_ratio = 0.70;
int    pipe_size_window = 10;
int    pipe_size_min_stable_samples = 5;
double pipe_size_jump_gate = 0.08;
std::string pipe_size_average_mode = "median"; // mean, median, trimmed_mean
double pipe_size_trim_ratio = 0.10;
bool   pipe_size_require_anchor = true;
bool   pipe_size_use_gravity_height = true;
bool   pipe_size_wall_history_enable = true;
int    pipe_size_wall_min_side_samples = 5;
int    pipe_size_min_axis_points = 20;
double pipe_size_wall_center_q_low = 0.15;
double pipe_size_wall_center_q_high = 0.85;
double pipe_size_span_q_low = 0.03;
double pipe_size_span_q_high = 0.97;
double pipe_size_wall_min_normal_cos = 0.35;
bool   pipe_size_accept_partial_pairs = true;
// If false, size samples can update the stable estimator even when the current
// frame is not reliable enough for a full pipe-axis observation. This avoids
// blocking valid W/H samples in degenerate straight-pipe frames.
bool   pipe_size_update_requires_axis = false;
// If false, valid size samples are not blocked by the axial update-zone gate.
// End-cap exclusion still removes points close to the outlet/entry face.
bool   pipe_size_update_zone_enable = false;
// Conservative size quantification: no full-span fallback. Width updates only
// from left/right wall pairs; height updates only from top/bottom wall pairs.
bool   pipe_size_accept_span_fallback = false;
// Publish policy for /pipe_size. When false, /pipe_size reports the current-frame
// raw W/H observation and does not reuse stable dimensions. Stable W/H are still
// maintained internally and written to CSV for debugging.
bool   pipe_size_publish_stable = false;
bool   pipe_size_publish_fallback_to_stable = false;
bool   pipe_width_require_both_side_walls = true;
bool   pipe_height_require_both_top_bottom_walls = true;
bool   pipe_width_hold_last_valid = true;
bool   pipe_height_hold_last_valid = true;
double pipe_last_raw_width = 0.0;
double pipe_last_raw_height = 0.0;
double pipe_last_anchor_length = -1.0;
bool   pipe_last_size_update_zone = false;
bool   pipe_last_size_sample_accepted = false;
bool   pipe_last_size_endcap_exclusion_used = false;
bool   pipe_stable_size_valid = false;
bool   pipe_stable_width_valid = false;
bool   pipe_stable_height_valid = false;
double pipe_stable_width = 0.0;
double pipe_stable_height = 0.0;
std::deque<double> pipe_width_hist;
std::deque<double> pipe_height_hist;
int pipe_last_width_sample_count = 0;
int pipe_last_height_sample_count = 0;
int pipe_last_width_hist_n = 0;
int pipe_last_height_hist_n = 0;
bool pipe_last_width_sample_accepted = false;
bool pipe_last_height_sample_accepted = false;
bool pipe_last_width_hold = false;
bool pipe_last_height_hold = false;
double pipe_last_width_sample = 0.0;
double pipe_last_height_sample = 0.0;

// Current-frame geometry observation for logging and size debugging.
// g_pipe_geom is allowed to keep the last usable axis for end-cap detection,
// so it must not be used as the per-frame measurement in CSV/log output.
bool   pipe_last_geom_updated = false;
int    pipe_last_geom_fail_code = 0;
bool   pipe_last_frame_valid = false;
bool   pipe_last_frame_axis_reliable = false;
bool   pipe_last_frame_box_reliable = false;
double pipe_last_frame_length = 0.0;
double pipe_last_frame_width = 0.0;
double pipe_last_frame_height = 0.0;
double pipe_last_frame_global_length = 0.0;
double pipe_last_frame_conf = 0.0;
int    pipe_last_frame_u_pos_count = 0;
int    pipe_last_frame_u_neg_count = 0;
int    pipe_last_frame_v_pos_count = 0;
int    pipe_last_frame_v_neg_count = 0;
bool   pipe_last_frame_four_wall_fit_ok = false;

// Simplified public pipe-size output. Only current-frame width and height
// after configured constraints are exposed. Other size variants remain internal.
bool   pipe_last_constrained_size_valid = false;
double pipe_last_constrained_width = 0.0;
double pipe_last_constrained_height = 0.0;
int    pipe_last_constrained_width_pair_count = 0;
int    pipe_last_constrained_height_pair_count = 0;

ofstream pipe_csv_log;

static void ensure_log_directory_exists(const string &root_dir)
{
    string log_dir = root_dir + "/Log";
    struct stat st;
    if (stat(log_dir.c_str(), &st) != 0)
    {
        if (mkdir(log_dir.c_str(), 0755) != 0)
        {
            ROS_WARN_STREAM("[LOG] failed to create log directory: " << log_dir);
        }
    }
}

void init_pipe_csv_log(const string &csv_path)
{
    if (pipe_csv_log.is_open()) return;

    pipe_csv_log.open(csv_path.c_str(), ios::out | ios::trunc);
    if (!pipe_csv_log.is_open())
    {
        ROS_WARN("failed to open pipe csv log: %s", csv_path.c_str());
        return;
    }

    pipe_csv_log << "stamp_abs,stamp_rel,size_valid,width,height,width_pair_count,height_pair_count,axis_reliable,geom_updated,geom_fail_code,pipe_s,s_odom,d_exit_scan,scan_valid,length_est,length_valid,robot_pose_valid,robot_x,robot_y,robot_z,robot_qx,robot_qy,robot_qz,robot_qw,robot_roll,robot_pitch,robot_yaw,robot_vx,robot_vy,robot_vz\n";
    pipe_csv_log.flush();
}

inline void append_pipe_csv_log(double stamp_abs, double stamp_rel)
{
    if (!pipe_csv_log.is_open()) return;

    bool robot_pose_valid = robot_pose_origin_enable && robot_origin_inited;
    double robot_roll = 0.0, robot_pitch = 0.0, robot_yaw = 0.0;
    if (robot_pose_valid)
    {
        Eigen::Quaterniond q_pipe(robotOdomOrigin.pose.pose.orientation.w,
                                  robotOdomOrigin.pose.pose.orientation.x,
                                  robotOdomOrigin.pose.pose.orientation.y,
                                  robotOdomOrigin.pose.pose.orientation.z);
        q_pipe.normalize();
        V3D eul_pipe = q_pipe.toRotationMatrix().eulerAngles(0, 1, 2);
        robot_roll = eul_pipe(0);
        robot_pitch = eul_pipe(1);
        robot_yaw = eul_pipe(2);
    }

    pipe_csv_log << fixed << setprecision(9)
                 << stamp_abs << ","
                 << stamp_rel << ","
                 << int(pipe_last_constrained_size_valid) << ","
                 << pipe_last_constrained_width << ","
                 << pipe_last_constrained_height << ","
                 << pipe_last_constrained_width_pair_count << ","
                 << pipe_last_constrained_height_pair_count << ","
                 << int(pipe_last_frame_axis_reliable) << ","
                 << int(pipe_last_geom_updated) << ","
                 << pipe_last_geom_fail_code << ","
                 << pipe_last_pipe_s << ","
                 << pipe_last_s_odom << ","
                 << pipe_last_d_exit_scan << ","
                 << int(pipe_last_d_exit_scan_valid) << ","
                 << pipe_length_est << ","
                 << int(pipe_length_est_valid) << ","
                 << int(robot_pose_valid) << ","
                 << (robot_pose_valid ? robotOdomOrigin.pose.pose.position.x : 0.0) << ","
                 << (robot_pose_valid ? robotOdomOrigin.pose.pose.position.y : 0.0) << ","
                 << (robot_pose_valid ? robotOdomOrigin.pose.pose.position.z : 0.0) << ","
                 << (robot_pose_valid ? robotOdomOrigin.pose.pose.orientation.x : 0.0) << ","
                 << (robot_pose_valid ? robotOdomOrigin.pose.pose.orientation.y : 0.0) << ","
                 << (robot_pose_valid ? robotOdomOrigin.pose.pose.orientation.z : 0.0) << ","
                 << (robot_pose_valid ? robotOdomOrigin.pose.pose.orientation.w : 1.0) << ","
                 << robot_roll << ","
                 << robot_pitch << ","
                 << robot_yaw << ","
                 << (robot_pose_valid ? robotOdomOrigin.twist.twist.linear.x : 0.0) << ","
                 << (robot_pose_valid ? robotOdomOrigin.twist.twist.linear.y : 0.0) << ","
                 << (robot_pose_valid ? robotOdomOrigin.twist.twist.linear.z : 0.0) << "\n";
    pipe_csv_log.flush();
}

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();


        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }
        if(lidar_type == MARSIM)
            lidar_end_time = meas.lidar_beg_time;

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    int map_skip_unselected = 0;
    int map_skip_residual = 0;
    int map_candidates = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

        /*
         * Pipe mode map protection:
         * after EKF is initialized, do not insert every downsampled point into ikdtree.
         * Only insert points that were actually accepted by the point-to-plane matcher
         * and whose final absolute plane residual is small enough. This prevents
         * a loose score gate from polluting the local map with thick walls or ghost points.
         */
        // Keep the original FAST-LIO map update alive when the current frame has
        // too few accepted point-to-plane rows. If we keep filtering the map by the
        // previous matcher result after effct_feat_num collapses, the ikd-tree can
        // stop growing and the system cannot recover.
        bool lio_match_healthy_for_map = (effct_feat_num >= std::max(1, pipe_map_filter_min_eff));
        if (pipe_prior_enable && pipe_map_filter_by_residual && flg_EKF_inited && lio_match_healthy_for_map)
        {
            if (pipe_map_use_selected_points && !point_selected_surf[i])
            {
                map_skip_unselected++;
                continue;
            }
            if (!(res_last[i] >= 0.0f && res_last[i] <= static_cast<float>(pipe_map_max_pd2)))
            {
                map_skip_residual++;
                continue;
            }
        }
        map_candidates++;

        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    if ((pipe_debug_match_stats || pipe_debug_log) && pipe_prior_enable && pipe_map_filter_by_residual && flg_EKF_inited)
    {
        ROS_WARN_STREAM_THROTTLE(1.0, "[DBG_MAP_FILTER] feats_down_size=" << feats_down_size
                                 << " eff=" << effct_feat_num
                                 << " healthy=" << int(effct_feat_num >= std::max(1, pipe_map_filter_min_eff))
                                 << " candidates=" << map_candidates
                                 << " skip_unselected=" << map_skip_unselected
                                 << " skip_residual=" << map_skip_residual
                                 << " map_max_pd2=" << pipe_map_max_pd2
                                 << " add_voxel=" << PointToAdd.size()
                                 << " add_direct=" << PointNoNeedDownsample.size());
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    auto P = kf.get_P();
    for (int i = 0; i < 36; ++i) odomAftMapped.pose.covariance[i] = 0.0;
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }
    odomAftMapped.twist.twist.linear.x = state_point.vel(0);
    odomAftMapped.twist.twist.linear.y = state_point.vel(1);
    odomAftMapped.twist.twist.linear.z = state_point.vel(2);
    pubOdomAftMapped.publish(odomAftMapped);

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void init_robot_origin_pose_if_needed()
{
    if (robot_origin_inited) return;

    robot_origin_R_w = state_point.rot.toRotationMatrix();
    robot_origin_t_w = state_point.pos;
    robot_origin_time = lidar_end_time;
    robot_origin_inited = true;

    robotPathOrigin.header.frame_id = robot_pose_origin_frame;
    robotPathOrigin.poses.clear();

    ROS_WARN_STREAM("[ROBOT POSE] origin initialized at t=" << robot_origin_time
                    << " pos=[" << robot_origin_t_w(0) << ","
                    << robot_origin_t_w(1) << "," << robot_origin_t_w(2) << "]");
}

bool compute_robot_pose_in_origin(V3D &p_rel, M3D &R_rel, V3D &v_rel)
{
    if (!robot_pose_origin_enable) return false;
    init_robot_origin_pose_if_needed();
    if (!robot_origin_inited) return false;

    M3D R0_inv = robot_origin_R_w.transpose();
    p_rel = R0_inv * (state_point.pos - robot_origin_t_w);
    R_rel = R0_inv * state_point.rot.toRotationMatrix();
    v_rel = R0_inv * state_point.vel;
    return true;
}

bool fill_robot_origin_odometry(nav_msgs::Odometry &odom)
{
    V3D p_rel = Zero3d, v_rel = Zero3d;
    M3D R_rel = Eye3d;
    if (!compute_robot_pose_in_origin(p_rel, R_rel, v_rel)) return false;

    Eigen::Quaterniond q_rel(R_rel);
    q_rel.normalize();

    odom.header.frame_id = robot_pose_origin_frame;
    odom.child_frame_id = robot_pose_origin_child_frame;
    odom.header.stamp = ros::Time().fromSec(lidar_end_time);

    odom.pose.pose.position.x = p_rel(0);
    odom.pose.pose.position.y = p_rel(1);
    odom.pose.pose.position.z = p_rel(2);
    odom.pose.pose.orientation.x = q_rel.x();
    odom.pose.pose.orientation.y = q_rel.y();
    odom.pose.pose.orientation.z = q_rel.z();
    odom.pose.pose.orientation.w = q_rel.w();

    for (int i = 0; i < 36; ++i) odom.pose.covariance[i] = odomAftMapped.pose.covariance[i];
    for (int i = 0; i < 36; ++i) odom.twist.covariance[i] = 0.0;

    odom.twist.twist.linear.x = v_rel(0);
    odom.twist.twist.linear.y = v_rel(1);
    odom.twist.twist.linear.z = v_rel(2);
    return true;
}

void append_robot_pose_origin_csv(const nav_msgs::Odometry &odom)
{
    if (!robot_pose_csv_log.is_open()) return;

    Eigen::Quaterniond q_csv(odom.pose.pose.orientation.w,
                             odom.pose.pose.orientation.x,
                             odom.pose.pose.orientation.y,
                             odom.pose.pose.orientation.z);
    q_csv.normalize();
    V3D eul_rel = q_csv.toRotationMatrix().eulerAngles(0, 1, 2);
    robot_pose_csv_log << std::fixed << std::setprecision(9)
                       << lidar_end_time << ","
                       << (lidar_end_time - robot_origin_time) << ","
                       << odom.pose.pose.position.x << ","
                       << odom.pose.pose.position.y << ","
                       << odom.pose.pose.position.z << ","
                       << odom.pose.pose.orientation.x << ","
                       << odom.pose.pose.orientation.y << ","
                       << odom.pose.pose.orientation.z << ","
                       << odom.pose.pose.orientation.w << ","
                       << eul_rel(0) << ","
                       << eul_rel(1) << ","
                       << eul_rel(2) << ","
                       << odom.twist.twist.linear.x << ","
                       << odom.twist.twist.linear.y << ","
                       << odom.twist.twist.linear.z << "\n";
    robot_pose_csv_log.flush();
}

void publish_robot_pose_origin(const ros::Publisher &pubRobotOdomOrigin,
                               const ros::Publisher &pubRobotPathOrigin)
{
    if (!robot_pose_origin_enable) return;

    if (!fill_robot_origin_odometry(robotOdomOrigin)) return;
    pubRobotOdomOrigin.publish(robotOdomOrigin);
    ROS_INFO_STREAM_THROTTLE(1.0, "[ROBOT POSE] origin pose x="
                             << robotOdomOrigin.pose.pose.position.x
                             << " y=" << robotOdomOrigin.pose.pose.position.y
                             << " z=" << robotOdomOrigin.pose.pose.position.z);

    if (robot_pose_origin_publish_tf)
    {
        static tf::TransformBroadcaster br_origin;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(robotOdomOrigin.pose.pose.position.x,
                                        robotOdomOrigin.pose.pose.position.y,
                                        robotOdomOrigin.pose.pose.position.z));
        q.setX(robotOdomOrigin.pose.pose.orientation.x);
        q.setY(robotOdomOrigin.pose.pose.orientation.y);
        q.setZ(robotOdomOrigin.pose.pose.orientation.z);
        q.setW(robotOdomOrigin.pose.pose.orientation.w);
        transform.setRotation(q);
        br_origin.sendTransform(tf::StampedTransform(transform,
                                robotOdomOrigin.header.stamp,
                                robot_pose_origin_frame,
                                robot_pose_origin_child_frame));
    }

    if (robot_pose_origin_path_enable)
    {
        static int path_counter = 0;
        path_counter++;
        int stride = std::max(1, robot_pose_origin_path_stride);
        if (path_counter % stride == 0)
        {
            robotPoseOriginStamped.header = robotOdomOrigin.header;
            robotPoseOriginStamped.pose = robotOdomOrigin.pose.pose;
            robotPathOrigin.header = robotOdomOrigin.header;
            robotPathOrigin.poses.push_back(robotPoseOriginStamped);
            pubRobotPathOrigin.publish(robotPathOrigin);
        }
    }

    append_robot_pose_origin_csv(robotOdomOrigin);
}


inline V3D pointBodyToWorldVec(const V3D &pi, const state_ikfom &s)
{
    return s.rot * (s.offset_R_L_I * pi + s.offset_T_L_I) + s.pos;
}

inline M3D skewMatrix(const V3D &v)
{
    M3D m;
    m << SKEW_SYM_MATRX(v);
    return m;
}

double quantile_from_sorted(vector<double> vec, double q)
{
    if (vec.empty()) return 0.0;
    std::sort(vec.begin(), vec.end());
    double idx = q * double(vec.size() - 1);
    int idx0 = int(std::floor(idx));
    int idx1 = std::min(idx0 + 1, int(vec.size() - 1));
    double t = idx - idx0;
    return vec[idx0] * (1.0 - t) + vec[idx1] * t;
}

double robust_trimmed_center(vector<double> vec, double q_low = 0.2, double q_high = 0.8)
{
    if (vec.empty()) return 0.0;
    std::sort(vec.begin(), vec.end());
    double low = quantile_from_sorted(vec, q_low);
    double high = quantile_from_sorted(vec, q_high);
    double sum = 0.0;
    int count = 0;
    for (double v : vec)
    {
        if (v >= low && v <= high)
        {
            sum += v;
            ++count;
        }
    }
    if (count <= 0) return quantile_from_sorted(vec, 0.5);
    return sum / double(count);
}


double pipe_effective_pipe_length()
{
    if (pipe_endcap_known_length > 0.0)
        return pipe_endcap_known_length;
    if (pipe_length_est_valid && pipe_length_est > 1e-6)
        return pipe_length_est;
    if (!pipe_exit_only_mode && pipe_origin_initialized && pipe_exit_cap_initialized)
        return (pipe_exit_cap_center_w - pipe_origin_w).dot(pipe_axis_anchor_w);
    return -1.0;
}

double pipe_anchor_length_for_size()
{
    double len = pipe_effective_pipe_length();
    pipe_last_anchor_length = len;
    return len;
}

bool pipe_point_near_endcap_for_size(const V3D &p_world)
{
    if (!pipe_size_exclude_endcap_points) return false;
    double margin = std::max(0.0, pipe_size_endcap_exclusion);
    if (margin <= 1e-6) return false;

    // Exit-only mode does not require a known length. Once the outlet plane is
    // locked, just remove points close to that plane so the end-cap does not
    // pollute W/H estimation.
    if (pipe_exit_only_mode)
    {
        if (!pipe_exit_cap_initialized) return false;
        V3D axis = pipe_axis_anchor_w;
        if (axis.norm() < 1e-6) return false;
        axis.normalize();
        double dist_to_exit_plane = std::fabs((p_world - pipe_exit_cap_center_w).dot(axis));
        return dist_to_exit_plane < margin;
    }

    if (!pipe_origin_initialized) return false;
    double s_val = (p_world - pipe_origin_w).dot(pipe_axis_anchor_w);
    if (s_val < margin) return true;

    double len = pipe_anchor_length_for_size();
    if (len > 0.0 && s_val > len - margin) return true;
    return false;
}

bool pipe_size_update_zone_ok()
{
    if (!pipe_size_stabilize_enable) return true;
    if (!pipe_size_update_zone_enable)
    {
        pipe_last_size_update_zone = true;
        return true;
    }

    pipe_last_size_update_zone = false;
    double len = pipe_anchor_length_for_size();

    if (len <= 1e-6)
    {
        pipe_last_size_update_zone = !pipe_size_require_anchor;
        return pipe_last_size_update_zone;
    }

    double s_for_zone = pipe_last_pipe_s;
    if (pipe_last_s_odom_valid)
        s_for_zone = pipe_last_s_odom;

    bool s_valid = pipe_last_pipe_s_valid || pipe_last_s_odom_valid;
    if (!s_valid)
    {
        pipe_last_size_update_zone = !pipe_size_require_anchor;
        return pipe_last_size_update_zone;
    }

    double ratio = s_for_zone / len;
    double r_min = std::max(0.0, std::min(1.0, pipe_size_update_s_min_ratio));
    double r_max = std::max(0.0, std::min(1.0, pipe_size_update_s_max_ratio));
    if (r_max < r_min) std::swap(r_min, r_max);
    pipe_last_size_update_zone = (ratio >= r_min && ratio <= r_max);
    return pipe_last_size_update_zone;
}

double pipe_median_from_deque(const std::deque<double> &dq)
{
    if (dq.empty()) return 0.0;
    vector<double> tmp(dq.begin(), dq.end());
    return quantile_from_sorted(tmp, 0.5);
}

double pipe_mean_from_deque(const std::deque<double> &dq)
{
    if (dq.empty()) return 0.0;
    double sum = 0.0;
    for (double v : dq) sum += v;
    return sum / double(dq.size());
}

double pipe_trimmed_mean_from_deque(const std::deque<double> &dq, double trim_ratio)
{
    if (dq.empty()) return 0.0;
    vector<double> tmp(dq.begin(), dq.end());
    std::sort(tmp.begin(), tmp.end());

    double r = std::max(0.0, std::min(0.40, trim_ratio));
    int n = int(tmp.size());
    int cut = int(std::floor(r * double(n)));
    int first = std::min(cut, std::max(0, n - 1));
    int last = std::max(first + 1, n - cut);

    double sum = 0.0;
    int cnt = 0;
    for (int i = first; i < last; ++i)
    {
        sum += tmp[i];
        cnt++;
    }
    return cnt > 0 ? sum / double(cnt) : 0.0;
}

double pipe_size_estimator_from_deque(const std::deque<double> &dq)
{
    if (pipe_size_average_mode == "mean") return pipe_mean_from_deque(dq);
    if (pipe_size_average_mode == "median") return pipe_median_from_deque(dq);
    return pipe_trimmed_mean_from_deque(dq, pipe_size_trim_ratio);
}

bool pipe_size_sample_close_to_history(const std::deque<double> &dq, double value, double gate)
{
    if (gate <= 1e-6) return true;
    // Do not reject during warm-up. Otherwise the first few noisy samples can
    // lock the stable estimator and prevent later valid observations from entering.
    if (int(dq.size()) < std::max(1, pipe_size_min_stable_samples)) return true;

    double ref = pipe_size_estimator_from_deque(dq);
    if (!(ref > 0.0)) return true;
    return std::fabs(value - ref) <= gate;
}

void pipe_update_stable_size_valid()
{
    pipe_last_width_hist_n = int(pipe_width_hist.size());
    pipe_last_height_hist_n = int(pipe_height_hist.size());

    pipe_stable_width = pipe_size_estimator_from_deque(pipe_width_hist);
    pipe_stable_height = pipe_size_estimator_from_deque(pipe_height_hist);

    pipe_stable_width_valid = (int(pipe_width_hist.size()) >= std::max(1, pipe_size_min_stable_samples) &&
                               pipe_stable_width > pipe_min_width && pipe_stable_width < pipe_max_width);

    pipe_stable_height_valid = (int(pipe_height_hist.size()) >= std::max(1, pipe_size_min_stable_samples) &&
                                pipe_stable_height > pipe_min_height && pipe_stable_height < pipe_max_height);

    pipe_stable_size_valid = pipe_stable_width_valid && pipe_stable_height_valid;
}

bool pipe_push_stable_width_sample(double width)
{
    if (!(width > pipe_min_width && width < pipe_max_width)) return false;

    // Gate against the current history only after enough samples have accumulated.
    // A non-positive gate disables jump rejection completely.
    if (!pipe_size_sample_close_to_history(pipe_width_hist, width, pipe_size_jump_gate)) return false;

    pipe_width_hist.push_back(width);
    int win = pipe_size_window;
    if (win > 0)
    {
        while (int(pipe_width_hist.size()) > win) pipe_width_hist.pop_front();
    }
    pipe_update_stable_size_valid();
    return true;
}

bool pipe_push_stable_height_sample(double height)
{
    if (!(height > pipe_min_height && height < pipe_max_height)) return false;

    if (!pipe_size_sample_close_to_history(pipe_height_hist, height, pipe_size_jump_gate)) return false;

    pipe_height_hist.push_back(height);
    int win = pipe_size_window;
    if (win > 0)
    {
        while (int(pipe_height_hist.size()) > win) pipe_height_hist.pop_front();
    }
    pipe_update_stable_size_valid();
    return true;
}

void pipe_push_stable_size_sample(double width, double height)
{
    pipe_push_stable_width_sample(width);
    pipe_push_stable_height_sample(height);
}

bool pipe_accept_size_sample(double width, double height, bool candidate_box_ok)
{
    pipe_last_size_sample_accepted = false;
    pipe_last_width_sample_accepted = false;
    pipe_last_height_sample_accepted = false;
    pipe_last_width_hold = false;
    pipe_last_height_hold = false;
    pipe_last_width_sample = width;
    pipe_last_height_sample = height;
    pipe_last_width_hist_n = int(pipe_width_hist.size());
    pipe_last_height_hist_n = int(pipe_height_hist.size());
    if (!pipe_size_stabilize_enable) return candidate_box_ok;
    if (!candidate_box_ok) return false;
    if (!pipe_size_update_zone_ok())
    {
        pipe_last_width_hold = pipe_width_hold_last_valid && pipe_stable_width_valid;
        pipe_last_height_hold = pipe_height_hold_last_valid && pipe_stable_height_valid;
        pipe_last_width_hist_n = int(pipe_width_hist.size());
        pipe_last_height_hist_n = int(pipe_height_hist.size());
        return false;
    }

    pipe_last_width_sample_accepted = pipe_push_stable_width_sample(width);
    pipe_last_height_sample_accepted = pipe_push_stable_height_sample(height);
    pipe_last_size_sample_accepted = pipe_last_width_sample_accepted && pipe_last_height_sample_accepted;
    return pipe_last_size_sample_accepted;
}

bool pipe_accept_size_observations(double width, bool width_ok,
                                   double height, bool height_ok,
                                   bool axis_ok)
{
    pipe_last_size_sample_accepted = false;
    pipe_last_width_sample_accepted = false;
    pipe_last_height_sample_accepted = false;
    pipe_last_width_hold = false;
    pipe_last_height_hold = false;
    pipe_last_width_sample = width;
    pipe_last_height_sample = height;
    pipe_last_width_hist_n = int(pipe_width_hist.size());
    pipe_last_height_hist_n = int(pipe_height_hist.size());

    if (!pipe_size_stabilize_enable) return axis_ok && width_ok && height_ok;
    if (pipe_size_update_requires_axis && !axis_ok) return false;
    if (!width_ok && !height_ok) return false;
    if (!pipe_size_update_zone_ok())
    {
        pipe_last_width_hold = pipe_width_hold_last_valid && pipe_stable_width_valid;
        pipe_last_height_hold = pipe_height_hold_last_valid && pipe_stable_height_valid;
        return false;
    }

    if (width_ok)
        pipe_last_width_sample_accepted = pipe_push_stable_width_sample(width);
    else
        pipe_last_width_hold = pipe_width_hold_last_valid && pipe_stable_width_valid;

    if (height_ok)
        pipe_last_height_sample_accepted = pipe_push_stable_height_sample(height);
    else
        pipe_last_height_hold = pipe_height_hold_last_valid && pipe_stable_height_valid;

    // Missing either wall pair never invalidates the previous stable dimension.
    // It simply does not add a new sample for that dimension.
    pipe_last_size_sample_accepted = pipe_last_width_sample_accepted || pipe_last_height_sample_accepted;
    return pipe_last_size_sample_accepted;
}

void pipe_apply_stable_size_to_geom(RectPipeGeomState &geom)
{
    if (!pipe_size_stabilize_enable) return;

    // Stable dimensions are publishing priors only. They must not turn a bad
    // single-frame rectangle into a reliable EKF geometry observation.
    if (pipe_stable_width_valid)
    {
        geom.width = pipe_stable_width;
        geom.u_min = -0.5 * pipe_stable_width;
        geom.u_max =  0.5 * pipe_stable_width;
    }

    if (pipe_stable_height_valid)
    {
        geom.height = pipe_stable_height;
        geom.v_min = -0.5 * pipe_stable_height;
        geom.v_max =  0.5 * pipe_stable_height;
    }
}

void pipe_reset_frame_geom_observation()
{
    pipe_last_geom_updated = false;
    pipe_last_geom_fail_code = 0;
    pipe_last_frame_valid = false;
    pipe_last_frame_axis_reliable = false;
    pipe_last_frame_box_reliable = false;
    pipe_last_frame_length = 0.0;
    pipe_last_frame_width = 0.0;
    pipe_last_frame_height = 0.0;
    pipe_last_frame_global_length = pipe_global_length;
    pipe_last_frame_conf = 0.0;
    pipe_last_frame_u_pos_count = 0;
    pipe_last_frame_u_neg_count = 0;
    pipe_last_frame_v_pos_count = 0;
    pipe_last_frame_v_neg_count = 0;
    pipe_last_frame_four_wall_fit_ok = false;
    pipe_last_constrained_size_valid = false;
    pipe_last_constrained_width = 0.0;
    pipe_last_constrained_height = 0.0;
    pipe_last_constrained_width_pair_count = 0;
    pipe_last_constrained_height_pair_count = 0;

    pipe_last_raw_width = 0.0;
    pipe_last_raw_height = 0.0;
    pipe_last_size_sample_accepted = false;
    pipe_last_width_sample_accepted = false;
    pipe_last_height_sample_accepted = false;
    pipe_last_width_sample_count = 0;
    pipe_last_height_sample_count = 0;
    pipe_last_width_sample = 0.0;
    pipe_last_height_sample = 0.0;
    pipe_last_width_hold = false;
    pipe_last_height_hold = false;
    pipe_last_size_update_zone = false;
    pipe_last_size_endcap_exclusion_used = false;
    pipe_last_width_hist_n = int(pipe_width_hist.size());
    pipe_last_height_hist_n = int(pipe_height_hist.size());
}

bool fit_rect_pipe_geometry(const state_ikfom &s, RectPipeGeomState &geom)
{
    geom.valid = false;
    geom.axis_reliable = false;
    geom.box_reliable = false;
    geom.length = 0.0;
    geom.width = 0.0;
    geom.height = 0.0;
    geom.global_length = pipe_global_length;
    geom.u_pos_count = 0;
    geom.u_neg_count = 0;
    geom.v_pos_count = 0;
    geom.v_neg_count = 0;
    geom.four_wall_fit_ok = false;
    pipe_last_constrained_size_valid = false;
    pipe_last_constrained_width = 0.0;
    pipe_last_constrained_height = 0.0;
    pipe_last_constrained_width_pair_count = 0;
    pipe_last_constrained_height_pair_count = 0;
    pipe_last_raw_width = 0.0;
    pipe_last_raw_height = 0.0;
    pipe_last_size_sample_accepted = false;
    pipe_last_width_sample_accepted = false;
    pipe_last_height_sample_accepted = false;
    pipe_last_width_sample_count = 0;
    pipe_last_height_sample_count = 0;
    pipe_last_size_update_zone = false;
    pipe_last_size_endcap_exclusion_used = false;
    pipe_anchor_length_for_size();
    if (effct_feat_num < 20) { pipe_last_geom_fail_code = 1; return false; }

    struct GeomCandidate
    {
        V3D p_world = Zero3d;
        V3D normal_w = Zero3d;
        double intensity = 0.0;
        double cos_incidence = 1.0;
    };

    vector<GeomCandidate> raw_candidates;
    raw_candidates.reserve(effct_feat_num);
    V3D lidar_pos_w = s.pos + s.rot * s.offset_T_L_I;

    for (int i = 0; i < effct_feat_num; ++i)
    {
        const PointType &laser_p = laserCloudOri->points[i];
        V3D p_body(laser_p.x, laser_p.y, laser_p.z);
        V3D p_world = pointBodyToWorldVec(p_body, s);
        const PointType &norm_p = corr_normvect->points[i];
        V3D n(norm_p.x, norm_p.y, norm_p.z);
        double n_norm = n.norm();
        if (n_norm < 1e-6) continue;
        n /= n_norm;

        GeomCandidate cand;
        cand.p_world = p_world;
        cand.normal_w = n;
        cand.intensity = laser_p.intensity;

        V3D ray = p_world - lidar_pos_w;
        double ray_norm = ray.norm();
        if (ray_norm > 1e-6)
        {
            cand.cos_incidence = std::fabs(ray.dot(n) / ray_norm);
        }

        if (pipe_point_near_endcap_for_size(cand.p_world))
        {
            pipe_last_size_endcap_exclusion_used = true;
            continue;
        }

        if (pipe_apply_incidence_filter_in_lio && cand.cos_incidence < pipe_min_incidence_cos)
            continue;

        raw_candidates.push_back(cand);
    }

    pipe_last_geom_raw_count = int(raw_candidates.size());
    pipe_last_geom_filtered_count = pipe_last_geom_raw_count;
    pipe_last_geom_used_intensity_trim = false;

    if (raw_candidates.size() < 20) { pipe_last_geom_fail_code = 2; return false; }

    vector<GeomCandidate> geom_candidates = raw_candidates;
    if (pipe_enable_intensity_trim && raw_candidates.size() >= 20)
    {
        vector<double> intensities;
        intensities.reserve(raw_candidates.size());
        for (const auto &cand : raw_candidates) intensities.push_back(cand.intensity);

        double q_low = std::max(0.0, std::min(0.49, pipe_intensity_quantile_low));
        double q_high = std::max(0.51, std::min(1.0, pipe_intensity_quantile_high));
        if (q_high > q_low)
        {
            double i_low = quantile_from_sorted(intensities, q_low);
            double i_high = quantile_from_sorted(intensities, q_high);
            vector<GeomCandidate> trimmed;
            trimmed.reserve(raw_candidates.size());
            for (const auto &cand : raw_candidates)
            {
                if (cand.intensity >= i_low && cand.intensity <= i_high)
                    trimmed.push_back(cand);
            }
            if (trimmed.size() >= 20)
            {
                geom_candidates.swap(trimmed);
                pipe_last_geom_used_intensity_trim = true;
            }
        }
    }

    pipe_last_geom_filtered_count = int(geom_candidates.size());
    if (pipe_debug_log && pipe_debug_print_filter_stats)
    {
        ROS_INFO_THROTTLE(1.0,
                          "pipe geom filter raw=%d filtered=%d trim=%d min_cos=%.3f",
                          pipe_last_geom_raw_count,
                          pipe_last_geom_filtered_count,
                          int(pipe_last_geom_used_intensity_trim),
                          pipe_min_incidence_cos);
    }

    vector<V3D> normals;
    vector<V3D> world_pts;
    normals.reserve(geom_candidates.size());
    world_pts.reserve(geom_candidates.size());

    V3D centroid = Zero3d;
    for (const auto &cand : geom_candidates)
    {
        world_pts.push_back(cand.p_world);
        normals.push_back(cand.normal_w);
        centroid += cand.p_world;
    }

    if (world_pts.size() < 20) { pipe_last_geom_fail_code = 3; return false; }
    centroid /= double(world_pts.size());

    M3D normal_cov = M3D::Zero();
    for (const auto &n : normals)
    {
        normal_cov += n * n.transpose();
    }
    normal_cov /= double(normals.size());

    Eigen::SelfAdjointEigenSolver<M3D> eig_solver(normal_cov);
    if (eig_solver.info() != Eigen::Success) { pipe_last_geom_fail_code = 4; return false; }

    auto eigvals = eig_solver.eigenvalues();
    M3D eigvecs = eig_solver.eigenvectors();
    V3D axis_w = eigvecs.col(0).normalized();

    // Keep axis direction consistent with LiDAR forward direction and across frames.
    V3D lidar_forward_w = s.rot * (s.offset_R_L_I * V3D(1.0, 0.0, 0.0));
    if (axis_w.dot(lidar_forward_w) < 0.0) axis_w = -axis_w;
    if (pipe_global_axis_initialized && axis_w.dot(pipe_global_axis_w) < 0.0) axis_w = -axis_w;

    double sum_eval = std::max(1e-9, eigvals(0) + eigvals(1) + eigvals(2));
    double normal_planarity = 1.0 - eigvals(0) / sum_eval;

    V3D u_w = Zero3d;
    V3D u_acc = Zero3d;
    double best_proj = -1.0;
    for (const auto &n : normals)
    {
        V3D proj = n - axis_w * (axis_w.dot(n));
        double nn = proj.norm();
        if (nn < 1e-3) continue;
        proj /= nn;
        if (nn > best_proj)
        {
            best_proj = nn;
            u_w = proj;
        }
        if (u_acc.norm() > 1e-9 && proj.dot(u_acc) < 0.0) proj = -proj;
        u_acc += proj;
    }
    if (u_acc.norm() > 1e-6) u_w = u_acc.normalized();
    if (u_w.norm() < 1e-6) { pipe_last_geom_fail_code = 5; return false; }

    V3D v_w = axis_w.cross(u_w);
    if (v_w.norm() < 1e-6) { pipe_last_geom_fail_code = 6; return false; }
    v_w.normalize();
    u_w = v_w.cross(axis_w).normalized();

    // In a rectangular pipe, W/H can swap when the point distribution changes.
    // Use gravity to keep the height direction stable when gravity is not
    // parallel to the pipe axis. The sign is still kept continuous below.
    if (pipe_size_use_gravity_height)
    {
        V3D g_w(s.grav[0], s.grav[1], s.grav[2]);
        if (g_w.norm() > 1e-6)
        {
            V3D height_dir = g_w - axis_w * axis_w.dot(g_w);
            if (height_dir.norm() > 1e-3)
            {
                v_w = height_dir.normalized();
                u_w = v_w.cross(axis_w);
                if (u_w.norm() > 1e-6)
                {
                    u_w.normalize();
                    v_w = axis_w.cross(u_w).normalized();
                }
            }
        }
    }

    // Keep cross-section basis orientation continuous across frames.
    if (prev_pipe_basis_valid)
    {
        if (u_w.dot(prev_pipe_u_w) < 0.0) u_w = -u_w;
        if (v_w.dot(prev_pipe_v_w) < 0.0) v_w = -v_w;
    }
    prev_pipe_u_w = u_w;
    prev_pipe_v_w = v_w;
    prev_pipe_basis_valid = true;

    // Local frame origin only matters in u/v for box priors; shifting along axis is irrelevant.
    V3D axis_point_w = centroid - axis_w * (axis_w.dot(centroid));

    struct PipeSample
    {
        double t = 0.0;
        double u = 0.0;
        double v = 0.0;
        double nu = 0.0;
        double nv = 0.0;
        bool wall_candidate = false;
    };

    vector<PipeSample> samples;
    vector<double> t_coords;
    vector<double> u_mid_coords, v_mid_coords;
    vector<double> u_pos_wall, u_neg_wall, v_pos_wall, v_neg_wall;
    samples.reserve(world_pts.size());
    t_coords.reserve(world_pts.size());
    u_mid_coords.reserve(world_pts.size());
    v_mid_coords.reserve(world_pts.size());
    u_pos_wall.reserve(world_pts.size());
    u_neg_wall.reserve(world_pts.size());
    v_pos_wall.reserve(world_pts.size());
    v_neg_wall.reserve(world_pts.size());

    for (size_t i = 0; i < world_pts.size(); ++i)
    {
        const V3D &p = world_pts[i];
        V3D d = p - axis_point_w;
        PipeSample ssmpl;
        ssmpl.t = d.dot(axis_w);
        ssmpl.u = d.dot(u_w);
        ssmpl.v = d.dot(v_w);
        t_coords.push_back(ssmpl.t);

        V3D n = normals[i] - axis_w * (axis_w.dot(normals[i]));
        double nn = n.norm();
        if (nn >= 1e-3)
        {
            n /= nn;
            ssmpl.nu = n.dot(u_w);
            ssmpl.nv = n.dot(v_w);
            double dom = std::max(std::fabs(ssmpl.nu), std::fabs(ssmpl.nv));
            ssmpl.wall_candidate = (dom >= pipe_size_wall_min_normal_cos);
        }
        samples.push_back(ssmpl);
    }

    double t_min = quantile_from_sorted(t_coords, 0.05);
    double t_max = quantile_from_sorted(t_coords, 0.95);
    double length = t_max - t_min;
    if (length < 1e-6) { pipe_last_geom_fail_code = 7; return false; }

    double keep_ratio = std::max(0.10, std::min(1.0, pipe_mid_section_keep_ratio));
    double t_center = 0.5 * (t_min + t_max);
    double half_keep = 0.5 * length * keep_ratio;
    double t_keep_min = t_center - half_keep;
    double t_keep_max = t_center + half_keep;

    for (const auto &ssmpl : samples)
    {
        if (ssmpl.t < t_keep_min || ssmpl.t > t_keep_max) continue;
        u_mid_coords.push_back(ssmpl.u);
        v_mid_coords.push_back(ssmpl.v);

        if (!ssmpl.wall_candidate) continue;
        if (std::fabs(ssmpl.nu) > std::fabs(ssmpl.nv))
        {
            if (ssmpl.nu >= 0.0) u_pos_wall.push_back(ssmpl.u);
            else                 u_neg_wall.push_back(ssmpl.u);
        }
        else
        {
            if (ssmpl.nv >= 0.0) v_pos_wall.push_back(ssmpl.v);
            else                 v_neg_wall.push_back(ssmpl.v);
        }
    }

    if (u_mid_coords.size() < size_t(std::max(8, pipe_size_min_axis_points / 2)) ||
        v_mid_coords.size() < size_t(std::max(8, pipe_size_min_axis_points / 2)))
    {
        pipe_last_geom_fail_code = 8;
        return false;
    }

    double q_low = std::max(0.0, std::min(0.20, pipe_size_span_q_low));
    double q_high = std::max(0.80, std::min(1.0, pipe_size_span_q_high));
    if (q_high <= q_low) { q_low = 0.03; q_high = 0.97; }

    double u_min = quantile_from_sorted(u_mid_coords, q_low);
    double u_max = quantile_from_sorted(u_mid_coords, q_high);
    double v_min = quantile_from_sorted(v_mid_coords, q_low);
    double v_max = quantile_from_sorted(v_mid_coords, q_high);
    double width = u_max - u_min;
    double height = v_max - v_min;

    const size_t min_wall_samples = size_t(std::max(2, pipe_size_wall_min_side_samples));
    bool width_wall_ok = (u_pos_wall.size() >= min_wall_samples &&
                          u_neg_wall.size() >= min_wall_samples);
    bool height_wall_ok = (v_pos_wall.size() >= min_wall_samples &&
                           v_neg_wall.size() >= min_wall_samples);
    bool four_wall_fit_ok = width_wall_ok && height_wall_ok;

    double wall_width = width;
    double wall_height = height;
    double u_center = 0.0;
    double v_center = 0.0;

    if (width_wall_ok)
    {
        // Precision mode: estimate opposite-wall distance from the median wall
        // centers, not from full-span extrema. This suppresses sparse edge
        // points, grazing reflections, and short-term map thickness.
        double u_pos = quantile_from_sorted(u_pos_wall, 0.5);
        double u_neg = quantile_from_sorted(u_neg_wall, 0.5);
        double u_low_raw = std::min(u_pos, u_neg);
        double u_high_raw = std::max(u_pos, u_neg);
        u_center = 0.5 * (u_low_raw + u_high_raw);
        u_min = u_low_raw - u_center;
        u_max = u_high_raw - u_center;
        wall_width = u_max - u_min;
        width = wall_width;
    }

    if (height_wall_ok)
    {
        // Same median-wall policy for height. The median is deliberately used
        // instead of max/min span so that a few wrong wall labels do not dominate.
        double v_pos = quantile_from_sorted(v_pos_wall, 0.5);
        double v_neg = quantile_from_sorted(v_neg_wall, 0.5);
        double v_low_raw = std::min(v_pos, v_neg);
        double v_high_raw = std::max(v_pos, v_neg);
        v_center = 0.5 * (v_low_raw + v_high_raw);
        v_min = v_low_raw - v_center;
        v_max = v_high_raw - v_center;
        wall_height = v_max - v_min;
        height = wall_height;
    }

    // Re-centre any dimension for which both opposite walls are visible. This
    // is important for a robot sliding on the lower wall: a single frame may
    // only provide a reliable width or a reliable height, but not always both.
    if (width_wall_ok || height_wall_ok)
    {
        axis_point_w = axis_point_w + u_center * u_w + v_center * v_w;
    }

    double u_low = std::min(u_max, u_min);
    double u_high = std::max(u_max, u_min);
    double v_low = std::min(v_max, v_min);
    double v_high = std::max(v_max, v_min);
    u_min = u_low;
    u_max = u_high;
    v_min = v_low;
    v_max = v_high;
    width = std::max(u_max, u_min) - std::min(u_max, u_min);
    height = std::max(v_max, v_min) - std::min(v_max, v_min);

    pipe_last_raw_width = width;
    pipe_last_raw_height = height;

    bool raw_size_ok = (width > pipe_min_width && width < pipe_max_width &&
                        height > pipe_min_height && height < pipe_max_height);
    bool width_size_ok = width_wall_ok && (wall_width > pipe_min_width && wall_width < pipe_max_width);
    bool height_size_ok = height_wall_ok && (wall_height > pipe_min_height && wall_height < pipe_max_height);

    // Keep disabled for precision. Full cross-section span is easily polluted by
    // outlet face points, robot body edges, and sparse grazing reflections.
    if (pipe_size_accept_span_fallback &&
        (!pipe_width_require_both_side_walls || !pipe_height_require_both_top_bottom_walls))
    {
        if (!pipe_width_require_both_side_walls && !width_size_ok &&
            u_mid_coords.size() >= size_t(std::max(8, pipe_size_min_axis_points)) &&
            width > pipe_min_width && width < pipe_max_width)
        {
            wall_width = width;
            width_size_ok = true;
        }
        if (!pipe_height_require_both_top_bottom_walls && !height_size_ok &&
            v_mid_coords.size() >= size_t(std::max(8, pipe_size_min_axis_points)) &&
            height > pipe_min_height && height < pipe_max_height)
        {
            wall_height = height;
            height_size_ok = true;
        }
    }
    double straight_conf = std::min(1.0, std::max(0.0, (normal_planarity - 0.5) / 0.5));
    bool axis_ok = (straight_conf > pipe_axis_conf_threshold);
    bool raw_box_ok = axis_ok && four_wall_fit_ok && raw_size_ok;

    pipe_last_width_sample_count = int(std::min(u_pos_wall.size(), u_neg_wall.size()));
    pipe_last_height_sample_count = int(std::min(v_pos_wall.size(), v_neg_wall.size()));

    // Size quantification is decoupled from strict single-frame box reliability.
    // First obtain the median opposite-wall observation for this frame, then
    // push it into the sliding history. The public output below is the 10-frame
    // median-filtered value when the current frame also satisfies the constraints.
    if (pipe_size_wall_history_enable && pipe_size_accept_partial_pairs)
        pipe_accept_size_observations(wall_width, width_size_ok, wall_height, height_size_ok, axis_ok);
    else
        pipe_accept_size_sample(width, height, raw_box_ok);

    bool current_wall_size_ok = width_size_ok && height_size_ok;
    bool filtered_size_ok = pipe_stable_width_valid && pipe_stable_height_valid;
    bool use_filtered_output = pipe_size_stabilize_enable && pipe_size_wall_history_enable;

    pipe_last_constrained_size_valid = current_wall_size_ok &&
                                       (!use_filtered_output || filtered_size_ok);
    pipe_last_constrained_width = 0.0;
    pipe_last_constrained_height = 0.0;
    if (pipe_last_constrained_size_valid)
    {
        pipe_last_constrained_width = use_filtered_output ? pipe_stable_width : wall_width;
        pipe_last_constrained_height = use_filtered_output ? pipe_stable_height : wall_height;
    }
    pipe_last_constrained_width_pair_count = pipe_last_width_sample_count;
    pipe_last_constrained_height_pair_count = pipe_last_height_sample_count;

    geom.axis_reliable = axis_ok;
    geom.box_reliable = raw_box_ok;
    geom.valid = geom.box_reliable;
    geom.axis_w = axis_w;
    geom.u_w = u_w;
    geom.v_w = v_w;
    geom.axis_point_w = axis_point_w;
    geom.centroid_w = centroid;
    geom.t_min = t_min;
    geom.t_max = t_max;
    geom.u_min = u_min;
    geom.u_max = u_max;
    geom.v_min = v_min;
    geom.v_max = v_max;
    geom.length = length;
    geom.width = width;
    geom.height = height;
    geom.u_pos_count = int(u_pos_wall.size());
    geom.u_neg_count = int(u_neg_wall.size());
    geom.v_pos_count = int(v_pos_wall.size());
    geom.v_neg_count = int(v_neg_wall.size());
    geom.four_wall_fit_ok = four_wall_fit_ok;
    geom.normal_planarity = normal_planarity;
    geom.straight_confidence = straight_conf;

    // Do not overwrite current-frame geometry with stable size here.
    // Stable W/H are only used by /pipe_size. The per-frame CSV columns must
    // remain raw observations, otherwise they appear frozen after the first
    // stable sample and hide whether the detector is actually updating.

    if (geom.axis_reliable)
    {
        if (!pipe_global_axis_initialized)
        {
            pipe_global_axis_w = axis_w;
            pipe_global_axis_initialized = true;
        }

        for (const auto &p : world_pts)
        {
            double t_global = p.dot(pipe_global_axis_w);
            pipe_global_t_min = std::min(pipe_global_t_min, t_global);
            pipe_global_t_max = std::max(pipe_global_t_max, t_global);
        }
        pipe_global_length = pipe_global_t_max - pipe_global_t_min;
        geom.global_length = pipe_global_length;
    }
    else
    {
        geom.global_length = pipe_global_length;
    }

    pipe_last_geom_updated = geom.axis_reliable;
    pipe_last_geom_fail_code = geom.axis_reliable ? 0 : 9;
    pipe_last_frame_valid = geom.valid;
    pipe_last_frame_axis_reliable = geom.axis_reliable;
    pipe_last_frame_box_reliable = geom.box_reliable;
    pipe_last_frame_length = geom.length;
    pipe_last_frame_width = width;
    pipe_last_frame_height = height;
    pipe_last_frame_global_length = geom.global_length;
    pipe_last_frame_conf = geom.straight_confidence;
    pipe_last_frame_u_pos_count = geom.u_pos_count;
    pipe_last_frame_u_neg_count = geom.u_neg_count;
    pipe_last_frame_v_pos_count = geom.v_pos_count;
    pipe_last_frame_v_neg_count = geom.v_neg_count;
    pipe_last_frame_four_wall_fit_ok = geom.four_wall_fit_ok;

    return geom.axis_reliable;
}

void analyze_degeneracy(const MatrixXd &Hx, const RectPipeGeomState &geom)
{
    pipe_last_min_eig = 0.0;
    pipe_last_cond_num = 0.0;
    pipe_last_axial_info = 0.0;
    pipe_last_lateral_info = 0.0;
    pipe_last_axial_ratio = 1.0;
    pipe_last_min_pos_rel = 1.0;
    pipe_last_u_info = 0.0;
    pipe_last_v_info = 0.0;
    pipe_last_deg_by_abs_eig = false;
    pipe_last_deg_by_rel_eig = false;
    pipe_last_deg_by_axial_ratio = false;
    pipe_last_deg_by_cond = false;
    pipe_degenerate = false;

    static int deg_hold = 0;
    if (Hx.rows() < 6)
    {
        deg_hold = std::max(0, deg_hold - 1);
        pipe_degenerate = (deg_hold > 0);
        return;
    }

    const double eps = 1e-12;
    Matrix<double, 6, 6> info = Hx.leftCols(6).transpose() * Hx.leftCols(6);
    Eigen::SelfAdjointEigenSolver<Matrix<double, 6, 6>> solver(info);
    if (solver.info() != Eigen::Success)
    {
        deg_hold = std::max(0, deg_hold - 1);
        pipe_degenerate = (deg_hold > 0);
        return;
    }

    const auto vals = solver.eigenvalues();
    pipe_last_min_eig = vals(0);
    pipe_last_cond_num = vals(5) / std::max(eps, vals(0));

    Matrix3d info_pos = info.block<3, 3>(0, 0);
    Eigen::SelfAdjointEigenSolver<Matrix3d> pos_solver(info_pos);
    if (pos_solver.info() != Eigen::Success)
    {
        deg_hold = std::max(0, deg_hold - 1);
        pipe_degenerate = (deg_hold > 0);
        return;
    }

    const auto pos_vals = pos_solver.eigenvalues();
    const auto pos_vecs = pos_solver.eigenvectors();
    const double pos_mean = std::max(eps, pos_vals.mean());
    pipe_last_min_pos_rel = pos_vals(0) / pos_mean;

    bool use_pipe_axis = false;
    V3D axis_for_test = pos_vecs.col(0).normalized();

    if (geom.axis_reliable && geom.axis_w.norm() > 0.5)
    {
        // The pipe axis is allowed to drive degeneracy detection even when the
        // rectangle size is not reliable enough for box priors.
        axis_for_test = geom.axis_w.normalized();
        use_pipe_axis = true;

        V3D u_for_test = geom.u_w;
        V3D v_for_test = geom.v_w;
        if (u_for_test.norm() > 0.5 && v_for_test.norm() > 0.5)
        {
            u_for_test.normalize();
            v_for_test.normalize();
            pipe_last_u_info = u_for_test.transpose() * info_pos * u_for_test;
            pipe_last_v_info = v_for_test.transpose() * info_pos * v_for_test;
            pipe_last_lateral_info = 0.5 * (pipe_last_u_info + pipe_last_v_info);
        }
        else
        {
            Matrix3d Pperp = Matrix3d::Identity() - axis_for_test * axis_for_test.transpose();
            pipe_last_lateral_info = (Pperp * info_pos * Pperp).trace() / 2.0;
            pipe_last_u_info = pipe_last_lateral_info;
            pipe_last_v_info = pipe_last_lateral_info;
        }

        pipe_last_axial_info = axis_for_test.transpose() * info_pos * axis_for_test;
    }
    else
    {
        // When the pipe axis is not reliable yet, do not force axial == lateral.
        // Use the weakest positional eigen-direction as a temporary degeneracy
        // direction, so the detector can still see ill-conditioned motion.
        pipe_last_axial_info = std::max(eps, pos_vals(0));
        pipe_last_lateral_info = std::max(eps, 0.5 * (pos_vals(1) + pos_vals(2)));
        pipe_last_u_info = pos_vals(1);
        pipe_last_v_info = pos_vals(2);
    }

    pipe_last_axial_info = std::max(eps, pipe_last_axial_info);
    pipe_last_lateral_info = std::max(eps, pipe_last_lateral_info);
    pipe_last_axial_ratio = pipe_last_axial_info / pipe_last_lateral_info;

    pipe_last_deg_by_abs_eig = pipe_last_min_eig < pipe_degenerate_min_eig;
    pipe_last_deg_by_rel_eig = pipe_last_min_pos_rel < pipe_degenerate_min_pos_rel;
    pipe_last_deg_by_axial_ratio = pipe_last_axial_ratio < pipe_degenerate_ratio;
    pipe_last_deg_by_cond = pipe_last_cond_num > pipe_degenerate_cond_thresh;

    bool deg_now = false;
    if (use_pipe_axis)
    {
        // In pipe mode, axial/lateral contrast is the most meaningful signal.
        deg_now = pipe_last_deg_by_axial_ratio ||
                  pipe_last_deg_by_rel_eig ||
                  pipe_last_deg_by_cond ||
                  pipe_last_deg_by_abs_eig;
    }
    else
    {
        // Before the pipe axis is reliable, avoid applying pipe priors, but still
        // expose degeneracy in the logs through relative eigenvalue and condition
        // number tests.
        deg_now = pipe_last_deg_by_rel_eig ||
                  pipe_last_deg_by_cond ||
                  pipe_last_deg_by_abs_eig;
    }

    int hold_frames = std::max(0, pipe_degenerate_hold_frames);
    if (deg_now)
        deg_hold = hold_frames;
    else
        deg_hold = std::max(0, deg_hold - 1);

    pipe_degenerate = deg_now || (deg_hold > 0);
}

void augment_with_pipe_priors(const state_ikfom &s,
                              const RectPipeGeomState &geom,
                              MatrixXd &Hx,
                              VectorXd &h)
{
    if (!pipe_prior_enable || !pipe_geometry_prior_enable || !pipe_degenerate || !geom.axis_reliable) return;

    vector<RowVectorXd> rows;
    vector<double> residuals;

    V3D lidar_forward_w = s.rot * (s.offset_R_L_I * V3D(1.0, 0.0, 0.0));
    V3D att_res = lidar_forward_w.cross(geom.axis_w);
    M3D d_att_d_theta = skewMatrix(geom.axis_w) * skewMatrix(lidar_forward_w);

    for (int j = 0; j < 3; ++j)
    {
        RowVectorXd row = RowVectorXd::Zero(12);
        row.block<1, 3>(0, 3) = pipe_axis_align_weight * d_att_d_theta.row(j);
        rows.push_back(row);
        residuals.push_back(-pipe_axis_align_weight * att_res(j));
    }

    V3D lidar_pos_w = s.pos + s.rot * s.offset_T_L_I;
    if (prev_lidar_pos_valid)
    {
        Matrix3d Pperp = Matrix3d::Identity() - geom.axis_w * geom.axis_w.transpose();
        V3D motion_res = Pperp * (lidar_pos_w - prev_lidar_pos_world);
        M3D dpos_dtheta = -skewMatrix(s.rot * s.offset_T_L_I);
        Matrix3d J_theta = Pperp * dpos_dtheta;
        Matrix3d J_pos = Pperp;

        for (int j = 0; j < 3; ++j)
        {
            RowVectorXd row = RowVectorXd::Zero(12);
            row.block<1, 3>(0, 0) = pipe_motion_weight * J_pos.row(j);
            row.block<1, 3>(0, 3) = pipe_motion_weight * J_theta.row(j);
            rows.push_back(row);
            residuals.push_back(-pipe_motion_weight * motion_res(j));
        }
    }

    if (geom.box_reliable)
    {
        // Soft rectangular box prior: keep LiDAR pose inside the fitted cross section.
        V3D rel = lidar_pos_w - geom.axis_point_w;
        double u = rel.dot(geom.u_w);
        double v = rel.dot(geom.v_w);
        double margin = 0.05;
        std::array<double, 4> box_res = {
            std::max(0.0, u - (geom.u_max + margin)),
            std::max(0.0, (geom.u_min - margin) - u),
            std::max(0.0, v - (geom.v_max + margin)),
            std::max(0.0, (geom.v_min - margin) - v)
        };
        std::array<V3D, 4> box_grads = {
            geom.u_w,
            -geom.u_w,
            geom.v_w,
            -geom.v_w
        };
        for (int j = 0; j < 4; ++j)
        {
            if (box_res[j] <= 0.0) continue;
            RowVectorXd row = RowVectorXd::Zero(12);
            row.block<1, 3>(0, 0) = pipe_prior_weight * box_grads[j].transpose();
            row.block<1, 3>(0, 3) = pipe_prior_weight * (box_grads[j].transpose() * (-skewMatrix(s.rot * s.offset_T_L_I)));
            rows.push_back(row);
            residuals.push_back(-pipe_prior_weight * box_res[j]);
        }
    }

    if (rows.empty()) return;

    int old_rows = Hx.rows();
    int add_rows = rows.size();
    MatrixXd Hx_aug(old_rows + add_rows, Hx.cols());
    VectorXd h_aug(old_rows + add_rows);
    Hx_aug.topRows(old_rows) = Hx;
    h_aug.head(old_rows) = h;
    for (int i = 0; i < add_rows; ++i)
    {
        Hx_aug.row(old_rows + i) = rows[i];
        h_aug(old_rows + i) = residuals[i];
    }
    Hx.swap(Hx_aug);
    h.swap(h_aug);
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void publish_pipe_size(const ros::Publisher &pubPipeSize)
{
    // Public output is intentionally reduced to one meaning only:
    // current-frame W/H after min/max and wall-pair constraints.
    // Vector3.x is kept as 0.0 because pipe length is no longer part of size output.
    if (!pipe_last_constrained_size_valid) return;

    geometry_msgs::Vector3 msg;
    msg.x = 0.0;
    msg.y = pipe_last_constrained_width;
    msg.z = pipe_last_constrained_height;
    pubPipeSize.publish(msg);
}

void publish_pipe_pose_suv(const ros::Publisher &pubPipePoseSUV)
{
    if (!pipe_last_pipe_s_valid) return;

    geometry_msgs::Vector3 msg;
    msg.x = pipe_last_pipe_s;  // axial position. In exit-only mode, s = known_length - distance_to_exit.
    msg.y = 0.0;               // lateral offset along pipe u direction.
    msg.z = 0.0;               // vertical offset along pipe v direction.

    if (g_pipe_geom.axis_reliable && g_pipe_geom.u_w.norm() > 0.5 && g_pipe_geom.v_w.norm() > 0.5)
    {
        V3D lidar_pos_w = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        V3D rel = lidar_pos_w - g_pipe_geom.axis_point_w;
        msg.y = rel.dot(g_pipe_geom.u_w.normalized());
        msg.z = rel.dot(g_pipe_geom.v_w.normalized());
    }

    pubPipePoseSUV.publish(msg);
}

void publish_pipe_bbox(const ros::Publisher &pubPipeBBox)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = "camera_init";
    marker.header.stamp = ros::Time().fromSec(lidar_end_time);
    marker.ns = "pipe_bbox";
    marker.id = 0;

    if (!g_pipe_geom.box_reliable)
    {
        marker.action = visualization_msgs::Marker::DELETE;
        pubPipeBBox.publish(marker);
        return;
    }

    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;

    V3D center = g_pipe_geom.axis_point_w
               + g_pipe_geom.axis_w * (0.5 * (g_pipe_geom.t_min + g_pipe_geom.t_max))
               + g_pipe_geom.u_w * (0.5 * (g_pipe_geom.u_min + g_pipe_geom.u_max))
               + g_pipe_geom.v_w * (0.5 * (g_pipe_geom.v_min + g_pipe_geom.v_max));

    M3D rot;
    rot.col(0) = g_pipe_geom.axis_w;
    rot.col(1) = g_pipe_geom.u_w;
    rot.col(2) = g_pipe_geom.v_w;
    Eigen::Quaterniond q(rot);
    q.normalize();

    marker.pose.position.x = center(0);
    marker.pose.position.y = center(1);
    marker.pose.position.z = center(2);
    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();
    marker.pose.orientation.w = q.w();

    marker.scale.x = std::max(1e-3, g_pipe_geom.length);
    marker.scale.y = std::max(1e-3, g_pipe_geom.width);
    marker.scale.z = std::max(1e-3, g_pipe_geom.height);

    marker.color.r = 0.1f;
    marker.color.g = 0.9f;
    marker.color.b = 0.2f;
    marker.color.a = 0.25f;
    marker.lifetime = ros::Duration(0.2);
    pubPipeBBox.publish(marker);
}

void update_pipe_debug_endcap_cloud(const state_ikfom &s, const vector<PipeEndCapState> &caps)
{
    pipe_debug_endcap_world->clear();
    pipe_last_endcap_candidate_count = int(caps.size());
    if (!pipe_debug_publish_endcap_points) return;

    for (const auto &cap : caps)
    {
        if (!cap.visible || cap.points_body.empty()) continue;
        float inten = 30.0f;
        if (cap.anchor_id == 1) inten = 100.0f;
        else if (cap.anchor_id == 2) inten = 200.0f;
        else if (cap.side < 0) inten = 60.0f;
        else if (cap.side > 0) inten = 160.0f;

        for (const auto &p_body : cap.points_body)
        {
            V3D p_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
            V3D p_world = s.rot * p_imu + s.pos;
            PointType pt;
            pt.x = p_world(0);
            pt.y = p_world(1);
            pt.z = p_world(2);
            pt.intensity = inten;
            pipe_debug_endcap_world->push_back(pt);
        }
    }
}

void publish_pipe_endcap_points(const ros::Publisher &pubPipeEndcapPoints)
{
    if (!pipe_debug_publish_endcap_points) return;
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*pipe_debug_endcap_world, msg);
    msg.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg.header.frame_id = "camera_init";
    pubPipeEndcapPoints.publish(msg);
}

// ---------------- Pipe entrance / exit end-cap landmark support ----------------
bool pipe_is_self_mask_point(const V3D &p_body)
{
    if (!pipe_self_mask_enable) return false;
    return (p_body.x() >= pipe_self_mask_x_min && p_body.x() <= pipe_self_mask_x_max &&
            p_body.y() >= pipe_self_mask_y_min && p_body.y() <= pipe_self_mask_y_max &&
            p_body.z() >= pipe_self_mask_z_min && p_body.z() <= pipe_self_mask_z_max);
}

inline V3D pipe_lidar_origin_world(const state_ikfom &s)
{
    return s.pos + s.rot * s.offset_T_L_I;
}

V3D pipe_lidar_axis_world(const state_ikfom &s)
{
    V3D axis_lidar(pipe_position_axis_lidar_x,
                   pipe_position_axis_lidar_y,
                   pipe_position_axis_lidar_z);
    if (axis_lidar.norm() < 1e-6) axis_lidar = V3D(1.0, 0.0, 0.0);
    axis_lidar.normalize();

    V3D forward_w = s.rot * (s.offset_R_L_I * axis_lidar);
    if (forward_w.norm() < 1e-6) forward_w = V3D(1.0, 0.0, 0.0);
    forward_w.normalize();
    return forward_w;
}

V3D pipe_axis_from_lidar_or_geom(const state_ikfom &s)
{
    V3D axis_w = Zero3d;
    if ((pipe_exit_cap_initialized || pipe_origin_initialized) && pipe_axis_anchor_w.norm() > 1e-6)
        axis_w = pipe_axis_anchor_w;
    else if (g_pipe_geom.axis_reliable && g_pipe_geom.axis_w.norm() > 1e-6)
        axis_w = g_pipe_geom.axis_w;
    else if (pipe_global_axis_initialized && pipe_global_axis_w.norm() > 1e-6)
        axis_w = pipe_global_axis_w;
    else
    {
        V3D axis_lidar(pipe_position_axis_lidar_x,
                       pipe_position_axis_lidar_y,
                       pipe_position_axis_lidar_z);
        if (axis_lidar.norm() < 1e-6) axis_lidar = V3D(1.0, 0.0, 0.0);
        axis_lidar.normalize();
        axis_w = s.rot * (s.offset_R_L_I * axis_lidar);
    }

    if (axis_w.norm() < 1e-6) axis_w = pipe_lidar_axis_world(s);
    axis_w.normalize();

    // Force the positive pipe coordinate to follow the configured LiDAR forward
    // direction. This prevents s_odom from becoming negative only because an
    // end-cap normal or PCA axis was selected with the opposite sign.
    V3D forward_w = pipe_lidar_axis_world(s);
    if (axis_w.dot(forward_w) < 0.0) axis_w = -axis_w;

    return axis_w;
}

void pipe_update_lidar_s_odom(const state_ikfom &s)
{
    pipe_last_s_odom_valid = false;

    V3D lidar_pos_w = pipe_lidar_origin_world(s);
    V3D axis_w = pipe_axis_from_lidar_or_geom(s);
    if (axis_w.norm() < 1e-6) return;
    axis_w.normalize();

    if (!pipe_axis_odom_initialized)
    {
        pipe_lidar_pos0_w = lidar_pos_w;
        pipe_axis0_w = axis_w;
        pipe_axis_odom_initialized = true;
    }

    pipe_last_s_odom = pipe_initial_lidar_s + (lidar_pos_w - pipe_lidar_pos0_w).dot(pipe_axis0_w);
    pipe_last_s_odom_valid = true;
}

bool pipe_estimate_d_exit_scan_from_cap(const state_ikfom &s,
                                         const PipeEndCapState &cap,
                                         double &d_exit_scan,
                                         double &plane_res)
{
    d_exit_scan = 0.0;
    plane_res = 0.0;

    if (!pipe_scan_exit_distance_enable) return false;
    if (!cap.reliable) return false;
    if (int(cap.points_body.size()) < std::max(6, pipe_endcap_min_points)) return false;

    V3D axis_w = pipe_axis_from_lidar_or_geom(s);
    if (axis_w.norm() < 1e-6) return false;
    axis_w.normalize();

    V3D axis_lidar = s.offset_R_L_I.conjugate() * (s.rot.conjugate() * axis_w);
    if (axis_lidar.norm() < 1e-6) return false;
    axis_lidar.normalize();

    vector<double> proj;
    proj.reserve(cap.points_body.size());
    V3D mean = Zero3d;
    for (const auto &p : cap.points_body)
    {
        proj.push_back(p.dot(axis_lidar));
        mean += p;
    }
    mean /= double(cap.points_body.size());

    double d_med = quantile_from_sorted(proj, 0.5);
    if (d_med < pipe_exit_min_forward_dist) return false;
    if (pipe_exit_max_forward_dist > 0.0 && d_med > pipe_exit_max_forward_dist) return false;

    // Keep the plane residual check from the end-cap points, but use the
    // median axial projection as distance. It is less sensitive to the sign of
    // the plane normal and to small angular errors in the ray-plane formula.
    M3D cov = M3D::Zero();
    for (const auto &p : cap.points_body)
    {
        V3D q = p - mean;
        cov += q * q.transpose();
    }

    Eigen::SelfAdjointEigenSolver<M3D> es(cov);
    if (es.info() == Eigen::Success)
    {
        V3D n = es.eigenvectors().col(0);
        if (n.norm() > 1e-6)
        {
            n.normalize();
            double abs_sum = 0.0;
            for (const auto &p : cap.points_body)
                abs_sum += std::fabs(n.dot(p - mean));
            plane_res = abs_sum / double(cap.points_body.size());
            if (pipe_endcap_max_plane_res > 0.0 && plane_res > pipe_endcap_max_plane_res) return false;
        }
    }

    d_exit_scan = d_med;
    return true;
}

void pipe_update_scan_exit_distance_output(const state_ikfom &s, const PipeEndCapState &cap)
{
    pipe_last_d_exit_scan_valid = false;
    pipe_last_d_exit_scan = 0.0;
    pipe_last_scan_plane_res = 0.0;

    double d_scan = 0.0;
    double res = 0.0;
    if (!pipe_estimate_d_exit_scan_from_cap(s, cap, d_scan, res)) return;

    pipe_last_d_exit_scan = d_scan;
    pipe_last_scan_plane_res = res;
    pipe_last_d_exit_scan_valid = true;
}

void pipe_update_length_estimate()
{
    if (!pipe_length_estimate_enable) return;
    if (!pipe_last_s_odom_valid || !pipe_last_d_exit_scan_valid) return;

    double L_meas = pipe_last_s_odom + pipe_last_d_exit_scan;
    pipe_last_length_meas = L_meas;

    if (!(L_meas > 0.05)) return;
    if (pipe_length_max > 0.0 && L_meas > pipe_length_max) return;

    if (!pipe_length_est_valid)
    {
        pipe_length_est = L_meas;
        pipe_length_est_valid = true;
        pipe_length_est_samples = 1;
        return;
    }

    double err = L_meas - pipe_length_est;
    if (pipe_length_update_gate > 0.0 && std::fabs(err) > pipe_length_update_gate)
    {
        if (pipe_debug_log || pipe_debug_match_stats)
        {
            ROS_WARN_STREAM_THROTTLE(0.5, "[PIPE_LEN] reject L_meas=" << L_meas
                                     << " L_est=" << pipe_length_est
                                     << " err=" << err
                                     << " gate=" << pipe_length_update_gate);
        }
        return;
    }

    double alpha = std::max(0.0, std::min(1.0, pipe_length_update_alpha));
    pipe_length_est = (1.0 - alpha) * pipe_length_est + alpha * L_meas;
    pipe_length_est_samples++;
}

void pipe_select_position_output_legacy(const state_ikfom &s)
{
    pipe_last_pipe_s_valid = false;
    if (!pipe_exit_cap_initialized) return;

    V3D axis = pipe_axis_anchor_w;
    if (axis.norm() < 1e-6) return;
    axis.normalize();

    V3D lidar_pos_w = pipe_lidar_origin_world(s);
    pipe_last_distance_to_exit = (pipe_exit_cap_center_w - lidar_pos_w).dot(axis);

    if (pipe_endcap_known_length > 0.0)
    {
        pipe_last_pipe_s = pipe_endcap_known_length - pipe_last_distance_to_exit;
        pipe_last_pipe_s_valid = true;
    }
}

void pipe_select_position_output(const state_ikfom &s)
{
    pipe_last_pipe_s_valid = false;
    pipe_last_pipe_length_output = pipe_effective_pipe_length();

    if (pipe_position_output_mode == 0)
    {
        pipe_select_position_output_legacy(s);
        return;
    }

    if (pipe_position_output_mode == 1 && pipe_endcap_known_length > 0.0 && pipe_last_d_exit_scan_valid)
    {
        pipe_last_pipe_s = pipe_endcap_known_length - pipe_last_d_exit_scan;
        pipe_last_distance_to_exit = pipe_last_d_exit_scan;
        pipe_last_pipe_s_valid = true;
        return;
    }

    // Unknown-length friendly output. s is the LiDAR center axial odometry from
    // the chosen initial point. The exit distance is a direct scan measurement
    // when the outlet is visible. Otherwise it is predicted from the online
    // length estimate, if available.
    if (pipe_last_s_odom_valid)
    {
        pipe_last_pipe_s = pipe_last_s_odom;
        pipe_last_pipe_s_valid = true;

        if (pipe_last_d_exit_scan_valid)
            pipe_last_distance_to_exit = pipe_last_d_exit_scan;
        else if (pipe_endcap_known_length > 0.0)
            pipe_last_distance_to_exit = pipe_endcap_known_length - pipe_last_s_odom;
        else if (pipe_length_est_valid)
            pipe_last_distance_to_exit = pipe_length_est - pipe_last_s_odom;
        else
            pipe_last_distance_to_exit = std::numeric_limits<double>::quiet_NaN();
    }
}

void pipe_update_exit_pose_outputs(const state_ikfom &s)
{
    pipe_update_lidar_s_odom(s);
    pipe_select_position_output(s);
}

void pipe_update_position_outputs_with_cap(const state_ikfom &s, const PipeEndCapState &cap)
{
    pipe_update_lidar_s_odom(s);
    pipe_update_scan_exit_distance_output(s, cap);
    pipe_update_length_estimate();
    pipe_select_position_output(s);
}

void pipe_reset_exit_candidate_track()
{
    pipe_exit_candidate_frames = 0;
    pipe_exit_candidate_valid = false;
    pipe_exit_candidate_center_w = Zero3d;
    pipe_exit_candidate_normal_w = V3D(1.0, 0.0, 0.0);
}

void update_exit_only_anchor(const state_ikfom &s, PipeEndCapState &cap, const V3D &axis_w)
{
    if (!cap.reliable) return;

    // Do not create a new outlet landmark while the basic FAST-LIO matcher is
    // already unhealthy. Otherwise a false high-slice plane can be locked, then
    // its EKF rows can pull the pose away from the map and make nn_ok collapse.
    if (pipe_endcap_require_lio_healthy && !pipe_exit_cap_initialized &&
        effct_feat_num < std::max(1, pipe_endcap_min_lio_eff))
    {
        cap.anchor_id = 0;
        cap.matched_anchor = false;
        cap.s_anchor = pipe_endcap_known_length > 0.0 ? pipe_endcap_known_length : 0.0;
        cap.s_meas = 0.0;
        return;
    }

    // Do not mark a candidate as anchor=2 before it has actually matched the
    // locked outlet landmark. This avoids logs such as anchor=2 with
    // endcap_rows=0 and makes append_endcap_rows rejection reasons explicit.
    cap.anchor_id = 0;
    cap.matched_anchor = false;
    cap.s_anchor = pipe_endcap_known_length > 0.0 ? pipe_endcap_known_length : 0.0;
    cap.s_meas = 0.0;

    V3D n = cap.normal_w.normalized();
    // Use a positive pipe axis from robot toward the exit.
    if (n.dot(axis_w) < 0.0) n = -n;
    cap.normal_w = n;

    bool same_candidate = true;
    if (!pipe_exit_candidate_valid)
    {
        same_candidate = false;
    }
    else
    {
        double dc = (cap.center_w - pipe_exit_candidate_center_w).norm();
        double cosn = std::fabs(cap.normal_w.dot(pipe_exit_candidate_normal_w));
        same_candidate = (dc < pipe_exit_lock_center_gate && cosn > pipe_exit_lock_normal_cos);
    }

    if (!same_candidate)
    {
        pipe_exit_candidate_frames = 1;
        pipe_exit_candidate_center_w = cap.center_w;
        pipe_exit_candidate_normal_w = cap.normal_w;
        pipe_exit_candidate_valid = true;
    }
    else
    {
        pipe_exit_candidate_frames++;
        // Light smoothing for the candidate state, not for a locked anchor.
        pipe_exit_candidate_center_w = 0.8 * pipe_exit_candidate_center_w + 0.2 * cap.center_w;
        pipe_exit_candidate_normal_w = (0.8 * pipe_exit_candidate_normal_w + 0.2 * cap.normal_w).normalized();
    }

    if (!pipe_exit_cap_initialized && pipe_exit_candidate_frames >= std::max(1, pipe_exit_lock_min_frames))
    {
        pipe_exit_cap_center_w = pipe_exit_candidate_center_w;
        pipe_axis_anchor_w = pipe_exit_candidate_normal_w.normalized();
        pipe_exit_cap_d = -pipe_axis_anchor_w.dot(pipe_exit_cap_center_w);
        pipe_exit_cap_initialized = true;
        ROS_WARN_STREAM("[EXIT_ONLY] locked exit end-cap. frames=" << pipe_exit_candidate_frames
                        << " center=(" << pipe_exit_cap_center_w(0) << ", " << pipe_exit_cap_center_w(1) << ", " << pipe_exit_cap_center_w(2) << ")"
                        << " axis=(" << pipe_axis_anchor_w(0) << ", " << pipe_axis_anchor_w(1) << ", " << pipe_axis_anchor_w(2) << ")");
    }

    if (pipe_exit_cap_initialized)
    {
        // A flat plane is not enough to be the exit landmark. After the exit is
        // locked, the current candidate must still be close to the fixed exit
        // plane. This prevents a local wall/body plane with a small fitting
        // residual from being mislabeled as anchor=2.
        V3D exit_axis = pipe_axis_anchor_w;
        if (exit_axis.norm() < 1e-6)
        {
            cap.matched_anchor = false;
            cap.anchor_id = 0;
            cap.s_meas = 0.0;
        }
        else
        {
            exit_axis.normalize();
            double exit_plane_err = std::fabs(exit_axis.dot(cap.center_w) + pipe_exit_cap_d);
            cap.s_anchor = pipe_endcap_known_length > 0.0 ? pipe_endcap_known_length : 0.0;
            cap.s_meas = (cap.center_w - pipe_exit_cap_center_w).dot(exit_axis) + cap.s_anchor;

            if (pipe_endcap_anchor_gate > 0.0 && exit_plane_err > pipe_endcap_anchor_gate)
            {
                cap.matched_anchor = false;
                cap.anchor_id = 0;
                if (pipe_debug_log || pipe_debug_match_stats)
                {
                    ROS_WARN_STREAM_THROTTLE(0.5, "[EXIT_ONLY] reject exit candidate by anchor gate. plane_err="
                                             << exit_plane_err << " gate=" << pipe_endcap_anchor_gate
                                             << " s_meas=" << cap.s_meas << " s_ref=" << cap.s_anchor
                                             << " pts=" << cap.point_count << " res=" << cap.mean_abs_res);
                }
            }
            else
            {
                cap.matched_anchor = true;
                cap.anchor_id = 2;
                pipe_last_endcap_matched = true;
            }
        }
    }
    else
    {
        cap.matched_anchor = false;
        cap.s_meas = 0.0;
    }

    pipe_update_exit_pose_outputs(s);
}

bool fit_endcap_candidate_from_indices(const state_ikfom &s,
                                       const vector<int> &indices,
                                       const vector<V3D> &pts_world,
                                       const vector<V3D> &pts_body,
                                       const V3D &axis_w,
                                       const V3D &u_w,
                                       const V3D &v_w,
                                       int side,
                                       PipeEndCapState &cap)
{
    cap = PipeEndCapState();
    cap.side = side;

    if (int(indices.size()) < pipe_endcap_min_points) return false;

    V3D centroid = Zero3d;
    for (int idx : indices) centroid += pts_world[idx];
    centroid /= double(indices.size());

    M3D cov = M3D::Zero();
    for (int idx : indices)
    {
        V3D d = pts_world[idx] - centroid;
        cov += d * d.transpose();
    }
    cov /= std::max(1.0, double(indices.size()));

    Eigen::SelfAdjointEigenSolver<M3D> solver(cov);
    if (solver.info() != Eigen::Success) return false;

    V3D n = solver.eigenvectors().col(0).normalized();
    if (n.dot(axis_w) < 0.0) n = -n;

    double axis_cos = std::fabs(n.dot(axis_w));
    if (axis_cos < pipe_endcap_axis_cos_thresh) return false;

    double mean_abs_res = 0.0;
    vector<double> us, vs;
    us.reserve(indices.size());
    vs.reserve(indices.size());
    for (int idx : indices)
    {
        V3D d = pts_world[idx] - centroid;
        mean_abs_res += std::fabs(n.dot(d));
        us.push_back(d.dot(u_w));
        vs.push_back(d.dot(v_w));
    }
    mean_abs_res /= double(indices.size());
    if (mean_abs_res > pipe_endcap_max_plane_res) return false;

    std::sort(us.begin(), us.end());
    std::sort(vs.begin(), vs.end());
    double span_u = quantile_from_sorted(us, 0.90) - quantile_from_sorted(us, 0.10);
    double span_v = quantile_from_sorted(vs, 0.90) - quantile_from_sorted(vs, 0.10);

    // A thin axial slice of the side walls can look like a plane whose normal is
    // parallel to the pipe axis. Requiring only max(span_u, span_v) lets that
    // false end-cap pass. A real rectangular end face should have visible extent
    // in both cross-section directions.
    if (pipe_endcap_require_2d_span)
    {
        if (span_u < pipe_endcap_min_cross_span || span_v < pipe_endcap_min_cross_span) return false;
    }
    else
    {
        if (std::max(span_u, span_v) < pipe_endcap_min_cross_span) return false;
    }

    cap.visible = true;
    cap.reliable = true;
    cap.normal_w = n;
    cap.center_w = centroid;
    cap.plane_d = -n.dot(centroid);
    cap.axis_cos = axis_cos;
    cap.mean_abs_res = mean_abs_res;
    cap.span_u = span_u;
    cap.span_v = span_v;
    cap.point_count = int(indices.size());
    cap.points_body.reserve(indices.size());
    for (int idx : indices) cap.points_body.push_back(pts_body[idx]);
    return true;
}

void update_endcap_anchor(const state_ikfom &s, PipeEndCapState &cap)
{
    if (!cap.reliable) return;

    if (pipe_endcap_init_origin && !pipe_origin_initialized)
    {
        pipe_origin_w = cap.center_w;
        pipe_axis_anchor_w = cap.normal_w.normalized();
        pipe_start_cap_center_w = pipe_origin_w;
        pipe_start_cap_d = -pipe_axis_anchor_w.dot(pipe_start_cap_center_w);
        pipe_origin_initialized = true;
        pipe_start_cap_initialized = true;
        cap.anchor_id = 1;
        cap.matched_anchor = true;
        cap.s_meas = 0.0;
        cap.s_anchor = 0.0;
        ROS_WARN_STREAM("[ENDCAP] initialized entrance end-cap origin. axis=("
                        << pipe_axis_anchor_w(0) << ", " << pipe_axis_anchor_w(1) << ", " << pipe_axis_anchor_w(2) << ")");
        return;
    }

    if (!pipe_origin_initialized) return;

    cap.s_meas = (cap.center_w - pipe_origin_w).dot(pipe_axis_anchor_w);
    pipe_last_pipe_s = (s.pos + s.rot * s.offset_T_L_I - pipe_origin_w).dot(pipe_axis_anchor_w);

    double dist_start = std::fabs(cap.s_meas);
    double dist_exit = std::numeric_limits<double>::infinity();
    if (pipe_endcap_known_length > 0.0)
        dist_exit = std::fabs(cap.s_meas - pipe_endcap_known_length);
    else if (pipe_exit_cap_initialized)
        dist_exit = std::fabs((cap.center_w - pipe_exit_cap_center_w).dot(pipe_axis_anchor_w));

    if (dist_start <= dist_exit && pipe_start_cap_initialized && dist_start < pipe_endcap_anchor_gate)
    {
        cap.anchor_id = 1;
        cap.matched_anchor = true;
        cap.s_anchor = 0.0;
        return;
    }

    if (pipe_endcap_known_length > 0.0 && dist_exit < pipe_endcap_anchor_gate)
    {
        pipe_exit_cap_center_w = pipe_origin_w + pipe_axis_anchor_w * pipe_endcap_known_length;
        pipe_exit_cap_d = -pipe_axis_anchor_w.dot(pipe_exit_cap_center_w);
        pipe_exit_cap_initialized = true;
        cap.anchor_id = 2;
        cap.matched_anchor = true;
        cap.s_anchor = pipe_endcap_known_length;
        return;
    }

    if (pipe_endcap_learn_exit && pipe_endcap_known_length <= 0.0 &&
        !pipe_exit_cap_initialized && cap.s_meas > pipe_endcap_exit_learn_min_s)
    {
        pipe_exit_cap_center_w = cap.center_w;
        pipe_exit_cap_d = -pipe_axis_anchor_w.dot(pipe_exit_cap_center_w);
        pipe_exit_cap_initialized = true;
        cap.anchor_id = 2;
        cap.matched_anchor = true;
        cap.s_anchor = cap.s_meas;
        ROS_WARN_STREAM("[ENDCAP] learned exit end-cap at s=" << cap.s_anchor << " m");
        return;
    }

    if (pipe_exit_cap_initialized && dist_exit < pipe_endcap_anchor_gate)
    {
        cap.anchor_id = 2;
        cap.matched_anchor = true;
        cap.s_anchor = (pipe_exit_cap_center_w - pipe_origin_w).dot(pipe_axis_anchor_w);
    }
}

bool detect_pipe_endcap(const state_ikfom &s, const RectPipeGeomState &geom, PipeEndCapState &best_cap)
{
    best_cap = PipeEndCapState();
    if (!pipe_prior_enable || !pipe_endcap_enable || feats_down_size < pipe_endcap_min_points) return false;

    V3D lidar_forward_w = s.rot * (s.offset_R_L_I * V3D(1.0, 0.0, 0.0));
    V3D axis_w = lidar_forward_w.normalized();
    if (geom.axis_reliable) axis_w = geom.axis_w.normalized();
    else if (pipe_origin_initialized) axis_w = pipe_axis_anchor_w.normalized();
    if (axis_w.dot(lidar_forward_w) < 0.0) axis_w = -axis_w;

    V3D u_w = geom.u_w;
    V3D v_w = geom.v_w;
    if (geom.axis_reliable == false)
    {
        V3D ref = std::fabs(axis_w.z()) < 0.9 ? V3D(0, 0, 1) : V3D(0, 1, 0);
        u_w = (ref - axis_w * axis_w.dot(ref)).normalized();
        v_w = axis_w.cross(u_w).normalized();
    }

    vector<V3D> pts_world;
    vector<V3D> pts_body;
    vector<double> t_vals;
    pts_world.reserve(feats_down_size);
    pts_body.reserve(feats_down_size);
    t_vals.reserve(feats_down_size);

    V3D lidar_pos_w = s.pos + s.rot * s.offset_T_L_I;
    pipe_last_self_mask_reject_count = 0;
    for (int i = 0; i < feats_down_size; ++i)
    {
        const PointType &pb = feats_down_body->points[i];
        V3D p_body(pb.x, pb.y, pb.z);
        if (pipe_is_self_mask_point(p_body))
        {
            pipe_last_self_mask_reject_count++;
            continue;
        }
        V3D p_world = pointBodyToWorldVec(p_body, s);
        pts_body.push_back(p_body);
        pts_world.push_back(p_world);
        t_vals.push_back((p_world - lidar_pos_w).dot(axis_w));
    }
    if (int(pts_world.size()) < pipe_endcap_min_points) return false;

    vector<double> t_sorted = t_vals;
    std::sort(t_sorted.begin(), t_sorted.end());
    double t05 = quantile_from_sorted(t_sorted, 0.05);
    double t95 = quantile_from_sorted(t_sorted, 0.95);
    double span_t = t95 - t05;
    if (span_t < 1e-3) return false;

    double slice = std::max(pipe_endcap_slice_min, span_t * std::max(0.05, pipe_endcap_slice_ratio));
    vector<int> low_idx, high_idx;
    low_idx.reserve(feats_down_size);
    high_idx.reserve(feats_down_size);
    for (size_t i = 0; i < t_vals.size(); ++i)
    {
        if (t_vals[i] <= t05 + slice) low_idx.push_back(int(i));
        if (t_vals[i] >= t95 - slice) high_idx.push_back(int(i));
    }

    PipeEndCapState low_cap, high_cap;
    bool low_ok = fit_endcap_candidate_from_indices(s, low_idx, pts_world, pts_body, axis_w, u_w, v_w, -1, low_cap);
    bool high_ok = fit_endcap_candidate_from_indices(s, high_idx, pts_world, pts_body, axis_w, u_w, v_w, +1, high_cap);

    double low_score = low_ok ? (low_cap.axis_cos * low_cap.point_count / std::max(1e-3, low_cap.mean_abs_res + 0.01)) : -1.0;
    double high_score = high_ok ? (high_cap.axis_cos * high_cap.point_count / std::max(1e-3, high_cap.mean_abs_res + 0.01)) : -1.0;

    if (!low_ok && !high_ok) return false;
    best_cap = (high_score > low_score) ? high_cap : low_cap;
    update_endcap_anchor(s, best_cap);
    return best_cap.reliable;
}

// Detect both entrance/exit end-cap candidates directly from the current frame.
// This function does not depend on successful map nearest-neighbor matching, so it can run before
// the "No Effective Points" early return and can provide a fallback landmark update.

bool build_locked_exit_plane_observation(const state_ikfom &s, PipeEndCapState &cap)
{
    cap = PipeEndCapState();
    pipe_last_exit_locked_plane_points = 0;

    if (!pipe_prior_enable || !pipe_endcap_enable) return false;
    if (!pipe_exit_only_mode || !pipe_exit_locked_plane_prior_enable) return false;
    if (!pipe_exit_cap_initialized) return false;
    if (feats_down_size < 1) return false;

    V3D n = pipe_axis_anchor_w;
    if (n.norm() < 1e-6) return false;
    n.normalize();

    cap.visible = false;
    cap.reliable = false;
    cap.matched_anchor = true;
    cap.side = +1;
    cap.anchor_id = 2;
    cap.normal_w = n;
    cap.center_w = pipe_exit_cap_center_w;
    cap.plane_d = pipe_exit_cap_d;
    cap.axis_cos = 1.0;
    cap.s_anchor = pipe_endcap_known_length > 0.0 ? pipe_endcap_known_length : 0.0;
    cap.s_meas = cap.s_anchor;

    int selfrej_local = 0;
    double abs_sum = 0.0;
    vector<V3D> accepted_body;
    accepted_body.reserve(feats_down_size);

    for (int i = 0; i < feats_down_size; ++i)
    {
        const PointType &pb = feats_down_body->points[i];
        V3D p_body(pb.x, pb.y, pb.z);
        if (pipe_is_self_mask_point(p_body))
        {
            selfrej_local++;
            continue;
        }

        V3D p_world = pointBodyToWorldVec(p_body, s);
        double r = n.dot(p_world) + pipe_exit_cap_d;
        if (std::fabs(r) > pipe_exit_prior_point_gate) continue;

        accepted_body.push_back(p_body);
        abs_sum += std::fabs(r);
    }

    // Keep the larger self-mask rejection count for the frame, since the regular
    // end-cap detector also reports this value.
    pipe_last_self_mask_reject_count = std::max(pipe_last_self_mask_reject_count, selfrej_local);
    pipe_last_exit_locked_plane_points = int(accepted_body.size());

    if (int(accepted_body.size()) < std::max(1, pipe_exit_prior_min_points))
    {
        return false;
    }

    cap.points_body.swap(accepted_body);
    cap.point_count = int(cap.points_body.size());
    cap.mean_abs_res = cap.point_count > 0 ? abs_sum / cap.point_count : 0.0;
    cap.visible = true;
    cap.reliable = true;
    return true;
}

bool detect_pipe_endcaps(const state_ikfom &s,
                         const RectPipeGeomState &geom,
                         vector<PipeEndCapState> &caps)
{
    caps.clear();
    if (!pipe_prior_enable || !pipe_endcap_enable || feats_down_size < pipe_endcap_min_points) return false;

    V3D lidar_forward_w = s.rot * (s.offset_R_L_I * V3D(1.0, 0.0, 0.0));
    if (lidar_forward_w.norm() < 1e-6) lidar_forward_w = V3D(1.0, 0.0, 0.0);
    V3D axis_w = lidar_forward_w.normalized();

    // In exit-only mode, do not use an entrance anchor at all. Prefer the current pipe axis
    // when it is reliable; otherwise use the LiDAR forward direction. Once the exit is locked,
    // use the locked exit axis only for distance output and prior residuals.
    if (!pipe_exit_only_mode)
    {
        if (pipe_origin_initialized) axis_w = pipe_axis_anchor_w.normalized();
        else if (geom.axis_reliable) axis_w = geom.axis_w.normalized();
        else if (pipe_global_axis_initialized) axis_w = pipe_global_axis_w.normalized();
    }
    else
    {
        // In exit-only mode, once the exit landmark is locked, keep using the
        // locked exit axis for end-cap slicing. Otherwise the second-stage pipe
        // geometry axis can jitter or flip, causing the detected end-cap used
        // for logging to differ from the one used for EKF augmentation.
        if (pipe_exit_cap_initialized && pipe_axis_anchor_w.norm() > 1e-6)
            axis_w = pipe_axis_anchor_w.normalized();
        else if (geom.axis_reliable)
            axis_w = geom.axis_w.normalized();
        else if (pipe_global_axis_initialized)
            axis_w = pipe_global_axis_w.normalized();
    }

    if (axis_w.dot(lidar_forward_w) < 0.0) axis_w = -axis_w;

    V3D u_w = geom.u_w;
    V3D v_w = geom.v_w;
    if (!geom.axis_reliable)
    {
        V3D ref = std::fabs(axis_w.z()) < 0.9 ? V3D(0, 0, 1) : V3D(0, 1, 0);
        u_w = ref - axis_w * axis_w.dot(ref);
        if (u_w.norm() < 1e-6) u_w = V3D(0, 1, 0);
        u_w.normalize();
        v_w = axis_w.cross(u_w);
        if (v_w.norm() < 1e-6) return false;
        v_w.normalize();
    }

    vector<V3D> pts_world;
    vector<V3D> pts_body;
    vector<double> t_vals;
    pts_world.reserve(feats_down_size);
    pts_body.reserve(feats_down_size);
    t_vals.reserve(feats_down_size);

    V3D lidar_pos_w = s.pos + s.rot * s.offset_T_L_I;
    pipe_last_self_mask_reject_count = 0;
    for (int i = 0; i < feats_down_size; ++i)
    {
        const PointType &pb = feats_down_body->points[i];
        V3D p_body(pb.x, pb.y, pb.z);
        if (pipe_is_self_mask_point(p_body))
        {
            pipe_last_self_mask_reject_count++;
            continue;
        }
        V3D p_world = pointBodyToWorldVec(p_body, s);
        pts_body.push_back(p_body);
        pts_world.push_back(p_world);
        t_vals.push_back((p_world - lidar_pos_w).dot(axis_w));
    }
    if (int(pts_world.size()) < pipe_endcap_min_points) return false;

    vector<double> t_sorted = t_vals;
    std::sort(t_sorted.begin(), t_sorted.end());
    double t05 = quantile_from_sorted(t_sorted, 0.05);
    double t95 = quantile_from_sorted(t_sorted, 0.95);
    double span_t = t95 - t05;
    if (span_t < 1e-3) return false;

    double slice = std::max(pipe_endcap_slice_min, span_t * std::max(0.05, pipe_endcap_slice_ratio));
    vector<int> low_idx, high_idx;
    low_idx.reserve(feats_down_size);
    high_idx.reserve(feats_down_size);
    for (size_t i = 0; i < t_vals.size(); ++i)
    {
        if (t_vals[i] <= t05 + slice) low_idx.push_back(int(i));
        if (t_vals[i] >= t95 - slice) high_idx.push_back(int(i));
    }

    PipeEndCapState low_cap, high_cap;
    bool low_ok = fit_endcap_candidate_from_indices(s, low_idx, pts_world, pts_body, axis_w, u_w, v_w, -1, low_cap);
    bool high_ok = fit_endcap_candidate_from_indices(s, high_idx, pts_world, pts_body, axis_w, u_w, v_w, +1, high_cap);

    if (pipe_exit_only_mode)
    {
        // Only the forward/high axial slice is allowed to become the exit landmark.
        // The lower/back slice is visual clutter for a dual-body robot and is ignored.
        if (!high_ok)
        {
            pipe_update_exit_pose_outputs(s);
            return false;
        }

        double forward_dist = (high_cap.center_w - lidar_pos_w).dot(axis_w);
        if (pipe_exit_only_front_side && forward_dist < pipe_exit_min_forward_dist)
        {
            pipe_update_exit_pose_outputs(s);
            return false;
        }
        if (pipe_exit_max_forward_dist > 0.0 && forward_dist > pipe_exit_max_forward_dist)
        {
            pipe_update_exit_pose_outputs(s);
            return false;
        }

        update_exit_only_anchor(s, high_cap, axis_w);
        caps.push_back(high_cap);
        return true;
    }

    if (!low_ok && !high_ok) return false;

    // Legacy two-end mode kept for comparison, but not recommended for the dual-body robot.
    if (!pipe_origin_initialized && pipe_endcap_init_origin)
    {
        PipeEndCapState *entry = nullptr;
        PipeEndCapState *other = nullptr;
        if (pipe_endcap_entry_side <= 0)
        {
            if (low_ok) entry = &low_cap;
            if (high_ok) other = &high_cap;
        }
        else
        {
            if (high_ok) entry = &high_cap;
            if (low_ok) other = &low_cap;
        }
        if (!entry)
        {
            if (low_ok) entry = &low_cap;
            else if (high_ok) entry = &high_cap;
        }
        if (entry) update_endcap_anchor(s, *entry);
        if (other && other->reliable) update_endcap_anchor(s, *other);
    }
    else
    {
        if (low_ok) update_endcap_anchor(s, low_cap);
        if (high_ok) update_endcap_anchor(s, high_cap);
    }

    if (low_ok) caps.push_back(low_cap);
    if (high_ok) caps.push_back(high_cap);

    return !caps.empty();
}

PipeEndCapState choose_best_endcap_for_log(const vector<PipeEndCapState> &caps)
{
    PipeEndCapState best;
    double best_score = -1.0;
    for (const auto &cap : caps)
    {
        if (!cap.reliable) continue;
        double anchor_bonus = cap.matched_anchor ? 10.0 : 1.0;
        double score = anchor_bonus * cap.axis_cos * double(cap.point_count) / std::max(1e-3, cap.mean_abs_res + 0.01);
        if (score > best_score)
        {
            best_score = score;
            best = cap;
        }
    }
    return best;
}

bool append_endcap_rows(const state_ikfom &s,
                        const PipeEndCapState &cap,
                        int cols,
                        double weight,
                        double gate,
                        vector<RowVectorXd> &rows,
                        vector<double> &residuals)
{
    if (!cap.reliable || cap.points_body.empty())
    {
        pipe_last_endcap_invalid_reject++;
        pipe_last_endcap_reject_code = 1;
        return false;
    }
    if (!cap.matched_anchor)
    {
        pipe_last_endcap_unmatched_reject++;
        pipe_last_endcap_reject_code = 2;
        return false;
    }
    if (weight <= 0.0 || cols < 6)
    {
        pipe_last_endcap_reject_code = 3;
        return false;
    }
    if (pipe_exit_only_mode && cap.anchor_id != 2)
    {
        pipe_last_endcap_reject_code = 4;
        return false;
    }
    if (pipe_exit_only_mode && pipe_exit_prior_requires_lock && !pipe_exit_cap_initialized)
    {
        pipe_last_endcap_lock_reject++;
        pipe_last_endcap_reject_code = 5;
        return false;
    }

    V3D n = pipe_axis_anchor_w.normalized();
    double d = pipe_start_cap_d;
    if (cap.anchor_id == 2) d = pipe_exit_cap_d;
    if (cap.anchor_id == 0) return false;

    int max_pts = std::max(1, pipe_endcap_max_prior_points);
    int step = std::max(1, int(cap.points_body.size()) / max_pts);
    int before = int(rows.size());

    for (size_t i = 0; i < cap.points_body.size(); i += step)
    {
        const V3D &p_body = cap.points_body[i];
        V3D p_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
        V3D p_world = s.rot * p_imu + s.pos;
        double r = n.dot(p_world) + d;
        if (gate > 0.0 && std::fabs(r) > gate)
        {
            pipe_last_endcap_gate_reject++;
            continue;
        }

        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(p_imu);
        V3D C(s.rot.conjugate() * n);
        V3D A(point_crossmat * C);

        RowVectorXd row = RowVectorXd::Zero(cols);
        if (extrinsic_est_en && cols >= 12)
        {
            M3D point_body_crossmat;
            point_body_crossmat << SKEW_SYM_MATRX(p_body);
            V3D B(point_body_crossmat * s.offset_R_L_I.conjugate() * C);
            row.block<1, 12>(0, 0) << n(0), n(1), n(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            row.block<1, 3>(0, 0) = n.transpose();
            row.block<1, 3>(0, 3) = A.transpose();
        }
        rows.push_back(weight * row);
        residuals.push_back(-weight * r);
    }

    if (int(rows.size()) <= before)
    {
        pipe_last_endcap_reject_code = 6;
        return false;
    }
    pipe_last_endcap_reject_code = 0;
    return true;
}

bool build_endcap_fallback_measurement(const state_ikfom &s,
                                       const vector<PipeEndCapState> &caps,
                                       MatrixXd &Hx,
                                       VectorXd &h)
{
    if (!pipe_prior_enable || !pipe_endcap_enable || !pipe_endcap_fallback_enable) return false;
    if (pipe_exit_only_mode && pipe_exit_fallback_requires_lock && !pipe_exit_cap_initialized) return false;
    if (caps.empty()) return false;

    vector<RowVectorXd> rows;
    vector<double> residuals;
    rows.reserve(pipe_endcap_max_prior_points * std::max(1, int(caps.size())));
    residuals.reserve(pipe_endcap_max_prior_points * std::max(1, int(caps.size())));

    const int cols = 12;
    for (const auto &cap : caps)
        append_endcap_rows(s, cap, cols, pipe_endcap_weight, pipe_endcap_fallback_gate, rows, residuals);

    pipe_last_endcap_fallback_rows = int(rows.size());
    pipe_last_endcap_fallback_used = !rows.empty();
    if (rows.empty()) return false;

    Hx = MatrixXd::Zero(int(rows.size()), cols);
    h.resize(int(rows.size()));
    for (int i = 0; i < int(rows.size()); ++i)
    {
        Hx.row(i) = rows[i];
        h(i) = residuals[i];
    }
    return true;
}


void augment_with_endcap_prior(const state_ikfom &s,
                               const PipeEndCapState &cap,
                               MatrixXd &Hx,
                               VectorXd &h)
{
    if (!pipe_prior_enable || !pipe_endcap_enable || !pipe_endcap_use_prior) return;
    if (pipe_endcap_weight <= 0.0) return;
    if (pipe_endcap_require_lio_healthy && effct_feat_num < std::max(1, pipe_endcap_min_lio_eff)) return;

    vector<RowVectorXd> rows;
    vector<double> residuals;
    rows.reserve(pipe_endcap_max_prior_points);
    residuals.reserve(pipe_endcap_max_prior_points);

    if (!append_endcap_rows(s, cap, Hx.cols(), pipe_endcap_weight, pipe_endcap_anchor_gate, rows, residuals)) return;

    int old_rows = Hx.rows();
    int add_rows = rows.size();
    pipe_last_endcap_prior_rows += add_rows;
    MatrixXd Hx_aug(old_rows + add_rows, Hx.cols());
    VectorXd h_aug(old_rows + add_rows);
    Hx_aug.topRows(old_rows) = Hx;
    h_aug.head(old_rows) = h;
    for (int i = 0; i < add_rows; ++i)
    {
        Hx_aug.row(old_rows + i) = rows[i];
        h_aug(old_rows + i) = residuals[i];
    }
    Hx.swap(Hx_aug);
    h.swap(h_aug);
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    int dbg_total = 0;
    int dbg_nn_ok = 0;
    int dbg_plane_ok = 0;
    int dbg_score_ok = 0;
    int dbg_abs_gate_reject = 0;
    int dbg_inc_reject = 0;
    int dbg_final = 0;
    int dbg_pd2_abs_lt_005 = 0;
    int dbg_pd2_abs_lt_010 = 0;
    int dbg_pd2_abs_lt_020 = 0;
    double dbg_pd2_abs_sum = 0.0;
    double dbg_pd2_abs_sum_score_ok = 0.0;
    pipe_last_endcap_candidate_count = 0;
    pipe_last_endcap_prior_rows = 0;
    pipe_last_endcap_fallback_rows = 0;
    pipe_last_endcap_fallback_used = false;
    pipe_last_endcap_matched = false;
    pipe_last_endcap_unmatched_reject = 0;
    pipe_last_endcap_lock_reject = 0;
    pipe_last_endcap_gate_reject = 0;
    pipe_last_endcap_invalid_reject = 0;
    pipe_last_endcap_reject_code = 0;
    pipe_last_exit_locked_plane_points = 0;
    const float dbg_score_gate = static_cast<float>(std::max(0.0, std::min(1.0, pipe_plane_score_gate)));
    const float dbg_abs_pd2_gate = static_cast<float>(pipe_plane_abs_pd2_gate);

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for reduction(+:dbg_total, dbg_nn_ok, dbg_plane_ok, dbg_score_ok, dbg_abs_gate_reject, dbg_inc_reject, dbg_final, dbg_pd2_abs_lt_005, dbg_pd2_abs_lt_010, dbg_pd2_abs_lt_020, dbg_pd2_abs_sum, dbg_pd2_abs_sum_score_ok)
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        dbg_total++;

        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (point_selected_surf[i]) dbg_nn_ok++;
        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            dbg_plane_ok++;

            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float abs_pd2 = fabs(pd2);
            float score = 1 - 0.9 * abs_pd2 / sqrt(p_body.norm());

            dbg_pd2_abs_sum += abs_pd2;
            if (abs_pd2 < 0.05f) dbg_pd2_abs_lt_005++;
            if (abs_pd2 < 0.10f) dbg_pd2_abs_lt_010++;
            if (abs_pd2 < 0.20f) dbg_pd2_abs_lt_020++;

            // Absolute residual gate for narrow rectangular pipes.
            // The original score gate is range-dependent and becomes too loose
            // at 0.5 to 1.0 m. This hard gate rejects thick-wall and ghost matches.
            if (dbg_abs_pd2_gate > 0.0f && abs_pd2 > dbg_abs_pd2_gate)
            {
                dbg_abs_gate_reject++;
                continue;
            }

            if (score > dbg_score_gate)
            {
                dbg_score_ok++;
                dbg_pd2_abs_sum_score_ok += abs_pd2;

                V3D plane_n(pabcd(0), pabcd(1), pabcd(2));
                double plane_n_norm = plane_n.norm();
                if (plane_n_norm < 1e-6) continue;
                plane_n /= plane_n_norm;

                if (pipe_apply_incidence_filter_in_lio)
                {
                    V3D lidar_pos_w = s.pos + s.rot * s.offset_T_L_I;
                    V3D ray_dir = p_global - lidar_pos_w;
                    double ray_norm = ray_dir.norm();
                    if (ray_norm < 1e-6) continue;
                    double cos_inc = std::fabs(ray_dir.dot(plane_n) / ray_norm);
                    if (cos_inc < pipe_min_incidence_cos)
                    {
                        dbg_inc_reject++;
                        continue;
                    }
                }

                point_selected_surf[i] = true;
                dbg_final++;
                normvec->points[i].x = plane_n(0);
                normvec->points[i].y = plane_n(1);
                normvec->points[i].z = plane_n(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (pipe_debug_match_stats || pipe_debug_log || effct_feat_num < 1)
    {
        double plane_mean_abs_pd2 = dbg_plane_ok > 0 ? dbg_pd2_abs_sum / dbg_plane_ok : 0.0;
        double score_ok_mean_abs_pd2 = dbg_score_ok > 0 ? dbg_pd2_abs_sum_score_ok / dbg_score_ok : 0.0;
        double plane_ratio = dbg_total > 0 ? static_cast<double>(dbg_plane_ok) / dbg_total : 0.0;
        double score_ratio_in_plane = dbg_plane_ok > 0 ? static_cast<double>(dbg_score_ok) / dbg_plane_ok : 0.0;

        ROS_WARN_STREAM("[DBG_MATCH] feats_down_size=" << feats_down_size
                        << " total=" << dbg_total
                        << " nn_ok=" << dbg_nn_ok
                        << " plane_ok=" << dbg_plane_ok
                        << " score_ok=" << dbg_score_ok
                        << " abs_gate_reject=" << dbg_abs_gate_reject
                        << " inc_reject=" << dbg_inc_reject
                        << " final=" << dbg_final
                        << " eff=" << effct_feat_num
                        << " kdtree_size=" << ikdtree.size()
                        << " score_gate=" << dbg_score_gate
                        << " abs_pd2_gate=" << dbg_abs_pd2_gate
                        << " plane_ratio=" << plane_ratio
                        << " score_ratio_in_plane=" << score_ratio_in_plane
                        << " mean_abs_pd2=" << plane_mean_abs_pd2
                        << " mean_abs_pd2_score_ok=" << score_ok_mean_abs_pd2
                        << " pd2_lt_0p05=" << dbg_pd2_abs_lt_005
                        << " pd2_lt_0p10=" << dbg_pd2_abs_lt_010
                        << " pd2_lt_0p20=" << dbg_pd2_abs_lt_020);
    }

    // Update pipe-coordinate outputs before geometry fitting. The stable W/H
    // filter uses pipe_last_pipe_s in exit-only mode.
    if (pipe_exit_only_mode)
    {
        pipe_update_exit_pose_outputs(s);
        pipe_anchor_length_for_size();
    }
    else if (pipe_origin_initialized)
    {
        V3D lidar_pos_w_for_size = s.pos + s.rot * s.offset_T_L_I;
        pipe_last_pipe_s = (lidar_pos_w_for_size - pipe_origin_w).dot(pipe_axis_anchor_w);
        pipe_last_pipe_s_valid = true;
        pipe_anchor_length_for_size();
    }

    // Fit current pipe geometry before end-cap detection so the end-cap slicer
    // uses the current frame axis when enough reliable LIO rows exist. The
    // current-frame measurement log is reset every iteration. If fitting fails,
    // keep g_pipe_geom only as an internal cached axis for end-cap detection, but
    // do not log that cached value as a new frame measurement.
    pipe_reset_frame_geom_observation();
    RectPipeGeomState geom_for_endcap = g_pipe_geom;
    if (effct_feat_num >= 20)
    {
        RectPipeGeomState current_geom = g_pipe_geom;
        if (fit_rect_pipe_geometry(s, current_geom))
        {
            g_pipe_geom = current_geom;
            geom_for_endcap = g_pipe_geom;
        }
    }
    else
    {
        pipe_last_geom_fail_code = 1;
    }

    // Detect the current-frame exit end-cap once and reuse this same observation
    // for logging, RViz debug, fallback, and normal EKF prior augmentation.
    vector<PipeEndCapState> frame_endcaps;
    detect_pipe_endcaps(s, geom_for_endcap, frame_endcaps);
    if (!frame_endcaps.empty())
        g_endcap = choose_best_endcap_for_log(frame_endcaps);
    else
        g_endcap = PipeEndCapState();
    // Once the outlet landmark is locked, also collect points near the fixed
    // outlet plane. This cap is appended to the same frame_endcaps vector, so
    // logging, RViz, fallback, and normal EKF prior all see the same observation.
    if (pipe_exit_only_mode && pipe_exit_cap_initialized && pipe_exit_locked_plane_prior_enable)
    {
        PipeEndCapState locked_exit_cap;
        if (build_locked_exit_plane_observation(s, locked_exit_cap))
        {
            frame_endcaps.push_back(locked_exit_cap);
            g_endcap = locked_exit_cap;
        }
    }
    pipe_last_endcap_candidate_count = int(frame_endcaps.size());
    pipe_last_endcap_matched = g_endcap.matched_anchor;
    pipe_update_position_outputs_with_cap(s, g_endcap);
    update_pipe_debug_endcap_cloud(s, frame_endcaps);

    if (effct_feat_num < 1)
    {
        if (build_endcap_fallback_measurement(s, frame_endcaps, ekfom_data.h_x, ekfom_data.h))
        {
            ekfom_data.valid = true;
            res_mean_last = 0.0;
            if (pipe_exit_only_mode)
            {
                pipe_update_exit_pose_outputs(s);
            }
            else if (pipe_origin_initialized)
            {
                V3D lidar_pos_w = s.pos + s.rot * s.offset_T_L_I;
                pipe_last_pipe_s = (lidar_pos_w - pipe_origin_w).dot(pipe_axis_anchor_w);
                pipe_last_pipe_s_valid = true;
            }
            if (pipe_debug_log || pipe_debug_match_stats)
            {
                ROS_WARN_STREAM("[ENDCAP_FALLBACK] using " << ekfom_data.h_x.rows()
                                << " end-cap rows. anchor=" << g_endcap.anchor_id
                                << " pts=" << g_endcap.point_count
                                << " s=" << g_endcap.s_meas
                                << " s_ref=" << g_endcap.s_anchor
                                << " res=" << g_endcap.mean_abs_res
                                << " pipe_s=" << pipe_last_pipe_s
                        << " pipe_s_valid=" << int(pipe_last_pipe_s_valid)
                        << " dist_exit=" << pipe_last_distance_to_exit);
            }
            return;
        }

        ekfom_data.valid = false;
        ROS_WARN_STREAM("No Effective Points! endcap_candidates=" << pipe_last_endcap_candidate_count
                        << " endcap_rows=" << pipe_last_endcap_prior_rows
                        << " fallback_rows=" << pipe_last_endcap_fallback_rows
                        << " fallback_used=" << int(pipe_last_endcap_fallback_used)
                        << " selfrej=" << pipe_last_self_mask_reject_count
                        << " entry_locked=" << int(pipe_start_cap_initialized)
                        << " exit_locked=" << int(pipe_exit_cap_initialized)
                        << " anchor=" << g_endcap.anchor_id
                        << " pts=" << g_endcap.point_count
                        << " res=" << g_endcap.mean_abs_res
                        << " pipe_s=" << pipe_last_pipe_s
                        << " pipe_s_valid=" << int(pipe_last_pipe_s_valid)
                        << " dist_exit=" << pipe_last_distance_to_exit);
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }

    // ---- Rectangular pipe degeneracy detection, end-cap landmark detection, and prior augmentation ----
    if (pipe_exit_only_mode)
    {
        pipe_update_exit_pose_outputs(s);
        pipe_anchor_length_for_size();
    }
    else if (pipe_origin_initialized)
    {
        V3D lidar_pos_w_for_size = s.pos + s.rot * s.offset_T_L_I;
        pipe_last_pipe_s = (lidar_pos_w_for_size - pipe_origin_w).dot(pipe_axis_anchor_w);
        pipe_last_pipe_s_valid = true;
        pipe_anchor_length_for_size();
    }
    // g_pipe_geom was already fitted before end-cap detection in this call.
    // Do not fit it a second time here, otherwise the observation model can
    // change inside the same EKF iteration.
    analyze_degeneracy(ekfom_data.h_x, g_pipe_geom);
    augment_with_pipe_priors(s, g_pipe_geom, ekfom_data.h_x, ekfom_data.h);
    for (const auto &cap : frame_endcaps)
        augment_with_endcap_prior(s, cap, ekfom_data.h_x, ekfom_data.h);

    if (pipe_exit_only_mode)
    {
        pipe_update_exit_pose_outputs(s);
    }
    else if (pipe_origin_initialized)
    {
        V3D lidar_pos_w = s.pos + s.rot * s.offset_T_L_I;
        pipe_last_pipe_s = (lidar_pos_w - pipe_origin_w).dot(pipe_axis_anchor_w);
        pipe_last_pipe_s_valid = true;
    }

    if (pipe_debug_log)
    {
        ROS_INFO_THROTTLE(0.5,
                          "pipe_size current_valid=%d width=%.3f height=%.3f width_pairs=%d height_pairs=%d axis=%d geom_up=%d fail=%d pose_s=%.3f s_odom=%.3f d_scan=%.3f L_est=%.3f robot_pose=%d",
                          int(pipe_last_constrained_size_valid),
                          pipe_last_constrained_width,
                          pipe_last_constrained_height,
                          pipe_last_constrained_width_pair_count,
                          pipe_last_constrained_height_pair_count,
                          int(pipe_last_frame_axis_reliable),
                          int(pipe_last_geom_updated),
                          pipe_last_geom_fail_code,
                          pipe_last_pipe_s,
                          pipe_last_s_odom,
                          pipe_last_d_exit_scan,
                          pipe_length_est,
                          int(robot_pose_origin_enable && robot_origin_inited));
    }


    solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<bool>("pipe_prior/enable", pipe_prior_enable, true);
    nh.param<bool>("pipe_prior/debug_log", pipe_debug_log, false);
    nh.param<bool>("pipe_prior/geometry_prior_enable", pipe_geometry_prior_enable, false);
    nh.param<bool>("pipe_prior/endcap_require_lio_healthy", pipe_endcap_require_lio_healthy, true);
    nh.param<int>("pipe_prior/endcap_min_lio_eff", pipe_endcap_min_lio_eff, 30);
    nh.param<bool>("pipe_prior/endcap_require_2d_span", pipe_endcap_require_2d_span, true);
    nh.param<double>("pipe_prior/exit_max_forward_dist", pipe_exit_max_forward_dist, 1.50);
    nh.param<int>("pipe_prior/map_filter_min_eff", pipe_map_filter_min_eff, 20);
    nh.param<double>("pipe_prior/weight_box", pipe_prior_weight, 8.0);
    nh.param<double>("pipe_prior/weight_axis_align", pipe_axis_align_weight, 3.0);
    nh.param<double>("pipe_prior/weight_motion", pipe_motion_weight, 1.5);
    nh.param<double>("pipe_prior/min_width", pipe_min_width, 0.06);
    nh.param<double>("pipe_prior/max_width", pipe_max_width, 1.00);
    nh.param<double>("pipe_prior/min_height", pipe_min_height, 0.06);
    nh.param<double>("pipe_prior/max_height", pipe_max_height, 1.00);
    nh.param<double>("pipe_prior/axis_conf_threshold", pipe_axis_conf_threshold, 0.35);
    nh.param<double>("pipe_prior/degenerate_min_eig", pipe_degenerate_min_eig, 1e-4);
    nh.param<double>("pipe_prior/degenerate_ratio", pipe_degenerate_ratio, 0.20);
    nh.param<double>("pipe_prior/degenerate_cond_thresh", pipe_degenerate_cond_thresh, 150.0);
    nh.param<double>("pipe_prior/degenerate_min_pos_rel", pipe_degenerate_min_pos_rel, 0.10);
    nh.param<int>("pipe_prior/degenerate_hold_frames", pipe_degenerate_hold_frames, 3);
    nh.param<double>("pipe_prior/mid_section_keep_ratio", pipe_mid_section_keep_ratio, 0.60);
    nh.param<double>("pipe_prior/plane_score_gate", pipe_plane_score_gate, 0.70);
    nh.param<double>("pipe_prior/plane_abs_pd2_gate", pipe_plane_abs_pd2_gate, 0.05);
    nh.param<bool>("pipe_prior/map_filter_by_residual", pipe_map_filter_by_residual, true);
    nh.param<bool>("pipe_prior/map_use_selected_points", pipe_map_use_selected_points, true);
    nh.param<double>("pipe_prior/map_max_pd2", pipe_map_max_pd2, 0.04);
    nh.param<double>("pipe_prior/min_incidence_cos", pipe_min_incidence_cos, 0.20);
    nh.param<bool>("pipe_prior/apply_incidence_filter_in_lio", pipe_apply_incidence_filter_in_lio, true);
    nh.param<bool>("pipe_prior/debug_match_stats", pipe_debug_match_stats, false);
    nh.param<bool>("pipe_prior/enable_intensity_trim", pipe_enable_intensity_trim, true);
    nh.param<double>("pipe_prior/intensity_quantile_low", pipe_intensity_quantile_low, 0.05);
    nh.param<double>("pipe_prior/intensity_quantile_high", pipe_intensity_quantile_high, 0.95);
    nh.param<bool>("pipe_prior/debug_print_filter_stats", pipe_debug_print_filter_stats, false);
    nh.param<bool>("pipe_prior/endcap_enable", pipe_endcap_enable, true);
    nh.param<bool>("pipe_prior/endcap_init_origin", pipe_endcap_init_origin, true);
    nh.param<bool>("pipe_prior/endcap_use_prior", pipe_endcap_use_prior, true);
    nh.param<bool>("pipe_prior/endcap_learn_exit", pipe_endcap_learn_exit, true);
    nh.param<double>("pipe_prior/endcap_weight", pipe_endcap_weight, 2.0);
    nh.param<int>("pipe_prior/endcap_min_points", pipe_endcap_min_points, 25);
    nh.param<int>("pipe_prior/endcap_max_prior_points", pipe_endcap_max_prior_points, 80);
    nh.param<double>("pipe_prior/endcap_axis_cos_thresh", pipe_endcap_axis_cos_thresh, 0.85);
    nh.param<double>("pipe_prior/endcap_max_plane_res", pipe_endcap_max_plane_res, 0.03);
    nh.param<double>("pipe_prior/endcap_slice_ratio", pipe_endcap_slice_ratio, 0.18);
    nh.param<double>("pipe_prior/endcap_slice_min", pipe_endcap_slice_min, 0.06);
    nh.param<double>("pipe_prior/endcap_min_cross_span", pipe_endcap_min_cross_span, 0.08);
    nh.param<double>("pipe_prior/endcap_anchor_gate", pipe_endcap_anchor_gate, 0.35);
    nh.param<double>("pipe_prior/endcap_exit_learn_min_s", pipe_endcap_exit_learn_min_s, 0.80);
    nh.param<double>("pipe_prior/endcap_known_length", pipe_endcap_known_length, -1.0);
    nh.param<int>("pipe_prior/position_output_mode", pipe_position_output_mode, 2);
    nh.param<double>("pipe_prior/initial_lidar_s", pipe_initial_lidar_s, 0.0);
    nh.param<bool>("pipe_prior/scan_exit_distance_enable", pipe_scan_exit_distance_enable, true);
    nh.param<double>("pipe_prior/position_axis_lidar_x", pipe_position_axis_lidar_x, 1.0);
    nh.param<double>("pipe_prior/position_axis_lidar_y", pipe_position_axis_lidar_y, 0.0);
    nh.param<double>("pipe_prior/position_axis_lidar_z", pipe_position_axis_lidar_z, 0.0);
    nh.param<bool>("pipe_prior/length_estimate_enable", pipe_length_estimate_enable, true);
    nh.param<double>("pipe_prior/length_update_alpha", pipe_length_update_alpha, 0.05);
    nh.param<double>("pipe_prior/length_update_gate", pipe_length_update_gate, 0.20);
    nh.param<double>("pipe_prior/length_max", pipe_length_max, 20.0);
    nh.param<bool>("pipe_prior/endcap_fallback_enable", pipe_endcap_fallback_enable, true);
    nh.param<double>("pipe_prior/endcap_fallback_gate", pipe_endcap_fallback_gate, 0.60);
    nh.param<int>("pipe_prior/endcap_entry_side", pipe_endcap_entry_side, -1);
    nh.param<bool>("pipe_prior/exit_only_mode", pipe_exit_only_mode, true);
    nh.param<bool>("pipe_prior/exit_only_front_side", pipe_exit_only_front_side, true);
    nh.param<double>("pipe_prior/exit_min_forward_dist", pipe_exit_min_forward_dist, 0.20);
    nh.param<int>("pipe_prior/exit_lock_min_frames", pipe_exit_lock_min_frames, 5);
    nh.param<double>("pipe_prior/exit_lock_center_gate", pipe_exit_lock_center_gate, 0.10);
    nh.param<double>("pipe_prior/exit_lock_normal_cos", pipe_exit_lock_normal_cos, 0.95);
    nh.param<bool>("pipe_prior/exit_prior_requires_lock", pipe_exit_prior_requires_lock, true);
    nh.param<bool>("pipe_prior/exit_fallback_requires_lock", pipe_exit_fallback_requires_lock, true);
    nh.param<bool>("pipe_prior/exit_locked_plane_prior_enable", pipe_exit_locked_plane_prior_enable, true);
    nh.param<double>("pipe_prior/exit_prior_point_gate", pipe_exit_prior_point_gate, 0.05);
    nh.param<int>("pipe_prior/exit_prior_min_points", pipe_exit_prior_min_points, 6);
    nh.param<bool>("pipe_prior/debug_publish_endcap_points", pipe_debug_publish_endcap_points, true);
    nh.param<bool>("pipe_prior/self_mask_enable", pipe_self_mask_enable, false);
    nh.param<double>("pipe_prior/self_mask_x_min", pipe_self_mask_x_min, -0.70);
    nh.param<double>("pipe_prior/self_mask_x_max", pipe_self_mask_x_max, -0.05);
    nh.param<double>("pipe_prior/self_mask_y_min", pipe_self_mask_y_min, -0.18);
    nh.param<double>("pipe_prior/self_mask_y_max", pipe_self_mask_y_max,  0.18);
    nh.param<double>("pipe_prior/self_mask_z_min", pipe_self_mask_z_min, -0.15);
    nh.param<double>("pipe_prior/self_mask_z_max", pipe_self_mask_z_max,  0.15);
    nh.param<bool>("pipe_prior/size_stabilize_enable", pipe_size_stabilize_enable, true);
    nh.param<bool>("pipe_prior/size_exclude_endcap_points", pipe_size_exclude_endcap_points, false);
    nh.param<double>("pipe_prior/size_endcap_exclusion", pipe_size_endcap_exclusion, 0.15);
    nh.param<double>("pipe_prior/size_update_s_min_ratio", pipe_size_update_s_min_ratio, 0.20);
    nh.param<double>("pipe_prior/size_update_s_max_ratio", pipe_size_update_s_max_ratio, 0.70);
    nh.param<int>("pipe_prior/size_window", pipe_size_window, 10);
    nh.param<int>("pipe_prior/size_min_stable_samples", pipe_size_min_stable_samples, 5);
    nh.param<double>("pipe_prior/size_jump_gate", pipe_size_jump_gate, 0.08);
    nh.param<string>("pipe_prior/size_average_mode", pipe_size_average_mode, string("median"));
    nh.param<double>("pipe_prior/size_trim_ratio", pipe_size_trim_ratio, 0.10);
    nh.param<bool>("pipe_prior/size_require_anchor", pipe_size_require_anchor, true);
    nh.param<bool>("pipe_prior/size_use_gravity_height", pipe_size_use_gravity_height, true);
    nh.param<bool>("pipe_prior/size_wall_history_enable", pipe_size_wall_history_enable, true);
    nh.param<int>("pipe_prior/size_wall_min_side_samples", pipe_size_wall_min_side_samples, 5);
    nh.param<int>("pipe_prior/size_min_axis_points", pipe_size_min_axis_points, 20);
    nh.param<double>("pipe_prior/size_wall_center_q_low", pipe_size_wall_center_q_low, 0.15);
    nh.param<double>("pipe_prior/size_wall_center_q_high", pipe_size_wall_center_q_high, 0.85);
    nh.param<double>("pipe_prior/size_span_q_low", pipe_size_span_q_low, 0.03);
    nh.param<double>("pipe_prior/size_span_q_high", pipe_size_span_q_high, 0.97);
    nh.param<double>("pipe_prior/size_wall_min_normal_cos", pipe_size_wall_min_normal_cos, 0.35);
    nh.param<bool>("pipe_prior/size_accept_partial_pairs", pipe_size_accept_partial_pairs, true);
    nh.param<bool>("pipe_prior/size_update_requires_axis", pipe_size_update_requires_axis, false);
    nh.param<bool>("pipe_prior/size_update_zone_enable", pipe_size_update_zone_enable, false);
    nh.param<bool>("pipe_prior/size_accept_span_fallback", pipe_size_accept_span_fallback, false);
    nh.param<bool>("pipe_prior/size_publish_stable", pipe_size_publish_stable, false);
    nh.param<bool>("pipe_prior/size_publish_fallback_to_stable", pipe_size_publish_fallback_to_stable, false);
    nh.param<bool>("pipe_prior/robot_pose_origin_enable", robot_pose_origin_enable, true);
    nh.param<bool>("pipe_prior/robot_pose_origin_publish_tf", robot_pose_origin_publish_tf, true);
    nh.param<bool>("pipe_prior/robot_pose_origin_path_enable", robot_pose_origin_path_enable, true);
    nh.param<int>("pipe_prior/robot_pose_origin_path_stride", robot_pose_origin_path_stride, 10);
    nh.param<string>("pipe_prior/robot_pose_origin_frame", robot_pose_origin_frame, string("robot_origin"));
    nh.param<string>("pipe_prior/robot_pose_origin_child_frame", robot_pose_origin_child_frame, string("body_in_robot_origin"));
    nh.param<bool>("pipe_prior/width_require_both_side_walls", pipe_width_require_both_side_walls, true);
    nh.param<bool>("pipe_prior/height_require_both_top_bottom_walls", pipe_height_require_both_top_bottom_walls, true);
    nh.param<bool>("pipe_prior/width_hold_last_valid", pipe_width_hold_last_valid, true);
    nh.param<bool>("pipe_prior/height_hold_last_valid", pipe_height_hold_last_valid, true);

    ROS_WARN_STREAM("[PIPE PARAM] geometry_prior_enable=" << pipe_geometry_prior_enable);
    ROS_WARN_STREAM("[PIPE PARAM] endcap_require_lio_healthy=" << pipe_endcap_require_lio_healthy
                    << " min_lio_eff=" << pipe_endcap_min_lio_eff
                    << " require_2d_span=" << pipe_endcap_require_2d_span
                    << " exit_max_forward_dist=" << pipe_exit_max_forward_dist
                    << " map_filter_min_eff=" << pipe_map_filter_min_eff);
    ROS_WARN_STREAM("[PIPE PARAM] apply_incidence_filter_in_lio=" << pipe_apply_incidence_filter_in_lio);
    ROS_WARN_STREAM("[PIPE PARAM] min_incidence_cos=" << pipe_min_incidence_cos);
    ROS_WARN_STREAM("[PIPE PARAM] axis_conf_threshold=" << pipe_axis_conf_threshold);
    ROS_WARN_STREAM("[PIPE PARAM] mid_section_keep_ratio=" << pipe_mid_section_keep_ratio);
    ROS_WARN_STREAM("[PIPE PARAM] degenerate_ratio=" << pipe_degenerate_ratio);
    ROS_WARN_STREAM("[PIPE PARAM] degenerate_cond_thresh=" << pipe_degenerate_cond_thresh);
    ROS_WARN_STREAM("[PIPE PARAM] degenerate_min_pos_rel=" << pipe_degenerate_min_pos_rel);
    ROS_WARN_STREAM("[PIPE PARAM] degenerate_hold_frames=" << pipe_degenerate_hold_frames);
    ROS_WARN_STREAM("[PIPE PARAM] plane_score_gate=" << pipe_plane_score_gate);
    ROS_WARN_STREAM("[PIPE PARAM] plane_abs_pd2_gate=" << pipe_plane_abs_pd2_gate);
    ROS_WARN_STREAM("[PIPE PARAM] map_filter_by_residual=" << pipe_map_filter_by_residual);
    ROS_WARN_STREAM("[PIPE PARAM] map_use_selected_points=" << pipe_map_use_selected_points);
    ROS_WARN_STREAM("[PIPE PARAM] map_max_pd2=" << pipe_map_max_pd2);
    ROS_WARN_STREAM("[PIPE PARAM] debug_match_stats=" << pipe_debug_match_stats);
    ROS_WARN_STREAM("[PIPE PARAM] endcap_enable=" << pipe_endcap_enable
                    << " endcap_use_prior=" << pipe_endcap_use_prior
                    << " endcap_weight=" << pipe_endcap_weight
                    << " fallback=" << pipe_endcap_fallback_enable
                    << " fallback_gate=" << pipe_endcap_fallback_gate
                    << " entry_side=" << pipe_endcap_entry_side
                    << " known_length=" << pipe_endcap_known_length);
    ROS_WARN_STREAM("[PIPE PARAM] exit_only_mode=" << pipe_exit_only_mode
                    << " front_only=" << pipe_exit_only_front_side
                    << " min_forward=" << pipe_exit_min_forward_dist
                    << " lock_frames=" << pipe_exit_lock_min_frames
                    << " known_length=" << pipe_endcap_known_length);
    ROS_WARN_STREAM("[PIPE PARAM] debug_publish_endcap_points=" << pipe_debug_publish_endcap_points);
    ROS_WARN_STREAM("[PIPE PARAM] self_mask_enable=" << pipe_self_mask_enable
                    << " x=[" << pipe_self_mask_x_min << "," << pipe_self_mask_x_max << "]"
                    << " y=[" << pipe_self_mask_y_min << "," << pipe_self_mask_y_max << "]"
                    << " z=[" << pipe_self_mask_z_min << "," << pipe_self_mask_z_max << "]");
    ROS_WARN_STREAM("[PIPE PARAM] size_stabilize=" << pipe_size_stabilize_enable
                    << " exclude_endcap=" << pipe_size_exclude_endcap_points
                    << " exclusion=" << pipe_size_endcap_exclusion
                    << " update_ratio=[" << pipe_size_update_s_min_ratio << "," << pipe_size_update_s_max_ratio << "]"
                    << " window=" << pipe_size_window
                    << " min_samples=" << pipe_size_min_stable_samples
                    << " jump_gate=" << pipe_size_jump_gate
                    << " avg_mode=" << pipe_size_average_mode
                    << " trim=" << pipe_size_trim_ratio
                    << " require_anchor=" << pipe_size_require_anchor
                    << " gravity_height=" << pipe_size_use_gravity_height
                    << " update_requires_axis=" << pipe_size_update_requires_axis
                    << " update_zone_enable=" << pipe_size_update_zone_enable
                    << " partial_pairs=" << pipe_size_accept_partial_pairs
                    << " span_fallback=" << pipe_size_accept_span_fallback);
    ROS_WARN_STREAM("[ROBOT POSE] origin_enable=" << robot_pose_origin_enable
                    << " frame=" << robot_pose_origin_frame
                    << " child=" << robot_pose_origin_child_frame
                    << " publish_tf=" << robot_pose_origin_publish_tf
                    << " path_enable=" << robot_pose_origin_path_enable
                    << " path_stride=" << robot_pose_origin_path_stride);

    p_pre->lidar_type = lidar_type;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->lidar_type = lidar_type;
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    ensure_log_directory_exists(root_dir);

    string pipe_csv_path = root_dir + "/Log/pipe_metrics.csv";
    init_pipe_csv_log(pipe_csv_path);

    string robot_pose_csv_path = root_dir + "/Log/robot_pose_origin.csv";
    robot_pose_csv_log.open(robot_pose_csv_path.c_str(), ios::out | ios::trunc);
    if (robot_pose_csv_log.is_open())
    {
        robot_pose_csv_log << "stamp_abs,stamp_rel,x,y,z,qx,qy,qz,qw,roll,pitch,yaw,vx,vy,vz\n";
    }
    else
    {
        ROS_WARN_STREAM("[ROBOT POSE] failed to open csv log: " << robot_pose_csv_path);
    }

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    ros::Publisher pubRobotOdomOrigin = nh.advertise<nav_msgs::Odometry>
            ("/robot_pose_origin", 100000);
    ros::Publisher pubRobotPathOrigin = nh.advertise<nav_msgs::Path>
            ("/robot_path_origin", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);
    ros::Publisher pubPipeSize      = nh.advertise<geometry_msgs::Vector3>
            ("/pipe_size", 1000);
    ros::Publisher pubPipeBBox      = nh.advertise<visualization_msgs::Marker>
            ("/pipe_bbox", 10);
    ros::Publisher pubPipePoseSUV   = nh.advertise<geometry_msgs::Vector3>
            ("/pipe_pose_suv", 1000);
    ros::Publisher pubPipeEndcapPoints = nh.advertise<sensor_msgs::PointCloud2>
            ("/pipe_endcap_points", 10);
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        if(sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);
            publish_robot_pose_origin(pubRobotOdomOrigin, pubRobotPathOrigin);
            prev_lidar_pos_world = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            prev_lidar_pos_valid = true;

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            publish_pipe_size(pubPipeSize);
            publish_pipe_pose_suv(pubPipePoseSUV);
            publish_pipe_bbox(pubPipeBBox);
            publish_pipe_endcap_points(pubPipeEndcapPoints);
            append_pipe_csv_log(Measures.lidar_beg_time, Measures.lidar_beg_time - first_lidar_time);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    if (pipe_csv_log.is_open()) pipe_csv_log.close();

    return 0;
}
