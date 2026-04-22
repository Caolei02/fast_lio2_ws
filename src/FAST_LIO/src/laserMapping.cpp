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
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
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

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

struct RectPipeGeomState
{
    bool valid = false;
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

RectPipeGeomState g_pipe_geom;
bool   pipe_prior_enable = true;
bool   pipe_debug_log = false;
bool   pipe_degenerate = false;
double pipe_prior_weight = 8.0;
double pipe_axis_align_weight = 3.0;
double pipe_motion_weight = 1.5;
double pipe_min_width = 0.25;
double pipe_max_width = 1.00;
double pipe_min_height = 0.25;
double pipe_max_height = 1.00;
double pipe_axis_conf_threshold = 0.35;
double pipe_degenerate_min_eig = 1e-4;
double pipe_degenerate_ratio = 1e-3;
double pipe_last_min_eig = 0.0;
double pipe_last_cond_num = 0.0;
double pipe_last_axial_info = 0.0;
double pipe_last_lateral_info = 0.0;
V3D    prev_lidar_pos_world = Zero3d;
bool   prev_lidar_pos_valid = false;
V3D    prev_pipe_u_w = V3D(0.0, 1.0, 0.0);
V3D    prev_pipe_v_w = V3D(0.0, 0.0, 1.0);
bool   prev_pipe_basis_valid = false;
double pipe_mid_section_keep_ratio = 0.60;
double pipe_min_incidence_cos = 0.20;
bool   pipe_apply_incidence_filter_in_lio = true;
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
ofstream pipe_csv_log;

void init_pipe_csv_log(const string &csv_path)
{
    if (pipe_csv_log.is_open()) return;

    pipe_csv_log.open(csv_path.c_str(), ios::out | ios::trunc);
    if (!pipe_csv_log.is_open())
    {
        ROS_WARN("failed to open pipe csv log: %s", csv_path.c_str());
        return;
    }

    pipe_csv_log << "stamp_abs,stamp_rel,valid,deg,eig_min,cond,axial,lateral,local_length,width,height,global_length,conf,u_pos_count,u_neg_count,v_pos_count,v_neg_count,four_wall_fit_ok,geom_raw_count,geom_filtered_count,intensity_trim_used\n";
    pipe_csv_log.flush();
}

inline void append_pipe_csv_log(double stamp_abs, double stamp_rel)
{
    if (!pipe_csv_log.is_open()) return;

    pipe_csv_log << fixed << setprecision(9)
                 << stamp_abs << ","
                 << stamp_rel << ","
                 << int(g_pipe_geom.valid) << ","
                 << int(pipe_degenerate) << ","
                 << pipe_last_min_eig << ","
                 << pipe_last_cond_num << ","
                 << pipe_last_axial_info << ","
                 << pipe_last_lateral_info << ","
                 << g_pipe_geom.length << ","
                 << g_pipe_geom.width << ","
                 << g_pipe_geom.height << ","
                 << g_pipe_geom.global_length << ","
                 << g_pipe_geom.straight_confidence << ","
                 << g_pipe_geom.u_pos_count << ","
                 << g_pipe_geom.u_neg_count << ","
                 << g_pipe_geom.v_pos_count << ","
                 << g_pipe_geom.v_neg_count << ","
                 << int(g_pipe_geom.four_wall_fit_ok) << ","
                 << pipe_last_geom_raw_count << ","
                 << pipe_last_geom_filtered_count << ","
                 << int(pipe_last_geom_used_intensity_trim) << "\n";
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
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
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

bool fit_rect_pipe_geometry(const state_ikfom &s, RectPipeGeomState &geom)
{
    geom.valid = false;
    geom.length = 0.0;
    geom.width = 0.0;
    geom.height = 0.0;
    geom.global_length = pipe_global_length;
    geom.u_pos_count = 0;
    geom.u_neg_count = 0;
    geom.v_pos_count = 0;
    geom.v_neg_count = 0;
    geom.four_wall_fit_ok = false;
    if (effct_feat_num < 20) return false;

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

        if (pipe_apply_incidence_filter_in_lio && cand.cos_incidence < pipe_min_incidence_cos)
            continue;

        raw_candidates.push_back(cand);
    }

    pipe_last_geom_raw_count = int(raw_candidates.size());
    pipe_last_geom_filtered_count = pipe_last_geom_raw_count;
    pipe_last_geom_used_intensity_trim = false;

    if (raw_candidates.size() < 20) return false;

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

    if (world_pts.size() < 20) return false;
    centroid /= double(world_pts.size());

    M3D normal_cov = M3D::Zero();
    for (const auto &n : normals)
    {
        normal_cov += n * n.transpose();
    }
    normal_cov /= double(normals.size());

    Eigen::SelfAdjointEigenSolver<M3D> eig_solver(normal_cov);
    if (eig_solver.info() != Eigen::Success) return false;

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
    if (u_w.norm() < 1e-6) return false;

    V3D v_w = axis_w.cross(u_w);
    if (v_w.norm() < 1e-6) return false;
    v_w.normalize();
    u_w = v_w.cross(axis_w).normalized();

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
            ssmpl.wall_candidate = (dom >= 0.35);
        }
        samples.push_back(ssmpl);
    }

    double t_min = quantile_from_sorted(t_coords, 0.05);
    double t_max = quantile_from_sorted(t_coords, 0.95);
    double length = t_max - t_min;
    if (length < 1e-6) return false;

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

    if (u_mid_coords.size() < 8 || v_mid_coords.size() < 8) return false;

    double u_min = quantile_from_sorted(u_mid_coords, 0.05);
    double u_max = quantile_from_sorted(u_mid_coords, 0.95);
    double v_min = quantile_from_sorted(v_mid_coords, 0.05);
    double v_max = quantile_from_sorted(v_mid_coords, 0.95);
    double width = u_max - u_min;
    double height = v_max - v_min;

    const size_t min_wall_samples = 6;
    bool four_wall_fit_ok = (u_pos_wall.size() >= min_wall_samples &&
                             u_neg_wall.size() >= min_wall_samples &&
                             v_pos_wall.size() >= min_wall_samples &&
                             v_neg_wall.size() >= min_wall_samples);

    if (four_wall_fit_ok)
    {
        double u_pos = robust_trimmed_center(u_pos_wall, 0.15, 0.85);
        double u_neg = robust_trimmed_center(u_neg_wall, 0.15, 0.85);
        double v_pos = robust_trimmed_center(v_pos_wall, 0.15, 0.85);
        double v_neg = robust_trimmed_center(v_neg_wall, 0.15, 0.85);

        double u_low_raw = std::min(u_pos, u_neg);
        double u_high_raw = std::max(u_pos, u_neg);
        double v_low_raw = std::min(v_pos, v_neg);
        double v_high_raw = std::max(v_pos, v_neg);

        // Recentre the local frame on the estimated pipe centerline in the cross section.
        double u_center = 0.5 * (u_low_raw + u_high_raw);
        double v_center = 0.5 * (v_low_raw + v_high_raw);
        axis_point_w = axis_point_w + u_center * u_w + v_center * v_w;

        u_min = u_low_raw - u_center;
        u_max = u_high_raw - u_center;
        v_min = v_low_raw - v_center;
        v_max = v_high_raw - v_center;
        width = std::max(u_max, u_min) - std::min(u_max, u_min);
        height = std::max(v_max, v_min) - std::min(v_max, v_min);
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

    bool size_ok = (width > pipe_min_width && width < pipe_max_width &&
                    height > pipe_min_height && height < pipe_max_height);
    double straight_conf = std::min(1.0, std::max(0.0, (normal_planarity - 0.5) / 0.5));

    geom.valid = size_ok && (straight_conf > pipe_axis_conf_threshold);
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

    if (geom.valid)
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

    return geom.valid;
}

void analyze_degeneracy(const MatrixXd &Hx, const RectPipeGeomState &geom)
{
    pipe_last_min_eig = 0.0;
    pipe_last_cond_num = 0.0;
    pipe_last_axial_info = 0.0;
    pipe_last_lateral_info = 0.0;
    pipe_degenerate = false;

    if (Hx.rows() < 6) return;

    Matrix<double, 6, 6> info = Hx.leftCols(6).transpose() * Hx.leftCols(6);
    Eigen::SelfAdjointEigenSolver<Matrix<double, 6, 6>> solver(info);
    if (solver.info() != Eigen::Success) return;

    auto vals = solver.eigenvalues();
    pipe_last_min_eig = vals(0);
    pipe_last_cond_num = vals(5) / std::max(1e-12, vals(0));

    Matrix3d info_pos = info.block<3, 3>(0, 0);
    if (geom.valid)
    {
        pipe_last_axial_info = geom.axis_w.transpose() * info_pos * geom.axis_w;
        Matrix3d Pperp = Matrix3d::Identity() - geom.axis_w * geom.axis_w.transpose();
        pipe_last_lateral_info = (Pperp * info_pos * Pperp).trace() / 2.0;
    }
    else
    {
        pipe_last_axial_info = info_pos.trace() / 3.0;
        pipe_last_lateral_info = pipe_last_axial_info;
    }

    bool min_eig_bad = pipe_last_min_eig < pipe_degenerate_min_eig;
    bool axial_ratio_bad = pipe_last_axial_info < std::max(1e-12, pipe_last_lateral_info) * pipe_degenerate_ratio;
    pipe_degenerate = min_eig_bad || axial_ratio_bad;
}

void augment_with_pipe_priors(const state_ikfom &s,
                              const RectPipeGeomState &geom,
                              MatrixXd &Hx,
                              VectorXd &h)
{
    if (!pipe_prior_enable || !pipe_degenerate || !geom.valid) return;

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
    if (!g_pipe_geom.valid) return;

    geometry_msgs::Vector3 msg;
    msg.x = g_pipe_geom.global_length > 0.0 ? g_pipe_geom.global_length : g_pipe_geom.length;
    msg.y = g_pipe_geom.width;
    msg.z = g_pipe_geom.height;
    pubPipeSize.publish(msg);
}

void publish_pipe_bbox(const ros::Publisher &pubPipeBBox)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = "camera_init";
    marker.header.stamp = ros::Time().fromSec(lidar_end_time);
    marker.ns = "pipe_bbox";
    marker.id = 0;

    if (!g_pipe_geom.valid)
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

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
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

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float score = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (score > 0.9)
            {
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
                    if (cos_inc < pipe_min_incidence_cos) continue;
                }

                point_selected_surf[i] = true;
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

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
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

    // ---- Rectangular pipe degeneracy detection and geometry-guided prior augmentation ----
    fit_rect_pipe_geometry(s, g_pipe_geom);
    analyze_degeneracy(ekfom_data.h_x, g_pipe_geom);
    augment_with_pipe_priors(s, g_pipe_geom, ekfom_data.h_x, ekfom_data.h);

    if (pipe_debug_log)
    {
        ROS_INFO_THROTTLE(0.5,
                          "pipe valid=%d deg=%d eig_min=%.3e cond=%.3e axial=%.3e lateral=%.3e local_size=(L=%.3f, W=%.3f, H=%.3f) global_L=%.3f conf=%.3f",
                          g_pipe_geom.valid,
                          pipe_degenerate,
                          pipe_last_min_eig,
                          pipe_last_cond_num,
                          pipe_last_axial_info,
                          pipe_last_lateral_info,
                          g_pipe_geom.length,
                          g_pipe_geom.width,
                          g_pipe_geom.height,
                          g_pipe_geom.global_length,
                          g_pipe_geom.straight_confidence);
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
    nh.param<double>("pipe_prior/weight_box", pipe_prior_weight, 8.0);
    nh.param<double>("pipe_prior/weight_axis_align", pipe_axis_align_weight, 3.0);
    nh.param<double>("pipe_prior/weight_motion", pipe_motion_weight, 1.5);
    nh.param<double>("pipe_prior/min_width", pipe_min_width, 0.25);
    nh.param<double>("pipe_prior/max_width", pipe_max_width, 1.00);
    nh.param<double>("pipe_prior/min_height", pipe_min_height, 0.25);
    nh.param<double>("pipe_prior/max_height", pipe_max_height, 1.00);
    nh.param<double>("pipe_prior/axis_conf_threshold", pipe_axis_conf_threshold, 0.35);
    nh.param<double>("pipe_prior/degenerate_min_eig", pipe_degenerate_min_eig, 1e-4);
    nh.param<double>("pipe_prior/degenerate_ratio", pipe_degenerate_ratio, 1e-3);
    nh.param<double>("pipe_prior/mid_section_keep_ratio", pipe_mid_section_keep_ratio, 0.60);
    nh.param<double>("pipe_prior/min_incidence_cos", pipe_min_incidence_cos, 0.20);
    nh.param<bool>("pipe_prior/apply_incidence_filter_in_lio", pipe_apply_incidence_filter_in_lio, true);
    nh.param<bool>("pipe_prior/enable_intensity_trim", pipe_enable_intensity_trim, true);
    nh.param<double>("pipe_prior/intensity_quantile_low", pipe_intensity_quantile_low, 0.05);
    nh.param<double>("pipe_prior/intensity_quantile_high", pipe_intensity_quantile_high, 0.95);
    nh.param<bool>("pipe_prior/debug_print_filter_stats", pipe_debug_print_filter_stats, false);

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

    string pipe_csv_path = root_dir + "/Log/pipe_metrics.csv";
    init_pipe_csv_log(pipe_csv_path);

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
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);
    ros::Publisher pubPipeSize      = nh.advertise<geometry_msgs::Vector3>
            ("/pipe_size", 1000);
    ros::Publisher pubPipeBBox      = nh.advertise<visualization_msgs::Marker>
            ("/pipe_bbox", 10);
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
            prev_lidar_pos_world = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            prev_lidar_pos_valid = true;

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            publish_pipe_size(pubPipeSize);
            publish_pipe_bbox(pubPipeBBox);
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
