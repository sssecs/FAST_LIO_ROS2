// FAST-LIO algorithm core (implementation). See laser_mapping_core.hpp.

#include <fast_lio/laser_mapping_core.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <omp.h>
#include <pcl/io/pcd_io.h>

// Length of the IMU-bias estimation window (s).
#define INIT_TIME (0.1)
// Measurement noise of a matched lidar point (m).
#define LASER_POINT_COV (0.001)

namespace
{
// FOV segment trigger threshold (relative to det_range).
constexpr float MOV_THRESHOLD = 1.5f;

// The esekfom library stores the measurement model as a raw function pointer,
// so the core instance is reached through this single file-local pointer.
fast_lio::LaserMappingCore * g_core = nullptr;
}  // namespace

namespace fast_lio
{

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

LaserMappingCore::LaserMappingCore(const Params & params, const rclcpp::Logger & logger)
  : params_(params), logger_(logger)
{
  g_core = this;

  imu_process_ = std::make_shared<ImuProcess>();
  num_max_iterations_ = params_.max_iteration;

  memset(point_selected_surf_, true, sizeof(point_selected_surf_));
  memset(res_last_, -1000.0f, sizeof(res_last_));
  downSizeFilterSurf_.setLeafSize(
    params_.filter_size_surf, params_.filter_size_surf, params_.filter_size_surf);
  downSizeFilterMap_.setLeafSize(
    params_.filter_size_map, params_.filter_size_map, params_.filter_size_map);

  if (params_.enable_localization)
  {
    load_map();
  }

  // --- debug logs ------------------------------------------------------------
  const std::string pos_log_dir = std::string(ROOT_DIR) + "/Log/pos_log.txt";
  fp_ = fopen(pos_log_dir.c_str(), "w");
  fout_pre_.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out_.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
  if (fout_pre_ && fout_out_)
  {
    std::cout << "~~~~" << ROOT_DIR << " file opened" << std::endl;
  }
  else
  {
    std::cout << "~~~~" << ROOT_DIR << " doesn't exist" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// EKF / seed initialization
// ---------------------------------------------------------------------------

void LaserMappingCore::init_filter()
{
  imu_process_->Reset();
  Lidar_T_wrt_IMU_ << VEC_FROM_ARRAY(params_.extrin_T);
  Lidar_R_wrt_IMU_ << MAT_FROM_ARRAY(params_.extrin_R);
  imu_process_->set_extrinsic(Lidar_T_wrt_IMU_, Lidar_R_wrt_IMU_);
  imu_process_->set_gyr_cov(V3D(params_.gyr_cov, params_.gyr_cov, params_.gyr_cov));
  imu_process_->set_acc_cov(V3D(params_.acc_cov, params_.acc_cov, params_.acc_cov));
  imu_process_->set_gyr_bias_cov(V3D(params_.b_gyr_cov, params_.b_gyr_cov, params_.b_gyr_cov));
  imu_process_->set_acc_bias_cov(V3D(params_.b_acc_cov, params_.b_acc_cov, params_.b_acc_cov));

  kf_ = std::make_unique<esekfom::esekf<state_ikfom, 12, input_ikfom>>();
  fill(epsi_, epsi_ + 23, 0.001);
  kf_->init_dyn_share(get_f, df_dx, df_dw, &LaserMappingCore::h_share_model, num_max_iterations_, epsi_);
}

void LaserMappingCore::set_initial_localization(const Eigen::Matrix4d & T_map_base)
{
  const Eigen::Vector3d p_map = T_map_base.block<3, 1>(0, 3);
  const Eigen::Matrix3d R_map = T_map_base.block<3, 3>(0, 0);
  Eigen::Quaterniond q_map(R_map);

  RCLCPP_INFO(
    logger_, "Init pose: pos=(%.3f, %.3f, %.3f), quat=(%.4f, %.4f, %.4f, %.4f)",
    p_map.x(), p_map.y(), p_map.z(), q_map.x(), q_map.y(), q_map.z(), q_map.w());

  auto init_state = kf_->get_x();
  init_state.rot = MTK::SO3(q_map);
  init_state.pos = p_map;
  kf_->change_x(init_state);
}

void LaserMappingCore::rearm_reinit(const Eigen::Vector3d & seed_pos, double lidar_beg_time)
{
  // Resume localization from this seed: clear any previous failure flag and
  // use the seed as the motion reference, so a teleport via /initialpose is
  // not mistaken for an implausible scan-to-scan jump by the health check.
  localization_failed_ = false;
  last_good_valid_ = true;
  last_good_pos_ = seed_pos;
  last_good_time_ = lidar_beg_time;
  // Arm the post-relocalization fly-away guard: the robot is expected to be
  // stationary right after a re-seed, so the pose must neither leave the
  // re-seeded position nor travel a long path during the verification window
  // (checked in localization_health_check()). A divergence means the
  // relocalization matched wrongly and the EKF is flying away -> the process
  // failed.
  relocalized_verify_ = params_.relocalization_verify_en;
  relocalized_pos_ = last_good_pos_;
  relocalized_time_ = last_good_time_;
  relocalized_path_ = 0.0;
}

void LaserMappingCore::set_first_lidar_time(double t)
{
  first_lidar_time_ = t;
  imu_process_->first_lidar_time = first_lidar_time_;
}

// ---------------------------------------------------------------------------
// Point transforms
// ---------------------------------------------------------------------------

void LaserMappingCore::point_body_to_world(PointType const * const pi, PointType * const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) + state_point_.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LaserMappingCore::RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) + state_point_.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LaserMappingCore::RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void LaserMappingCore::points_cache_collect()
{
  PointVector points_history;
  ikdtree_.acquire_removed_points(points_history);
}

// ---------------------------------------------------------------------------
// Map FOV segmentation / incremental update
// ---------------------------------------------------------------------------

void LaserMappingCore::lasermap_fov_segment()
{
  cub_needrm_.clear();
  kdtree_delete_counter_ = 0;
  kdtree_delete_time_ = 0.0;
  point_body_to_world(XAxisPoint_body_, XAxisPoint_world_);
  V3D pos_LiD = pos_lid_;
  if (!Localmap_Initialized_)
  {
    for (int i = 0; i < 3; i++)
    {
      LocalMap_Points_.vertex_min[i] = pos_LiD(i) - params_.cube_side_length / 2.0;
      LocalMap_Points_.vertex_max[i] = pos_LiD(i) + params_.cube_side_length / 2.0;
    }
    Localmap_Initialized_ = true;
    return;
  }
  float dist_to_map_edge[3][2];
  bool need_move = false;
  for (int i = 0; i < 3; i++)
  {
    dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points_.vertex_min[i]);
    dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points_.vertex_max[i]);
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * params_.det_range ||
      dist_to_map_edge[i][1] <= MOV_THRESHOLD * params_.det_range)
    {
      need_move = true;
    }
  }
  if (!need_move) return;

  BoxPointType New_LocalMap_Points, tmp_boxpoints;
  New_LocalMap_Points = LocalMap_Points_;
  float mov_dist = std::max(
    (params_.cube_side_length - 2.0 * MOV_THRESHOLD * params_.det_range) * 0.5 * 0.9,
    double(params_.det_range * (MOV_THRESHOLD - 1)));
  for (int i = 0; i < 3; i++)
  {
    tmp_boxpoints = LocalMap_Points_;
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * params_.det_range)
    {
      New_LocalMap_Points.vertex_max[i] -= mov_dist;
      New_LocalMap_Points.vertex_min[i] -= mov_dist;
      tmp_boxpoints.vertex_min[i] = LocalMap_Points_.vertex_max[i] - mov_dist;
      cub_needrm_.push_back(tmp_boxpoints);
    }
    else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * params_.det_range)
    {
      New_LocalMap_Points.vertex_max[i] += mov_dist;
      New_LocalMap_Points.vertex_min[i] += mov_dist;
      tmp_boxpoints.vertex_max[i] = LocalMap_Points_.vertex_min[i] + mov_dist;
      cub_needrm_.push_back(tmp_boxpoints);
    }
  }
  LocalMap_Points_ = New_LocalMap_Points;

  points_cache_collect();
  double delete_begin = omp_get_wtime();
  if (cub_needrm_.size() > 0)
  {
    kdtree_delete_counter_ = ikdtree_.Delete_Point_Boxes(cub_needrm_);
  }
  kdtree_delete_time_ = omp_get_wtime() - delete_begin;
}

void LaserMappingCore::map_incremental()
{
  PointVector PointToAdd;
  PointVector PointNoNeedDownsample;
  PointToAdd.reserve(feats_down_size_);
  PointNoNeedDownsample.reserve(feats_down_size_);
  for (int i = 0; i < feats_down_size_; i++)
  {
    /* transform to world frame */
    point_body_to_world(&(feats_down_body_->points[i]), &(feats_down_world_->points[i]));
    /* decide if need add to map */
    if (!Nearest_Points_[i].empty() && flg_EKF_inited_)
    {
      const PointVector & points_near = Nearest_Points_[i];
      bool need_add = true;
      BoxPointType Box_of_Point;
      PointType downsample_result, mid_point;
      mid_point.x =
        floor(feats_down_world_->points[i].x / params_.filter_size_map) * params_.filter_size_map +
        0.5 * params_.filter_size_map;
      mid_point.y =
        floor(feats_down_world_->points[i].y / params_.filter_size_map) * params_.filter_size_map +
        0.5 * params_.filter_size_map;
      mid_point.z =
        floor(feats_down_world_->points[i].z / params_.filter_size_map) * params_.filter_size_map +
        0.5 * params_.filter_size_map;
      float dist = calc_dist(feats_down_world_->points[i], mid_point);
      if (fabs(points_near[0].x - mid_point.x) > 0.5 * params_.filter_size_map &&
        fabs(points_near[0].y - mid_point.y) > 0.5 * params_.filter_size_map &&
        fabs(points_near[0].z - mid_point.z) > 0.5 * params_.filter_size_map)
      {
        PointNoNeedDownsample.push_back(feats_down_world_->points[i]);
        continue;
      }
      for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
      {
        if (points_near.size() < NUM_MATCH_POINTS) break;
        if (calc_dist(points_near[readd_i], mid_point) < dist)
        {
          need_add = false;
          break;
        }
      }
      if (need_add) PointToAdd.push_back(feats_down_world_->points[i]);
    }
    else
    {
      PointToAdd.push_back(feats_down_world_->points[i]);
    }
  }

  double st_time = omp_get_wtime();
  add_point_size_ = ikdtree_.Add_Points(PointToAdd, true);
  ikdtree_.Add_Points(PointNoNeedDownsample, false);
  add_point_size_ = PointToAdd.size() + PointNoNeedDownsample.size();
  kdtree_incremental_time_ = omp_get_wtime() - st_time;
}

void LaserMappingCore::load_map()
{
  RCLCPP_INFO(logger_, "Loading GlobalMap from %s ...", params_.map_path.c_str());

  PointCloudXYZI::Ptr localization_map(new PointCloudXYZI());
  if (pcl::io::loadPCDFile(params_.map_path, *localization_map) < 0)
  {
    RCLCPP_ERROR(logger_, "Failed to load map file: %s", params_.map_path.c_str());
    throw std::runtime_error("localization map file not found: " + params_.map_path);
  }

  // Keep the full-resolution map for the node to feed the Open3D relocalizer.
  map_raw_ = localization_map;

  // Downsample the loaded map once; the scan matching uses this resolution.
  PointCloudXYZI::Ptr map_ds(new PointCloudXYZI());
  pcl::VoxelGrid<PointType> voxel_grid_filter;
  voxel_grid_filter.setLeafSize(params_.leaf_size, params_.leaf_size, params_.leaf_size);
  voxel_grid_filter.setInputCloud(localization_map);
  voxel_grid_filter.filter(*map_ds);
  RCLCPP_INFO(logger_, "GlobalMap ds point num: %zu", map_ds->size());

  PointVector points_to_add;
  points_to_add.reserve(map_ds->points.size());
  for (size_t i = 0; i < map_ds->points.size(); i++)
  {
    points_to_add.push_back(map_ds->points[i]);
  }

  downSizeFilterSurf_.setInputCloud(map_ds);
  downSizeFilterSurf_.filter(*feats_down_body_);
  feats_down_size_ = feats_down_body_->points.size();

  if (ikdtree_.Root_Node == nullptr)
  {
    RCLCPP_INFO(logger_, "Initialize the map kdtree");
    if (feats_down_size_ > 5)
    {
      ikdtree_.set_downsample_param(params_.filter_size_map);
      feats_down_world_->resize(feats_down_size_);
      for (int i = 0; i < feats_down_size_; i++)
      {
        point_body_to_world(&(feats_down_body_->points[i]), &(feats_down_world_->points[i]));
      }
      ikdtree_.Build(feats_down_world_->points);
    }
    return;
  }

  ikdtree_.Add_Points(points_to_add, false);  // load whole map.
  int featsFromMapNum = ikdtree_.validnum();
  kdtree_size_st_ = ikdtree_.size();

  RCLCPP_INFO(logger_, "Loaded GlobalMap from %s!", params_.map_path.c_str());
}

// ---------------------------------------------------------------------------
// Main per-scan processing loop
// ---------------------------------------------------------------------------

bool LaserMappingCore::process_scan(const MeasureGroup & meas, ScanTiming & timing)
{
  match_time_ = 0;
  kdtree_search_time_ = 0.0;
  solve_time_ = 0;
  solve_const_H_time_ = 0;

  imu_process_->Process(meas, *kf_, feats_undistort_);
  timing.t_imu_end = omp_get_wtime();
  state_point_ = kf_->get_x();
  pos_lid_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;

  if (feats_undistort_->empty() || (feats_undistort_ == NULL))
  {
    RCLCPP_WARN(logger_, "No point, skip this scan!\n");
    return false;
  }

  flg_EKF_inited_ =
    (meas.lidar_beg_time - first_lidar_time_) < INIT_TIME ? false : true;
  /*** Segment the map in lidar FOV ***/
  if (params_.enable_map_incremental)
  {
    lasermap_fov_segment();
  }
  timing.t_fov_end = omp_get_wtime();

  /*** downsample the feature points in a scan ***/
  downSizeFilterSurf_.setInputCloud(feats_undistort_);
  downSizeFilterSurf_.filter(*feats_down_body_);
  timing.t1 = omp_get_wtime();
  feats_down_size_ = feats_down_body_->points.size();
  /*** initialize the map kdtree ***/
  if (ikdtree_.Root_Node == nullptr)
  {
    RCLCPP_INFO(logger_, "Initialize the map kdtree");
    if (feats_down_size_ > 5)
    {
      ikdtree_.set_downsample_param(params_.filter_size_map);
      feats_down_world_->resize(feats_down_size_);
      for (int i = 0; i < feats_down_size_; i++)
      {
        point_body_to_world(&(feats_down_body_->points[i]), &(feats_down_world_->points[i]));
      }
      ikdtree_.Build(feats_down_world_->points);
    }
    return false;
  }
  int featsFromMapNum = ikdtree_.validnum();
  kdtree_size_st_ = ikdtree_.size();

  /*** ICP and iterated Kalman filter update ***/
  if (feats_down_size_ < 5)
  {
    RCLCPP_WARN(logger_, "No point, skip this scan!\n");
    return false;
  }

  normvec_->resize(feats_down_size_);
  feats_down_world_->resize(feats_down_size_);

  V3D ext_euler = SO3ToEuler(state_point_.offset_R_L_I);
  fout_pre_ << setw(20) << meas.lidar_beg_time - first_lidar_time_ << " "
            << euler_cur_.transpose() << " " << state_point_.pos.transpose() << " "
            << ext_euler.transpose() << " " << state_point_.offset_T_L_I.transpose() << " "
            << state_point_.vel.transpose() << " " << state_point_.bg.transpose() << " "
            << state_point_.ba.transpose() << " " << state_point_.grav << endl;

  pointSearchInd_surf_.resize(feats_down_size_);
  Nearest_Points_.resize(feats_down_size_);
  int rematch_num = 0;
  bool nearest_search_en = true;  //

  timing.t2 = omp_get_wtime();

  /*** iterated state estimation ***/
  double t_update_start = omp_get_wtime();
  double solve_H_time = 0;
  kf_->update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
  state_point_ = kf_->get_x();
  euler_cur_ = SO3ToEuler(state_point_.rot);
  pos_lid_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;
  geoQuat_.x = state_point_.rot.coeffs()[0];
  geoQuat_.y = state_point_.rot.coeffs()[1];
  geoQuat_.z = state_point_.rot.coeffs()[2];
  geoQuat_.w = state_point_.rot.coeffs()[3];

  timing.t_update_start = t_update_start;
  timing.t_update_end = omp_get_wtime();
  timing.solve_H_time = solve_H_time;

  return true;
}

bool LaserMappingCore::localization_health_check(double lidar_beg_time)
{
  // Only active for static-map localization; in mapping mode the map is being
  // built so the support check does not apply.
  if (!params_.enable_localization || !params_.failure_detect_en)
  {
    return true;
  }

  // A non-finite pose means the filter diverged -> lost.
  if (!state_point_.pos.allFinite())
  {
    localization_failed_ = true;
    RCLCPP_WARN(
      logger_,
      "Localization FAILED: pose is non-finite; stopping localization, map->base "
      "TF suppressed until a new initial pose.");
    return false;
  }

  // Post-relocalization fly-away guard: the robot is expected to stay put right
  // after a re-seed (relocalization is done with the robot stationary), so the
  // pose must stay near the re-seeded position during a short verification
  // window. Both the straight-line drift from the seed and the total path the
  // pose travels are bounded: net drift alone misses an oscillating pose (or one
  // that moves away and comes back), which ends near the seed but has traveled a
  // long path. If either bound is exceeded, the relocalization itself matched
  // wrongly and the EKF is diverging -> judge the process failed (same lost-state
  // machinery: TF suppressed, next /initialpose re-arms the guard).
  if (relocalized_verify_)
  {
    const double since_seed = lidar_beg_time - relocalized_time_;
    if (since_seed > params_.relocalization_verify_window)
    {
      relocalized_verify_ = false;  // verification window passed -> normal checks
    }
    else
    {
      // Accumulate the pose path traveled since the re-seed. last_good_pos_ is
      // the last healthy position, which the re-init branch seeded to the
      // re-seeded position, so the first step is the move away from the seed.
      relocalized_path_ += (state_point_.pos - last_good_pos_).norm();
      const double drift = (state_point_.pos - relocalized_pos_).norm();
      if (drift > params_.relocalization_verify_max_drift ||
          relocalized_path_ > params_.relocalization_verify_max_path)
      {
        relocalized_verify_ = false;
        localization_failed_ = true;
        RCLCPP_WARN(
          logger_,
          "Localization FAILED: relocalization drifted %.2f m from the re-seeded "
          "position / traveled %.2f m in %.1f s; stopping localization, map->base "
          "TF suppressed until a new initial pose.",
          drift, relocalized_path_, since_seed);
        return false;
      }
    }
  }

  // Map support: Euclidean distance from the pose to the nearest map point.
  PointType pose_pt;
  pose_pt.x = static_cast<float>(state_point_.pos(0));
  pose_pt.y = static_cast<float>(state_point_.pos(1));
  pose_pt.z = static_cast<float>(state_point_.pos(2));
  PointVector points_near;
  std::vector<float> point_sqdis;
  ikdtree_.Nearest_Search(pose_pt, 1, points_near, point_sqdis);
  const double nearest_dist = points_near.empty()
                                ? std::numeric_limits<double>::infinity()
                                : std::sqrt(static_cast<double>(point_sqdis[0]));

  // Motion: pose displacement rate since the last healthy update. A jump that
  // is impossible for the robot in the elapsed time means the match is wrong.
  double speed = 0.0;
  if (last_good_valid_)
  {
    const double dt = lidar_beg_time - last_good_time_;
    if (dt > 1e-6)
    {
      speed = (state_point_.pos - last_good_pos_).norm() / dt;
    }
  }

  const bool ok =
    nearest_dist <= params_.max_pose_to_map_dist && speed <= params_.max_pose_speed;

  if (ok)
  {
    last_good_valid_ = true;
    last_good_pos_ = state_point_.pos;
    last_good_time_ = lidar_beg_time;
    if (localization_failed_)
    {
      localization_failed_ = false;
      RCLCPP_INFO(
        logger_,
        "Localization recovered: nearest map point %.2f m, speed %.2f m/s.",
        nearest_dist, speed);
    }
  }
  else
  {
    localization_failed_ = true;
    RCLCPP_WARN(
      logger_,
      "Localization FAILED: nearest map point %.2f m (> %.2f) or speed %.2f m/s "
      "(> %.2f); stopping localization, map->base TF suppressed until a new "
      "initial pose.",
      nearest_dist, params_.max_pose_to_map_dist, speed, params_.max_pose_speed);
  }
  return ok;
}

// ---------------------------------------------------------------------------
// Measurement model (called by the esekfom update through the trampoline)
// ---------------------------------------------------------------------------

void LaserMappingCore::h_share_model(
  state_ikfom & s, esekfom::dyn_share_datastruct<double> & ekfom_data)
{
  g_core->h_share_model_impl(s, ekfom_data);
}

void LaserMappingCore::h_share_model_impl(
  state_ikfom & s, esekfom::dyn_share_datastruct<double> & ekfom_data)
{
  double match_start = omp_get_wtime();
  laserCloudOri_->clear();
  corr_normvect_->clear();
  total_residual_ = 0.0;

  /** closest surface search and residual computation **/
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < feats_down_size_; i++)
  {
    PointType & point_body = feats_down_body_->points[i];
    PointType & point_world = feats_down_world_->points[i];

    /* transform to world frame */
    V3D p_body(point_body.x, point_body.y, point_body.z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    point_world.x = p_global(0);
    point_world.y = p_global(1);
    point_world.z = p_global(2);
    point_world.intensity = point_body.intensity;

    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

    auto & points_near = Nearest_Points_[i];

    if (ekfom_data.converge)
    {
      /** Find the closest surfaces in the map **/
      ikdtree_.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
      point_selected_surf_[i] =
        points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
    }

    if (!point_selected_surf_[i]) continue;

    VF(4) pabcd;
    point_selected_surf_[i] = false;
    if (esti_plane(pabcd, points_near, 0.1f))
    {
      float pd2 =
        pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
      float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

      if (s > 0.9)
      {
        point_selected_surf_[i] = true;
        normvec_->points[i].x = pabcd(0);
        normvec_->points[i].y = pabcd(1);
        normvec_->points[i].z = pabcd(2);
        normvec_->points[i].intensity = pd2;
        res_last_[i] = abs(pd2);
      }
    }
  }

  effct_feat_num_ = 0;

  for (int i = 0; i < feats_down_size_; i++)
  {
    if (point_selected_surf_[i])
    {
      laserCloudOri_->points[effct_feat_num_] = feats_down_body_->points[i];
      corr_normvect_->points[effct_feat_num_] = normvec_->points[i];
      total_residual_ += res_last_[i];
      effct_feat_num_++;
    }
  }

  if (effct_feat_num_ < 1)
  {
    ekfom_data.valid = false;
    std::cerr << "No Effective Points!" << std::endl;
    return;
  }

  res_mean_last_ = total_residual_ / effct_feat_num_;
  match_time_ += omp_get_wtime() - match_start;
  double solve_start_ = omp_get_wtime();

  /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
  ekfom_data.h_x = MatrixXd::Zero(effct_feat_num_, 12);  // 23
  ekfom_data.h.resize(effct_feat_num_);

  for (int i = 0; i < effct_feat_num_; i++)
  {
    const PointType & laser_p = laserCloudOri_->points[i];
    V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
    M3D point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
    V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    /*** get the normal vector of closest surface/corner ***/
    const PointType & norm_p = corr_normvect_->points[i];
    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

    /*** calculate the Measuremnt Jacobian matrix H ***/
    V3D C(s.rot.conjugate() * norm_vec);
    V3D A(point_crossmat * C);
    if (params_.extrinsic_est_en)
    {
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // s.rot.conjugate()*norm_vec);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A),
        VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
    }
    else
    {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -norm_p.intensity;
  }
  solve_time_ += omp_get_wtime() - solve_start_;
}

// ---------------------------------------------------------------------------
// Runtime performance log
// ---------------------------------------------------------------------------

void LaserMappingCore::log_scan_runtime(
  const MeasureGroup & meas, double t0, double t3, double t5, const ScanTiming & timing)
{
  if (!params_.runtime_pos_log_enable) return;

  double t1 = timing.t1;
  double t_update_start = timing.t_update_start;
  double t_update_end = timing.t_update_end;
  double solve_H_time = timing.solve_H_time;

  frame_num_++;
  kdtree_size_end_ = ikdtree_.size();
  aver_time_consu_ =
    aver_time_consu_ * (frame_num_ - 1) / frame_num_ + (t5 - t0) / frame_num_;
  aver_time_icp_ =
    aver_time_icp_ * (frame_num_ - 1) / frame_num_ + (t_update_end - t_update_start) / frame_num_;
  aver_time_match_ =
    aver_time_match_ * (frame_num_ - 1) / frame_num_ + (match_time_) / frame_num_;
  aver_time_incre_ =
    aver_time_incre_ * (frame_num_ - 1) / frame_num_ + (kdtree_incremental_time_) / frame_num_;
  aver_time_solve_ =
    aver_time_solve_ * (frame_num_ - 1) / frame_num_ + (solve_time_ + solve_H_time) / frame_num_;
  aver_time_const_H_time_ =
    aver_time_const_H_time_ * (frame_num_ - 1) / frame_num_ + solve_time_ / frame_num_;
  T1_[time_log_counter_] = meas.lidar_beg_time;
  s_plot_[time_log_counter_] = t5 - t0;
  s_plot2_[time_log_counter_] = feats_undistort_->points.size();
  s_plot3_[time_log_counter_] = kdtree_incremental_time_;
  s_plot4_[time_log_counter_] = kdtree_search_time_;
  s_plot5_[time_log_counter_] = kdtree_delete_counter_;
  s_plot6_[time_log_counter_] = kdtree_delete_time_;
  s_plot7_[time_log_counter_] = kdtree_size_st_;
  s_plot8_[time_log_counter_] = kdtree_size_end_;
  s_plot9_[time_log_counter_] = aver_time_consu_;
  s_plot10_[time_log_counter_] = add_point_size_;
  time_log_counter_++;
  printf(
    "[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f "
    " ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",
    t1 - t0, aver_time_match_, aver_time_solve_, t3 - t1, t5 - t3, aver_time_consu_,
    aver_time_icp_, aver_time_const_H_time_);
  V3D ext_euler = SO3ToEuler(state_point_.offset_R_L_I);
  fout_out_ << setw(20) << meas.lidar_beg_time - first_lidar_time_ << " "
            << euler_cur_.transpose() << " " << state_point_.pos.transpose() << " "
            << ext_euler.transpose() << " " << state_point_.offset_T_L_I.transpose() << " "
            << state_point_.vel.transpose() << " " << state_point_.bg.transpose() << " "
            << state_point_.ba.transpose() << " " << state_point_.grav << " "
            << feats_undistort_->points.size() << endl;
  dump_lio_state_to_log(fp_, meas.lidar_beg_time);
}

void LaserMappingCore::dump_lio_state_to_log(FILE * fp, double lidar_beg_time)
{
  V3D rot_ang(Log(state_point_.rot.toRotationMatrix()));
  fprintf(fp, "%lf ", lidar_beg_time - first_lidar_time_);
  fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                      // Angle
  fprintf(fp, "%lf %lf %lf ", state_point_.pos(0), state_point_.pos(1), state_point_.pos(2));  // Pos
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                           // omega
  fprintf(fp, "%lf %lf %lf ", state_point_.vel(0), state_point_.vel(1), state_point_.vel(2));  // Vel
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                           // Acc
  fprintf(fp, "%lf %lf %lf ", state_point_.bg(0), state_point_.bg(1), state_point_.bg(2));  // Bias_g
  fprintf(fp, "%lf %lf %lf ", state_point_.ba(0), state_point_.ba(1), state_point_.ba(2));  // Bias_a
  fprintf(fp, "%lf %lf %lf ", state_point_.grav[0], state_point_.grav[1], state_point_.grav[2]);
  fprintf(fp, "\r\n");
  fflush(fp);
}

// ---------------------------------------------------------------------------
// Map publishing / shutdown
// ---------------------------------------------------------------------------

PointCloudXYZI::Ptr LaserMappingCore::map_to_publish()
{
  PointCloudXYZI::Ptr laserCloudFullRes(params_.dense_publish_en ? feats_undistort_ : feats_down_body_);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++)
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_wait_pub_ += *laserCloudWorld;

  // Downsample the accumulated map before publishing to RViz to keep it
  // responsive. The full-resolution map in pcl_wait_pub_ is kept intact for
  // save_to_pcd().
  PointCloudXYZI::Ptr laserCloudFullResMap(new PointCloudXYZI());
  downSizeFilterMap_.setInputCloud(pcl_wait_pub_);
  downSizeFilterMap_.filter(*laserCloudFullResMap);

  return laserCloudFullResMap;
}

void LaserMappingCore::record_preprocess_time(int scan_count, double seconds)
{
  s_plot11_[scan_count] = seconds;
}

void LaserMappingCore::on_shutdown()
{
  fout_out_.close();
  fout_pre_.close();
  if (fp_)
  {
    fclose(fp_);
  }

  // Save the accumulated map (only populated when map incremental / pcd save is enabled).
  if (!pcl_wait_save_->empty() && params_.pcd_save_en)
  {
    const std::string file_name = "scans.pcd";
    const std::string all_points_dir = std::string(ROOT_DIR) + "PCD/" + file_name;
    pcl::PCDWriter pcd_writer;
    std::cout << "current scan saved to /PCD/" << file_name << std::endl;
    pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
  }

  // Write the runtime performance log.
  if (params_.runtime_pos_log_enable)
  {
    std::vector<double> t, s_vec, s_vec2, s_vec3, s_vec5;
    FILE * fp2 = nullptr;
    const std::string log_dir = std::string(ROOT_DIR) + "/Log/fast_lio_time_log.csv";
    fp2 = fopen(log_dir.c_str(), "w");
    fprintf(
      fp2,
      "time_stamp, total time, scan point size, incremental time, search time, delete size, delete "
      "time, tree size st, tree size end, add point size, preprocess time\n");
    for (int i = 0; i < time_log_counter_; i++)
    {
      fprintf(
        fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n", T1_[i], s_plot_[i],
        int(s_plot2_[i]), s_plot3_[i], s_plot4_[i], int(s_plot5_[i]), s_plot6_[i],
        int(s_plot7_[i]), int(s_plot8_[i]), int(s_plot10_[i]), s_plot11_[i]);
      t.push_back(T1_[i]);
      s_vec.push_back(s_plot9_[i]);
      s_vec2.push_back(s_plot3_[i] + s_plot6_[i]);
      s_vec3.push_back(s_plot4_[i]);
      s_vec5.push_back(s_plot_[i]);
    }
    fclose(fp2);
  }
}

}  // namespace fast_lio
