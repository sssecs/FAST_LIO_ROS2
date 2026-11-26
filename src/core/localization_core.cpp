// FAST-LIO localization algorithm core (implementation). See localization_core.hpp.

#include <fast_lio/core/localization_core.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <omp.h>

// Measurement noise of a matched lidar point (m).
#define LASER_POINT_COV (0.001)

namespace
{
// The esekfom library stores the measurement model as a raw function pointer,
// so the core instance is reached through this single file-local pointer. It is
// cleared by the destructor so a destroyed core is never called back.
fast_lio::LocalizationCore * g_core = nullptr;
}  // namespace

namespace fast_lio
{

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

LocalizationCore::LocalizationCore(const Params & params, const rclcpp::Logger & logger)
  : params_(params), logger_(logger)
{
  g_core = this;

  imu_process_ = std::make_shared<ImuProcess>();
  num_max_iterations_ = params_.max_iteration;

  memset(point_selected_surf_, true, sizeof(point_selected_surf_));
  memset(res_last_, -1000.0f, sizeof(res_last_));
  downSizeFilterSurf_.setLeafSize(
    params_.filter_size_surf, params_.filter_size_surf, params_.filter_size_surf);
}

LocalizationCore::~LocalizationCore()
{
  if (g_core == this)
  {
    g_core = nullptr;
  }
}

// ---------------------------------------------------------------------------
// EKF / seed initialization
// ---------------------------------------------------------------------------

void LocalizationCore::init_filter()
{
  imu_process_->Reset();
  V3D Lidar_T;
  M3D Lidar_R;
  Lidar_T << VEC_FROM_ARRAY(params_.extrin_T);
  Lidar_R << MAT_FROM_ARRAY(params_.extrin_R);
  imu_process_->set_extrinsic(Lidar_T, Lidar_R);
  imu_process_->set_gyr_cov(V3D(params_.gyr_cov, params_.gyr_cov, params_.gyr_cov));
  imu_process_->set_acc_cov(V3D(params_.acc_cov, params_.acc_cov, params_.acc_cov));
  imu_process_->set_gyr_bias_cov(V3D(params_.b_gyr_cov, params_.b_gyr_cov, params_.b_gyr_cov));
  imu_process_->set_acc_bias_cov(V3D(params_.b_acc_cov, params_.b_acc_cov, params_.b_acc_cov));

  kf_ = std::make_unique<esekfom::esekf<state_ikfom, 12, input_ikfom>>();
  fill(epsi_, epsi_ + 23, 0.001);
  kf_->init_dyn_share(get_f, df_dx, df_dw, &LocalizationCore::h_share_model, num_max_iterations_, epsi_);
}

void LocalizationCore::set_initial_localization(const Eigen::Matrix4d & T_map_base)
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

void LocalizationCore::rearm_reinit(const Eigen::Vector3d & seed_pos, double lidar_beg_time)
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
  // (checked in localization_health_check()).
  relocalized_verify_ = params_.relocalization_verify_en;
  relocalized_pos_ = last_good_pos_;
  relocalized_time_ = last_good_time_;
  relocalized_path_ = 0.0;
}

void LocalizationCore::set_first_lidar_time(double t)
{
  first_lidar_time_ = t;
  imu_process_->first_lidar_time = first_lidar_time_;
}

// ---------------------------------------------------------------------------
// Point transforms
// ---------------------------------------------------------------------------

void LocalizationCore::point_body_to_world(PointType const * const pi, PointType * const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) + state_point_.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LocalizationCore::RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) + state_point_.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LocalizationCore::RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

// ---------------------------------------------------------------------------
// Main per-scan processing
// ---------------------------------------------------------------------------

bool LocalizationCore::process_scan(const MeasureGroup & meas, ScanTiming & timing)
{
  if (!map_)
  {
    return false;
  }

  match_time_ = 0;
  solve_time_ = 0;

  imu_process_->Process(meas, *kf_, feats_undistort_);
  timing.t_imu_end = omp_get_wtime();
  state_point_ = kf_->get_x();
  pos_lid_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;

  if (feats_undistort_->empty())
  {
    RCLCPP_WARN(logger_, "No point, skip this scan!");
    return false;
  }

  /*** downsample the feature points in a scan ***/
  downSizeFilterSurf_.setInputCloud(feats_undistort_);
  downSizeFilterSurf_.filter(*feats_down_body_);
  timing.t1 = omp_get_wtime();
  feats_down_size_ = feats_down_body_->points.size();

  if (feats_down_size_ < 5)
  {
    RCLCPP_WARN(logger_, "Too few feature points, skip this scan!");
    return false;
  }

  normvec_->resize(feats_down_size_);
  feats_down_world_->resize(feats_down_size_);
  pointSearchInd_surf_.resize(feats_down_size_);
  Nearest_Points_.resize(feats_down_size_);

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

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

HealthReport LocalizationCore::localization_health_check(double lidar_beg_time)
{
  HealthReport report;

  if (!params_.failure_detect_en)
  {
    return report;
  }

  // A non-finite pose means the filter diverged -> lost.
  if (!state_point_.pos.allFinite())
  {
    localization_failed_ = true;
    report.ok = false;
    report.reason = HealthReport::Reason::kNonFinitePose;
    return report;
  }

  // Post-relocalization fly-away guard: the robot is expected to stay put right
  // after a re-seed, so the pose must stay near the re-seeded position during a
  // short verification window. Both the straight-line drift from the seed and
  // the total path the pose travels are bounded.
  if (relocalized_verify_)
  {
    const double since_seed = lidar_beg_time - relocalized_time_;
    report.verify_drift = (state_point_.pos - relocalized_pos_).norm();
    report.verify_path = relocalized_path_;
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
      report.verify_path = relocalized_path_;
      if (
        report.verify_drift > params_.relocalization_verify_max_drift ||
        relocalized_path_ > params_.relocalization_verify_max_path)
      {
        relocalized_verify_ = false;
        localization_failed_ = true;
        report.ok = false;
        report.reason = HealthReport::Reason::kVerifyDrift;
        return report;
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
  map_->nearestSearch(pose_pt, 1, points_near, point_sqdis);
  report.nearest_dist = points_near.empty()
                          ? std::numeric_limits<double>::infinity()
                          : std::sqrt(static_cast<double>(point_sqdis[0]));

  // Motion: pose displacement rate since the last healthy update. A jump that
  // is impossible for the robot in the elapsed time means the match is wrong.
  if (last_good_valid_)
  {
    const double dt = lidar_beg_time - last_good_time_;
    if (dt > 1e-6)
    {
      report.speed = (state_point_.pos - last_good_pos_).norm() / dt;
    }
  }

  if (report.nearest_dist > params_.max_pose_to_map_dist)
  {
    localization_failed_ = true;
    report.ok = false;
    report.reason = HealthReport::Reason::kOffMap;
    return report;
  }
  if (report.speed > params_.max_pose_speed)
  {
    localization_failed_ = true;
    report.ok = false;
    report.reason = HealthReport::Reason::kSpeedJump;
    return report;
  }

  last_good_valid_ = true;
  last_good_pos_ = state_point_.pos;
  last_good_time_ = lidar_beg_time;
  return report;
}

// ---------------------------------------------------------------------------
// Measurement model (called by the esekfom update through the trampoline)
// ---------------------------------------------------------------------------

void LocalizationCore::h_share_model(
  state_ikfom & s, esekfom::dyn_share_datastruct<double> & ekfom_data)
{
  g_core->h_share_model_impl(s, ekfom_data);
}

void LocalizationCore::h_share_model_impl(
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
      map_->nearestSearch(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
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

}  // namespace fast_lio
