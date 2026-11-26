// Parameter declaration and loading for the localization node.
//
// declare_params() runs in the constructor (UNCONFIGURED): every key used from
// the yaml must be declared — the node is created with default NodeOptions, so
// yaml overrides are not auto-declared and a bare get_parameter would silently
// keep the struct defaults. load_params() runs in on_configure and reads the
// (possibly overridden) values into params_.

#include <fast_lio/node/localization_node.hpp>

namespace fast_lio
{

void LocalizationNode::declare_params()
{
  // topics
  this->declare_parameter<std::string>("common.lid_topic", "/livox/lidar");
  this->declare_parameter<std::string>("common.imu_topic", "/livox/imu");
  this->declare_parameter<bool>("common.time_sync_en", false);
  this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
  // frames
  this->declare_parameter<std::string>("frame.map_frame", "map");
  this->declare_parameter<std::string>("frame.base_frame", "base_link");
  this->declare_parameter<std::string>("frame.lidar_frame", "livox_frame");
  // preprocess
  this->declare_parameter<double>("preprocess.blind", 0.5);
  // EKF
  this->declare_parameter<double>("mapping.gyr_cov", 0.1);
  this->declare_parameter<double>("mapping.acc_cov", 0.1);
  this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
  this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
  this->declare_parameter<double>("filter_size_surf", 0.5);
  this->declare_parameter<int>("max_iteration", 4);
  this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
  this->declare_parameter<std::vector<double>>("mapping.extrinsic_T", std::vector<double>());
  this->declare_parameter<std::vector<double>>("mapping.extrinsic_R", std::vector<double>());
  // publish
  this->declare_parameter<bool>("publish.path_en", true);
  this->declare_parameter<bool>("publish.effect_map_en", false);
  this->declare_parameter<bool>("publish.map_en", false);
  this->declare_parameter<bool>("publish.scan_publish_en", true);
  this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
  // logging
  this->declare_parameter<bool>("debug_time_usage", false);
  // localization
  this->declare_parameter<double>("localization.leaf_size", 0.05);
  this->declare_parameter<double>("localization.map_publish_leaf", 0.2);
  this->declare_parameter<std::string>("localization.map_path", "");
  this->declare_parameter<double>("localization.init_z", 1.2);
  this->declare_parameter<double>("localization.tf_lookup_timeout", 5.0);
  this->declare_parameter<std::string>("localization.initial_pose_topic", "/initialpose");
  // Open3D relocalization
  this->declare_parameter<bool>("localization.relocalization.enable", true);
  this->declare_parameter<double>("localization.relocalization.voxel_size", 0.2);
  this->declare_parameter<double>("localization.relocalization.map_voxel_size", 0.2);
  this->declare_parameter<std::vector<double>>("localization.relocalization.scale", std::vector<double>{1.0, 4.0, 6.0});
  this->declare_parameter<int>("localization.relocalization.icp_method", 1);
  this->declare_parameter<int>("localization.relocalization.icp_iteration", 30);
  this->declare_parameter<bool>("localization.relocalization.use_fpfh", true);
  this->declare_parameter<int>("localization.relocalization.ransac_max_iteration", 100000);
  this->declare_parameter<bool>("localization.relocalization.mutual_filter", true);
  this->declare_parameter<bool>("localization.relocalization.statistical_filter", true);
  this->declare_parameter<int>("localization.relocalization.filter_neighbors", 50);
  this->declare_parameter<double>("localization.relocalization.filter_std_ratio", 3.0);
  this->declare_parameter<double>("localization.relocalization.crop_radius", 60.0);
  this->declare_parameter<double>("localization.relocalization.min_fitness", 0.5);
  this->declare_parameter<double>("localization.relocalization.max_translation_delta", 5.0);
  this->declare_parameter<double>("localization.relocalization.max_rotation_delta", 0.785);
  // Localization failure detection.
  this->declare_parameter<bool>("localization.failure_detect_en", true);
  this->declare_parameter<double>("localization.max_pose_to_map_dist", 3.0);
  this->declare_parameter<double>("localization.max_pose_speed", 3.0);
  // Post-relocalization fly-away guard.
  this->declare_parameter<bool>("localization.relocalization_verify_en", true);
  this->declare_parameter<double>("localization.relocalization_verify_window", 2.0);
  this->declare_parameter<double>("localization.relocalization_verify_max_drift", 1.0);
  this->declare_parameter<double>("localization.relocalization_verify_max_path", 2.0);
}

void LocalizationNode::load_params()
{
  // reads
  this->get_parameter("common.lid_topic", params_.lidar_topic);
  this->get_parameter("common.imu_topic", params_.imu_topic);
  this->get_parameter("common.time_sync_en", params_.time_sync_en);
  this->get_parameter("common.time_offset_lidar_to_imu", params_.time_offset_lidar_to_imu);
  this->get_parameter("frame.map_frame", params_.map_frame);
  this->get_parameter("frame.base_frame", params_.base_frame);
  this->get_parameter("frame.lidar_frame", params_.lidar_frame);
  this->get_parameter("preprocess.blind", params_.blind);
  this->get_parameter("mapping.gyr_cov", params_.gyr_cov);
  this->get_parameter("mapping.acc_cov", params_.acc_cov);
  this->get_parameter("mapping.b_gyr_cov", params_.b_gyr_cov);
  this->get_parameter("mapping.b_acc_cov", params_.b_acc_cov);
  this->get_parameter("filter_size_surf", params_.filter_size_surf);
  this->get_parameter("max_iteration", params_.max_iteration);
  this->get_parameter("mapping.extrinsic_est_en", params_.extrinsic_est_en);
  this->get_parameter("mapping.extrinsic_T", params_.extrin_T);
  this->get_parameter("mapping.extrinsic_R", params_.extrin_R);
  this->get_parameter("publish.path_en", params_.path_en);
  this->get_parameter("publish.effect_map_en", params_.effect_map_en);
  this->get_parameter("publish.map_en", params_.map_en);
  this->get_parameter("publish.scan_publish_en", params_.scan_publish_en);
  this->get_parameter("publish.scan_bodyframe_pub_en", params_.scan_bodyframe_pub_en);
  this->get_parameter("debug_time_usage", params_.debug_time_usage);
  this->get_parameter("localization.leaf_size", params_.leaf_size);
  this->get_parameter("localization.map_publish_leaf", params_.map_publish_leaf);
  this->get_parameter("localization.map_path", params_.map_path);
  this->get_parameter("localization.init_z", params_.init_z);
  this->get_parameter("localization.tf_lookup_timeout", params_.tf_lookup_timeout);
  this->get_parameter("localization.initial_pose_topic", params_.initial_pose_topic);
  this->get_parameter("localization.relocalization.enable", params_.relocalization_en);
  this->get_parameter("localization.relocalization.voxel_size", params_.open3d.voxel_size);
  this->get_parameter("localization.relocalization.map_voxel_size", params_.open3d.map_voxel_size);
  this->get_parameter("localization.relocalization.scale", params_.open3d.scale);
  this->get_parameter("localization.relocalization.icp_method", params_.open3d.icp_method);
  this->get_parameter("localization.relocalization.icp_iteration", params_.open3d.icp_iteration);
  this->get_parameter("localization.relocalization.use_fpfh", params_.open3d.use_fpfh);
  this->get_parameter("localization.relocalization.ransac_max_iteration", params_.open3d.ransac_max_iteration);
  this->get_parameter("localization.relocalization.mutual_filter", params_.open3d.mutual_filter);
  this->get_parameter("localization.relocalization.statistical_filter", params_.open3d.statistical_filter);
  this->get_parameter("localization.relocalization.filter_neighbors", params_.open3d.filter_neighbors);
  this->get_parameter("localization.relocalization.filter_std_ratio", params_.open3d.filter_std_ratio);
  this->get_parameter("localization.relocalization.crop_radius", params_.open3d.crop_radius);
  this->get_parameter("localization.relocalization.min_fitness", params_.open3d.min_fitness);
  this->get_parameter("localization.relocalization.max_translation_delta", params_.open3d.max_translation_delta);
  this->get_parameter("localization.relocalization.max_rotation_delta", params_.open3d.max_rotation_delta);
  this->get_parameter("localization.failure_detect_en", params_.failure_detect_en);
  this->get_parameter("localization.max_pose_to_map_dist", params_.max_pose_to_map_dist);
  this->get_parameter("localization.max_pose_speed", params_.max_pose_speed);
  this->get_parameter("localization.relocalization_verify_en", params_.relocalization_verify_en);
  this->get_parameter("localization.relocalization_verify_window", params_.relocalization_verify_window);
  this->get_parameter("localization.relocalization_verify_max_drift", params_.relocalization_verify_max_drift);
  this->get_parameter("localization.relocalization_verify_max_path", params_.relocalization_verify_max_path);

  preprocess_->blind = params_.blind;
  localizer_.setParams(params_.open3d);
}

}  // namespace fast_lio
