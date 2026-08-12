// Mid-360 localization node for FAST-LIO.
//
// This node runs the FAST-LIO iterated-EKF against a pre-built point-cloud map
// (loaded once at startup) instead of building the map online. The lidar input
// is a Livox Mid-360 streamed as sensor_msgs/PointCloud2 and the sensor data
// (lidar + IMU) is transformed from the lidar frame to the robot base frame
// through a static TF that must be available at startup.
//
// The FAST-LIO algorithm itself lives in LaserMappingCore (laser_mapping_core);
// this node owns only the ROS plumbing — parameters, TF, subscribers/publishers,
// sensor callbacks, buffer synchronization, per-scan orchestration and message
// publishing — and calls into the core for every algorithm step.
//
// Localization flow:
//   1. load a downsampled PCD map into the ikd-Tree,
//   2. wait for an initial pose on /initialpose (RViz "2D Pose Estimate"),
//   3. run the standard FAST-LIO update against the (static) map.

#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <common_lib.h>
#include <fast_lio/laser_mapping_core.hpp>
#include <fast_lio/preprocess.hpp>
#include <fast_lio/open3d_localizer.hpp>

namespace fast_lio
{

class LaserMappingNode : public rclcpp::Node
{
public:
  explicit LaserMappingNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~LaserMappingNode() override;

private:
  void load_params();

  LaserMappingCore::Params params_;
  std::unique_ptr<LaserMappingCore> core_;

  MeasureGroup measures_;

  // ---------------------------------------------------------------------------
  // Sensor buffers and synchronization
  // ---------------------------------------------------------------------------
  std::mutex mtx_buffer_;
  std::deque<double> time_buffer_;
  std::deque<PointCloudXYZI::Ptr> lidar_buffer_;
  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer_;
  bool lidar_pushed_ = false;
  bool is_first_lidar_ = true;
  double last_timestamp_lidar_ = 0.0;
  double last_timestamp_imu_ = -1.0;
  double timediff_lidar_wrt_imu_ = 0.0;
  bool timediff_set_flg_ = false;
  double lidar_mean_scantime_ = 0.0;
  int scan_num_ = 0;
  int scan_count_ = 0;
  int publish_count_ = 0;
  double lidar_end_time_ = 0.0;

  bool sync_packages(MeasureGroup & meas);

  // ---------------------------------------------------------------------------
  // Preprocessing
  // ---------------------------------------------------------------------------
  std::shared_ptr<Preprocess> preprocess_;

  // ---------------------------------------------------------------------------
  // Messages
  // ---------------------------------------------------------------------------
  nav_msgs::msg::Path path_;
  nav_msgs::msg::Odometry odomAftMapped_;
  geometry_msgs::msg::PoseStamped msg_body_pose_;

  // ---------------------------------------------------------------------------
  // ROS handles
  // ---------------------------------------------------------------------------
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_full_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_full_body_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_effect_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_map_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_init_pose_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr map_pub_timer_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  geometry_msgs::msg::TransformStamped tf_lidar_to_base_;

  // ---------------------------------------------------------------------------
  // Localization state
  // ---------------------------------------------------------------------------
  // True when a fresh /initialpose is waiting to be consumed by the re-init
  // branch in timer_callback(). In localization mode it starts false so the node
  // does nothing until the operator's first /initialpose; the constructor sets
  // it true for mapping mode (self-start from the origin on the first scan).
  std::atomic<bool> initial_pose_set_{false};
  geometry_msgs::msg::PoseWithCovarianceStamped init_pose_;
  // Open3D registration core; refines the /initialpose guess against the map.
  Open3DLocalizer localizer_;
  bool localizer_ready_ = false;  // set once the loaded map has been fed to localizer_

  // ---------------------------------------------------------------------------
  // Callbacks
  // ---------------------------------------------------------------------------
  void timer_callback();
  void map_publish_callback();
  void map_save_callback(
    const std_srvs::srv::Trigger::Request::SharedPtr req,
    std_srvs::srv::Trigger::Response::SharedPtr res);
  void initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg);
  void imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr & msg_in);

  // ---------------------------------------------------------------------------
  // Publishing helpers
  // ---------------------------------------------------------------------------
  void publish_odometry();
  void publish_path();
  void publish_frame_world();
  void publish_frame_body();
  void publish_effect_world();
  void publish_map();
  void save_to_pcd();
};

}  // namespace fast_lio
