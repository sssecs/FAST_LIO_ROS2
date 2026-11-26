// Mid-360 static-map localization node (ROS 2 Lifecycle).
//
// The node owns only the ROS plumbing: parameters, TF, subscribers/publishers,
// sensor callbacks, buffer synchronization, per-scan orchestration, the
// localization business state machine and message publishing. The algorithm
// lives in LocalizationCore, the map asset in PriorMap and the Open3D
// relocalization in Open3DLocalizer.
//
// Two orthogonal state machines:
//   - the ROS lifecycle (outer) manages infrastructure: the constructor only
//     declares parameters; on_configure() does IO (static TF, prior map) and
//     creates publishers; on_activate() creates subscriptions and the main
//     loop; on_deactivate() stops them but KEEPS the map and static TF;
//     on_cleanup() releases everything.
//   - LocalizationState (inner) describes the runtime localization semantics
//     and only advances while the node is ACTIVE.
//
// Localization flow:
//   1. configure: load a downsampled PCD map into a PriorMap,
//   2. activate: wait for an initial pose on /initialpose,
//   3. refine it against the map with the Open3D relocalizer,
//   4. run the standard FAST-LIO update against the (static) map.

#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <pcl_conversions/pcl_conversions.h>

#include <fast_lio/common/common_lib.h>
#include <fast_lio/common/params.hpp>
#include <fast_lio/core/localization_core.hpp>
#include <fast_lio/core/prior_map.hpp>
#include <fast_lio/imu/imu_processing.hpp>
#include <fast_lio/node/localization_state.hpp>
#include <fast_lio/node/telemetry.hpp>
#include <fast_lio/preprocess/preprocess.hpp>
#include <fast_lio/relocalization/open3d_localizer.hpp>

namespace fast_lio
{

class LocalizationNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  explicit LocalizationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  LocalizationState state() const { return state_.load(); }

  // ---------------------------------------------------------------------------
  // Lifecycle transitions
  // ---------------------------------------------------------------------------
  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_error(const rclcpp_lifecycle::State & state) override;

private:
  // Constructor: declare all parameters (no IO, no ROS graph entities).
  void declare_params();
  // on_configure: read the declared parameters into params_.
  void load_params();
  // on_configure: build the PriorMap from params_.map_path and inject it into
  // the core and the relocalizer. Throws on failure.
  void load_map();
  void set_state(LocalizationState state);

  Params params_;
  std::unique_ptr<LocalizationCore> core_;
  PriorMapPtr map_;  // active prior map (also held by the core)

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
  double lidar_mean_scantime_ = 0.0;
  int scan_num_ = 0;
  double lidar_end_time_ = 0.0;

  bool sync_packages(MeasureGroup & meas);
  void clear_buffers();

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
  // ROS handles. Publishers are created in on_configure and only publish while
  // ACTIVE; subscriptions and timers exist only while ACTIVE.
  // ---------------------------------------------------------------------------
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_full_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_full_body_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_effect_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_map_;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_init_pose_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  geometry_msgs::msg::TransformStamped tf_lidar_to_base_;

  // ---------------------------------------------------------------------------
  // Localization state
  // ---------------------------------------------------------------------------
  std::atomic<LocalizationState> state_{LocalizationState::kNoMap};
  // True when a fresh /initialpose is waiting to be consumed by the re-init
  // branch in timer_callback().
  std::atomic<bool> initial_pose_set_{false};
  geometry_msgs::msg::PoseWithCovarianceStamped init_pose_;
  // Open3D registration core; refines the /initialpose guess against the map.
  Open3DLocalizer localizer_;
  bool localizer_ready_ = false;  // set once the loaded map has been fed to localizer_

  // ---------------------------------------------------------------------------
  // Diagnostics (node/diagnostics.cpp)
  // ---------------------------------------------------------------------------
  std::unique_ptr<diagnostic_updater::Updater> updater_;
  Telemetry telemetry_;
  void setup_diagnostics();
  void fill_sensor_diagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat);
  void fill_state_diagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat);
  void fill_tracking_diagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat);
  void fill_relocation_diagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat);
  static double steady_now_sec()
  {
    return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
  }

  // ---------------------------------------------------------------------------
  // Callbacks (node/sensor_callbacks.cpp, node/processing.cpp)
  // ---------------------------------------------------------------------------
  void timer_callback();
  void initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg);
  void imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr & msg_in);

  // ---------------------------------------------------------------------------
  // Publishing helpers (node/publishers.cpp)
  // ---------------------------------------------------------------------------
  void publish_odometry();
  void publish_path();
  void publish_frame_world();
  void publish_frame_body();
  void publish_effect_world();
  void publish_map();
};

}  // namespace fast_lio
