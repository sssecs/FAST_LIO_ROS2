// Localization node: lifecycle transitions and map loading.
//
// Lifecycle contract (see localization_node.hpp):
//   constructor    declare parameters only — no IO, no TF, no ROS graph.
//   on_configure   read params, static TF, load the prior map, create pubs.
//   on_activate    create subscriptions + main loop, reset the filter.
//   on_deactivate  stop the main loop and subscriptions, KEEP map + static TF.
//   on_cleanup     release everything (map, core, ROS handles).
//
// Sensor callbacks live in node/sensor_callbacks.cpp, per-scan orchestration
// in node/processing.cpp and publishing helpers in node/publishers.cpp.

#include <fast_lio/node/localization_node.hpp>

#include <functional>

namespace fast_lio
{

using CallbackReturn = LocalizationNode::CallbackReturn;

// ---------------------------------------------------------------------------
// Construction (UNCONFIGURED): parameters only
// ---------------------------------------------------------------------------

LocalizationNode::LocalizationNode(const rclcpp::NodeOptions & options)
  : rclcpp_lifecycle::LifecycleNode(
      "fastlio_localization", "/TCT_Docent/locomotion/fastlio", options)
{
  preprocess_ = std::make_shared<Preprocess>();
  declare_params();
}

// ---------------------------------------------------------------------------
// on_configure: params, static TF, prior map, publishers
// ---------------------------------------------------------------------------

CallbackReturn LocalizationNode::on_configure(const rclcpp_lifecycle::State &)
{
  load_params();

  RCLCPP_INFO(
    get_logger(), "lidar topic: %s, imu topic: %s, frames: %s <- %s -> %s",
    params_.lidar_topic.c_str(), params_.imu_topic.c_str(), params_.map_frame.c_str(),
    params_.base_frame.c_str(), params_.lidar_frame.c_str());

  path_.header.stamp = get_clock()->now();
  path_.header.frame_id = params_.map_frame;

  // --- static lidar -> base TF -----------------------------------------------
  // Required to express the lidar scan and the IMU in the robot base frame.
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  try
  {
    tf_lidar_to_base_ = tf_buffer_->lookupTransform(
      params_.base_frame, params_.lidar_frame, tf2::TimePointZero,
      std::chrono::seconds(static_cast<int64_t>(params_.tf_lookup_timeout)));
    RCLCPP_INFO(
      get_logger(), "Static TF %s -> %s available.", params_.lidar_frame.c_str(),
      params_.base_frame.c_str());
  }
  catch (const tf2::TransformException & ex)
  {
    RCLCPP_ERROR(
      get_logger(), "Could not get static TF %s -> %s: %s", params_.lidar_frame.c_str(),
      params_.base_frame.c_str(), ex.what());
    on_cleanup(rclcpp_lifecycle::State());
    return CallbackReturn::FAILURE;
  }

  // --- algorithm core, relocalizer and prior map ------------------------------
  core_ = std::make_unique<LocalizationCore>(params_, get_logger());
  localizer_ = std::make_unique<Open3DLocalizer>(params_.open3d);
  init_z_ = params_.init_z;
  if (!params_.map_path.empty())
  {
    try
    {
      load_map();
    }
    catch (const std::exception & ex)
    {
      RCLCPP_ERROR(get_logger(), "Failed to load prior map: %s", ex.what());
      on_cleanup(rclcpp_lifecycle::State());
      return CallbackReturn::FAILURE;
    }
  }
  else
  {
    RCLCPP_WARN(get_logger(), "No map_path configured; the node will stay in NO_MAP.");
  }

  // --- callback groups ---------------------------------------------------------
  sub_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  main_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  heavy_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // --- map switch service (available from INACTIVE on) -------------------------
  switch_map_srv_ = create_service<fast_lio::srv::SwitchMap>(
    "~/switch_map",
    std::bind(
      &LocalizationNode::switch_map_cbk, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, heavy_group_);

  // --- publishers (inactive until on_activate) --------------------------------
  // Relative names resolve under the node namespace
  // (/TCT_Docent/locomotion/fastlio).
  pub_laser_cloud_full_ = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered", 20);
  pub_laser_cloud_full_body_ = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered_body", 20);
  pub_laser_cloud_effect_ = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_effected", 20);
  // transient_local so RViz (which subscribes after the map was published once)
  // still receives the loaded map.
  pub_laser_cloud_map_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    "Laser_map", rclcpp::QoS(1).transient_local());
  pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("Odometry", 20);
  pub_path_ = create_publisher<nav_msgs::msg::Path>("path", 20);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  setup_diagnostics();

  RCLCPP_INFO(get_logger(), "on_configure done.");
  return CallbackReturn::SUCCESS;
}

// ---------------------------------------------------------------------------
// on_activate: subscriptions, main loop, filter reset
// ---------------------------------------------------------------------------

CallbackReturn LocalizationNode::on_activate(const rclcpp_lifecycle::State & state)
{
  pub_laser_cloud_full_->on_activate();
  pub_laser_cloud_full_body_->on_activate();
  pub_laser_cloud_effect_->on_activate();
  pub_laser_cloud_map_->on_activate();
  pub_odom_->on_activate();
  pub_path_->on_activate();

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = sub_group_;
  sub_pcl_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    params_.lidar_topic, rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg) { this->standard_pcl_cbk(msg); },
    sub_options);
  sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
    params_.imu_topic, 10,
    [this](const sensor_msgs::msg::Imu::ConstSharedPtr & msg) { this->imu_cbk(msg); },
    sub_options);
  // /initialpose only stores the message and sets a flag; the heavy work
  // (relocalization) happens in the main loop. Default callback group.
  sub_init_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    params_.initial_pose_topic, 10,
    [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) { this->initial_pose_callback(msg); });

  // Fresh start: drop any pose from a previous active period; the operator
  // must provide a new /initialpose.
  clear_buffers();
  {
    std::lock_guard<std::mutex> lock(pipeline_mtx_);
    core_->reset_filter();
  }
  initial_pose_set_.store(false);
  path_.poses.clear();
  set_state(std::atomic_load(&map_) ? LocalizationState::kUnlocalized : LocalizationState::kNoMap);

  // Publish the loaded map once so RViz can display it before the first pose.
  // transient_local caches the sample, so late subscribers still receive it.
  if (std::atomic_load(&map_) && params_.map_en)
  {
    publish_map();
  }

  // Create the main loop LAST so it cannot fire while activation is still
  // resetting state.
  auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
  timer_ = rclcpp::create_timer(
    this, get_clock(), period_ms, std::bind(&LocalizationNode::timer_callback, this), main_group_);

  RCLCPP_INFO(get_logger(), "on_activate done. State: %s.", toString(state_.load()));
  return CallbackReturn::SUCCESS;
}

// ---------------------------------------------------------------------------
// on_deactivate: stop the main loop, keep the map and static TF
// ---------------------------------------------------------------------------

CallbackReturn LocalizationNode::on_deactivate(const rclcpp_lifecycle::State &)
{
  timer_.reset();
  sub_pcl_.reset();
  sub_imu_.reset();
  sub_init_pose_.reset();
  clear_buffers();
  // Wait out any in-flight main-loop iteration before reporting inactive.
  {
    std::lock_guard<std::mutex> lock(pipeline_mtx_);
  }

  pub_laser_cloud_full_->on_deactivate();
  pub_laser_cloud_full_body_->on_deactivate();
  pub_laser_cloud_effect_->on_deactivate();
  pub_laser_cloud_map_->on_deactivate();
  pub_odom_->on_deactivate();
  pub_path_->on_deactivate();

  RCLCPP_INFO(get_logger(), "on_deactivate done (map and static TF kept).");
  return CallbackReturn::SUCCESS;
}

// ---------------------------------------------------------------------------
// on_cleanup / on_shutdown / on_error: release everything
// ---------------------------------------------------------------------------

CallbackReturn LocalizationNode::on_cleanup(const rclcpp_lifecycle::State &)
{
  timer_.reset();
  sub_pcl_.reset();
  sub_imu_.reset();
  sub_init_pose_.reset();
  switch_map_srv_.reset();
  clear_buffers();

  pub_laser_cloud_full_.reset();
  pub_laser_cloud_full_body_.reset();
  pub_laser_cloud_effect_.reset();
  pub_laser_cloud_map_.reset();
  pub_odom_.reset();
  pub_path_.reset();

  tf_broadcaster_.reset();
  tf_listener_.reset();
  tf_buffer_.reset();

  {
    // Serializes with any in-flight main-loop iteration or map switch.
    std::lock_guard<std::mutex> lock(pipeline_mtx_);
    localizer_.reset();
    localizer_ready_ = false;
    std::atomic_store(&map_, PriorMapPtr{});
    core_.reset();  // also clears the esekfom trampoline pointer
  }
  sub_group_.reset();
  main_group_.reset();
  heavy_group_.reset();
  updater_.reset();
  telemetry_.reset();
  initial_pose_set_.store(false);
  set_state(LocalizationState::kNoMap);

  RCLCPP_INFO(get_logger(), "on_cleanup done.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn LocalizationNode::on_shutdown(const rclcpp_lifecycle::State &)
{
  timer_.reset();
  sub_pcl_.reset();
  sub_imu_.reset();
  sub_init_pose_.reset();
  return CallbackReturn::SUCCESS;
}

CallbackReturn LocalizationNode::on_error(const rclcpp_lifecycle::State &)
{
  RCLCPP_ERROR(get_logger(), "on_error: releasing resources.");
  on_cleanup(rclcpp_lifecycle::State());
  return CallbackReturn::SUCCESS;
}

// ---------------------------------------------------------------------------
// Map loading (called from on_configure)
// ---------------------------------------------------------------------------

void LocalizationNode::load_map()
{
  RCLCPP_INFO(
    get_logger(), "Loading prior map from %s (leaf %.3f m)...", params_.map_path.c_str(),
    params_.leaf_size);
  auto map = std::make_shared<PriorMap>(params_.map_path, params_.leaf_size, params_.map_publish_leaf);
  RCLCPP_INFO(
    get_logger(), "Prior map ready: %zu raw -> %zu matching / %zu display points.",
    map->info().raw_points, map->info().map_points, map->info().publish_points);

  core_->set_map(map);

  if (params_.relocalization_en)
  {
    localizer_->setMap(map->cloudXyz());
    localizer_ready_ = true;
    RCLCPP_INFO(get_logger(), "Open3D relocalizer map ready: %zu points.", localizer_->mapPoints());
  }

  std::atomic_store(&map_, PriorMapPtr(std::move(map)));
}

void LocalizationNode::set_state(LocalizationState state)
{
  const auto prev = state_.exchange(state);
  if (prev != state)
  {
    RCLCPP_INFO(get_logger(), "State: %s -> %s", toString(prev), toString(state));
  }
}

void LocalizationNode::clear_buffers()
{
  std::lock_guard<std::mutex> lock(mtx_buffer_);
  lidar_buffer_.clear();
  time_buffer_.clear();
  imu_buffer_.clear();
  lidar_pushed_ = false;
  is_first_lidar_ = true;
  last_timestamp_lidar_ = 0.0;
  last_timestamp_imu_ = -1.0;
  lidar_mean_scantime_ = 0.0;
  scan_num_ = 0;
}

}  // namespace fast_lio
