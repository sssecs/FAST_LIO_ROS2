// Sensor-side callbacks of the localization node: lidar/IMU ingestion (with
// the static lidar->base transform applied), the /initialpose intake and the
// lidar+IMU buffer synchronization. These are only wired while the node is in
// the ACTIVE lifecycle state.

#include <fast_lio/node/localization_node.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

namespace fast_lio
{

void LocalizationNode::standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
{
  std::lock_guard<std::mutex> lock(mtx_buffer_);

  // Transform the scan from the lidar frame to the base frame. A failure here
  // must not kill the node: drop the frame and report it instead.
  sensor_msgs::msg::PointCloud2 cloud_transformed;
  try
  {
    tf2::doTransform(*msg, cloud_transformed, tf_lidar_to_base_);
  }
  catch (const tf2::TransformException & ex)
  {
    RCLCPP_ERROR(get_logger(), "LiDAR transform failed, dropping frame: %s", ex.what());
    return;
  }

  double cur_time = get_time_sec(msg->header.stamp);

  if (!is_first_lidar_ && cur_time < last_timestamp_lidar_)
  {
    RCLCPP_WARN(get_logger(), "lidar loop back, clear buffer");
    lidar_buffer_.clear();
    std::lock_guard<std::mutex> tlock(telemetry_.mtx);
    telemetry_.d.lidar_regressions++;
  }
  is_first_lidar_ = false;

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  preprocess_->process(cloud_transformed, ptr);

  lidar_buffer_.push_back(ptr);
  time_buffer_.push_back(cur_time);
  last_timestamp_lidar_ = cur_time;

  // Diagnostics: frame counters, rate EMA and queue depth.
  const double now = steady_now_sec();
  std::lock_guard<std::mutex> tlock(telemetry_.mtx);
  if (telemetry_.d.last_lidar_wall > 0.0)
  {
    const double dt = now - telemetry_.d.last_lidar_wall;
    if (dt > 1e-6)
    {
      const double inst = 1.0 / dt;
      telemetry_.d.lidar_hz =
        telemetry_.d.lidar_hz <= 0.0 ? inst : 0.9 * telemetry_.d.lidar_hz + 0.1 * inst;
    }
  }
  telemetry_.d.last_lidar_wall = now;
  telemetry_.d.lidar_frames++;
  telemetry_.d.lidar_queue = lidar_buffer_.size();
}

void LocalizationNode::imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr & msg_in)
{
  // Rotate the IMU measurement from the lidar frame to the base frame using the
  // cached static transform.
  sensor_msgs::msg::Imu imu_transformed = *msg_in;

  tf2::Quaternion q_tf;
  tf2::fromMsg(tf_lidar_to_base_.transform.rotation, q_tf);

  tf2::Quaternion q_imu;
  tf2::fromMsg(msg_in->orientation, q_imu);
  imu_transformed.orientation = tf2::toMsg(q_tf * q_imu);

  tf2::Vector3 gyro(
    msg_in->angular_velocity.x, msg_in->angular_velocity.y, msg_in->angular_velocity.z);
  gyro = tf2::quatRotate(q_tf, gyro);
  imu_transformed.angular_velocity.x = gyro.x();
  imu_transformed.angular_velocity.y = gyro.y();
  imu_transformed.angular_velocity.z = gyro.z();

  tf2::Vector3 accel(
    msg_in->linear_acceleration.x, msg_in->linear_acceleration.y, msg_in->linear_acceleration.z);
  accel = tf2::quatRotate(q_tf, accel);
  imu_transformed.linear_acceleration.x = accel.x();
  imu_transformed.linear_acceleration.y = accel.y();
  imu_transformed.linear_acceleration.z = accel.z();

  auto msg = std::make_shared<sensor_msgs::msg::Imu>(imu_transformed);
  msg->header.stamp =
    get_ros_time(get_time_sec(msg_in->header.stamp) - params_.time_offset_lidar_to_imu);

  if (std::abs(timediff_lidar_wrt_imu_) > 0.1 && params_.time_sync_en)
  {
    msg->header.stamp = rclcpp::Time(timediff_lidar_wrt_imu_ + get_time_sec(msg_in->header.stamp));
  }

  double timestamp = get_time_sec(msg->header.stamp);

  {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (timestamp < last_timestamp_imu_)
    {
      RCLCPP_WARN(get_logger(), "IMU loop back, clear buffer");
      imu_buffer_.clear();
    }
    last_timestamp_imu_ = timestamp;
    imu_buffer_.push_back(msg);
  }

  // Diagnostics: frame counters, rate EMA and queue depth.
  const double now = steady_now_sec();
  std::lock_guard<std::mutex> tlock(telemetry_.mtx);
  if (telemetry_.d.last_imu_wall > 0.0)
  {
    const double dt = now - telemetry_.d.last_imu_wall;
    if (dt > 1e-6)
    {
      const double inst = 1.0 / dt;
      telemetry_.d.imu_hz =
        telemetry_.d.imu_hz <= 0.0 ? inst : 0.9 * telemetry_.d.imu_hz + 0.1 * inst;
    }
  }
  telemetry_.d.last_imu_wall = now;
  telemetry_.d.imu_frames++;
  telemetry_.d.imu_queue = imu_buffer_.size();
}

void LocalizationNode::initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
  if (state_.load() == LocalizationState::kNoMap)
  {
    RCLCPP_WARN(get_logger(), "Initial pose ignored: no map loaded.");
    return;
  }
  init_pose_ = *msg;
  initial_pose_set_.store(true);
  RCLCPP_INFO(get_logger(), "Initial pose received and queued for relocalization.");
}

// ---------------------------------------------------------------------------
// Sensor synchronization
// ---------------------------------------------------------------------------

bool LocalizationNode::sync_packages(MeasureGroup & meas)
{
  if (lidar_buffer_.empty() || imu_buffer_.empty())
  {
    return false;
  }

  /*** push a lidar scan ***/
  if (!lidar_pushed_)
  {
    meas.lidar = lidar_buffer_.front();
    meas.lidar_beg_time = time_buffer_.front();
    if (meas.lidar->points.size() <= 1)  // time too little
    {
      lidar_end_time_ = meas.lidar_beg_time + lidar_mean_scantime_;
      RCLCPP_WARN(get_logger(), "Too few input points in scan.");
    }
    else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime_)
    {
      lidar_end_time_ = meas.lidar_beg_time + lidar_mean_scantime_;
    }
    else
    {
      scan_num_++;
      lidar_end_time_ = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
      lidar_mean_scantime_ +=
        (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime_) / scan_num_;
    }

    meas.lidar_end_time = lidar_end_time_;

    lidar_pushed_ = true;
  }

  if (last_timestamp_imu_ < lidar_end_time_)
  {
    return false;
  }

  /*** push imu data, and pop from imu buffer ***/
  double imu_time = get_time_sec(imu_buffer_.front()->header.stamp);
  meas.imu.clear();
  while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_))
  {
    imu_time = get_time_sec(imu_buffer_.front()->header.stamp);
    if (imu_time > lidar_end_time_) break;
    meas.imu.push_back(imu_buffer_.front());
    imu_buffer_.pop_front();
  }

  lidar_buffer_.pop_front();
  time_buffer_.pop_front();
  lidar_pushed_ = false;
  return true;
}

}  // namespace fast_lio
