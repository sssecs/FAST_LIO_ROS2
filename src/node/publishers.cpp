// Publishing helpers of the localization node: odometry + TF, path, scan
// clouds and the /Laser_map prior-map display cloud.

#include <fast_lio/node/localization_node.hpp>

namespace fast_lio
{

void LocalizationNode::publish_odometry()
{
  odomAftMapped_.header.frame_id = params_.map_frame;
  odomAftMapped_.child_frame_id = params_.base_frame;
  odomAftMapped_.header.stamp = get_ros_time(lidar_end_time_);
  core_->set_posestamp(odomAftMapped_.pose);

  // Fill the covariance BEFORE publishing (the old code published first and
  // wrote the covariance afterwards, so every message carried the previous
  // scan's covariance).
  auto P = core_->get_P();
  for (int i = 0; i < 6; i++)
  {
    int k = i < 3 ? i + 3 : i - 3;
    odomAftMapped_.pose.covariance[i * 6 + 0] = P(k, 3);
    odomAftMapped_.pose.covariance[i * 6 + 1] = P(k, 4);
    odomAftMapped_.pose.covariance[i * 6 + 2] = P(k, 5);
    odomAftMapped_.pose.covariance[i * 6 + 3] = P(k, 0);
    odomAftMapped_.pose.covariance[i * 6 + 4] = P(k, 1);
    odomAftMapped_.pose.covariance[i * 6 + 5] = P(k, 2);
  }
  pub_odom_->publish(odomAftMapped_);

  geometry_msgs::msg::TransformStamped trans;
  trans.header.frame_id = params_.map_frame;
  trans.header.stamp = odomAftMapped_.header.stamp;
  trans.child_frame_id = params_.base_frame;
  trans.transform.translation.x = odomAftMapped_.pose.pose.position.x;
  trans.transform.translation.y = odomAftMapped_.pose.pose.position.y;
  trans.transform.translation.z = odomAftMapped_.pose.pose.position.z;
  trans.transform.rotation.w = odomAftMapped_.pose.pose.orientation.w;
  trans.transform.rotation.x = odomAftMapped_.pose.pose.orientation.x;
  trans.transform.rotation.y = odomAftMapped_.pose.pose.orientation.y;
  trans.transform.rotation.z = odomAftMapped_.pose.pose.orientation.z;
  tf_broadcaster_->sendTransform(trans);
}

void LocalizationNode::publish_path()
{
  core_->set_posestamp(msg_body_pose_);
  msg_body_pose_.header.stamp = get_ros_time(lidar_end_time_);
  msg_body_pose_.header.frame_id = params_.map_frame;

  /*** if path is too large, the rviz will crash ***/
  static int jjj = 0;
  jjj++;
  if (jjj % 10 == 0)
  {
    path_.poses.push_back(msg_body_pose_);
    pub_path_->publish(path_);
  }
}

void LocalizationNode::publish_frame_world()
{
  PointCloudXYZI::Ptr laserCloudFullRes(core_->feats_down_body());
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++)
  {
    core_->RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
  laserCloudmsg.header.stamp = get_ros_time(lidar_end_time_);
  laserCloudmsg.header.frame_id = params_.map_frame;
  pub_laser_cloud_full_->publish(laserCloudmsg);
}

void LocalizationNode::publish_frame_body()
{
  int size = core_->feats_undistort()->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++)
  {
    core_->RGBpointBodyLidarToIMU(
      &core_->feats_undistort()->points[i], &laserCloudIMUBody->points[i]);
  }

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  laserCloudmsg.header.stamp = get_ros_time(lidar_end_time_);
  laserCloudmsg.header.frame_id = params_.base_frame;
  pub_laser_cloud_full_body_->publish(laserCloudmsg);
}

void LocalizationNode::publish_effect_world()
{
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(core_->effct_feat_num(), 1));
  for (int i = 0; i < core_->effct_feat_num(); i++)
  {
    core_->RGBpointBodyToWorld(&core_->laser_cloud_ori()->points[i], &laserCloudWorld->points[i]);
  }
  sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time_);
  laserCloudFullRes3.header.frame_id = params_.map_frame;
  pub_laser_cloud_effect_->publish(laserCloudFullRes3);
}

void LocalizationNode::publish_map()
{
  // The /Laser_map cloud comes straight from the immutable prior map (coarse
  // display resolution). The old implementation accumulated the current scan
  // into a publish cloud and depended on leftover scan-buffer state.
  const auto map = std::atomic_load(&map_);
  if (!map)
  {
    return;
  }
  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(map->publishCloud(), laserCloudmsg);
  laserCloudmsg.header.stamp = this->get_clock()->now();
  laserCloudmsg.header.frame_id = params_.map_frame;
  pub_laser_cloud_map_->publish(laserCloudmsg);
}

}  // namespace fast_lio
