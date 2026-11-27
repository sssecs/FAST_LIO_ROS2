// Synchronous relocalization service (~/relocalize).
//
// The /initialpose topic stays the fire-and-forget interface (RViz "2D Pose
// Estimate"); this service is the programmatic one: the caller provides the
// same initial-pose guess and blocks until the main loop has refined it
// against the loaded map (Open3D registration) and seeded — or refused to
// seed — the EKF, receiving the match quality in the response.
//
// The service does NOT run the alignment itself. It queues the guess into the
// same single-slot initial-pose pipeline as the topic (newest request wins)
// and waits on reloc_cv_ for the main loop's re-init branch to record the
// outcome under a matching sequence number. Waiting never holds
// pipeline_mtx_, so the main loop keeps processing scans; a map switch in
// flight simply delays the outcome. The 10 s timeout covers a stalled sensor
// stream (the re-init branch only runs when a synchronized scan arrives).

#include <fast_lio/node/localization_node.hpp>

#include <lifecycle_msgs/msg/state.hpp>

namespace fast_lio
{

geometry_msgs::msg::Pose LocalizationNode::matrix_to_pose(const Eigen::Matrix4d & T)
{
  geometry_msgs::msg::Pose p;
  p.position.x = T(0, 3);
  p.position.y = T(1, 3);
  p.position.z = T(2, 3);
  const Eigen::Quaterniond q(Eigen::Matrix3d(T.block<3, 3>(0, 0)));
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
  p.orientation.w = q.w();
  return p;
}

void LocalizationNode::relocalize_cbk(
  const std::shared_ptr<fast_lio::srv::Relocalize::Request> req,
  std::shared_ptr<fast_lio::srv::Relocalize::Response> res)
{
  if (get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE)
  {
    res->success = false;
    res->message = "node is not ACTIVE";
    return;
  }
  if (!std::atomic_load(&map_))
  {
    res->success = false;
    res->message = "no map loaded";
    return;
  }
  const auto & frame = req->initial_pose.header.frame_id;
  if (!frame.empty() && frame != params_.map_frame)
  {
    res->success = false;
    res->message = "initial_pose is in frame '" + frame + "', expected '" + params_.map_frame + "'";
    return;
  }

  uint64_t my_seq = 0;
  {
    std::lock_guard<std::mutex> lock(reloc_mtx_);
    if (reloc_waiter_)
    {
      res->success = false;
      res->message = "another relocalization request is still pending";
      return;
    }
    reloc_waiter_ = true;
    my_seq = ++reloc_req_seq_;
  }

  {
    std::lock_guard<std::mutex> lock(init_pose_mtx_);
    init_pose_ = req->initial_pose;
  }
  seamless_seed_.store(false);  // a service request always goes through scan matching
  initial_pose_set_.store(true);
  RCLCPP_INFO(
    get_logger(), "Relocalize service: initial pose queued (seq %lu), waiting for scan matching...",
    static_cast<unsigned long>(my_seq));

  std::unique_lock<std::mutex> lock(reloc_mtx_);
  const bool done = reloc_cv_.wait_for(lock, std::chrono::seconds(10), [&] {
    return reloc_done_seq_ >= my_seq;
  });
  reloc_waiter_ = false;
  if (!done)
  {
    res->success = false;
    res->message = "timed out after 10 s waiting for a synchronized scan";
    return;
  }
  *res = reloc_outcome_;
  RCLCPP_INFO(get_logger(), "Relocalize service (seq %lu): %s",
              static_cast<unsigned long>(my_seq),
              res->success ? "localized" : res->message.c_str());
}

void LocalizationNode::complete_pending_reloc(uint64_t seq, fast_lio::srv::Relocalize::Response outcome)
{
  if (seq == 0)
  {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(reloc_mtx_);
    reloc_outcome_ = std::move(outcome);
    reloc_done_seq_ = seq;
  }
  reloc_cv_.notify_all();
}

}  // namespace fast_lio
