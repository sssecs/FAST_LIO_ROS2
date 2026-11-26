// Runtime map switching (SwitchMap service).
//
// Double-buffered transaction:
//   1. a reentrancy guard rejects concurrent switches;
//   2. the new PriorMap AND its Open3D relocalizer are built entirely OUTSIDE
//      the pipeline lock — the main loop keeps tracking on the old map, and a
//      failure here leaves the old map fully usable (strong guarantee);
//   3. a short locked section atomically swaps map/core/localizer, resets the
//      filter, buffers and path, and applies the new init_z;
//   4. when ACTIVE, the new map is published on /Laser_map and an optional
//      initial pose from the request is queued for relocalization.
//
// The service exists from on_configure onwards; switching while INACTIVE only
// swaps the map (no publishing, no relocalization) for externally orchestrated
// deactivate -> switch -> activate sequences. Switching never changes the
// lifecycle state.

#include <fast_lio/node/localization_node.hpp>

#include <chrono>

#include <lifecycle_msgs/msg/state.hpp>

namespace fast_lio
{

void LocalizationNode::switch_map_cbk(
  const std::shared_ptr<fast_lio::srv::SwitchMap::Request> req,
  std::shared_ptr<fast_lio::srv::SwitchMap::Response> res)
{
  bool expected = false;
  if (!switching_.compare_exchange_strong(expected, true))
  {
    res->success = false;
    res->message = "another map switch is already in progress";
    return;
  }
  struct Guard
  {
    std::atomic<bool> & flag;
    ~Guard() { flag.store(false); }
  } guard{switching_};

  if (req->map_path.empty())
  {
    res->success = false;
    res->message = "map_path is empty";
    return;
  }

  // ---- build the new map assets outside the pipeline lock --------------------
  RCLCPP_INFO(
    get_logger(), "Switching map: building '%s' (leaf %.3f m)...", req->map_path.c_str(),
    params_.leaf_size);
  const auto t_start = std::chrono::steady_clock::now();
  PriorMapPtr new_map;
  try
  {
    new_map = std::make_shared<PriorMap>(req->map_path, params_.leaf_size, params_.map_publish_leaf);
  }
  catch (const std::exception & ex)
  {
    res->success = false;
    res->message = std::string("map load failed, old map kept: ") + ex.what();
    RCLCPP_ERROR(get_logger(), "Map switch aborted: %s", ex.what());
    return;
  }

  // Build the matching Open3D relocalizer for the new map (also outside the
  // lock — this converts and downsamples millions of points).
  std::unique_ptr<Open3DLocalizer> new_localizer;
  if (params_.relocalization_en)
  {
    new_localizer = std::make_unique<Open3DLocalizer>(params_.open3d);
    new_localizer->setMap(new_map->cloudXyz());
  }
  const double load_time =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  RCLCPP_INFO(
    get_logger(), "New map built in %.2f s: %zu matching points. Swapping...",
    load_time, new_map->info().map_points);

  // ---- short locked swap ------------------------------------------------------
  const bool active =
    get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE;
  const PriorMapPtr old_map = std::atomic_load(&map_);
  {
    std::lock_guard<std::mutex> lock(pipeline_mtx_);
    std::atomic_store(&map_, new_map);
    core_->set_map(new_map);
    if (new_localizer)
    {
      localizer_ = std::move(new_localizer);
      localizer_ready_ = true;
    }
    else
    {
      localizer_.reset();
      localizer_ready_ = false;
    }

    // Clean slate on the new map: the old pose is meaningless here.
    clear_buffers();
    core_->reset_filter();
    initial_pose_set_.store(false);
    path_.poses.clear();
    if (req->override_init_z)
    {
      init_z_ = req->init_z;
      RCLCPP_INFO(get_logger(), "init_z overridden to %.2f m for the new map.", init_z_);
    }
    if (active)
    {
      set_state(LocalizationState::kUnlocalized);
    }
  }
  // The old map (and its kd-tree) is destroyed here, outside the lock.

  // ---- post-swap actions (ACTIVE only) ----------------------------------------
  if (active)
  {
    if (params_.map_en)
    {
      publish_map();
    }
    if (req->has_initial_pose)
    {
      geometry_msgs::msg::PoseWithCovarianceStamped seed;
      seed.header.frame_id = params_.map_frame;
      seed.header.stamp = get_clock()->now();
      seed.pose.pose = req->initial_pose;
      {
        std::lock_guard<std::mutex> lock(init_pose_mtx_);
        init_pose_ = seed;
      }
      initial_pose_set_.store(true);
      RCLCPP_INFO(get_logger(), "Initial pose from the switch request queued for relocalization.");
    }
  }

  res->success = true;
  res->message = active ? "map switched; waiting for relocalization" : "map switched (node inactive)";
  res->map_points = static_cast<uint32_t>(new_map->info().map_points);
  res->load_time_sec = load_time;
  (void)old_map;
}

}  // namespace fast_lio
