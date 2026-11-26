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
  if (req->seamless && !req->has_initial_pose)
  {
    res->success = false;
    res->message = "seamless switching requires has_initial_pose with the pose in the new map";
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
  // Seamless: the caller vouches for the pose in the new map, so the filter
  // (and the sensor stream) must survive the swap — no buffer/filter reset.
  const bool seamless = active && req->seamless && req->has_initial_pose;
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

    if (!seamless)
    {
      // Clean slate on the new map: the old pose is meaningless here.
      clear_buffers();
      core_->reset_filter();
      initial_pose_set_.store(false);
      path_.poses.clear();
    }
    if (req->override_init_z)
    {
      init_z_ = req->init_z;
      RCLCPP_INFO(get_logger(), "init_z overridden to %.2f m for the new map.", init_z_);
    }

    // Queue the requested initial pose INSIDE the lock: the main loop holds
    // the same lock, so it can never process a scan against the new map with
    // the old pose still in the filter.
    if (active && req->has_initial_pose)
    {
      geometry_msgs::msg::PoseWithCovarianceStamped seed;
      seed.header.frame_id = params_.map_frame;
      seed.header.stamp = get_clock()->now();
      seed.pose.pose = req->initial_pose;
      {
        std::lock_guard<std::mutex> lock(init_pose_mtx_);
        init_pose_ = seed;
      }
      seamless_seed_.store(req->seamless);
      initial_pose_set_.store(true);
      if (req->seamless)
      {
        RCLCPP_INFO(get_logger(), "Seamless initial pose queued (direct seed, no scan matching).");
      }
      else
      {
        RCLCPP_INFO(get_logger(), "Initial pose from the switch request queued for relocalization.");
      }
    }

    if (active && !seamless)
    {
      set_state(LocalizationState::kUnlocalized);
    }
  }
  // The old map (and its kd-tree) is destroyed here, outside the lock.

  // ---- post-swap actions (ACTIVE only) ----------------------------------------
  if (active && params_.map_en)
  {
    publish_map();
  }

  res->success = true;
  if (!active)
  {
    res->message = req->has_initial_pose
                     ? "map switched (node inactive); initial pose ignored"
                     : "map switched (node inactive)";
  }
  else if (seamless)
  {
    res->message = "map switched; seamless pose seed queued";
  }
  else if (req->has_initial_pose)
  {
    res->message = "map switched; initial pose queued for relocalization";
  }
  else
  {
    res->message = "map switched; waiting for relocalization";
  }
  res->map_points = static_cast<uint32_t>(new_map->info().map_points);
  res->load_time_sec = load_time;
  (void)old_map;
}

}  // namespace fast_lio
