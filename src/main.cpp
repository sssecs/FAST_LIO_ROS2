// Entry point for the FAST-LIO localization node.

#include <csignal>
#include <atomic>
#include <iostream>

#include <rclcpp/rclcpp.hpp>

#include <fast_lio/node/localization_node.hpp>

namespace
{
std::atomic<bool> flg_exit{false};

void SigHandle(int sig)
{
  flg_exit = true;
  std::cout << "catch sig %d" << sig << std::endl;
  rclcpp::shutdown();
}
}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  signal(SIGINT, SigHandle);

  try
  {
    // The node throws std::runtime_error when required configuration (the
    // static lidar -> base TF or the prior map) is missing at startup, so exit
    // cleanly instead of terminating.
    rclcpp::spin(std::make_shared<fast_lio::LocalizationNode>());
  }
  catch (const std::runtime_error & ex)
  {
    RCLCPP_ERROR(rclcpp::get_logger("main"), "Startup failed: %s", ex.what());
    rclcpp::shutdown();
    return 1;
  }

  if (rclcpp::ok())
  {
    rclcpp::shutdown();
  }

  return 0;
}
