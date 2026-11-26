// Entry point for the FAST-LIO localization node (ROS 2 Lifecycle).
//
// The node lives in the /TCT_Docent/locomotion/fastlio namespace and starts
// UNCONFIGURED; bring it up with:
//   ros2 lifecycle set /TCT_Docent/locomotion/fastlio/fastlio_localization configure
//   ros2 lifecycle set /TCT_Docent/locomotion/fastlio/fastlio_localization activate
// (launch/localization.launch.py does this automatically).

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include <fast_lio/node/localization_node.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<fast_lio::LocalizationNode>();
  // Three callback groups (sensors / main loop / map switch) plus the default
  // group (lifecycle, diagnostics, /initialpose) need real parallelism so a
  // long relocalization or map load never starves the monitors.
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
  executor.add_node(node->get_node_base_interface());
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
