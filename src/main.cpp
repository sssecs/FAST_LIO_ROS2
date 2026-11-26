// Entry point for the FAST-LIO localization node (ROS 2 Lifecycle).
//
// The node starts UNCONFIGURED; bring it up with:
//   ros2 lifecycle set /fastlio_localization configure
//   ros2 lifecycle set /fastlio_localization activate
// (launch/localization.launch.py does this automatically).

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include <fast_lio/node/localization_node.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<fast_lio::LocalizationNode>();
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
