import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, LogInfo, RegisterEventHandler
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState, matches_node_name

from lifecycle_msgs.msg import Transition

NODE_NAME = 'fastlio_localization'
NODE_NS = '/TCT_Docent/locomotion/fastlio'
NODE_FQN = NODE_NS + '/' + NODE_NAME


def generate_launch_description():
    package_path = get_package_share_directory('fast_lio')
    default_config_path = os.path.join(package_path, 'config')

    use_sim_time = LaunchConfiguration('use_sim_time')
    config_path = LaunchConfiguration('config_path')
    config_file = LaunchConfiguration('config_file')
    map_path = LaunchConfiguration('map_path')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    declare_config_path_cmd = DeclareLaunchArgument(
        'config_path', default_value=default_config_path,
        description='Yaml config file path'
    )
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='mid360.yaml',
        description='Config file'
    )
    declare_map_path_cmd = DeclareLaunchArgument(
        'map_path', default_value='',
        description='Prior map PCD to load at configure time (overrides the yaml value). '
                    'Empty: start without a map (NO_MAP) and select one later via the '
                    'switch_map service'
    )

    fast_lio_node = LifecycleNode(
        package='fast_lio',
        executable='fastlio_localization',
        name=NODE_NAME,
        namespace=NODE_NS,
        parameters=[PathJoinSubstitution([config_path, config_file]),
                    {'use_sim_time': use_sim_time,
                     'localization.map_path': map_path}],
        output='screen'
    )

    lidar_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_to_base_tf',
        arguments=[
            '0', '0', '0',
            '0', '0', '3.14159265359',
            'base_link',
            'livox_frame'
        ]
    )

    # Bring the lifecycle node up automatically: configure as soon as the node
    # is up, activate as soon as it reports inactive (configured).
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_node_name(node_name=NODE_FQN),
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )
    activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=fast_lio_node,
            goal_state='inactive',
            entities=[
                LogInfo(msg='[lifecycle] {} configured, activating...'.format(NODE_FQN)),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_node_name(node_name=NODE_FQN),
                    transition_id=Transition.TRANSITION_ACTIVATE,
                )),
            ],
        )
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_map_path_cmd)
    ld.add_action(lidar_to_base)
    ld.add_action(fast_lio_node)
    ld.add_action(configure_event)
    ld.add_action(activate_event)

    return ld
