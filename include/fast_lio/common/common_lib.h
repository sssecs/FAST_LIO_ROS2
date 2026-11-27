// Shared types, macros and small math helpers for the FAST-LIO localization
// pipeline (point types, the lidar+IMU measurement group, plane fitting).

#ifndef FAST_LIO_COMMON_COMMON_LIB_H
#define FAST_LIO_COMMON_COMMON_LIB_H

#include <deque>
#include <limits>
#include <rclcpp/rclcpp.hpp>
#include <fast_lio/common/so3_math.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <fast_lio/msg/pose6_d.hpp>
#include <sensor_msgs/msg/imu.hpp>

using namespace std;
using namespace Eigen;

#define PI_M (3.14159265358)
#define G_m_s2 (9.81)  // gravity constant [m/s^2]
#define NUM_MATCH_POINTS (5)

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]

typedef fast_lio::msg::Pose6D Pose6D;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;

#define MD(a, b) Matrix<double, (a), (b)>
#define VF(a) Matrix<float, (a), 1>

inline M3D Eye3d(M3D::Identity());
inline V3D Zero3d(0, 0, 0);

// Lidar scan + the IMU samples covering it, synchronized by the node.
struct MeasureGroup
{
  MeasureGroup() { this->lidar.reset(new PointCloudXYZI()); }
  double lidar_beg_time = 0.0;
  double lidar_end_time = 0.0;
  PointCloudXYZI::Ptr lidar;
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu;
};

template <typename T>
auto set_pose6d(
  const double t, const Matrix<T, 3, 1> & a, const Matrix<T, 3, 1> & g, const Matrix<T, 3, 1> & v,
  const Matrix<T, 3, 1> & p, const Matrix<T, 3, 3> & R)
{
  Pose6D rot_kp;
  rot_kp.offset_time = t;
  for (int i = 0; i < 3; i++)
  {
    rot_kp.acc[i] = a(i);
    rot_kp.gyr[i] = g(i);
    rot_kp.vel[i] = v(i);
    rot_kp.pos[i] = p(i);
    for (int j = 0; j < 3; j++) rot_kp.rot[i * 3 + j] = R(i, j);
  }
  return move(rot_kp);
}

/* Fit a plane Ax + By + Cz + D = 0 through NUM_MATCH_POINTS points.
   Returns false when any point lies farther than `threshold` from the plane.
   The normal is the smallest-eigenvalue eigenvector of the neighbors'
   covariance (fit in centroid-centered coordinates): an explicit origin
   offset — as in the old A n = -1 solve — degenerates whenever the plane
   passes through (or extremely close to) the map frame origin, e.g. a floor
   saved exactly at z = 0, and produced garbage fits or NaN normals. */
template <typename T>
bool esti_plane(Matrix<T, 4, 1> & pca_result, const PointVector & point, const T & threshold)
{
  Matrix<T, 3, 1> centroid(0, 0, 0);
  for (const auto & p : point)
  {
    centroid += Matrix<T, 3, 1>(p.x, p.y, p.z);
  }
  centroid /= static_cast<T>(point.size());

  Matrix<T, 3, 3> cov;
  cov.setZero();
  for (const auto & p : point)
  {
    const Matrix<T, 3, 1> q(p.x - centroid(0), p.y - centroid(1), p.z - centroid(2));
    cov += q * q.transpose();
  }

  SelfAdjointEigenSolver<Matrix<T, 3, 3>> es(cov);
  if (es.info() != Success)
  {
    return false;
  }
  // Reject coincident/collinear neighborhoods: the middle eigenvalue must
  // carry real in-plane spread, otherwise the normal direction is arbitrary.
  const T eval_middle = es.eigenvalues()(1);
  if (eval_middle < std::numeric_limits<T>::epsilon())
  {
    return false;
  }

  const Matrix<T, 3, 1> n = es.eigenvectors().col(0);
  pca_result(0) = n(0);
  pca_result(1) = n(1);
  pca_result(2) = n(2);
  pca_result(3) = -n.dot(centroid);

  for (int j = 0; j < NUM_MATCH_POINTS; j++)
  {
    if (
      fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z +
           pca_result(3)) > threshold)
    {
      return false;
    }
  }
  return true;
}

inline double get_time_sec(const builtin_interfaces::msg::Time & time)
{
  return rclcpp::Time(time).seconds();
}

inline rclcpp::Time get_ros_time(double timestamp)
{
  int32_t sec = std::floor(timestamp);
  auto nanosec_d = (timestamp - std::floor(timestamp)) * 1e9;
  uint32_t nanosec = nanosec_d;
  return rclcpp::Time(sec, nanosec);
}

#endif  // FAST_LIO_COMMON_COMMON_LIB_H
