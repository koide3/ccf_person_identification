#ifndef LIGHT_VIEWER_HPP
#define LIGHT_VIEWER_HPP

#include <functional>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include <sensor_msgs/LaserScan.h>

class LightViewer {
public:
  LightViewer(const std::string& canvas_frame_id = "base_link", const cv::Size& size = cv::Size(512, 512), double m2pix = 512.0 / 10.0, const cv::Point& center = cv::Point(256, 256))
    : canvas_frame_id(canvas_frame_id),
      canvas(size, CV_8UC3, cv::Scalar::all(255))
  {
    canvas_frame2canvas.setZero();
    canvas_frame2canvas(0, 1) = -m2pix;
    canvas_frame2canvas(1, 0) = -m2pix;
    canvas_frame2canvas.block<3, 1>(0, 3) << center.x, center.y, 0.0;
    canvas_frame2canvas(2, 2) = canvas_frame2canvas(3, 3) = 1.0;

    std::cout << "--- canvas_frame2canvas ---" << std::endl;
    std::cout << canvas_frame2canvas << std::endl;

    prev_frame_id.clear();
    prev_frame2canvas = canvas_frame2canvas;

    robot_mark = Eigen::MatrixXd(4, 5);
    robot_mark << 0.1, 0.5,  0.1, -0.3, -0.3,
                  0.4, 0.0, -0.4, -0.4,  0.4,
                  0.0, 0.0,  0.0,  0.0,  0.0,
                  1.0, 1.0,  1.0,  1.0,  1.0;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:

  void clear() {
    canvas = cv::Scalar::all(255);
    prev_frame_id.clear();
    prev_frame2canvas = canvas_frame2canvas;
  }

  void drawLine(const Eigen::Vector2d& pt0, const Eigen::Vector2d& pt1, const cv::Scalar& color, int thickness) {
    cv::line(canvas, project(pt0), project(pt1), color, thickness);
  }

  void drawCircle(const Eigen::Vector2d& center, double radius, const cv::Scalar& color, int thickness) {
    int r = radius * prev_frame2canvas.block<1, 2>(0, 0).norm();
    cv::circle(canvas, project(center), r, color, thickness);
  }

  void drawEqualProbCircle(const Eigen::Vector2d& center, const Eigen::Matrix2d& cov, double c, const cv::Scalar& color, int thickness) {
    float r = std::sqrt( cov(0, 0) * c );
    drawCircle(center, r, color, thickness);
  }

  void drawLaserScan(const sensor_msgs::LaserScan& scan, const cv::Scalar& color) {
    setFrame(scan.header.frame_id);

    Eigen::MatrixXd cartesian(4, scan.ranges.size());
    cartesian.row(2).setZero();
    cartesian.row(3).setConstant(1.0);

    for(int i=0; i<scan.ranges.size(); i++) {
      double rad = scan.angle_min + scan.angle_increment * i;
      cartesian(0, i) = std::cos(rad) * scan.ranges[i];
      cartesian(1, i) = std::sin(rad) * scan.ranges[i];
    }

    Eigen::MatrixXd projected = prev_frame2canvas * cartesian;
    for(int i=0; i<scan.ranges.size(); i++) {
      cv::circle(canvas, cv::Point(projected(0, i), projected(1, i)), 2, color, -1);
    }
  }

  void drawRobot(const std::string& base_id, double size, const cv::Scalar& color) {
    setFrame(base_id);

    Eigen::MatrixXd scaled = robot_mark;
    scaled.block<2, 5>(0, 0) = scaled.block<2, 5>(0, 0) * size;

    Eigen::MatrixXd projected = prev_frame2canvas * scaled;
    std::vector<cv::Point> points(projected.cols());
    for(int i=0; i<projected.cols(); i++) {
      points[i].x = projected(0, i);
      points[i].y = projected(1, i);
    }

    cv::fillConvexPoly(canvas, points, color);
  }

  void drawCameraView(const std::string& base_id, double angle_of_view_deg, double length, const cv::Scalar& color, int thickness) {
    setFrame(base_id);

    double rad = angle_of_view_deg * M_PI / 180.0;

    double c = std::cos(rad);
    double s = std::sin(rad);

    Eigen::MatrixXd view = Eigen::MatrixXd::Zero(4, 3);
    view <<  s, -s, 0.0,
           0.0, 0.0, 0.0,
             c,  c, 0.0,
           1.0, 1.0, 1.0;

    view.block<3, 3>(0, 0) = view.block<3, 3>(0, 0) * length;

    Eigen::MatrixXd projected = prev_frame2canvas * view;
    cv::line(canvas, cv::Point(projected(0, 0), projected(1, 0)), cv::Point(projected(0, 2), projected(1, 2)), color, thickness);
    cv::line(canvas, cv::Point(projected(0, 1), projected(1, 1)), cv::Point(projected(0, 2), projected(1, 2)), color, thickness);
  }

  void drawCoordinate(const std::string& base_id, double length) {
    setFrame(base_id);

    Eigen::MatrixXd coordinate = Eigen::MatrixXd::Identity(4, 4) * length;
    coordinate.row(3).setOnes();

    Eigen::MatrixXd projected = prev_frame2canvas * coordinate;
    for(int i=0; i<3; i++) {
      cv::Scalar color = cv::Scalar::all(0);
      color[2 - i] = 255;
      cv::line(canvas, cv::Point(projected(0, i), projected(1, i)), cv::Point(projected(0, 3), projected(1, 3)), color, 1);
    }

    cv::putText(canvas, base_id, cv::Point(projected(0, 3), projected(1, 3)) + cv::Point(2, 6), CV_FONT_HERSHEY_PLAIN, 0.8, cv::Scalar::all(92));
  }

  void show() const {
    cv::imshow("light_viewer", canvas);
  }

public:
  cv::Point project(const Eigen::Vector2d& v) const {
    Eigen::Vector2d projected = (prev_frame2canvas * Eigen::Vector4d(v[0], v[1], 0.0, 1.0)).head<2>();
    return cv::Point(projected[0], projected[1]);
  }

  Eigen::Vector2d unproject(const cv::Point& pt) const {
    Eigen::Vector2d unprojected = (prev_frame2canvas.inverse() * Eigen::Vector4d(pt.x, pt.y, 0.0, 1.0)).head<2>();
    return unprojected;
  }

public:
  void setFrame(const std::string& frame_id) {
    if(frame_id == prev_frame_id) {
      return;
    }

    tf::StampedTransform trans;
    tf_listner.waitForTransform(canvas_frame_id, frame_id, ros::Time(0), ros::Duration(5.0));
    tf_listner.lookupTransform(canvas_frame_id, frame_id, ros::Time(0), trans);

    Eigen::Isometry3d mat;
    tf::transformTFToEigen(trans, mat);

    prev_frame_id = frame_id;
    prev_frame2canvas = canvas_frame2canvas * mat.matrix();
  }

private:
  tf::TransformListener tf_listner;

  cv::Mat canvas;

  std::string canvas_frame_id;
  Eigen::Matrix4d canvas_frame2canvas;

  std::string prev_frame_id;
  Eigen::Matrix4d prev_frame2canvas;

  Eigen::MatrixXd robot_mark;
};


#endif // LIGHT_VIEWER_HPP
