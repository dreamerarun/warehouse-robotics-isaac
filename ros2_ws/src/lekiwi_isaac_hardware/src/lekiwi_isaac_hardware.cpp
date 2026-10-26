// ============================================================================
// lekiwi_isaac_hardware.cpp
// ----------------------------------------------------------------------------
// ros2_control SystemInterface that bridges controller_manager to Isaac Sim
// 5.1 using ONLY the officially shipped Isaac Sim ROS 2 bridge nodes.
//
// Isaac Sim 5.1 does NOT ship a ros2_control hardware plugin (verified: no
// `ros2_control` content anywhere in the installation).  The supported control
// path in Isaac Sim 5.1 is:
//
//     isaacsim.ros2.bridge.ROS2PublishJointState   (state  -> ROS 2)
//     isaacsim.ros2.bridge.ROS2SubscribeJointState  (ROS 2  -> commands)
//              -> isaacsim.core.nodes.IsaacArticulationController
//              -> /World/LeKiwi
//
// so this plugin provides the ros2_control side of that bridge:
//
//     read()  : takes the newest sensor_msgs/JointState received from Isaac
//               Sim on <state_topic> and exposes it as state interfaces
//     write() : converts the claimed command interfaces into two
//               sensor_msgs/JointState messages
//                 <velocity_command_topic> : wheel velocities
//                 <position_command_topic> : arm positions
//               and publishes them for Isaac Sim's ROS2SubscribeJointState
//
// The plugin owns its own rclcpp node + executor thread so that it never
// depends on controller_manager's executor behaviour.
// ============================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace lekiwi_isaac_hardware
{

using hardware_interface::CallbackReturn;
using hardware_interface::return_type;

namespace
{
double to_double(const std::string & s, double fallback)
{
  if (s.empty()) {
    return fallback;
  }
  try {
    return std::stod(s);
  } catch (const std::exception &) {
    return fallback;
  }
}
}  // namespace

class LekIwiIsaacSystem : public hardware_interface::SystemInterface
{
public:
  LekIwiIsaacSystem() = default;
  ~LekIwiIsaacSystem() override;

  CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & previous) override;

  return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  void on_state_message(const sensor_msgs::msg::JointState::SharedPtr msg);
  void start_spin();
  void stop_spin();
  void publish_commands(bool force);

  // ---- configuration ------------------------------------------------------
  size_t n_{0};
  std::vector<std::string> joint_names_;
  std::vector<bool> has_pos_cmd_;
  std::vector<bool> has_vel_cmd_;
  std::vector<double> pos_min_;
  std::vector<double> pos_max_;
  std::vector<double> vel_max_;
  std::vector<size_t> velocity_group_;   // joints with a velocity command interface
  std::vector<size_t> position_group_;   // joints with a position  command interface

  std::string state_topic_{"/lekIwi/joint_states_isaac"};
  std::string vel_topic_{"/lekIwi/wheel_command"};
  std::string pos_topic_{"/lekIwi/arm_command"};
  double state_timeout_{5.0};
  int publish_decimation_{1};

  // ---- data ---------------------------------------------------------------
  std::vector<double> pos_state_;
  std::vector<double> vel_state_;
  std::vector<double> pos_cmd_;
  std::vector<double> vel_cmd_;

  std::vector<double> rx_pos_;          // newest message, guarded by mutex_
  std::vector<double> rx_vel_;
  bool has_rx_{false};
  bool warned_no_state_{false};
  std::mutex mutex_;

  // ---- ros ---------------------------------------------------------------
  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_thread_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr vel_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pos_pub_;

  std::atomic<bool> active_{false};
  std::atomic<bool> spin_running_{false};
  int cycle_{0};
  std::chrono::steady_clock::time_point last_state_wall_time_;
};

CallbackReturn LekIwiIsaacSystem::on_init(const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  n_ = info_.joints.size();
  if (n_ == 0) {
    RCLCPP_FATAL(rclcpp::get_logger("LekIwiIsaacSystem"), "no joints declared in <ros2_control>");
    return CallbackReturn::ERROR;
  }

  joint_names_.reserve(n_);
  has_pos_cmd_.assign(n_, false);
  has_vel_cmd_.assign(n_, false);
  pos_min_.assign(n_, -1e30);
  pos_max_.assign(n_, 1e30);
  vel_max_.assign(n_, 1e30);

  for (size_t i = 0; i < n_; ++i) {
    const auto & ji = info_.joints[i];
    joint_names_.push_back(ji.name);

    for (const auto & ci : ji.command_interfaces) {
      if (ci.name == "position") {
        has_pos_cmd_[i] = true;
        pos_min_[i] = to_double(ci.min, -1e30);
        pos_max_[i] = to_double(ci.max, 1e30);
        position_group_.push_back(i);
      } else if (ci.name == "velocity") {
        has_vel_cmd_[i] = true;
        vel_max_[i] = to_double(ci.max, 1e30);
        velocity_group_.push_back(i);
      } else {
        RCLCPP_ERROR(
          rclcpp::get_logger("LekIwiIsaacSystem"),
          "unsupported command interface '%s' on joint '%s'",
          ci.name.c_str(), ji.name.c_str());
        return CallbackReturn::ERROR;
      }
    }

    for (const auto & si : ji.state_interfaces) {
      if (si.name != "position" && si.name != "velocity") {
        RCLCPP_ERROR(
          rclcpp::get_logger("LekIwiIsaacSystem"),
          "unsupported state interface '%s' on joint '%s'",
          si.name.c_str(), ji.name.c_str());
        return CallbackReturn::ERROR;
      }
    }
  }

  pos_state_.assign(n_, 0.0);
  vel_state_.assign(n_, 0.0);
  pos_cmd_.assign(n_, 0.0);
  vel_cmd_.assign(n_, 0.0);
  rx_pos_.assign(n_, 0.0);
  rx_vel_.assign(n_, 0.0);

  // ---- hardware parameters (from the <hardware><param> block) -------------
  const auto & p = info_.hardware_parameters;
  auto get = [&p](const std::string & k, const std::string & dflt) -> std::string {
      auto it = p.find(k);
      return it == p.end() ? dflt : it->second;
    };
  state_topic_ = get("state_topic", state_topic_);
  vel_topic_ = get("velocity_command_topic", vel_topic_);
  pos_topic_ = get("position_command_topic", pos_topic_);
  state_timeout_ = to_double(get("state_timeout", "5.0"), 5.0);
  publish_decimation_ = static_cast<int>(to_double(get("publish_decimation", "1"), 1.0));
  if (publish_decimation_ < 1) {
    publish_decimation_ = 1;
  }

  RCLCPP_INFO(
    rclcpp::get_logger("LekIwiIsaacSystem"),
    "LeKiwi Isaac Sim bridge initialised: %zu joints "
    "(%zu velocity-commanded, %zu position-commanded)",
    n_, velocity_group_.size(), position_group_.size());
  RCLCPP_INFO(
    rclcpp::get_logger("LekIwiIsaacSystem"),
    "  state   <- %s", state_topic_.c_str());
  RCLCPP_INFO(
    rclcpp::get_logger("LekIwiIsaacSystem"),
    "  wheels  -> %s", vel_topic_.c_str());
  RCLCPP_INFO(
    rclcpp::get_logger("LekIwiIsaacSystem"),
    "  arm     -> %s", pos_topic_.c_str());

  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
LekIwiIsaacSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> out;
  out.reserve(2 * n_);
  for (size_t i = 0; i < n_; ++i) {
    for (const auto & si : info_.joints[i].state_interfaces) {
      double * ptr = nullptr;
      if (si.name == "position") {
        ptr = &pos_state_[i];
      } else if (si.name == "velocity") {
        ptr = &vel_state_[i];
      }
      if (ptr == nullptr) {
        return {};
      }
      out.emplace_back(joint_names_[i], si.name, ptr);
    }
  }
  return out;
}

std::vector<hardware_interface::CommandInterface>
LekIwiIsaacSystem::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> out;
  out.reserve(n_);
  for (size_t i = 0; i < n_; ++i) {
    for (const auto & ci : info_.joints[i].command_interfaces) {
      double * ptr = nullptr;
      if (ci.name == "position") {
        ptr = &pos_cmd_[i];
      } else if (ci.name == "velocity") {
        ptr = &vel_cmd_[i];
      }
      if (ptr == nullptr) {
        return {};
      }
      out.emplace_back(joint_names_[i], ci.name, ptr);
    }
  }
  return out;
}

CallbackReturn LekIwiIsaacSystem::on_configure(const rclcpp_lifecycle::State &)
{
  try {
    node_ = std::make_shared<rclcpp::Node>("lekIwi_isaac_hw_bridge");
  } catch (const std::exception & ex) {
    RCLCPP_FATAL(
      rclcpp::get_logger("LekIwiIsaacSystem"), "failed to create node: %s", ex.what());
    return CallbackReturn::ERROR;
  }

  // Best-effort subscription: compatible with a reliable OR best-effort
  // publisher, so it matches whatever QoS Isaac Sim uses.
  auto state_qos = rclcpp::QoS(rclcpp::KeepLast(20)).best_effort();
  // Reliable publications: compatible with any subscriber QoS Isaac Sim uses.
  auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

  state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    state_topic_, state_qos,
    std::bind(&LekIwiIsaacSystem::on_state_message, this, std::placeholders::_1));
  vel_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(vel_topic_, cmd_qos);
  pos_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(pos_topic_, cmd_qos);

  start_spin();
  RCLCPP_INFO(rclcpp::get_logger("LekIwiIsaacSystem"), "configured (bridge node spinning)");
  return CallbackReturn::SUCCESS;
}

CallbackReturn LekIwiIsaacSystem::on_activate(const rclcpp_lifecycle::State &)
{
  // command buffers are zeroed: stationary wheels, arm holds its current pose
  for (size_t i = 0; i < n_; ++i) {
    vel_cmd_[i] = 0.0;
    pos_cmd_[i] = pos_state_[i];
  }
  cycle_ = 0;
  active_.store(true);
  RCLCPP_INFO(rclcpp::get_logger("LekIwiIsaacSystem"), "activated");
  return CallbackReturn::SUCCESS;
}

CallbackReturn LekIwiIsaacSystem::on_deactivate(const rclcpp_lifecycle::State &)
{
  active_.store(false);
  // safety: tell Isaac Sim to stop the wheels immediately
  for (size_t i = 0; i < n_; ++i) {
    vel_cmd_[i] = 0.0;
  }
  publish_commands(true);
  RCLCPP_INFO(rclcpp::get_logger("LekIwiIsaacSystem"), "deactivated (zero velocity sent)");
  return CallbackReturn::SUCCESS;
}

CallbackReturn LekIwiIsaacSystem::on_cleanup(const rclcpp_lifecycle::State &)
{
  active_.store(false);
  stop_spin();
  state_sub_.reset();
  vel_pub_.reset();
  pos_pub_.reset();
  node_.reset();
  RCLCPP_INFO(rclcpp::get_logger("LekIwiIsaacSystem"), "cleaned up");
  return CallbackReturn::SUCCESS;
}

void LekIwiIsaacSystem::on_state_message(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  const size_t count = std::min(msg->name.size(), n_);
  for (size_t k = 0; k < count; ++k) {
    // messages may arrive in any joint order -> map by name
    auto it = std::find(joint_names_.begin(), joint_names_.end(), msg->name[k]);
    if (it == joint_names_.end()) {
      continue;
    }
    const size_t i = static_cast<size_t>(it - joint_names_.begin());
    if (k < msg->position.size()) {
      rx_pos_[i] = msg->position[k];
    }
    if (k < msg->velocity.size()) {
      rx_vel_[i] = msg->velocity[k];
    }
  }
  has_rx_ = true;
  last_state_wall_time_ = std::chrono::steady_clock::now();
}

return_type LekIwiIsaacSystem::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  bool fresh = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (has_rx_) {
      pos_state_ = rx_pos_;
      vel_state_ = rx_vel_;
      const auto age = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - last_state_wall_time_).count();
      fresh = age <= state_timeout_;
    }
  }

  if (fresh) {
    if (warned_no_state_) {
      RCLCPP_INFO(
        rclcpp::get_logger("LekIwiIsaacSystem"),
        "joint state stream from Isaac Sim is alive");
      warned_no_state_ = false;
    }
  } else if (!warned_no_state_) {
    RCLCPP_WARN(
      rclcpp::get_logger("LekIwiIsaacSystem"),
      "no joint state received from Isaac Sim on '%s' yet "
      "(is the simulation playing and is the action graph present?)",
      state_topic_.c_str());
    warned_no_state_ = true;
  }
  return return_type::OK;
}

return_type LekIwiIsaacSystem::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (!active_.load()) {
    return return_type::OK;
  }

  // ---- enforce joint limits ---------------------------------------------
  for (size_t i = 0; i < n_; ++i) {
    if (has_vel_cmd_[i]) {
      double v = std::isfinite(vel_cmd_[i]) ? vel_cmd_[i] : 0.0;
      const double lim = vel_max_[i];
      if (std::isfinite(lim) && lim > 0.0) {
        v = std::max(-lim, std::min(lim, v));
      }
      vel_cmd_[i] = v;
    }
    if (has_pos_cmd_[i]) {
      double p = std::isfinite(pos_cmd_[i]) ? pos_cmd_[i] : 0.0;
      p = std::max(pos_min_[i], std::min(pos_max_[i], p));
      pos_cmd_[i] = p;
    }
  }

  ++cycle_;
  if (publish_decimation_ > 1 && (cycle_ % publish_decimation_) != 0) {
    return return_type::OK;
  }

  publish_commands(false);
  return return_type::OK;
}

void LekIwiIsaacSystem::publish_commands(bool /*force*/)
{
  if (!vel_pub_ || !pos_pub_) {
    return;
  }

  if (!velocity_group_.empty()) {
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = node_->now();
    msg.name.reserve(velocity_group_.size());
    msg.velocity.reserve(velocity_group_.size());
    for (size_t i : velocity_group_) {
      msg.name.push_back(joint_names_[i]);
      msg.velocity.push_back(vel_cmd_[i]);
    }
    // position[] intentionally left EMPTY: Isaac Sim's
    // isaacsim.core.nodes.IsaacArticulationController skips arrays with size 0,
    // so the wheels are driven in velocity-only mode.
    vel_pub_->publish(msg);
  }

  if (!position_group_.empty()) {
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = node_->now();
    msg.name.reserve(position_group_.size());
    msg.position.reserve(position_group_.size());
    for (size_t i : position_group_) {
      msg.name.push_back(joint_names_[i]);
      msg.position.push_back(pos_cmd_[i]);
    }
    // velocity[] intentionally left EMPTY -> position control only for the arm
    pos_pub_->publish(msg);
  }
}

void LekIwiIsaacSystem::start_spin()
{
  if (spin_running_.load()) {
    return;
  }
  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_->get_node_base_interface());
  spin_running_.store(true);
  spin_thread_ = std::thread([this]() {
    try {
      executor_->spin();
    } catch (const std::exception & ex) {
      RCLCPP_ERROR(
        rclcpp::get_logger("LekIwiIsaacSystem"), "executor stopped: %s", ex.what());
    }
  });
}

void LekIwiIsaacSystem::stop_spin()
{
  if (!spin_running_.load()) {
    return;
  }
  spin_running_.store(false);
  if (executor_) {
    executor_->cancel();
  }
  if (spin_thread_.joinable()) {
    spin_thread_.join();
  }
  if (executor_) {
    executor_->remove_node(node_->get_node_base_interface());
    executor_.reset();
  }
}

LekIwiIsaacSystem::~LekIwiIsaacSystem()
{
  active_.store(false);
  stop_spin();
}

}  // namespace lekiwi_isaac_hardware

PLUGINLIB_EXPORT_CLASS(
  lekiwi_isaac_hardware::LekIwiIsaacSystem, hardware_interface::SystemInterface)
