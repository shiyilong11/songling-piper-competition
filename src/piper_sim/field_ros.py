"""ROS 2 Humble transport for the official PiPER single-arm driver."""

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from piper_msgs.msg import PiperStatusMsg
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState

from piper_sim.control import canonical_state, driver_command

RGB_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
STATUS_TOPIC = "/arm_status"


def stamp_seconds(msg):
    stamp = msg.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


@dataclass
class JointSample:
    stamp: float
    received: float
    state: np.ndarray
    velocity: np.ndarray


class FieldROS(Node):
    def __init__(self, interface, calibration):
        super().__init__("songling_l1_l2")
        self.interface, self.calibration = interface, calibration
        topics = interface["ros2"]
        self.command_topic = self.resolve_topic_name(topics["driver_command_topic"])
        self.state_topic = self.resolve_topic_name(topics["driver_joint_state_topic"])
        self.status_topic = self.resolve_topic_name(STATUS_TOPIC)
        self.condition = threading.Condition()
        self.joints = deque(maxlen=128)
        self.images = {"rgb": deque(maxlen=6), "depth": deque(maxlen=6)}
        self.status = None
        self.feedback_error = None
        self.publisher = None
        self.multiplier = None
        self.guard = None
        self.last_target = None
        self.last_publish = 0.0
        self.stopped = None
        self.stop_report = None
        self.zero = time.monotonic()
        self.deadline = None
        self.create_subscription(
            JointState, topics["driver_joint_state_topic"], self._joint, qos_profile_sensor_data
        )
        self.create_subscription(
            PiperStatusMsg, STATUS_TOPIC, self._status, qos_profile_sensor_data
        )
        for kind, topic in (("rgb", RGB_TOPIC), ("depth", DEPTH_TOPIC)):
            self.create_subscription(
                Image, topic, lambda msg, kind=kind: self._image(kind, msg), qos_profile_sensor_data
            )
        self.create_timer(0.05, self._watchdog)
        self._field_executor = SingleThreadedExecutor()
        self._field_executor.add_node(self)
        self.worker = threading.Thread(target=self._field_executor.spin, daemon=True)
        self.worker.start()

    def clock(self):
        return time.monotonic() - self.zero

    def ros_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _joint(self, msg):
        try:
            state, velocity = canonical_state(
                list(msg.name), list(msg.position), list(msg.velocity), self.interface
            )
            low, high = self.interface["joints"]["lower_rad"], self.interface["joints"]["upper_rad"]
            if np.any(state[:6] < np.asarray(low) - 0.02) or np.any(
                state[:6] > np.asarray(high) + 0.02
            ):
                raise ValueError("Joint feedback outside the packaged PiPER limits")
            if not -0.002 <= state[6] <= 0.074:
                raise ValueError("Gripper feedback outside the packaged 0-70 mm range")
            sample = JointSample(stamp_seconds(msg), time.monotonic(), state, velocity)
            with self.condition:
                if not self.joints or sample.stamp > self.joints[-1].stamp:
                    self.joints.append(sample)
                    self.feedback_error = None
                    self.condition.notify_all()
        except (TypeError, ValueError) as exc:
            with self.condition:
                self.feedback_error = str(exc)
                self.condition.notify_all()

    def _status(self, msg):
        with self.condition:
            self.status = (time.monotonic(), int(msg.arm_status), int(msg.err_code))
            self.condition.notify_all()

    def _image(self, kind, msg):
        with self.condition:
            self.images[kind].append((stamp_seconds(msg), msg))
            self.condition.notify_all()

    def _latest(self):
        timeout = self.interface["ros2"]["state_timeout_s"]
        if self.feedback_error:
            raise RuntimeError(self.feedback_error)
        if not self.joints:
            raise RuntimeError("No PiPER joint feedback")
        sample = self.joints[-1]
        age = self.ros_now() - sample.stamp
        if not -0.05 <= age <= timeout or time.monotonic() - sample.received > timeout:
            raise RuntimeError("PiPER joint feedback is stale or ROS clocks are unsynchronised")
        return sample

    def _healthy(self):
        if self.stopped:
            raise RuntimeError(self.stopped)
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Round wall-clock deadline reached")
        sample = self._latest()
        if self.status is None or time.monotonic() - self.status[0] > 0.30:
            raise RuntimeError("PiPER arm_status is missing or stale")
        if self.status[1:] != (0, 0):
            raise RuntimeError(
                f"PiPER reports arm_status={self.status[1]}, err_code={self.status[2]}"
            )
        if self.deadline is not None:
            now = self.ros_now()
            for kind, images in self.images.items():
                if not images or not -0.05 <= now - images[-1][0] <= 0.50:
                    raise RuntimeError(f"{kind} stream stopped or became stale")
        return sample

    def wait_ready(self, timeout=8.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                try:
                    return self._healthy()
                except RuntimeError as exc:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"ROS preflight failed: {exc}") from exc
                    self.condition.wait(0.05)

    def read_state(self):
        with self.condition:
            return self.clock(), self._healthy().state.copy()

    def capture(self, robot, timeout=2.0):
        start_stamp = self.ros_now()
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                self._healthy()
                match = self._match_images(start_stamp)
                if match is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "No fresh RGB/registered depth/joint sample within sync tolerance"
                    )
                self.condition.wait(0.03)
        rgb_stamp, rgb, depth, sample = match
        timestamp = self.clock() - (self.ros_now() - rgb_stamp)
        obs = self.calibration.observation(
            rgb, depth, sample.state, sample.velocity, timestamp, robot.end_pose(sample.state[:6])
        )
        robot.sync()
        return obs

    def preview_pair(self):
        """Read a fresh paired camera frame without sending commands or waiting for joints."""
        with self.condition:
            now = self.ros_now()
            skew = min(self.interface["observation"]["max_sensor_skew_s"], 0.035)
            pair = None
            for stamp, rgb in reversed(self.images["rgb"]):
                if not -0.05 <= now - stamp <= 0.20 or not self.images["depth"]:
                    continue
                depth_stamp, depth = min(
                    self.images["depth"], key=lambda entry: abs(entry[0] - stamp)
                )
                if abs(depth_stamp - stamp) <= skew:
                    pair = stamp, rgb, depth
                    break
        if pair is None:
            return None
        stamp, rgb_msg, depth_msg = pair
        rgb, depth = self.calibration.aligned_images(rgb_msg, depth_msg)
        timestamp = self.clock() - (self.ros_now() - stamp)
        return stamp, timestamp, rgb, depth

    def _match_images(self, after):
        max_skew = self.interface["observation"]["max_sensor_skew_s"]
        now = self.ros_now()
        for rgb_stamp, rgb in reversed(self.images["rgb"]):
            if rgb_stamp <= after or not -0.05 <= now - rgb_stamp <= 0.20:
                continue
            for depth_stamp, depth in reversed(self.images["depth"]):
                if abs(depth_stamp - rgb_stamp) > min(max_skew, 0.035):
                    continue
                sample = min(self.joints, key=lambda item: abs(item.stamp - rgb_stamp))
                if (
                    max(rgb_stamp, depth_stamp, sample.stamp)
                    - min(rgb_stamp, depth_stamp, sample.stamp)
                    <= max_skew
                ):
                    return rgb_stamp, rgb, depth, sample
        return None

    def driver_parameters(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            endpoints = self.get_subscriptions_info_by_topic(self.command_topic)
            if len(endpoints) == 1:
                endpoint = endpoints[0]
                if endpoint.topic_type != "sensor_msgs/msg/JointState":
                    raise RuntimeError("PiPER command subscriber is not sensor_msgs/JointState")
                break
            if len(endpoints) > 1:
                raise RuntimeError(
                    "Multiple subscribers on the arm command topic; use the bench namespace"
                )
            time.sleep(0.05)
        else:
            raise RuntimeError(f"No PiPER driver subscribes to {self.command_topic}")
        driver = f"{endpoint.node_namespace.rstrip('/')}/{endpoint.node_name}"
        for topic in (self.state_topic, self.status_topic):
            publishers = self.get_publishers_info_by_topic(topic)
            if len(publishers) != 1:
                raise RuntimeError(f"Expected one PiPER feedback publisher on {topic}")
            source = publishers[0]
            if (source.node_namespace, source.node_name) != (
                endpoint.node_namespace,
                endpoint.node_name,
            ):
                raise RuntimeError(
                    f"Feedback on {topic} does not come from command driver {driver}"
                )
        client = self.create_client(GetParameters, f"{driver}/get_parameters")
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"Cannot read {driver} gripper_val_mutiple")
            request = GetParameters.Request()
            request.names = ["gripper_val_mutiple"]
            future = client.call_async(request)
            deadline = time.monotonic() + timeout
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not future.done() or future.result() is None:
                raise RuntimeError("PiPER gripper parameter request timed out")
            values = future.result().values
            if len(values) != 1 or values[0].type != ParameterType.PARAMETER_INTEGER:
                raise RuntimeError(
                    "Driver did not expose its gripper_val_mutiple integer parameter"
                )
            multiplier = values[0].integer_value
            if not 1 <= multiplier <= 10:
                raise RuntimeError(
                    "PiPER gripper multiplier disables or invalidates gripper control"
                )
            return driver, multiplier
        finally:
            self.destroy_client(client)

    def begin(self, limit_s, guard):
        with self.condition:
            self._healthy()
        driver, self.multiplier = self.driver_parameters()
        if self.count_publishers(self.command_topic):
            raise RuntimeError("Another node is already publishing PiPER commands")
        self.guard = guard
        self.publisher = self.create_publisher(
            JointState, self.interface["ros2"]["driver_command_topic"], QoSProfile(depth=1)
        )
        end = time.monotonic() + 3.0
        while self.publisher.get_subscription_count() != 1:
            if time.monotonic() >= end:
                raise RuntimeError("PiPER command publisher failed to match exactly one driver")
            time.sleep(0.05)
        self.zero = time.monotonic()
        self.deadline = self.zero + limit_s
        return {
            "driver": driver,
            "gripper_val_mutiple": self.multiplier,
            "command_topic": self.command_topic,
        }

    def _publish(self, target):
        data = driver_command(target, self.interface)
        data["position"][6] /= self.multiplier
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.interface["frames"]["base"]
        msg.name, msg.position, msg.velocity, msg.effort = (
            data["name"],
            data["position"],
            data["velocity"],
            data["effort"],
        )
        self.publisher.publish(msg)
        self.last_publish = time.monotonic()
        self.last_target = np.asarray(target).copy()

    def send(self, target):
        if self.publisher is None:
            raise RuntimeError("Read-only ROS session")
        period = 1.0 / self.interface["control"]["policy_hz"]
        wait = self.last_publish + period - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        with self.condition:
            state = self._healthy().state
            if (
                self.count_publishers(self.command_topic) != 1
                or self.publisher.get_subscription_count() != 1
                or self.count_publishers(self.state_topic) != 1
                or self.count_publishers(self.status_topic) != 1
            ):
                raise RuntimeError("PiPER command graph changed during execution")
            self.guard(state, target)
            self._publish(target)

    def _hold_target(self):
        state = self._latest().state.copy()
        state[:6] = np.clip(
            state[:6], self.interface["joints"]["lower_rad"], self.interface["joints"]["upper_rad"]
        )
        width = state[6] if self.last_target is None else self.last_target[6]
        state[6] = np.clip(width, *self.interface["control"]["gripper_width_range_m"])
        return state

    def pause(self, _target):
        # Motion skills may pause and then recheck a grasp. Keep the previous
        # gripper target and hold the latest measured arm pose.
        with self.condition:
            if self.stopped or self.publisher is None:
                return
            try:
                self._healthy()
                self._publish(self._hold_target())
            except (RuntimeError, TimeoutError, ValueError) as exc:
                self._stop_locked(str(exc))

    def _stop_locked(self, reason):
        if self.stopped is not None:
            return
        self.stopped = reason
        self.stop_report = {
            "reason": reason,
            "hold_published": False,
            "hardware_stop_confirmed": False,
        }
        if self.publisher is None or self.last_target is None:
            return
        try:
            if (
                self.count_publishers(self.command_topic) == 1
                and self.publisher.get_subscription_count() == 1
            ):
                self._publish(self._hold_target())
                self.stop_report["hold_published"] = True
        except (RuntimeError, ValueError):
            pass

    def stop(self, reason):
        with self.condition:
            self._stop_locked(reason)
        return self.stop_report

    def _watchdog(self):
        with self.condition:
            if self.publisher is None or self.stopped or self.last_target is None:
                return
            try:
                sample = self._healthy()
                if time.monotonic() - self.last_publish > 0.75:
                    error = np.max(np.abs(sample.state[:6] - self.last_target[:6]))
                    if error > 0.03 or np.max(np.abs(sample.velocity)) > 0.08:
                        raise RuntimeError("Arm still moving during a command pause")
            except (RuntimeError, TimeoutError) as exc:
                self._stop_locked(str(exc))

    def close(self):
        self._field_executor.shutdown(timeout_sec=2.0)
        self.worker.join(timeout=2.0)
        self.destroy_node()
