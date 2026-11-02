
import os
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped
from ament_index_python.packages import get_package_share_directory

JOINTS = ["joint1", "joint2", "joint3"]
WAYPOINTS = [np.array([-1.0, 0.45, -0.95]), np.array([1.0, 1.15, -0.45]), np.array([0.2, 0.75, -1.35])]
SEG_T, HOLD = 1.6, 0.4


def quintic(q0, q1, T, t):
    s = np.clip(t / T, 0, 1)
    h = 10 * s**3 - 15 * s**4 + 6 * s**5
    hd = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    hdd = (60 * s - 180 * s**2 + 120 * s**3) / T**2
    return q0 + (q1 - q0) * h, (q1 - q0) * hd, (q1 - q0) * hdd


def reference(t):
    """Pick-and-place loop through the waypoints (and back), repeated forever."""
    pts = WAYPOINTS + [WAYPOINTS[0]]
    seg = SEG_T + HOLD
    k = int(t // seg) % (len(pts) - 1)
    return quintic(pts[k], pts[k + 1], SEG_T, t % seg)


class MujocoBridge(Node):
    def __init__(self):
        super().__init__("mujoco_bridge")
        share = get_package_share_directory("mbd_arm_sim")
        self.declare_parameter("model_path", os.path.join(share, "model", "arm.xml"))
        self.declare_parameter("mode", "ct")
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("real_time_factor", 1.0)
        path = self.get_parameter("model_path").value
        self.mode = self.get_parameter("mode").value
        rate = float(self.get_parameter("publish_rate").value)
        self.rtf = float(self.get_parameter("real_time_factor").value)

        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        self.m_nom = mujoco.MjModel.from_xml_path(path)      # controller's nominal model
        self.d_nom = mujoco.MjData(self.m_nom)
        self.d.qpos[:] = reference(0.0)[0]
        mujoco.mj_forward(self.m, self.d)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self.lo, self.hi = self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1]
        wn = 25.0
        self.Kp, self.Kd = wn**2, 2 * wn
        self.tau_ext = np.zeros(self.m.nu)

        self.pub_js = self.create_publisher(JointState, "joint_states", 10)
        self.pub_tip = self.create_publisher(PointStamped, "tip_position", 10)
        self.create_subscription(Float64MultiArray, "joint_effort_cmd", self.on_cmd, 10)
        self.period = 1.0 / rate
        self.steps = max(1, int(round(self.rtf * self.period / self.m.opt.timestep)))
        self.create_timer(self.period, self.tick)
        self.get_logger().info(f"MuJoCo bridge up: {path}, mode={self.mode}, {self.steps} physics steps per {self.period*1000:.0f} ms")

    def on_cmd(self, msg):
        if len(msg.data) != self.m.nu:
            self.get_logger().warn(f"expected {self.m.nu} torques, got {len(msg.data)}")
            return
        self.tau_ext = np.asarray(msg.data, float)

    def control(self):
        if self.mode == "external":
            return self.tau_ext
        qr, qdr, qddr = reference(self.d.time)
        a = qddr + self.Kp * (qr - self.d.qpos) + self.Kd * (qdr - self.d.qvel)
        self.d_nom.qpos[:] = self.d.qpos; self.d_nom.qvel[:] = self.d.qvel; self.d_nom.qacc[:] = a
        mujoco.mj_inverse(self.m_nom, self.d_nom)
        return self.d_nom.qfrc_inverse

    def tick(self):
        for _ in range(self.steps):
            self.d.ctrl[:] = np.clip(self.control(), self.lo, self.hi)
            mujoco.mj_step(self.m, self.d)
        now = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = now
        js.name = JOINTS
        js.position = self.d.qpos.tolist()
        js.velocity = self.d.qvel.tolist()
        js.effort = self.d.actuator_force.tolist()
        self.pub_js.publish(js)
        tip = PointStamped()
        tip.header.stamp = now; tip.header.frame_id = "base_link"   # URDF root (fixed at the MuJoCo world origin)
        tip.point.x, tip.point.y, tip.point.z = map(float, self.d.site_xpos[self.site])
        self.pub_tip.publish(tip)


def main():
    rclpy.init()
    node = MujocoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
