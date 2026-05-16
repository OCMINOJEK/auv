#!/usr/bin/env python3
"""AUV PID Autopilot v45 | Реверс при цели сзади + след траектории в RViz"""
import rclpy, math, time
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32, Float64
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

P_Z0  = 101325.0
RHO_G = 9810.0

class AUVController(Node):
    def __init__(self):
        super().__init__('auv_ctrl')
        self.pub_lt   = self.create_publisher(Float64, '/model/submarine/joint/left_propeller_joint/cmd_force', 10)
        self.pub_rt   = self.create_publisher(Float64, '/model/submarine/joint/right_propeller_joint/cmd_force', 10)
        self.pub_vert = self.create_publisher(Float64, '/model/submarine/joint/vertical_rudder/cmd_position', 10)
        self.pub_hl   = self.create_publisher(Float64, '/model/submarine/joint/horizontal_rudder_left/cmd_position', 10)
        self.pub_hr   = self.create_publisher(Float64, '/model/submarine/joint/horizontal_rudder_right/cmd_position', 10)

        # ── След траектории для визуализации в RViz ──────────────────
        # В RViz добавь Display: Path, topic: /auv/trail, fixed_frame: world
        self.pub_trail = self.create_publisher(Path, '/auv/trail', 10)
        self.trail = Path()
        self.trail.header.frame_id = 'world'
        self.trail_max_points = 2000   # ограничение размера

        self.create_subscription(Odometry, '/model/submarine/odometry', self.odom_cb, 10)
        qos_s = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Float32, '/model/submarine/pressure', self.press_cb, qos_s)

        self.state = 'INIT'
        self.pos   = [0.0, 0.0, 0.0]
        self.baro_z = 0.0
        self.vel   = 0.0
        self.rpy   = [0.0, 0.0, 0.0]
        self.prev_rpy = [0.0, 0.0, 0.0]
        self.prev_baro_z = 0.0
        self.target = [0.0, 0.0, 0.0]
        self.dist_2d = 1000.0
        self.bearing = 0.0
        self.stable_t = 0.0
        self.last_log_t = 0.0
        self.last_trail_t = 0.0

        # ── Коэффициенты ─────────────────────────────────────────────
        self.Kp_z = 3.0; self.Kd_z = 1.4
        self.Kp_yaw = 2.5; self.Kd_yaw = 0.7
        self.K_diff = 3.5            # сильный разворот
        self.Kp_roll = 14.0; self.Kd_roll = 4.5
        self.roll_bias = 0.04

        # Скорости
        self.max_speed = 2.0
        self.min_speed = 0.6

        # Углы курса
        self.YAW_BAD_DEG  = 90.0   # > 90° — цель сзади, тормозим в ноль
        self.YAW_WARN_DEG = 45.0   # > 45° — медленно ползём
        self.YAW_OK_DEG   = 20.0   # < 20° — полная скорость

        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.loop)

    def press_cb(self, msg):
        self.baro_z = (P_Z0 - msg.data) / RHO_G

    def odom_cb(self, msg):
        self.pos[0] = msg.pose.pose.position.x
        self.pos[1] = msg.pose.pose.position.y
        self.pos[2] = self.baro_z
        self.vel = msg.twist.twist.linear.x
        q = msg.pose.pose.orientation
        self.rpy[0] = math.atan2(2*(q.w*q.x + q.y*q.z), 1 - 2*(q.x**2 + q.y**2))
        self.rpy[1] = math.asin(max(-1.0, min(1.0, 2*(q.w*q.y - q.z*q.x))))
        self.rpy[2] = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

        if self.state == 'INIT':
            self.target = [self.raw_tx, self.raw_ty, self.raw_tz]
            self.prev_rpy = list(self.rpy)
            self.prev_baro_z = self.baro_z
            self.state = 'STAB'
            print(f"\n🎯 Цель: X={self.target[0]} Y={self.target[1]} Z={self.target[2]}")
            print(f"   Старт: X={self.pos[0]:.1f} Y={self.pos[1]:.1f} Z={self.pos[2]:.1f}")
            print(f"   След пишется в /auv/trail (RViz: Add → Path → topic /auv/trail)")

        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]
        self.dist_2d = math.hypot(dx, dy)
        self.bearing = math.atan2(dy, dx)

        # Записываем точку в след каждые 0.2с
        now = time.time()
        if now - self.last_trail_t >= 0.2:
            self.last_trail_t = now
            p = PoseStamped()
            p.header.frame_id = 'world'
            p.header.stamp = self.get_clock().now().to_msg()
            p.pose.position.x = self.pos[0]
            p.pose.position.y = self.pos[1]
            p.pose.position.z = self.pos[2]
            p.pose.orientation = msg.pose.pose.orientation
            self.trail.poses.append(p)
            if len(self.trail.poses) > self.trail_max_points:
                self.trail.poses.pop(0)
            self.trail.header.stamp = p.header.stamp
            self.pub_trail.publish(self.trail)

    def loop(self):
        if self.state == 'INIT': return

        z_err = self.pos[2] - self.target[2]
        dz_dt = (self.pos[2] - self.prev_baro_z) / self.dt
        self.prev_baro_z = self.pos[2]

        # ── PD по Z ───────────────────────────────────────────────────
        raw_h = -(self.Kp_z * z_err + self.Kd_z * dz_dt)
        rudder_h = max(-0.5, min(0.5, raw_h))

        # ── Курс ──────────────────────────────────────────────────────
        yaw_err = math.atan2(math.sin(self.bearing - self.rpy[2]),
                              math.cos(self.bearing - self.rpy[2]))
        d_yaw = (self.rpy[2] - self.prev_rpy[2]) / self.dt
        rudder_v = max(-0.45, min(0.45, self.Kp_yaw * yaw_err + self.Kd_yaw * d_yaw))

        # ── Крен ──────────────────────────────────────────────────────
        roll_err = self.rpy[0]
        d_roll = (self.rpy[0] - self.prev_rpy[0]) / self.dt
        roll_pid = self.Kp_roll * roll_err + self.Kd_roll * d_roll

        cmd_hl = max(-0.6, min(0.6, rudder_h - roll_pid - self.roll_bias))
        cmd_hr = max(-0.6, min(0.6, rudder_h + roll_pid + self.roll_bias))

        self.prev_rpy = list(self.rpy)

        cmd_lt = 0.0; cmd_rt = 0.0
        yaw_err_deg = abs(math.degrees(yaw_err))

        if self.state == 'STAB':
            if abs(roll_err) < 0.12:
                self.stable_t += self.dt
            else:
                self.stable_t = 0.0
            if self.stable_t >= 1.5:
                self.state = 'NAV'
                print("\n🚀 STAB → NAV")
            cmd_lt = -1.5; cmd_rt = -1.5

        elif self.state == 'NAV':
            # ── Выбор режима по ошибке курса ──────────────────────────
            if yaw_err_deg > self.YAW_BAD_DEG:
                # Цель СЗАДИ. Гасим скорость и разворачиваемся на месте.
                if self.vel > 0.2:
                    # Активное торможение (противоход)
                    thrust = +1.5 * 3.3   # ПОЛОЖИТЕЛЬНОЕ = назад в этой SDF
                else:
                    # Почти стоим — разворот дифференциалом
                    thrust = 0.0
                diff = self.K_diff * 1.5 * yaw_err   # усиленный разворот
                cmd_lt = thrust + diff
                cmd_rt = thrust - diff

            elif yaw_err_deg > self.YAW_WARN_DEG:
                # Курс сильно неправильный — ползём очень медленно
                target_speed = 0.4
                thrust = -target_speed * 3.3
                diff = self.K_diff * yaw_err
                cmd_lt = thrust + diff
                cmd_rt = thrust - diff

            else:
                # Курс приемлемый — нормальная навигация
                base_speed = max(self.min_speed, min(self.max_speed, self.dist_2d * 0.2))
                if yaw_err_deg > self.YAW_OK_DEG:
                    base_speed *= 0.6   # плавное снижение скорости
                thrust = -base_speed * 3.3
                diff = self.K_diff * yaw_err
                cmd_lt = thrust + diff
                cmd_rt = thrust - diff

            if self.dist_2d < 2.0 and abs(z_err) < 1.5:
                self.state = 'HOLD'
                print(f"\n✅ ЦЕЛЬ ДОСТИГНУТА  X={self.pos[0]:.2f} Y={self.pos[1]:.2f} Z={self.pos[2]:.2f}")

        elif self.state == 'HOLD':
            thrust = -0.4 * 3.3
            diff = self.K_diff * yaw_err
            cmd_lt = thrust + diff
            cmd_rt = thrust - diff

        self.pub_lt.publish(Float64(data=float(cmd_lt)))
        self.pub_rt.publish(Float64(data=float(cmd_rt)))
        self.pub_vert.publish(Float64(data=float(rudder_v)))
        self.pub_hl.publish(Float64(data=float(cmd_hl)))
        self.pub_hr.publish(Float64(data=float(cmd_hr)))

        now = time.time()
        if now - self.last_log_t >= 1.0:
            self.last_log_t = now
            yaw_deg = math.degrees(yaw_err)
            mode = ('REVERSE' if yaw_err_deg > self.YAW_BAD_DEG else
                    'SLOW'    if yaw_err_deg > self.YAW_WARN_DEG else
                    'NORM')
            print(f"[{self.state:5}|{mode:7}] "
                  f"Pos:[{self.pos[0]:+6.1f} {self.pos[1]:+6.1f} {self.pos[2]:+6.2f}] | "
                  f"D2D:{self.dist_2d:5.1f}m  Z_err:{z_err:+6.2f}m | "
                  f"Yaw_err:{yaw_deg:+5.1f}° | "
                  f"V:{self.vel:+.2f}  Roll:{math.degrees(roll_err):+5.1f}°")

    def run(self):
        try:
            print("=" * 60)
            print("🚢 AUV v45 — реверс при цели сзади + след в RViz")
            print("=" * 60)
            self.raw_tx = float(input("X цели: "))
            self.raw_ty = float(input("Y цели: "))
            self.raw_tz = float(input("Z цели: "))
            rclpy.spin(self)
        except (KeyboardInterrupt, SystemExit):
            for p in [self.pub_lt, self.pub_rt, self.pub_vert, self.pub_hl, self.pub_hr]:
                p.publish(Float64(data=0.0))

def main():
    rclpy.init()
    node = AUVController()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
