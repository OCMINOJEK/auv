#!/usr/bin/env python3
"""AUV PID Autopilot v37.0 | Без сальто: rate-limit на руле Z + HOLD с фиксацией курса"""
import rclpy, math, sys
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float64
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

P_Z0  = 101325.0
RHO_G = 9810.0

XY_TOL       = 2.0    # м
Z_TOL        = 1.5    # м
CLIMB_RADIUS = 12.0   # м
CLIMB_SPEED  = 1.0    # м/с

# Максимальная скорость изменения руля высоты за один тик (0.05 с)
# Ограничивает резкие броски носа → предотвращает сальто
RUDDER_H_RATE_LIMIT = 0.04   # рад/тик ≈ 0.8 рад/с

class AUVController(Node):
    def __init__(self):
        super().__init__('auv_ctrl')

        self.pub_lt   = self.create_publisher(Float64, '/model/submarine/joint/left_propeller_joint/cmd_force',       10)
        self.pub_rt   = self.create_publisher(Float64, '/model/submarine/joint/right_propeller_joint/cmd_force',      10)
        self.pub_vert = self.create_publisher(Float64, '/model/submarine/joint/vertical_rudder/cmd_position',         10)
        self.pub_hl   = self.create_publisher(Float64, '/model/submarine/joint/horizontal_rudder_left/cmd_position',  10)
        self.pub_hr   = self.create_publisher(Float64, '/model/submarine/joint/horizontal_rudder_right/cmd_position', 10)

        self.create_subscription(Odometry, '/model/submarine/odometry', self.odom_cb, 10)
        qos_s = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Float32, '/model/submarine/pressure', self.press_cb, qos_s)

        self.state       = 'INIT'
        self.pos         = [0.0, 0.0, 0.0]
        self.baro_z      = 0.0
        self.vel         = 0.0
        self.rpy         = [0.0, 0.0, 0.0]
        self.prev_rpy    = [0.0, 0.0, 0.0]
        self.prev_baro_z = 0.0
        self.target      = [0.0, 0.0, 0.0]
        self.dist_2d     = 1000.0
        self.bearing     = 0.0
        self.hold_bearing = 0.0   # зафиксированный курс в HOLD

        # Текущее положение руля высоты (для rate-limit)
        self.rudder_h_cur = 0.0

        self.max_cruise  = 2.2
        self.min_cruise  = 0.6

        # PD по Z
        # Уменьшены коэффициенты — большой Kd был причиной сальто
        self.Kp_z = 2.5
        self.Kd_z = 0.8   # было 1.6 — слишком резкий отклик на dz_dt

        # PD по курсу
        self.Kp_yaw      = 1.8
        self.Kd_yaw      = 0.5
        self.K_diff_base = 3.0

        # Крен
        self.Kp_roll  = 16.0
        self.Kd_roll  = 5.0
        self.roll_bias = 0.04

        # PD по радиусу (Z_CLIMB)
        self.Kp_r       = 0.06
        self.Kd_r       = 0.18
        self.prev_r_err = 0.0

        self.stable_t = 0.0
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.loop)

    # ─────────────────────────────────────────────────────────────────
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
            self.target       = [self.raw_target_x, self.raw_target_y, self.raw_target_z]
            self.prev_rpy     = list(self.rpy)
            self.prev_baro_z  = self.baro_z
            self.prev_r_err   = 0.0
            self.hold_bearing = self.rpy[2]
            self.state = 'STAB'
            print(f"\n🎯 AUV v37.0 | Цель:  X={self.target[0]:.1f} Y={self.target[1]:.1f} Z={self.target[2]:.1f}")
            print(f"               Старт: X={self.pos[0]:.1f} Y={self.pos[1]:.1f} Z={self.pos[2]:.1f}")

        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]
        self.dist_2d = math.hypot(dx, dy)

        if self.state == 'Z_CLIMB':
            angle_to_sub = math.atan2(
                self.pos[1] - self.target[1],
                self.pos[0] - self.target[0]
            )
            r_err  = self.dist_2d - CLIMB_RADIUS
            dr_err = (r_err - self.prev_r_err) / self.dt
            self.prev_r_err = r_err
            correction = -(self.Kp_r * r_err + self.Kd_r * dr_err)
            correction  = max(-0.7, min(0.7, correction))
            self.bearing = angle_to_sub + math.pi / 2 + correction

        elif self.state == 'HOLD':
            # В HOLD смотрим в сторону зафиксированного курса — не уплываем
            self.bearing = self.hold_bearing

        else:
            self.bearing = math.atan2(dy, dx)

    # ─────────────────────────────────────────────────────────────────
    def loop(self):
        if self.state not in ('STAB', 'NAV', 'Z_CLIMB', 'HOLD'):
            return

        # ── PD по Z с rate-limit ──────────────────────────────────────
        z_err    = self.pos[2] - self.target[2]
        dz_dt    = (self.pos[2] - self.prev_baro_z) / self.dt

        # Антивсплеск D-члена: фильтруем dz_dt экспоненциально
        if not hasattr(self, 'dz_filt'):
            self.dz_filt = 0.0
        self.dz_filt = 0.6 * self.dz_filt + 0.4 * dz_dt   # α=0.4

        raw_h = -(self.Kp_z * z_err + self.Kd_z * self.dz_filt)
        raw_h = max(-0.55, min(0.55, raw_h))

        # Rate-limit: руль не может двигаться быстрее RUDDER_H_RATE_LIMIT за тик
        delta = raw_h - self.rudder_h_cur
        delta = max(-RUDDER_H_RATE_LIMIT, min(RUDDER_H_RATE_LIMIT, delta))
        self.rudder_h_cur += delta
        rudder_h = self.rudder_h_cur

        self.prev_baro_z = self.pos[2]

        # ── PD по курсу ───────────────────────────────────────────────
        yaw_err = math.atan2(
            math.sin(self.bearing - self.rpy[2]),
            math.cos(self.bearing - self.rpy[2])
        )
        if abs(math.degrees(yaw_err)) < 1.0:
            yaw_err = 0.0
        d_yaw    = (self.rpy[2] - self.prev_rpy[2]) / self.dt
        rudder_v = max(-0.45, min(0.45, self.Kp_yaw * yaw_err + self.Kd_yaw * d_yaw))

        # ── Крен ──────────────────────────────────────────────────────
        roll_err = self.rpy[0]
        d_roll   = (self.rpy[0] - self.prev_rpy[0]) / self.dt
        roll_pid = self.Kp_roll * roll_err + self.Kd_roll * d_roll
        cmd_hl   = max(-0.6, min(0.6, rudder_h - roll_pid - self.roll_bias))
        cmd_hr   = max(-0.6, min(0.6, rudder_h + roll_pid + self.roll_bias))
        self.prev_rpy = list(self.rpy)

        thrust = 0.0; cmd_lt = 0.0; cmd_rt = 0.0

        # ══════════════════════════════════════════════════════════════

        if self.state == 'STAB':
            if abs(roll_err) < 0.12:
                self.stable_t += self.dt
            else:
                self.stable_t = 0.0
            if self.stable_t >= 1.5:
                self.state = 'NAV'
                print("\n🚀 STAB → NAV")
            cmd_hl = max(-0.15, min(0.15, -roll_pid - self.roll_bias))
            cmd_hr = max(-0.15, min(0.15,  roll_pid + self.roll_bias))

        elif self.state == 'NAV':
            target_speed = max(self.min_cruise, min(self.max_cruise, self.dist_2d * 0.35))
            if self.vel > target_speed + 0.2:
                thrust = 0.8
            else:
                thrust = -target_speed * 3.3

            k_diff = self.K_diff_base * (1.0 + abs(self.vel))
            diff   = k_diff * yaw_err
            cmd_lt = thrust + diff
            cmd_rt = thrust - diff

            if self.dist_2d < XY_TOL:
                if abs(z_err) > Z_TOL:
                    self.prev_r_err = self.dist_2d - CLIMB_RADIUS
                    self.state = 'Z_CLIMB'
                    print(f"\n🔄 NAV → Z_CLIMB  Z_err={z_err:+.2f}м")
                else:
                    self.hold_bearing = self.rpy[2]
                    self.state = 'HOLD'
                    print(f"\n✅ NAV → HOLD  pos={self.pos}")

        elif self.state == 'Z_CLIMB':
            if self.vel > CLIMB_SPEED + 0.1:
                thrust = 1.0
            else:
                thrust = -CLIMB_SPEED * 3.3

            k_diff = self.K_diff_base * (1.0 + abs(self.vel))
            diff   = k_diff * yaw_err
            cmd_lt = thrust + diff
            cmd_rt = thrust - diff

            if abs(math.degrees(roll_err)) > 35.0:
                cmd_lt   = thrust
                cmd_rt   = thrust
                rudder_v = 0.0

            if abs(z_err) < Z_TOL:
                self.hold_bearing = self.rpy[2]   # фиксируем текущий курс
                self.state = 'HOLD'
                print(f"\n✅ Z_CLIMB → HOLD  pos={self.pos}")

        elif self.state == 'HOLD':
            # Минимальная тяга вперёд чтобы рули работали
            # Тяга НЕ масштабируется с z_err — это было причиной разгона и перелёта
            hold_speed = 0.5
            if self.vel > hold_speed + 0.05:
                thrust = 0.5
            else:
                thrust = -hold_speed * 3.3

            # Рулим к зафиксированному курсу — не уплываем в сторону
            diff   = self.K_diff_base * yaw_err
            cmd_lt = thrust + diff
            cmd_rt = thrust - diff

        self._pub(cmd_lt, cmd_rt, rudder_v, cmd_hl, cmd_hr)
        print(
            f"\r[{self.state:10}] "
            f"Pos:[{self.pos[0]:+5.1f} {self.pos[1]:+5.1f} {self.pos[2]:+6.2f}] | "
            f"D2D:{self.dist_2d:5.1f}m Z_err:{z_err:+5.2f}m | "
            f"rh:{rudder_h:+.2f} dz:{self.dz_filt:+.2f} | "
            f"V:{self.vel:+.2f} Roll:{math.degrees(roll_err):+4.1f}°",
            end='', flush=True
        )

    # ─────────────────────────────────────────────────────────────────
    def _pub(self, lt, rt, rv, hl, hr):
        self.pub_lt.publish(Float64(data=float(lt)))
        self.pub_rt.publish(Float64(data=float(rt)))
        self.pub_vert.publish(Float64(data=float(rv)))
        self.pub_hl.publish(Float64(data=float(hl)))
        self.pub_hr.publish(Float64(data=float(hr)))

    def run(self):
        try:
            print("=" * 60)
            print("🚢 AUV v37.0 — Без сальто")
            print("   Ось Z вверх+. Всплытие = большее Z.")
            print("=" * 60)
            self.raw_target_x = float(input("📍 X цели (м): "))
            self.raw_target_y = float(input("📍 Y цели (м): "))
            self.raw_target_z = float(input("📍 Z цели (м, вверх+): "))
            rclpy.spin(self)
        except (KeyboardInterrupt, SystemExit):
            self._pub(0, 0, 0, 0, 0)

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
