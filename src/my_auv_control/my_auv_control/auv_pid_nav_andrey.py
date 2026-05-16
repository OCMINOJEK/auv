#!/usr/bin/env python3
"""AUV PID Autopilot v42.0 | Стоп-разворот + правильное удержание курса"""
import rclpy, math, time
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float64
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

P_Z0  = 101325.0
RHO_G = 9810.0

XY_TOL       = 2.0    # м — XY достигнуто
Z_TOL        = 1.5    # м — Z достигнуто
CLIMB_RADIUS = 12.0   # м
CLIMB_SPEED  = 0.8    # м/с

# Углы курса: если yaw_err больше этого — сначала разворачиваемся, потом едем
YAW_DRIVE_DEG = 20.0   # °: при ошибке курса > 20° — стоп и поворот

ROLL_RECOV_DEG = 15.0  # ° — порог RECOVERY (снижен для ранней реакции)

def fwd(speed):
    return -abs(speed) * 3.3

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

        self.state        = 'INIT'
        self.prev_state   = 'INIT'
        self.pos          = [0.0, 0.0, 0.0]
        self.baro_z       = 0.0
        self.vel          = 0.0
        self.rpy          = [0.0, 0.0, 0.0]
        self.prev_rpy     = [0.0, 0.0, 0.0]
        self.prev_baro_z  = 0.0
        self.target       = [0.0, 0.0, 0.0]
        self.dist_2d      = 1000.0
        self.bearing      = 0.0
        self.hold_bearing = 0.0
        self.rudder_h_cur = 0.0
        self.dz_filt      = 0.0
        self.recov_t      = 0.0
        self.stable_t     = 0.0
        self.prev_r_err   = 0.0
        self.last_log_t   = 0.0

        # PD по Z
        self.Kp_z = 2.5; self.Kd_z = 0.8

        # PD по курсу (вертикальный руль)
        self.Kp_yaw = 2.0; self.Kd_yaw = 0.6

        # Разворот на месте через дифференциал моторов
        # Используется ТОЛЬКО когда аппарат почти остановлен
        self.K_turn = 4.0

        # Крен (горизонтальные рули дифференциально)
        self.Kp_roll = 20.0; self.Kd_roll = 6.0
        self.roll_bias = 0.04

        # PD по радиусу (Z_CLIMB)
        self.Kp_r = 0.06; self.Kd_r = 0.18

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
            self.target       = [self.raw_target_x, self.raw_target_y, self.raw_target_z]
            self.prev_rpy     = list(self.rpy)
            self.prev_baro_z  = self.baro_z
            self.hold_bearing = self.rpy[2]
            self.state = 'STAB'
            print(f"\n🎯 AUV v42.0")
            print(f"   Цель:  X={self.target[0]:.1f} Y={self.target[1]:.1f} Z={self.target[2]:.1f}")
            print(f"   Старт: X={self.pos[0]:.1f} Y={self.pos[1]:.1f} Z={self.pos[2]:.1f}")

        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]
        self.dist_2d = math.hypot(dx, dy)

        if self.state == 'Z_CLIMB':
            angle_to_sub = math.atan2(self.pos[1] - self.target[1],
                                       self.pos[0] - self.target[0])
            r_err  = self.dist_2d - CLIMB_RADIUS
            dr_err = (r_err - self.prev_r_err) / self.dt
            self.prev_r_err = r_err
            correction = max(-0.5, min(0.5,
                -(self.Kp_r * r_err + self.Kd_r * dr_err)))
            self.bearing = angle_to_sub + math.pi / 2 + correction
        elif self.state == 'HOLD':
            self.bearing = self.hold_bearing
        elif self.state != 'RECOVERY':
            self.bearing = math.atan2(dy, dx)

    def loop(self):
        if self.state not in ('STAB', 'NAV', 'Z_CLIMB', 'HOLD', 'RECOVERY'):
            return

        z_err    = self.pos[2] - self.target[2]
        roll_rad = self.rpy[0]
        roll_deg = math.degrees(roll_rad)
        now      = time.time()

        # ── RECOVERY ─────────────────────────────────────────────────
        if self.state != 'RECOVERY' and abs(roll_deg) > ROLL_RECOV_DEG:
            self.prev_state = self.state
            self.recov_t    = 0.0
            self.state      = 'RECOVERY'
            print(f"\n⚠️  RECOVERY (крен {roll_deg:+.1f}°) ← {self.prev_state}")

        if self.state == 'RECOVERY':
            self.recov_t += self.dt
            d_roll = (roll_rad - self.prev_rpy[0]) / self.dt
            roll_err_signed = math.atan2(math.sin(-roll_rad), math.cos(-roll_rad))
            roll_cmd = max(-0.6, min(0.6,
                self.Kp_roll * roll_err_signed - self.Kd_roll * d_roll))
            cmd_hl = max(-0.6, min(0.6, -roll_cmd - self.roll_bias))
            cmd_hr = max(-0.6, min(0.6,  roll_cmd + self.roll_bias))
            # Стоп — при нулевой скорости нет кренящего момента
            t = fwd(0.4)
            self.prev_rpy = list(self.rpy)
            if abs(roll_deg) < 5.0 and self.recov_t > 0.5:
                self.rudder_h_cur = 0.0
                self.dz_filt      = 0.0
                self.state = self.prev_state
                print(f"\n✅ RECOVERY → {self.state} (крен {roll_deg:+.1f}°)")
            self._pub(t, t, 0.0, cmd_hl, cmd_hr)
            if now - self.last_log_t >= 1.0:
                self.last_log_t = now
                print(f"\n[RECOVERY   ] крен:{roll_deg:+.1f}° cmd:{roll_cmd:+.2f} t:{self.recov_t:.1f}с")
            return

        # ── PD по Z ───────────────────────────────────────────────────
        dz_dt = (self.pos[2] - self.prev_baro_z) / self.dt
        self.dz_filt = 0.6 * self.dz_filt + 0.4 * dz_dt
        raw_h = -(self.Kp_z * z_err + self.Kd_z * self.dz_filt)
        raw_h = max(-0.55, min(0.55, raw_h))
        delta = max(-0.03, min(0.03, raw_h - self.rudder_h_cur))
        self.rudder_h_cur += delta
        rudder_h = self.rudder_h_cur
        self.prev_baro_z = self.pos[2]

        # ── Курс ──────────────────────────────────────────────────────
        yaw_err = math.atan2(math.sin(self.bearing - self.rpy[2]),
                              math.cos(self.bearing - self.rpy[2]))
        if abs(math.degrees(yaw_err)) < 1.0:
            yaw_err = 0.0
        d_yaw    = (self.rpy[2] - self.prev_rpy[2]) / self.dt
        rudder_v = max(-0.45, min(0.45,
            self.Kp_yaw * yaw_err + self.Kd_yaw * d_yaw))

        # ── Крен ──────────────────────────────────────────────────────
        d_roll = (roll_rad - self.prev_rpy[0]) / self.dt
        roll_err_signed = math.atan2(math.sin(-roll_rad), math.cos(-roll_rad))
        roll_pid = self.Kp_roll * roll_err_signed - self.Kd_roll * d_roll
        cmd_hl = max(-0.6, min(0.6, rudder_h - roll_pid - self.roll_bias))
        cmd_hr = max(-0.6, min(0.6, rudder_h + roll_pid + self.roll_bias))
        self.prev_rpy = list(self.rpy)

        cmd_lt = 0.0; cmd_rt = 0.0

        # ══════════════════════════════════════════════════════════════

        if self.state == 'STAB':
            if abs(roll_deg) < 3.0:
                self.stable_t += self.dt
            else:
                self.stable_t = 0.0
            if self.stable_t >= 1.5:
                self.state = 'NAV'
                print("\n🚀 STAB → NAV")
            cmd_hl = max(-0.15, min(0.15, -roll_pid - self.roll_bias))
            cmd_hr = max(-0.15, min(0.15,  roll_pid + self.roll_bias))
            t = fwd(0.4); cmd_lt = t; cmd_rt = t

        elif self.state == 'NAV':
            yaw_deg = abs(math.degrees(yaw_err))

            if yaw_deg > YAW_DRIVE_DEG:
                # ── РАЗВОРОТ НА МЕСТЕ ─────────────────────────────────
                # Тормозим до нуля, потом крутимся дифференциалом
                if abs(self.vel) > 0.1:
                    # Торможение: противоход
                    brake = min(abs(self.vel) * 3.3, 5.0)
                    t = brake if self.vel < 0 else -brake
                    cmd_lt = t; cmd_rt = t
                else:
                    # Стоим — крутимся дифференциалом моторов
                    # При нулевой скорости дифференциал не кренит
                    turn = max(-6.0, min(6.0, self.K_turn * yaw_err))
                    cmd_lt = -turn   # знак: положительный yaw_err = повернуть влево
                    cmd_rt =  turn
            else:
                # ── ЕДЕМ ВПЕРЁД К ЦЕЛИ ────────────────────────────────
                target_speed = max(0.5, min(1.2, self.dist_2d * 0.15))
                if self.vel < target_speed - 0.15:
                    t = fwd(target_speed)
                elif self.vel > target_speed + 0.15:
                    t = fwd(target_speed * 0.3)
                else:
                    t = fwd(target_speed * 0.7)
                cmd_lt = t; cmd_rt = t

            if self.dist_2d < XY_TOL:
                if abs(z_err) > Z_TOL:
                    self.prev_r_err = self.dist_2d - CLIMB_RADIUS
                    self.state = 'Z_CLIMB'
                    print(f"\n🔄 NAV → Z_CLIMB  Z_err={z_err:+.2f}м")
                else:
                    self.hold_bearing = self.rpy[2]
                    self.state = 'HOLD'
                    print(f"\n✅ HOLD  X={self.pos[0]:.2f} Y={self.pos[1]:.2f} Z={self.pos[2]:.2f}")

        elif self.state == 'Z_CLIMB':
            yaw_deg = abs(math.degrees(yaw_err))
            if yaw_deg > YAW_DRIVE_DEG:
                if abs(self.vel) > 0.1:
                    brake = min(abs(self.vel) * 3.3, 5.0)
                    t = brake if self.vel < 0 else -brake
                    cmd_lt = t; cmd_rt = t
                else:
                    turn = max(-6.0, min(6.0, self.K_turn * yaw_err))
                    cmd_lt = -turn; cmd_rt = turn
            else:
                if self.vel < CLIMB_SPEED - 0.1:
                    t = fwd(CLIMB_SPEED)
                elif self.vel > CLIMB_SPEED + 0.1:
                    t = fwd(CLIMB_SPEED * 0.4)
                else:
                    t = fwd(CLIMB_SPEED * 0.7)
                cmd_lt = t; cmd_rt = t

            if abs(z_err) < Z_TOL:
                self.hold_bearing = self.rpy[2]
                self.state = 'HOLD'
                print(f"\n✅ HOLD  X={self.pos[0]:.2f} Y={self.pos[1]:.2f} Z={self.pos[2]:.2f}")

        elif self.state == 'HOLD':
            # Удерживаем позицию — минимальная тяга, держим курс и Z
            hold_speed = 0.4
            if self.vel < hold_speed - 0.05:
                t = fwd(hold_speed)
            elif self.vel > hold_speed + 0.05:
                t = fwd(hold_speed * 0.3)
            else:
                t = fwd(hold_speed * 0.6)
            cmd_lt = t; cmd_rt = t

        self._pub(cmd_lt, cmd_rt, rudder_v, cmd_hl, cmd_hr)

        if now - self.last_log_t >= 1.0:
            self.last_log_t = now
            yaw_deg_log = math.degrees(math.atan2(
                math.sin(self.bearing - self.rpy[2]),
                math.cos(self.bearing - self.rpy[2])))
            print(f"\n[{self.state:10}] "
                  f"Pos:[{self.pos[0]:+6.1f} {self.pos[1]:+6.1f} {self.pos[2]:+7.2f}] | "
                  f"D2D:{self.dist_2d:6.1f}m  Z_err:{z_err:+7.2f}m | "
                  f"Yaw_err:{yaw_deg_log:+5.1f}° rh:{rudder_h:+.2f} | "
                  f"V:{self.vel:+.2f}  Roll:{roll_deg:+5.1f}°")

    def _pub(self, lt, rt, rv, hl, hr):
        self.pub_lt.publish(Float64(data=float(lt)))
        self.pub_rt.publish(Float64(data=float(rt)))
        self.pub_vert.publish(Float64(data=float(rv)))
        self.pub_hl.publish(Float64(data=float(hl)))
        self.pub_hr.publish(Float64(data=float(hr)))

    def run(self):
        try:
            print("=" * 60)
            print("🚢 AUV v42.0 — стоп-разворот, крен ≤5°")
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
