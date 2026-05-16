#!/usr/bin/env python3
"""AUV PID Autopilot v38.0 | Emergency Roll Recovery + корректный знак крена"""
import rclpy, math, sys
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float64
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

P_Z0  = 101325.0
RHO_G = 9810.0

XY_TOL       = 2.0
Z_TOL        = 1.5
CLIMB_RADIUS = 12.0
CLIMB_SPEED  = 1.0

RUDDER_H_RATE  = 0.04   # рад/тик — rate-limit руля высоты

# Пороги крена
ROLL_WARN_DEG    = 25.0   # предупреждение, начинаем активнее стабилизировать
ROLL_RECOV_DEG   = 50.0   # входим в режим RECOVERY — всё бросаем, выравниваемся

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
        self.prev_state   = 'INIT'   # куда вернуться после RECOVERY
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
        self.recov_t      = 0.0   # время в режиме RECOVERY

        self.max_cruise  = 2.2
        self.min_cruise  = 0.6

        # PD по Z
        self.Kp_z = 2.5
        self.Kd_z = 0.8

        # PD по курсу
        self.Kp_yaw      = 1.8
        self.Kd_yaw      = 0.5
        self.K_diff_base = 3.0

        # Стабилизация крена
        # Ключевое: крен вычисляется как минимальный угол до нуля,
        # знак всегда правильный независимо от того перевёрнут аппарат или нет
        self.Kp_roll  = 8.0    # уменьшен — агрессивный roll PID сам вызывал проблемы
        self.Kd_roll  = 3.0
        self.roll_bias = 0.04

        # PD по радиусу (Z_CLIMB)
        self.Kp_r       = 0.06
        self.Kd_r       = 0.18
        self.prev_r_err = 0.0

        self.stable_t = 0.0
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
            print(f"\n🎯 AUV v38.0 | Цель:  X={self.target[0]:.1f} Y={self.target[1]:.1f} Z={self.target[2]:.1f}")
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
            correction = max(-0.7, min(0.7,
                -(self.Kp_r * r_err + self.Kd_r * dr_err)))
            self.bearing = angle_to_sub + math.pi / 2 + correction
        elif self.state == 'HOLD':
            self.bearing = self.hold_bearing
        elif self.state == 'RECOVERY':
            # В RECOVERY летим прямо — не меняем bearing
            pass
        else:
            self.bearing = math.atan2(dy, dx)

    def loop(self):
        if self.state not in ('STAB', 'NAV', 'Z_CLIMB', 'HOLD', 'RECOVERY'):
            return

        z_err = self.pos[2] - self.target[2]
        roll_deg = math.degrees(self.rpy[0])

        # ══════════════════════════════════════════════════════════════
        # RECOVERY: аппарат сильно накренился — выравниваем крен в первую очередь
        # ══════════════════════════════════════════════════════════════
        if self.state != 'RECOVERY' and abs(roll_deg) > ROLL_RECOV_DEG:
            self.prev_state = self.state
            self.recov_t    = 0.0
            self.state      = 'RECOVERY'
            print(f"\n⚠️  RECOVERY (крен {roll_deg:+.1f}°) ← {self.prev_state}")

        if self.state == 'RECOVERY':
            self.recov_t += self.dt

            # Крен: используем кратчайший путь к нулю
            # math.atan2(sin(roll), cos(roll)) = roll для малых углов,
            # но для roll=±π это будет ±π — именно то что нам нужно.
            # Однако при roll=137° правильная команда: повернуть к нулю через -43°
            # roll_err_signed даёт нам правильный знак всегда:
            roll_rad = self.rpy[0]
            d_roll   = (roll_rad - self.prev_rpy[0]) / self.dt
            # Целимся в roll=0 через кратчайший путь
            roll_err_signed = math.atan2(math.sin(-roll_rad), math.cos(-roll_rad))
            roll_cmd = self.Kp_roll * roll_err_signed - self.Kd_roll * d_roll
            roll_cmd = max(-0.6, min(0.6, roll_cmd))

            # Горизонтальные рули: только выравнивание крена, без Z
            cmd_hl = max(-0.6, min(0.6, -roll_cmd - self.roll_bias))
            cmd_hr = max(-0.6, min(0.6,  roll_cmd + self.roll_bias))

            # Вертикальный руль: нейтраль — не пытаемся рулить курсом
            rudder_v = 0.0

            # Тяга минимальная — рули должны работать, но не разгоняемся
            thrust = -0.8 * 3.3
            cmd_lt = thrust
            cmd_rt = thrust

            self.prev_rpy = list(self.rpy)

            # Выходим из RECOVERY когда крен нормализован и прошло хотя бы 1с
            if abs(roll_deg) < ROLL_WARN_DEG and self.recov_t > 1.0:
                self.rudder_h_cur = 0.0   # сброс руля высоты
                self.dz_filt      = 0.0
                self.state = self.prev_state
                print(f"\n✅ RECOVERY завершён (крен {roll_deg:+.1f}°) → {self.state}")

            self._pub(cmd_lt, cmd_rt, rudder_v, cmd_hl, cmd_hr)
            print(
                f"\r[RECOVERY   ] крен:{roll_deg:+6.1f}° → cmd:{roll_cmd:+.2f} | "
                f"V:{self.vel:+.2f} t:{self.recov_t:.1f}с",
                end='', flush=True
            )
            return

        # ══════════════════════════════════════════════════════════════
        # Обычное управление
        # ══════════════════════════════════════════════════════════════

        # ── PD по Z с фильтрацией и rate-limit ───────────────────────
        dz_dt = (self.pos[2] - self.prev_baro_z) / self.dt
        self.dz_filt = 0.6 * self.dz_filt + 0.4 * dz_dt
        raw_h = -(self.Kp_z * z_err + self.Kd_z * self.dz_filt)
        raw_h = max(-0.55, min(0.55, raw_h))
        delta = max(-RUDDER_H_RATE, min(RUDDER_H_RATE, raw_h - self.rudder_h_cur))
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

        # ── Крен: масштабируем силу стабилизации по величине крена ───
        roll_rad = self.rpy[0]
        d_roll   = (roll_rad - self.prev_rpy[0]) / self.dt
        # Кратчайший путь к roll=0
        roll_err_signed = math.atan2(math.sin(-roll_rad), math.cos(-roll_rad))
        roll_pid = self.Kp_roll * roll_err_signed - self.Kd_roll * d_roll

        # При большом крене (>WARN) усиливаем корректирующий момент
        if abs(roll_deg) > ROLL_WARN_DEG:
            boost = 1.0 + (abs(roll_deg) - ROLL_WARN_DEG) / 20.0
            roll_pid *= min(boost, 2.5)

        cmd_hl = max(-0.6, min(0.6, rudder_h - roll_pid - self.roll_bias))
        cmd_hr = max(-0.6, min(0.6, rudder_h + roll_pid + self.roll_bias))
        self.prev_rpy = list(self.rpy)

        thrust = 0.0; cmd_lt = 0.0; cmd_rt = 0.0

        # ══════════════════════════════════════════════════════════════

        if self.state == 'STAB':
            if abs(roll_deg) < math.degrees(0.12):
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
                    print(f"\n✅ HOLD  X={self.pos[0]:.2f} Y={self.pos[1]:.2f} Z={self.pos[2]:.2f}")

        elif self.state == 'Z_CLIMB':
            if self.vel > CLIMB_SPEED + 0.1:
                thrust = 1.0
            else:
                thrust = -CLIMB_SPEED * 3.3

            k_diff = self.K_diff_base * (1.0 + abs(self.vel))
            diff   = k_diff * yaw_err
            cmd_lt = thrust + diff
            cmd_rt = thrust - diff

            if abs(z_err) < Z_TOL:
                self.hold_bearing = self.rpy[2]
                self.state = 'HOLD'
                print(f"\n✅ HOLD  X={self.pos[0]:.2f} Y={self.pos[1]:.2f} Z={self.pos[2]:.2f}")

        elif self.state == 'HOLD':
            hold_speed = 0.5
            if self.vel > hold_speed + 0.05:
                thrust = 0.5
            else:
                thrust = -hold_speed * 3.3
            diff   = self.K_diff_base * yaw_err
            cmd_lt = thrust + diff
            cmd_rt = thrust - diff

        self._pub(cmd_lt, cmd_rt, rudder_v, cmd_hl, cmd_hr)
        print(
            f"\r[{self.state:10}] "
            f"Pos:[{self.pos[0]:+5.1f} {self.pos[1]:+5.1f} {self.pos[2]:+6.2f}] | "
            f"D2D:{self.dist_2d:5.1f}m Z_err:{z_err:+5.2f}m | "
            f"rh:{rudder_h:+.2f} V:{self.vel:+.2f} Roll:{roll_deg:+5.1f}°",
            end='', flush=True
        )

    def _pub(self, lt, rt, rv, hl, hr):
        self.pub_lt.publish(Float64(data=float(lt)))
        self.pub_rt.publish(Float64(data=float(rt)))
        self.pub_vert.publish(Float64(data=float(rv)))
        self.pub_hl.publish(Float64(data=float(hl)))
        self.pub_hr.publish(Float64(data=float(hr)))

    def run(self):
        try:
            print("=" * 60)
            print("🚢 AUV v38.0 — Emergency Roll Recovery")
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
