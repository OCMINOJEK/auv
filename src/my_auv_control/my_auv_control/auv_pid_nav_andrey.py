#!/usr/bin/env python3
"""AUV PID Autopilot v35.1 | Fixed depth sign + hold on finish"""
import rclpy, math, time, sys
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float64
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

P_Z0   = 101325.0
RHO_G  = 9810.0

ORBIT_RADIUS     = 15.0
PREDICTIVE_ZONE  = ORBIT_RADIUS * 1.2   # было 1.5 → сократили, входим позже

class AUVController(Node):
    def __init__(self):
        super().__init__('auv_ctrl')
        self.pub_lt   = self.create_publisher(Float64, '/model/submarine/joint/left_propeller_joint/cmd_force',  10)
        self.pub_rt   = self.create_publisher(Float64, '/model/submarine/joint/right_propeller_joint/cmd_force', 10)
        self.pub_vert = self.create_publisher(Float64, '/model/submarine/joint/vertical_rudder/cmd_position',    10)
        self.pub_hl   = self.create_publisher(Float64, '/model/submarine/joint/horizontal_rudder_left/cmd_position',  10)
        self.pub_hr   = self.create_publisher(Float64, '/model/submarine/joint/horizontal_rudder_right/cmd_position', 10)

        self.create_subscription(Odometry, '/model/submarine/odometry', self.odom_cb, 10)

        qos_s = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Float32, '/model/submarine/pressure', self.press_cb, qos_s)

        self.state   = 'INIT'
        self.pos     = [0.0, 0.0, 0.0]
        self.baro_z  = 0.0
        self.vel     = 0.0
        self.rpy     = [0.0, 0.0, 0.0]
        self.prev_rpy    = [0.0, 0.0, 0.0]
        self.target_global = [0.0, 0.0, 0.0]
        self.bearing    = 0.0
        self.dist_2d    = 1000.0
        self.prev_baro_z = 0.0

        # ── Крейсерская скорость ──────────────────────────────────────
        self.max_cruise_speed = 2.2
        self.min_cruise_speed = 0.6
        self.brake_threshold  = 0.2

        # ── PID Z ─────────────────────────────────────────────────────
        self.Kp_z = 3.0; self.Kd_z = 1.4

        # ── PID курса ─────────────────────────────────────────────────
        self.Kp_yaw = 1.8; self.Kd_yaw = 0.5
        self.K_diff_base = 3.0

        # ── Стабилизация крена ────────────────────────────────────────
        self.Kp_roll = 16.0; self.Kd_roll = 5.0
        self.roll_bias = 0.04

        # ── Орбита: PD по радиусу ─────────────────────────────────────
        # Ошибка радиуса → поправка к bearing (добавочный угол от касательной)
        self.Kp_orbit_r = 0.06   # пропорциональная часть (рад / м ошибки)
        self.Kd_orbit_r = 0.18   # деривативная часть     (рад / (м/с))
        self.prev_radius_err = 0.0

        # Скорость орбиты чуть выше, чтобы рули крена и курса работали
        self.orbit_speed = 0.9

        self.stable_t = 0.0
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.loop)

    # ─────────────────────────────────────────────────────────────────
    def press_cb(self, msg):
        # fake_barometer публикует: pressure = P_Z0 - RHO_G * gz_z
        # → gz_z = (P_Z0 - pressure) / RHO_G
        # gz_z отрицательный когда аппарат под водой (Gazebo Z вверх)
        self.baro_z = (P_Z0 - msg.data) / RHO_G

    def odom_cb(self, msg):
        self.pos[0] = msg.pose.pose.position.x
        self.pos[1] = msg.pose.pose.position.y
        self.pos[2] = self.baro_z

        self.vel = msg.twist.twist.linear.x
        q = msg.pose.pose.orientation
        self.rpy[0] = math.atan2(2*(q.w*q.x + q.y*q.z), 1-2*(q.x**2 + q.y**2))
        self.rpy[1] = math.asin(max(-1.0, min(1.0, 2*(q.w*q.y - q.z*q.x))))
        self.rpy[2] = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y**2 + q.z**2))

        if self.state == 'INIT':
            self.target_global = [self.raw_target_x, self.raw_target_y, self.raw_target_z]
            self.prev_rpy        = list(self.rpy)
            self.prev_baro_z     = self.baro_z
            self.prev_radius_err = 0.0
            self.state = 'STAB'
            print(f"\n🎯 AUV v35.1 | Цель: "
                  f"X={self.target_global[0]:.2f} Y={self.target_global[1]:.2f} "
                  f"Z={self.target_global[2]:.2f}м (ось вверх)")

        dx = self.target_global[0] - self.pos[0]
        dy = self.target_global[1] - self.pos[1]
        self.dist_2d = math.hypot(dx, dy)

        # ── Вычисление bearing ────────────────────────────────────────
        if self.state == 'ORBIT':
            # Угол от цели до аппарата
            angle_to_sub  = math.atan2(self.pos[1] - self.target_global[1],
                                        self.pos[0] - self.target_global[0])
            # Ошибка радиуса и её производная
            radius_err    = self.dist_2d - ORBIT_RADIUS
            d_radius_err  = (radius_err - self.prev_radius_err) / self.dt
            self.prev_radius_err = radius_err

            # PD поправка: если далеко от орбиты — доворачиваем нос внутрь/наружу
            # Знак: radius_err > 0 → аппарат снаружи → нужно довернуть ВНУТРЬ (-correction)
            correction = -(self.Kp_orbit_r * radius_err + self.Kd_orbit_r * d_radius_err)
            correction  = max(-0.7, min(0.7, correction))

            # Касательная к орбите + PD-поправка
            self.bearing = angle_to_sub + math.pi/2 + correction
        else:
            self.bearing = math.atan2(dy, dx)

    # ─────────────────────────────────────────────────────────────────
    def loop(self):
        if self.state not in ['STAB', 'NAV', 'ORBIT', 'FINAL_LOCK', 'HOLD']:
            return

        # ── Высота ───────────────────────────────────────────────────
        z_err  = self.pos[2] - self.target_global[2]
        dz_dt  = (self.pos[2] - self.prev_baro_z) / self.dt
        raw_h  = -(self.Kp_z * z_err + self.Kd_z * dz_dt)
        # Увеличен лимит руля: большая z_err требует большего отклонения
        rudder_h = max(-0.45, min(0.45, raw_h))
        self.prev_baro_z = self.pos[2]

        # ── Курс ─────────────────────────────────────────────────────
        yaw_err = math.atan2(math.sin(self.bearing - self.rpy[2]),
                              math.cos(self.bearing - self.rpy[2]))
        if abs(math.degrees(yaw_err)) < 1.0:
            yaw_err = 0.0
        d_yaw    = (self.rpy[2] - self.prev_rpy[2]) / self.dt
        rudder_v = max(-0.45, min(0.45, self.Kp_yaw * yaw_err + self.Kd_yaw * d_yaw))

        # ── Стабилизация крена ────────────────────────────────────────
        roll_err = self.rpy[0]
        d_roll   = (self.rpy[0] - self.prev_rpy[0]) / self.dt
        roll_pid = self.Kp_roll * roll_err + self.Kd_roll * d_roll

        cmd_hl = max(-0.6, min(0.6, rudder_h - roll_pid - self.roll_bias))
        cmd_hr = max(-0.6, min(0.6, rudder_h + roll_pid + self.roll_bias))
        self.prev_rpy = list(self.rpy)

        thrust = 0.0; cmd_lt = 0.0; cmd_rt = 0.0

        # ═════════════════ АВТОМАТ СОСТОЯНИЙ ═════════════════════════

        if self.state == 'STAB':
            if abs(roll_err) < 0.12:
                self.stable_t += self.dt
            else:
                self.stable_t = 0.0
            if self.stable_t >= 1.5:
                self.state = 'NAV'
            cmd_hl = max(-0.15, min(0.15, -roll_pid - self.roll_bias))
            cmd_hr = max(-0.15, min(0.15,  roll_pid + self.roll_bias))

        elif self.state == 'NAV':
            target_speed = max(self.min_cruise_speed,
                               min(self.max_cruise_speed, self.dist_2d * 0.35))
            if self.vel > target_speed + self.brake_threshold:
                thrust = 0.8
            else:
                thrust = -target_speed * 3.3

            # Предиктивный вход в орбиту
            if self.dist_2d < PREDICTIVE_ZONE and abs(z_err) >= 1.5:
                abs_dz  = max(abs(dz_dt), 0.05)
                abs_vel = max(abs(self.vel), 0.1)
                if (abs(z_err) / abs_dz) > (self.dist_2d / abs_vel):
                    self.state = 'ORBIT'
                    self.prev_radius_err = self.dist_2d - ORBIT_RADIUS
                    sys.stdout.write("\n🔮 PREDICT → ORBIT\n"); sys.stdout.flush()

            k_diff = self.K_diff_base * (1.0 + abs(self.vel))
            diff   = k_diff * yaw_err
            cmd_lt = thrust + diff; cmd_rt = thrust - diff

        elif self.state == 'ORBIT':
            # Удерживаем постоянную скорость орбиты
            if self.vel > self.orbit_speed + 0.1:
                thrust = 1.0
            else:
                thrust = -self.orbit_speed * 3.3

            # Адаптивный дифференциал: масштабируем скоростью, как в NAV
            # Это главное исправление — без него на малой скорости не хватает
            # момента для удержания кривой нужного радиуса
            k_diff = self.K_diff_base * (1.0 + abs(self.vel))
            diff   = k_diff * yaw_err
            cmd_lt = thrust + diff; cmd_rt = thrust - diff

            # Аварийный предохранитель от переворота
            if abs(math.degrees(roll_err)) > 35.0:
                cmd_lt  = thrust; cmd_rt = thrust
                rudder_v = 0.0

            if abs(z_err) < 1.5:
                self.state = 'FINAL_LOCK'
                sys.stdout.write("\n🎯 FINAL_LOCK\n"); sys.stdout.flush()

        elif self.state == 'FINAL_LOCK':
            target_speed = max(self.min_cruise_speed, min(1.0, self.dist_2d * 0.4))
            if self.vel > target_speed + self.brake_threshold:
                thrust = 0.2
            else:
                thrust = -target_speed * 3.3

            diff   = self.K_diff_base * yaw_err
            cmd_lt = thrust + diff; cmd_rt = thrust - diff

            # Переходим в HOLD как только XY достигнуто — Z доберём в HOLD
            if self.dist_2d < 2.0:
                self.state = 'HOLD'
                print(f"\n✅ XY достигнуто → HOLD (Z_err={z_err:+.2f}м) | "
                      f"X={self.pos[0]:.2f} Y={self.pos[1]:.2f} Z={self.pos[2]:.2f}")

        elif self.state == 'HOLD':
            # Скорость зависит от оставшейся z-ошибки:
            # если ещё далеко по Z — держим достаточный ход чтобы рули работали
            z_hold_err = abs(z_err)
            if z_hold_err > 5.0:
                hold_speed = 1.2   # рули эффективны, активно всплываем/тонем
            elif z_hold_err > 1.0:
                hold_speed = 0.7
            else:
                hold_speed = 0.4   # почти на месте, минимальный ход

            if self.vel > hold_speed + 0.05:
                thrust = 0.5
            else:
                thrust = -hold_speed * 3.3
            diff   = self.K_diff_base * yaw_err
            cmd_lt = thrust + diff; cmd_rt = thrust - diff

        self._pub(cmd_lt, cmd_rt, rudder_v, cmd_hl, cmd_hr)
        print(f"\r[{self.state:12}] Pos:[{self.pos[0]:+.1f},{self.pos[1]:+.1f},gz={self.pos[2]:+.2f}m] | "
              f"D2D:{self.dist_2d:.1f}m R_err:{self.dist_2d - ORBIT_RADIUS:+.1f}m | "
              f"V:{self.vel:+.2f} Z_err:{z_err:+.2f} | "
              f"Roll:{math.degrees(roll_err):+.1f}°",
              end='', flush=True)

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
            print("🚢 AUV v35.1 — Fixed Depth Sign + Hold")
            print("=" * 60)
            self.raw_target_x = float(input("📍 X цели (м): "))
            self.raw_target_y = float(input("📍 Y цели (м): "))
            self.raw_target_z = float(input("📍 Z цели (Gazebo, ось вверх, м): "))
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
