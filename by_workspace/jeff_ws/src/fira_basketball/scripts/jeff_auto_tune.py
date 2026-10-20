#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import random
import numpy as np
from sensor_msgs.msg import Imu
from op3_utils import Robot_weightlift

class WalkingTuner:
    def __init__(self):
        rospy.init_node('walking_ai_tuner', anonymous=True)
        
        # 使用您原本的機器人控制類別
        self.robot = Robot_weightlift('fira_basketball')
        
        # 訂閱 IMU 來感受搖晃程度
        self.imu_sub = rospy.Subscriber('/robotis/open_cr/imu', Imu, self.imu_callback)
        
        self.current_gyro_x = []
        self.current_gyro_y = []
        self.is_recording = False
        
        rospy.sleep(2) # 等待系統連線

    def imu_callback(self, msg):
        # 記錄角速度 (X軸是前後搖晃，Y軸是左右側翻)
        if self.is_recording:
            self.current_gyro_x.append(abs(msg.angular_velocity.x))
            self.current_gyro_y.append(abs(msg.angular_velocity.y))

    def evaluate_params(self, z_amp, y_amp):
        rospy.loginfo("=======================================")
        rospy.loginfo("正在測試 -> Z(抬腿): {:.4f}, Y(搖晃): {:.4f}".format(z_amp, y_amp))
        
        # 1. 載入基本參數並覆寫我們要測試的這兩個變數
        param = self.robot.loadWalkingParams('pickup_param.yaml')
        param.z_move_amplitude = z_amp
        param.y_move_amplitude = y_amp
        param.balance_enable = True # 草地必備
        
        # 2. 切換模組並套用參數
        self.robot.setGeneralControlModule("walking_module")
        self.robot.walk_set_param_pub.publish(param)
        rospy.sleep(1.0) # 等待重心轉移

        # 3. 開始走路並紀錄 IMU
        self.current_gyro_x = []
        self.current_gyro_y = []
        self.is_recording = True
        
        self.robot.walkStart()
        rospy.sleep(3.0) # 讓它走 3 秒測穩定度
        
        self.robot.walkStop()
        self.is_recording = False
        rospy.sleep(3.0) # 讓它站穩，準備下一回合
        
        # 4. 計算搖晃懲罰分數 (越低越好)
        if not self.current_gyro_x or not self.current_gyro_y:
            return 999.0 
            
        score = np.mean(self.current_gyro_x) + np.mean(self.current_gyro_y)
        rospy.loginfo(">> 此組參數搖晃分數: {:.4f}".format(score))
        return score

if __name__ == '__main__':
    tuner = WalkingTuner()
    
    # 初始化機器人姿勢
    tuner.robot.setGeneralControlModule("action_module")
    tuner.robot.playMotion(1, wait_for_end=True)
    rospy.sleep(1.0)
    
    best_score = float('inf')
    best_params = {'z': 0, 'y': 0}
    
    # 測試回合數
    NUM_TRIALS = 20 
    
    rospy.loginfo("啟動隨機演算法調參，請用雙手在兩側虛扶機器人！")
    rospy.sleep(3)
    
    for i in range(NUM_TRIALS):
        rospy.loginfo("--- 第 {}/{} 回合 ---".format(i+1, NUM_TRIALS))
        
        # AI 隨機生成合理範圍內的參數
        # Z: 抬腿高度 (限制在 0.035 ~ 0.045 之間)
        test_z = random.uniform(0.035, 0.045)
        # Y: 左右搖晃 (限制在 0.010 ~ 0.025 之間)
        test_y = random.uniform(0.010, 0.025)
        
        score = tuner.evaluate_params(test_z, test_y)
        
        # 紀錄歷史最佳
        if score < best_score:
            best_score = score
            best_params['z'] = test_z
            best_params['y'] = test_y
            rospy.loginfo("🌟 發現新最佳參數！(分數: {:.4f})".format(best_score))
            
    rospy.loginfo("========= 測試結束 =========")
    rospy.loginfo("🏆 最穩定起步參數:")
    rospy.loginfo("Z (z_move_amplitude): {:.4f}".format(best_params['z']))
    rospy.loginfo("Y (y_move_amplitude): {:.4f}".format(best_params['y']))
    rospy.loginfo("最低搖晃分數: {:.4f}".format(best_score))