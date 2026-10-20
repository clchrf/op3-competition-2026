#!/usr/bin/env python
import rospy
import time
import cv2
import math
import numpy as np
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
from op3_ros_utils import getWalkingParams, Robot
from copy import copy
from std_msgs.msg import Int32, String, Float32, Float64, Bool, Float32MultiArray


first=True
fixg=False
z_gyro = 0.0
z_list=[]
z_gyro_offset = 0.0
clock=0
loop_count=0
def read_IMU(data):#imu_get_yaw_by_integral 
    global first
    global fixg
    global old_z
    global z_gyro
    global z_list
    global z_gyro_offset
    global clock
    global loop_count
    global argsIMU
    alpha = 0.7
    
    rospy.sleep(0.001)
    loop_count+=1 
    if clock==0:
        clock=time.time()
        ltime=0.0079
    else:
        
        loop=time.time()-clock
        ltime=loop/loop_count
    
    

    #print(ltime)
    
    angular_velocity = data.angular_velocity
    z=angular_velocity.z

    if first:     
        z=angular_velocity.z
    else:
        z=(alpha*old_z)+((1-alpha)*angular_velocity.z)
    old_z=z
    
    #print(z)
    #z+=0.003
    z=math.floor(z*100)/100
    

    if fixg:
        error=z-z_gyro_offset
        dg=(error)*ltime
        dg=dg*180/np.pi
        z_gyro = z_gyro - dg
        #print(z_gyro)
    else:
        z_list+=[z]
        if len(z_list)>800:
            del z_list[0]
        avg=sum(z_list)/(len(z_list))
        avg=math.floor(avg*100)/100
        if abs(avg-z_list[-1])<0.001 and len(z_list)>=500:
            print("ok")
            z_gyro_offset=sum(z_list)/(len(z_list))
            z_gyro_offset=math.floor(z_gyro_offset*100)/100
            fixg=True
    





rospy.init_node("action_test")

robot=Robot()
cap=cv2.VideoCapture(0)


robot.setGeneralControlModule("walking_module")
rospy.sleep(1)

robot.setGeneralControlModule("action_module")
robot.setJointsControlModule(["head_pan","head_tilt"],["none","none"])
robot.setJointsControlModule(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],["none","none","none","none"])
robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[0,0.78,-0.78,0])
rospy.sleep(2)

fixg=False
z_gyro = 0.0
z_list=[]
print("init gyro")
rospy.Subscriber("/robotis/open_cr/imu", Imu, read_IMU, queue_size=1)
rospy.sleep(1)
print("imu ok")

robot.setJointPos(["head_pan","head_tilt"],[0,-0.5])
rospy.sleep(1)


"""vfov 73.74 hfov 90"""
situation="wait"
gyro_offset=0
add=0.01
pan_angle=0
tilt_angle=-0.5
state="find"
mode="not move"
move_mode="h"
theta_list=[]
while not rospy.is_shutdown():

    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#
    if situation=="wait":
        print(z_gyro)
        if robot.buttonCheck("user"):
            gyro_offset=z_gyro
            # situation="tt"   #find_ball
            situation="find_ball"

    if situation=="find_ball":
        
        # upper1=np.array([180,255,255])
        # lower1=np.array([70,165,230])

        # upper2=np.array([170,110,165])
        # lower2=np.array([15,65,130])


        # upper3=np.array([40,200,255])
        # lower3=np.array([5,50,110])

        # lower1=np.array([15,130,139])
        # upper1=np.array([50,255,255])

        upper1=np.array([32,255,255])
        lower1=np.array([16,64,213])

        upper2=np.array([0,0,0])
        lower2=np.array([0,0,0])

        upper3=np.array([0,0,0])
        lower3=np.array([0,0,0])

        mask1=cv2.inRange(frame,lower1,upper1)
        mask2=cv2.inRange(frame,lower2,upper2)
        mask3=cv2.inRange(frame,lower3,upper3)
        mask=cv2.bitwise_or(mask1,mask2)
        mask=cv2.bitwise_or(mask,mask3)

        kn = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        mask = cv2.erode(mask,kn,iterations=2)
        kn = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        mask = cv2.dilate(mask,kn,iterations=5)
        cv2.imshow("img2",mask)
        """find contours"""
        mask,contours,_ =cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x=0
        y=0
        if len(contours)!=0:
            state="nofind" 
            for contour in contours:
                area = cv2.contourArea(contour)
                if area>500:
                    state="find"
                    x,y,w,h=cv2.boundingRect(contour)
                    x=int(x+w/2)
                    y=int(y+h/2)
                    
                    if mode=="stop":
                        rospy.sleep(0.01)

                    frame=cv2.circle(frame,(x,y),5,(255,0,0),-1)

        else:
            state="nofind"
            

        """cal angle"""
        if state=="find":
            mode="move"
            
            if tilt_angle>=-0.5:
                tilt_angle=-0.5

            if tilt_angle<-1.2:  #arriver
                mode="stop"
               
 
                
                # --------------------------------------------------------------------------------------------------
                # --------------------------------------------------------------------------------------------------
                # --------------------------------------------------------------------------------------------------
            if  x!=320:  # x>=330 or x<=310
                fix=(320-x)*45/320
                fix=float(fix)/180*np.pi
            else:
                fix=0
        
            pan_angle=float(pan_angle)+(fix*0.1)
            
            if y!=120:
                fix=(120-y)*36.87/240
                fix=float(fix)/180*np.pi
            else:
                fix=0
            tilt_angle=float(tilt_angle)+(fix*0.1)
        else:
            mode="not move"

            if abs(pan_angle)>1:
                add=add*-1
            pan_angle+=add
            
            

        """move"""      
        if mode=="move":
            if len(theta_list)>=50:
                del theta_list[0]
            theta_list+=[pan_angle*20]
            
            robot.setGeneralControlModule("walking_module")

            robot.walkVelocities(x=-0.1,y=0, th=np.clip(sum(theta_list)/len(theta_list),-5,5),hip_pitch=10,balance=True)

            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------
            robot.setJointPos(["head_pan","head_tilt"],[pan_angle+0.32,tilt_angle]) #0.36 34
            robot.walkStart()
            
        elif mode=="stop":
            robot.walkStop()
            robot.setJointPos(["head_pan"],[0])
            rospy.sleep(2)
            ckk=time.time()        
            situation="watch_ball"
            
            #robot.setJointPos(["head_pan","head_tilt"],[0,tilt_angle])
        else:
            robot.walkStop()

            robot.setJointPos(["head_pan","head_tilt"],[pan_angle,tilt_angle])

        

    if situation=="watch_ball":
        
        # upper1=np.array([180,255,255])
        # lower1=np.array([70,165,230])

        # upper2=np.array([170,110,165])
        # lower2=np.array([15,65,130])


        # upper3=np.array([40,200,255])
        # lower3=np.array([5,50,110])

        # lower1=np.array([15,130,139])
        # upper1=np.array([50,255,255])

        upper1=np.array([29,255,255])
        lower1=np.array([6,105,138])

        upper2=np.array([0,0,0])
        lower2=np.array([0,0,0])


        upper3=np.array([0,0,0])
        lower3=np.array([0,0,0])

        mask1=cv2.inRange(frame,lower1,upper1)
        mask2=cv2.inRange(frame,lower2,upper2)
        mask3=cv2.inRange(frame,lower3,upper3)
        mask=cv2.bitwise_or(mask1,mask2)
        mask=cv2.bitwise_or(mask,mask3)

        kn = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        mask = cv2.erode(mask,kn,iterations=2)
        kn = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        mask = cv2.dilate(mask,kn,iterations=5)

        mask,contours,_ =cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x=0
        y=0

        if len(contours)>0:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area>200:
                    state="find"
                    x,y,w,h=cv2.boundingRect(contour)
                    x=int(x+w/2)
                    y=int(y+h/2)
                    frame=cv2.circle(frame,(x,y),5,(255,0,0),-1)
        else:
            state="no_find"

        if state=="find":
            if y!=200:
                fix=(200-y)*36.87/240
                fix=float(fix)/180*np.pi
            else:
                fix=0
            tilt_angle=float(tilt_angle)+(fix*0.1)
            robot.setJointPos(["head_pan","head_tilt"],[0,tilt_angle])
        if (time.time()-ckk)>3:
            situation="pick_ball"


    if situation=="pick_ball":
        err=tilt_angle+1
        sh=-1.25-err
        err=tilt_angle-(-1.15)
        fxa=-1.2-(err/0.09*0.1)
        ab=(3.14+2*fxa)

        rospy.sleep(2)
        robot.setJointsControlModule(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],["none","none","none","none"])
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[0,0.78,-0.58,0])
        rospy.sleep(1)
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[sh,0.78,-0.58,-1]) #l_gripper = -1
        rospy.sleep(1)
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[sh,-fxa-0.1,-ab,-1]) #l_gripper = -1
        rospy.sleep(1)
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[sh,-fxa-0.1,-ab,0.1])
        rospy.sleep(1)
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[sh,0.78,-0.78,0.1])
        rospy.sleep(1)
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[0,0.78,-0.78,0.1])
        rospy.sleep(1)
        robot.setJointPos(["head_pan","head_tilt"],[0,0])
        rospy.sleep(1)

        situation="turn"
        state="find"
        mode="not move"
        tilt_angle=0
        pan_angle=0
        theta_list=[]
    
    if situation=="turn":
        robot.setGeneralControlModule("walking_module")
        thr=(0-(z_gyro-gyro_offset))*-0.3
        robot.walkVelocities(x=0,y=0, th=thr ,hip_pitch=10)
        robot.walkStart()
        print("abs(0-(z_gyro-gyro_offset)): ",abs(0-(z_gyro-gyro_offset)))
        if abs(0-(z_gyro-gyro_offset))<10:
            robot.walkStop()
            rospy.sleep(2)
            situation="find_board"

    if situation=="find_board":

        upper1=np.array([10,255,255])
        lower1=np.array([0,173,70])

        upper2=np.array([179,255,255])
        lower2=np.array([165,173,70])

        upper3=np.array([0,0,0])
        lower3=np.array([0,0,0])

	

        # upper1=np.array([90,70,210])
        # lower1=np.array([0,0,110])
        # upper1=np.array([35,50,200])
        # lower1=np.array([0,0,100])

        # upper2=np.array([35,50,200])
        # lower2=np.array([0,0,100])

        # upper3=np.array([35,50,200])
        # lower3=np.array([0,0,100])

        # upper1=np.array([35,255,255])
        # lower1=np.array([0,150,150])

        # upper2=np.array([0,0,0])
        # lower2=np.array([0,0,100])

        # upper3=np.array([0,0,0])
        # lower3=np.array([0,0,100])
        mask1=cv2.inRange(frame,lower1,upper1)
        mask2=cv2.inRange(frame,lower2,upper2)
        mask3=cv2.inRange(frame,lower3,upper3)
        mask=cv2.bitwise_or(mask1,mask2)
        mask=cv2.bitwise_or(mask,mask3)

        kn = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        mask = cv2.erode(mask,kn,iterations=2)
        kn = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        mask = cv2.dilate(mask,kn,iterations=5)


        """find contours"""
        mask,contours,_ =cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x=0
        y=0
        if len(contours)!=0:
            state="nofind"
            for contour in contours:
                area = cv2.contourArea(contour)
                if area>1400:
                    state="find"
                    x,y,w,h=cv2.boundingRect(contour)
                    x=int(x+w/2)
                    y=int(y+h/2)
                    frame=cv2.circle(frame,(x,y),5,(255,0,0),-1)

        else:
            state="nofind"
            

        """cal angle"""
        if state=="find":
            mode="move"
            if tilt_angle<-0.75:  #arriver 0.75
                mode="not move"
                situation="throw"
                

            if  x!=300:  # x>=330 or x<=310
                fix=(250-x)*45/320
                fix=float(fix)/180*np.pi
            else:
                fix=0
        
            pan_angle=float(pan_angle)+(fix*0.1)
            
            if y!=200:
                fix=(200-y)*36.87/240
                fix=float(fix)/180*np.pi
            else:
                fix=0
            tilt_angle=float(tilt_angle)+(fix*0.1)
        else:
            mode="not move"

            if abs(pan_angle)>1.8:
                add=add*-1
            pan_angle+=add
            
            

        """move"""      
        if mode=="move":
            if len(theta_list)>=5:
                del theta_list[0]
            theta_list+=[pan_angle*12]

            robot.setGeneralControlModule("walking_module")
            mv=sum(theta_list)/len(theta_list)
            mv=np.clip(mv,-3,3)
            thr=(0-(z_gyro-gyro_offset))*-0.6#-0.3
            print("thr, mv", thr, mv, situation)
            # robot.walkVelocities(x=0.3,y=(mv*1.5), th=np.clip((-60)*pan_angle,-3,3) ,hip_pitch=10,balance=True)
            robot.walkVelocities(x=0.3,y=(mv*2.5), th=np.clip(thr,-3,3) ,hip_pitch=10,balance=True)#mv*1.5   clip(thr,-3, 3)
            print("pam_angle: ",pan_angle)
            robot.walkStart()
        else:
            robot.walkStop()

        robot.setJointPos(["head_pan","head_tilt"],[pan_angle,tilt_angle])

    if situation=="tt":
        robot.setGeneralControlModule("walking_module")
        thr=(0-(z_gyro-gyro_offset))*-0.3
        robot.walkVelocities(x=10,y=0, th=0 ,hip_pitch=22,balance=True)
        robot.walkStart()


    if situation=="throw":
        rospy.sleep(2)
        robot.setJointsControlModule(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],["none","none","none","none"])
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[0,0.78,-0.58,0])
        rospy.sleep(1)
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[0,0.78,-0.58,0])
        rospy.sleep(1)
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[-1.57,0,-0.58,0])
        rospy.sleep(1)
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[-2.5,0,-0.58,0])
        rospy.sleep(1)
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[-4.5,1.3,-0.9,0]) #l_el = -0.55 #l_sho_pitch = -2.5
        rospy.sleep(2)
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[-4.5,1.3,-0.9,-0.8]) #l_gripper = -1 #l_el = -0.55
        rospy.sleep(1)
        situation="stop"

    #cv2.imshow("img",frame)
    #cv2.imshow("img2",mask)


    cv2.waitKey(1)
    if robot.buttonCheck("user") and situation!="wait":
        robot.walkStop()
        break


	