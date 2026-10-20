#! /usr/bin/python
import rospy
from op3_ros_utils import getWalkingParams, Robot
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# for archery
import cv2
import numpy as np
import math
import time
from sklearn.cluster import KMeans    

# init rosnode
rospy.init_node("fira_Archery_PF")
# rospy.init_node("fira_Archery_PF", anonymous=True)


class States:
    INIT = -1
    READY = 0
    CLAC_ROTATION = 1
    SHOOT_STATE = 2
    END = 99

'''-------------------------ROBOT Motions-----------------------------'''

'''-------------------------PF Functions-----------------------------'''
def gaussian1D(m, s):
    r = np.random.normal(m, s, 1)  # s: standard deviation
    return r

def gaussian2D(m, s):
    r1 = np.random.normal(m, s, 1)  # s: standard deviation
    r2 = np.random.normal(m, s, 1)  # s: standard deviation
    return np.array([r1[0], r2[0]])

#when the detection is wrong:
def reinitiate_particles(particles, numParticles, start_x, start_y, focus_W, focus_H):
    particles = np.array([np.random.randint(start_x, start_x + focus_W - 1, numParticles),
                      np.random.randint(start_y, start_y + focus_H - 1, numParticles),
                      3 * np.random.randn(numParticles) + 3,
                      3 * np.random.randn(numParticles)]).T
    return particles

def update_particles(particles, numParticles, weights, colorSigma, targetColor):
    # calculate likelihood of the particles by pixel color and update the particle's weight
    for i in range(numParticles):
        # Ensure particle position is within valid range
        # px = int(max(0, min(particles[i, 0], imgW - 1)))      #for raw frame size
        # py = int(max(0, min(particles[i, 1], imgH - 1)))      #for raw frame sizeq

        px = int(max(start_x, min(particles[i, 0], start_x + focus_W - 1)))        #for focus area size
        py = int(max(start_y, min(particles[i, 1], start_y + focus_H - 1)))        #for focus area size

        # Calculate color likelihood
        color = frame[py, px]
        colorDiff = np.sum((color - targetColor) ** 2)
        weights[i] = np.exp(-colorDiff / (2 * colorSigma ** 2))

    # Resampling: the particle with higher weight will be duplicated more times
    # Normalize weights
    weights /= np.sum(weights)

    # Create temporary particle array
    particleTemp = np.zeros_like(particles)

    # Resampling loop
    for k in range(numParticles):
        particleIndex = np.random.choice(np.arange(numParticles), p=weights)
        particleTemp[k] = particles[particleIndex]

    # Copy particleTemp back to particles
    particles = np.copy(particleTemp)
    return particles, weights

def predict_particles(particles, predMat, posNoise, velNoise):
    for i in range(numParticles_R):
        particles[i, :2] = np.dot(predMat[:2, :2], particles[i, :2]) + gaussian2D(0, posNoise)  #org, pred new_p pos(x, y)
        particles[i, 2:] = np.dot(predMat[2:, 2:], particles[i, 2:]) + gaussian2D(0, velNoise)  #org, ped new_p vel(vx, vy)
    return particles

#draw and show particles:
def draw_particles(numParticles, particles, frame, color):
    for i in range(numParticles):
        x = int(particles[i, 0])
        y = int(particles[i, 1])
        cv2.circle(frame, (x, y), 1, color, -1)

'''-------------------------Add for Archery Functions-----------------------------'''

#1.retun target center. 2."plot bounding box from the range of particles!"
def calc_Target_center(particles, targetColor):
    #1. calc (mean_px, mean_py)-------------------------------
    mean_px = 0
    mean_py = 0

    #(1-1) calc from 4-ending points of particles to get center
    max_px, max_py, min_px, min_py = 0, 0, 0, 0
    
    max_px = int(np.max(particles[:, 0]))
    max_py = int(np.max(particles[:, 1]))
    min_px = int(np.min(particles[:, 0]))
    min_py = int(np.min(particles[:, 1]))

    #for checking the 4-ending points:
    if np.array_equal(targetColor, targetColor_R):
        #for tracking red, regtangle color = white
        cv2.rectangle(frame, (int(min_px), int(min_py)), (int(max_px), int(max_py)), color = (255, 255, 255), thickness = 2)
    elif np.array_equal(targetColor, targetColor_B):
        #for tracking blue, regtangle color = black
        cv2.rectangle(frame, (int(min_px), int(min_py)), (int(max_px), int(max_py)), color = (0, 0, 0), thickness = 2)
    else:
        print('targetColor Error !!! target Color = ', targetColor)
        return [None, None]

    #get mean coord:
    mean_px = int((max_px + min_px)/2)
    mean_py = int((max_py + min_py)/2)
    
    '''
    #get mean coord:
    #if the particles spread to much, it's not tracking correctly!
    if max_px - min_px > 200 or max_py - min_py >200:
        mean_px, mean_py = None, None
    else:
        #(2-1)4-end points to calc target's center:   --shake less.
        mean_px = int((max_px + min_px)/2)
        mean_py = int((max_py + min_py)/2)  

        #(2-2)calc mean from all particles as target's center:     --shake more. 
        # mean_px = int(np.mean(particles[:, 0]))
        # mean_py = int(np.mean(particles[:, 1]))
        # mean_px = np.mean(particles[:, 0]).astype(np.int)
        # mean_py = np.mean(particles[:, 1]).astype(np.int)
    '''

    # print("mean(px, py) = ", "(", mean_px, ",", mean_py, ")")

    #2.return Target center info:---------------------------------
    return [mean_px, mean_py]

#only draw the center of target:
def draw_Target_center(coord, frame):
    x, y =  coord[0], coord[1]
    #3.plot X for center:------------------------------------
    if (x != None and y != None):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        # cv2.putText(frame, 'X', (mean_px, mean_py), font, font_scale, color, thickness)   #text will be on top-right corner
        cv2.circle(frame, (x, y), radius=8, color=(0,255,255), thickness=-1)    #fill = -1
        cv2.circle(frame, (x, y), radius=8, color=(0,0,0), thickness=2)         #contur = 2

        # cv2.imshow('frame', frame)        #display in main!
        # cv2.waitKey(1)

#only use for check:
def draw_Rotation_trace(target_rotate_trace):
    for target_center in target_rotate_trace:
        cv2.circle(frame, (target_center[0], target_center[1]), 2, color = (150, 200, 180), thickness=-1)

'''
#if there are too many None in list, re-record the data.
def check_num_of_None_data(tareget_rotate_trace, max_limit, start_time):
    none_count = tareget_rotate_trace.count(None)
    if none_count > max_limit:
        tareget_rotate_trace = []
        start_time = time.time() # Reset the timer to 0
        return tareget_rotate_trace, start_time
    else:
        return tareget_rotate_trace, start_time
'''

#use KMeans to calc: center, r
def calc_Rotational_center(target_rotate_trace):
    # Fit a circle to the data points using K-means clustering
    kmeans = KMeans(n_clusters=1, n_init=10).fit(target_rotate_trace)

    # Get the center and radius of the fitted circle
    cx, cy = kmeans.cluster_centers_[0]
    cx, cy = int(cx), int(cy)
    trace_nparray = np.array(target_rotate_trace)
    r = np.mean(np.linalg.norm(trace_nparray - [cx, cy], axis=1))
    r = int(r)
    return [cx, cy], r

def draw_Rotation_circle(center, r, circle_color):
    #rotational center:
    cv2.circle(frame, (center[0], center[1]), 5, color = circle_color, thickness=-1)
    #rotational circle:
    cv2.circle(frame, (center[0], center[1]), r, color = circle_color, thickness=2)

###----------for predict target movement--------------###

#rotational_factor: CW = 1. CCW = -1.
def calc_mean_w(target_rotate_trace, rotate_c, rotational_factor, r):
    
    vt_list = []        #for debug
    tc_rad_list = []     #record all tc in 0-2pi (0-360)
    diff_rad_list = []      #record all diff rad hwile rotating

    # print('-'*10)
    # print('rc = ', rotate_c)
    for target_c in target_rotate_trace:
        vt = [target_c[0] - rotate_c[0], target_c[1] - rotate_c[1]]
        # Calculate the angle in radians
        vt_rad = - (np.arctan2(vt[1], vt[0]))       # '-'(Negative) for making +/- pi (180 degree)
        if vt_rad < 0:                              # change to 0-2*pi (0-360 degree)
            vt_rad = np.pi + (np.pi + vt_rad)

        vt_list.append(vt)
        tc_rad_list.append(vt_rad)

    # print('vt_list = ', '\n', vt_list)

    vt_1_data = np.array(tc_rad_list[0:-1])
    vt_data = np.array(tc_rad_list[1:])
    diff_rad_list = np.around(vt_data - vt_1_data, 4)          #record every diff degree. used to calc mean_w

    #filtered those outlier data (while pass 0-degree, data will change a lot but not real.)
    filt_diff_list = list(filter(lambda x: np.abs(x) < 1, diff_rad_list))    #if diff_rad > 1(rad): drop data.

    # print('diff_rad = ', '\n', filt_diff_list)
    mean_w = np.mean(filt_diff_list)
    # print('mean_w = ', mean_w)
    mean_w *= rotational_factor
    mean_w = np.around(mean_w, 3)
    return mean_w

#calc the diff-angle between aim-coord & target-center
def calc_aim_tc_angle(rotate_c, target_c, aim_coord):
    vt = [target_c[0] - rotate_c[0], target_c[1] - rotate_c[1]]
    va = [aim_coord[0] - rotate_c[0], aim_coord[1] - rotate_c[1]]

    # print('-'*10)       #debug
    # print('tc = ', target_c, '\t', 'r_c = ', rotate_c)
    # print("vt = ", vt, '\t', 'va = ', va)               

    # Calculate the angle in radians
    vt_rad = - (np.arctan2(vt[1], vt[0]))       # '-'(Negative) for making +/- pi (180 degree)
    if vt_rad < 0:                              # change to 0-2*pi (0-360 degree)
        vt_rad = np.pi + (np.pi + vt_rad)
    va_rad = - (np.arctan2(va[1], va[0]))       # '-'(Negative) for making +/- pi (180 degree)
    if va_rad < 0:                              # change to 0-2*pi (0-360 degree)
        va_rad = np.pi + (np.pi + va_rad)

    # Convert the angle to degrees
    vt_A = np.around(math.degrees(vt_rad),3)
    va_A = np.around(math.degrees(va_rad),3)

    #debug
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    cv2.putText(frame, 'vt_ang = {0}'.format(vt_A), (start_x, start_y), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(frame, 'va_ang = {0}'.format(va_A), (start_x, start_y - start_y//5), font, font_scale, (255, 255, 255), thickness)

    diff_angle_rad = vt_rad - va_rad
    if diff_angle_rad < 0:
        diff_angle_rad += 2*np.pi       # change to 0-2*pi (0-360 degree)
    
    return diff_angle_rad

def shooting_decision(diff_angle, mean_w, TTG):      #TTG-unit: frame
    #if target rotate in CW, mean_w *(-1). CCW no need to *(-1)
    mean_w *= -1
    #dicide to shoot or not:
    if diff_angle <= mean_w * TTG:
        return True
    else:
        return False

#draw: aim_coord, mean_w, diff_angle_rad in cam
def draw_shooting_para(diff_angle, mean_w, rotate_c, target_c, aim_coord, r, TTG, shooting_decision):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    diff_angle = np.around(np.degrees(diff_angle), 3)
    mean_w = np.around(np.degrees(mean_w), 3)

    # cv2.putText(frame, 'aim', (aim_coord[0] - 10, aim_coord[1] + 20), font, font_scale, (255, 255, 255), thickness)
    cv2.circle(frame, tuple(aim_coord), 5, color = (255, 255, 255), thickness=2)
    cv2.putText(frame, 'diff_ang={0}'.format(diff_angle), (start_x + focus_W*4//7, start_y), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(frame, 'mean_w = {0}'.format(mean_w), (frame_w//2 , start_y + focus_H*11//10), font, font_scale, (255, 255, 255), 2)
    cv2.putText(frame, 'TTG = {0}'.format(TTG), (frame_w//2 - focus_W//2, start_y + focus_H*11//10), font, font_scale, (255, 255, 255), 2)
    
    #Don't Shoot! --> white
    if shoot_decision != True:
        cv2.putText(frame, 'Shoot = {0}'.format(shooting_decision), (frame_w//2, 50), font, font_scale, (255, 255, 255), 2)
    #Shoot! --> Green
    else:
        cv2.putText(frame, 'Shoot = {0}'.format(shooting_decision), (frame_w//2, start_y + focus_H*12//10), font, font_scale, (0, 255, 0), 2)
    
    # for chekcing the angle calc
    cv2.line(frame, tuple(rotate_c), tuple(target_c), color =(200, 200, 0), thickness=2)
    cv2.line(frame, tuple(rotate_c), tuple(aim_coord), color = (200, 200, 0) , thickness=2)
    
    # for drawing good shooting angle arc
    arc_r = r + 3
    va = [aim_coord[0] - rotate_c[0], aim_coord[1] - rotate_c[1]]
    va_rad = - (np.arctan2(va[1], va[0]))       # '-'(Negative) for making +/- pi (180 degree)
    if va_rad < 0:                              # change to 0-2*pi (0-360 degree)
        va_rad = np.pi + (np.pi + va_rad)
    mean_w *= -1    #if rotate in CW. Otherwise no need.
    start_angle_rad = va_rad + mean_w * TTG
    end_angle_rad = va_rad

    # cv2.ellipse(image, center, (radius, radius), 0, start_angle, end_angle, color, thickness)     #example
    # cv2.ellipse(frame, rotate_c, (arc_r, arc_r), 0, start_angle_rad, end_angle_rad, (0,255,0), 3)

    # for cv2 ver 3.3.1:
    # start_angle_rad = int(np.degrees(start_angle_rad))
    # end_angle_rad = int(np.degrees(end_angle_rad))

    # cv2.ellipse(frame, rotate_c, (arc_r, arc_r), 0, int(np.degrees(start_angle_rad)), int(np.degrees(end_angle_rad)),color = (0, 255, 0), thickness = 3)
    # cv2.ellipse(frame, rotate_c, (arc_r, arc_r), 0, start_angle_rad, end_angle_rad, (0,255,0), 3)  
        

###----------for predict target movement--------------###


'''---------------------------------------PARAS---------------------------------------'''
#Drop frame:
drop_frame_threashold = 4               #drop evry Nrd frame. [Good delay: numParticles * d_f_t ~= 1500] 
#need to check the cam first!:
imgH = 480      #It's to make sure  there no overflow pixel info in particles. the H & W should be org-frame size.
imgW = 640      
'''
imgH= 848:v1~3, 960:v4~6, 640:v7~8
imgW= 464:v1~3, 540:v4~6, 360:v7~8
'''

#only focus the rotate motion in a smaller area:
#also only spread particles in this region
#if no need focus area, just set stat_x/y = 0 & focus_H/W = imgH/W.
start_x = 50        #50       #focus area = [start_x : start_x + focus_W, start_y : start_y + focus_H] 
start_y = 50        #50
focus_H = 380       #380
focus_W = 540      #540

targetColor_R = np.array([0, 0, 255])  # red
targetColor_B = np.array([255, 0, 0])  # blue

colorSigma_R = 15  # set sigma of color likelihood function, org:50. Slower more accurate to select target color.
posNoise_R = 10  # set position uncertainty after prediction, org:15
velNoise_R = 10  # set velocity uncertainty after prediction, org:5

colorSigma_B = 15  # set sigma of color likelihood function, org:50. Slower more accurate to select target color.
posNoise_B = 10  # set position uncertainty after prediction, org:15
velNoise_B = 10  # set velocity uncertainty after prediction, org:5

numParticles_R = 200      #50?          
numParticles_B = 200     #50?
RB_dist_threshold = 150         # for check if the tracking is wrong or not. 

# particle state vectors and initialization
# for raw frame size:
# particles = np.array([np.random.randint(0, imgW - 1, numParticles),
#                       np.random.randint(0, imgH - 1, numParticles),
#                       3 * np.random.randn(numParticles) + 3,
#                       3 * np.random.randn(numParticles)]).T

#for focus area size:(init particles)
particles_R = np.array([np.random.randint(start_x, start_x + focus_W - 1, numParticles_R),
                      np.random.randint(start_y, start_y + focus_H - 1, numParticles_R),
                      3 * np.random.randn(numParticles_R) + 3,
                      3 * np.random.randn(numParticles_R)]).T

particles_B = np.array([np.random.randint(start_x, start_x + focus_W - 1, numParticles_B),
                      np.random.randint(start_y, start_y + focus_H - 1, numParticles_B),
                      3 * np.random.randn(numParticles_B) + 3,
                      3 * np.random.randn(numParticles_B)]).T

# weights and initialization
weights_R = np.zeros(numParticles_R) + 1.0 / numParticles_R
weights_B = np.zeros(numParticles_B) + 1.0 / numParticles_B
# prediction matrix (constant velocity model)
predMat = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])


### ------------- Archery paras ------------------- ###
tc_rotate_trace = []        # record target center rotating trace
filtered_trace = []         # filter None data from tc_rotate_trace = [] 
target_rotate_factor = 1    # dicide by manual!! (CW = 1, CCW = -1)
TTG =  72  #ORG:53                     # time to target (unit: frame)
rotate_center_coord, rotate_r = None, None      #init as None, only calc once!
aim_coord = [320, 848 - 150]       # set the aimming-coordinate but is not real aiming coordinate.
# aim_coord = [260, 323]       # real shooting corrd but no use.by testing.
reinitiate_count = 0
start_time_checker = True          #to make sure start_time only set in once. This value will set to false after start recording. 
start_time = None
debug_frame = 1    # 0 is debug mode, 1 is normal
init_pos_count = 0


# Create robot
robot = Robot()
#rospy.init_node("fira_Archery_PF")     #change to line 13
rospy.sleep(3) # Make sure every publisher has registered to their topic,
               # avoiding lost messages
currState = States.INIT

'''     ###Edit to create RosNode###
# Subscribe to cv_camera topic with vision system
rospy.Subscriber("/cv_camera/image_raw", Image, vision.read, queue_size=1)
# rospy.Subscriber("/cv_camera/image_raw", Image, iimg, queue_size=1)
# Iinitialize Node
rospy.init_node("fira_sprint")
'''

# video_path = 'Target_video/Target_04.mp4'       #4,
# cap = cv2.VideoCapture(video_path)              #for test with video
cap = cv2.VideoCapture(0)                         #for camera
frameCount = 0

'''----------------------------------------Main-----------------------------------------------'''
if __name__ == '__main__':
    while not rospy.is_shutdown():

        # Capture frame-by-frame
        ret, frame = cap.read()
        # rawframe = np.copy(frame)
        # frame = frame[start_x:start_x + focus_W, start_y:start_y + focus_H]     #for raw frame size
        # frame = cv2.flip(frame, -1)     #only for videos: v7 & v8
        if ret:
            frameCount = frameCount + 1
            
            if frameCount < 60 :  # start the particle filter from the 30th(org) frames, simplify the process
                continue

            if frameCount % (drop_frame_threashold) == 0 :
                continue        #only process every (d_f_t +1) frame
            
            #for focus area size. The Margine of where can spread particles.
            cv2.rectangle(frame, (start_x, start_y), (start_x + focus_W, start_y + focus_H ), (0, 0, 0), 2)

        if currState == States.INIT:
            print("State = [INIT]", ' init_pos_count = {0}'.format(init_pos_count))
            
            
            #if (cv2.waitKey(1) & 0xFF == ord('q')) and init_pos_count == 0 :
            if init_pos_count == 0 and robot.buttonCheck("start") :
                robot.setGeneralControlModule("action_module")
                robot.playMotion(90, wait_for_end=True)     #wait pos
                init_pos_count += 1
                time.sleep(1)

            #if (cv2.waitKey(1) & 0xFF == ord('q')) and init_pos_count == 1:
            if init_pos_count == 1 and robot.buttonCheck("start") :
                robot.setGeneralControlModule("action_module")
                robot.playMotion(91, wait_for_end=True)     #grip bow
                init_pos_count += 1
                time.sleep(1)

            #if (cv2.waitKey(1) & 0xFF == ord('q')) and init_pos_count == 2:
            if init_pos_count == 2 and robot.buttonCheck("start") :
                robot.setGeneralControlModule("action_module")
                robot.playMotion(92, wait_for_end=True)     #grip string's handle
                init_pos_count += 1
                time.sleep(1)


            if init_pos_count >= 3:
                currState = States.READY

        elif currState == States.READY:
            print("State = [READY]")
            cv2.circle(frame, (imgW/2, imgH/2), radius=15, color=(0,0,0), thickness=2)
            cv2.circle(frame, (260,323), radius=15, color=(200,200,200), thickness=2)
            
            if robot.buttonCheck("start") :
                currState = States.CLAC_ROTATION

        elif currState == States.CLAC_ROTATION:
            if start_time == None:
                print("State = [CLAC_ROTOTION_INFO]")
            else:
                # struct_start_time = time.localtime(start_time)
                # start_time_string = time.strftime("%Y-%m-%d %H:%M:%S", struct_start_time)
                # print("State = [CLAC_ROTOTION_INFO]", 'start_T = {0}'.format(start_time_string))
                print("State = [CLAC_ROTOTION_INFO]")

            #record start time for select data in a duration, then use data to fit rotation circle: 
            if start_time_checker  == True:
                start_time = time.time()
                start_time_checker = False
            timer_ = 16              # n sec

            #Update particles info:
            particles_R, weights_R = update_particles(particles_R, numParticles_R, weights_R, colorSigma_R, targetColor_R)
            particles_B, weights_B = update_particles(particles_B, numParticles_B, weights_B, colorSigma_B, targetColor_B)

            '''-----------------------------------------------------------------------------------------'''
            # draw_particles(numParticles_R, particles_R, frame, (255, 255, 255))  # draw and show particles
            # draw_particles(numParticles_B, particles_B, frame, (255, 255, 255))  # draw and show particles

            #target center info:
            tc_x, tc_y = calc_Target_center(particles_R, targetColor_R)
            blue_cx, blue_cy =   calc_Target_center(particles_B, targetColor_B)

            coord_tc = np.array([tc_x, tc_y])
            coord_blue_c = np.array([blue_cx, blue_cy])

            #if the tracking is wrong:
            if np.linalg.norm(coord_tc - coord_blue_c) > RB_dist_threshold:
                coord_tc = [None, None]
                particles_R = reinitiate_particles(particles_R, numParticles_R, start_x, start_y, focus_W, focus_H)
                particles_B = reinitiate_particles(particles_B, numParticles_B, start_x, start_y, focus_W, focus_H)
                reinitiate_count += 1
                print('-'*5)
                print("Reinitiate  Particles!")
                # print("Reinitiate Red Particles!")
                # print("Reinitiate Blue Particles!")
                # rawframe = cv2.resize(rawframe, (405, 720))
                # cv2.imshow('frame', rawframe)
                # cv2.waitKey(1)
                continue

            draw_Target_center(coord_tc, frame)

            #check timer to continue/stop recording rotating-trace info:
            if time.time() - start_time < timer_:
                tc_rotate_trace.append(coord_tc)
                # print('Collecting Data......')
                # print('pass_T = {0}'.format(np.around(time.time() - start_time), 3))
            
            #reinitiate > 10, restart recording data.
            #comment this elif when only need to record 1 time.
            elif reinitiate_count > 10 :
                print("Reinitiate count = {0}".format(reinitiate_count))
                print('\n')
                print("Restart recording the trace...")
                start_time = time.time()
                start_time_checker = True
                reinitiate_count = 0
            
            #finish record -->
            #1. calc rotational-circle info.
            #2. calc mean-W from recorded list.
            #3. predict when to shoot by (aim_coord(t), tc(t), mean_w) & show infoW
            else:
                print('Finished Collecting Data!!')

                stop_time = time.time()
                struct_stop_time = time.localtime(stop_time)
                stop_time_string = time.strftime("%Y-%m-%d %H:%M:%S", struct_stop_time)

                struct_start_time = time.localtime(start_time)
                start_time_string = time.strftime("%Y-%m-%d %H:%M:%S", struct_start_time)

                print('start_T = {0}'.format(start_time_string), '----->', 'stop_T = {0}'.format(stop_time_string))
                print('pass_T = {0}'.format(np.around(time.time() - start_time), 3))

                reinitiate_count = 0        #force to not re-record the data!
                # Use filter() to remove None values
                filtered_trace = list(filter(lambda x: x[0] is not None and x[1] is not None, tc_rotate_trace))
                # draw_Rotation_trace(filtered_trace)
                #Change to Shoot State!
                currState = States.SHOOT_STATE

                #step 1.
                if rotate_center_coord == None and  rotate_r == None:
                    rotate_center_coord, rotate_r = calc_Rotational_center(filtered_trace)
                    aim_coord = [rotate_center_coord[0], rotate_center_coord[1] + rotate_r]     #New Aim_Coord !!
                else:
                    pass
                

                '''
                #debug:--------------------------
                debug_frame = 1                #set 0 to debug
                
                #debug:--------------------------
                '''
            
            #display:
            # frame = cv2.resize(frame, (405, 720))
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            # cv2.waitKey(debug_frame)       #only for debug
            '''---------------------------------------------------------------------------------------------'''

            # Perdict and Update particles:
            # predict new position of a particle by its velocity and old position (plus noise) (constant velocity model)
            particles_R = predict_particles(particles_R, predMat, posNoise_R, velNoise_R)
            particles_B = predict_particles(particles_B, predMat, posNoise_B, velNoise_B)

            # Press Q on keyboard to exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

        elif currState == States.SHOOT_STATE:
            print("State = [SHOOT_STATE]")


            #Update particles info:
            particles_R, weights_R = update_particles(particles_R, numParticles_R, weights_R, colorSigma_R, targetColor_R)
            particles_B, weights_B = update_particles(particles_B, numParticles_B, weights_B, colorSigma_B, targetColor_B)

            '''-----------------------------------------------------------------------------------------'''
            # draw_particles(numParticles_R, particles_R, frame, (255, 255, 255))  # draw and show particles
            # draw_particles(numParticles_B, particles_B, frame, (255, 255, 255))  # draw and show particles

            #target center info:
            tc_x, tc_y = calc_Target_center(particles_R, targetColor_R)
            blue_cx, blue_cy =   calc_Target_center(particles_B, targetColor_B)

            coord_tc = np.array([tc_x, tc_y])
            coord_blue_c = np.array([blue_cx, blue_cy])

            #if the tracking is wrong:
            if np.linalg.norm(coord_tc - coord_blue_c) > RB_dist_threshold:
                coord_tc = [None, None]
                particles_R = reinitiate_particles(particles_R, numParticles_R, start_x, start_y, focus_W, focus_H)
                particles_B = reinitiate_particles(particles_B, numParticles_B, start_x, start_y, focus_W, focus_H)
                
                print('-'*5)
                print("Reinitiate  Particles!")
                # print("Reinitiate Red Particles!")
                # print("Reinitiate Blue Particles!")
                # rawframe = cv2.resize(rawframe, (405, 720))
                # cv2.imshow('frame', rawframe)
                # cv2.waitKey(1)
                continue

            #calc from obs
            draw_Target_center(coord_tc, frame)

            draw_Rotation_circle(rotate_center_coord, rotate_r, (255, 255, 0))
            mean_w = calc_mean_w(filtered_trace, rotate_center_coord, target_rotate_factor, rotate_r)
            aim_coord = [rotate_center_coord[0], rotate_center_coord[1] + 80]         #only for testing.
            diff_angle = calc_aim_tc_angle(rotate_center_coord, coord_tc, aim_coord)
            shoot_decision = shooting_decision(diff_angle, mean_w, TTG)     #have error!!
            draw_shooting_para(diff_angle, mean_w, rotate_center_coord, coord_tc, aim_coord, rotate_r, TTG, shoot_decision)

            #debug:--------------------------
            debug_frame = 1                #set 0 to debug
            
            #debug:--------------------------

            #display:
            # frame = cv2.resize(frame, (405, 720))
            # cv2.imshow('frame', frame)
            # cv2.waitKey(1)
            # cv2.waitKey(debug_frame)       #only for debug
            '''---------------------------------------------------------------------------------------------'''

            # Perdict and Update particles:
            # predict new position of a particle by its velocity and old position (plus noise) (constant velocity model)
            particles_R = predict_particles(particles_R, predMat, posNoise_R, velNoise_R)
            particles_B = predict_particles(particles_B, predMat, posNoise_B, velNoise_B)


            #shoot motion:
            if shoot_decision :
                robot.setGeneralControlModule("action_module")
                robot.playMotion(93, wait_for_end=True)     #shooting motion
                currState == States.END

        elif currState == States.END:
            print("State = [END]")
    
        #show frame:
        #print(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    # finished:
    cap.release()
    cv2.destroyAllWindows()


