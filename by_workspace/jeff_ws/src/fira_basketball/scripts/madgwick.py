import numpy as np

class MadgwickAHRS:
    def __init__(self, sample_period=1/256, beta=0.1):
        self.sample_period = sample_period
        self.beta = beta
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0]) 

    def update_imu(self, gyroscope, accelerometer):
        q1, q2, q3, q4 = self.quaternion
        gx, gy, gz = gyroscope
        ax, ay, az = accelerometer

        if np.linalg.norm(accelerometer) == 0:
            return

        
        a_norm = np.linalg.norm(accelerometer)
        ax, ay, az = ax / a_norm, ay / a_norm, az / a_norm

        
        f = np.array([
            2*(q2*q4 - q1*q3) - ax,
            2*(q1*q2 + q3*q4) - ay,
            2*(0.5 - q2**2 - q3**2) - az
        ])
        j = np.array([
            [-2*q3,  2*q4, -2*q1,  2*q2],
            [ 2*q2,  2*q1,  2*q4,  2*q3],
            [ 0,    -4*q2, -4*q3,  0]
        ])
        step = np.dot(j.T, f)
        step /= np.linalg.norm(step)

       
        g = np.array([gx, gy, gz]) * (np.pi / 180.0)
        q_dot = 0.5 * np.array([
            -q2*g[0] - q3*g[1] - q4*g[2],
             q1*g[0] + q3*g[2] - q4*g[1],
             q1*g[1] - q2*g[2] + q4*g[0],
             q1*g[2] + q2*g[1] - q3*g[0]
        ])
        q_dot -= self.beta * step

       
        self.quaternion += q_dot * self.sample_period
        self.quaternion /= np.linalg.norm(self.quaternion)

    def get_euler_angles(self):
        q1, q2, q3, q4 = self.quaternion
        roll = np.arctan2(2*(q1*q2 + q3*q4), 1 - 2*(q2**2 + q3**2))
        pitch = np.arcsin(2*(q1*q3 - q4*q2))
        yaw = np.arctan2(2*(q1*q4 + q2*q3), 1 - 2*(q3**2 + q4**2))
        return np.degrees([roll, pitch, yaw])
