import numpy as np

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) 
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T)) 
        K = np.dot(np.dot(self.F, (np.dot(self.P, self.H.T))), np.linalg.inv(S)) #Kalman Gain
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def example():
	dt = 1.0/60
	F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]) #Dynamics Matrix
	B = np.array([[(dt*dt)/2, 0], [0, (dt*dt)/2], [dt, 0], [0, dt]]) #Control Matrix
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) #I_4x4
	Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]]) #Process noise
	R = np.array([0.5]).reshape(1, 1) #Observation Noise
    
    # Make these changes as you get depth map
    # Qk = Qd[1−γQ/2*tanh(sd) + 1+γQ/2]
    # R = Rd[1−γR/2*tanh(sd) + 1+γR/2]

    # Add x0 = Initial Positions of the Objects 
	kf = KalmanFilter(F = F, B = B, H = H, Q = Q, R = R, x0)
	predictions = []

if __name__ == '__main__':
    example()