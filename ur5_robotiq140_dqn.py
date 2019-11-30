#!/usr/bin/env python3
import pybullet as p
import pybullet_data

import os
import time
import pdb
import utils_robotiq_140
from collections import deque
from collections import namedtuple
from itertools import count
from PIL import Image
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# define world in pybullet
serverMode = p.GUI # GUI/DIRECT
sisbotUrdfPath = "./urdf/ur5_robotiq_140.urdf"

# connect to engine servers
physicsClient = p.connect(serverMode)
# add search path for loadURDFs
p.setAdditionalSearchPath(pybullet_data.getDataPath())

def reset_env():

    p.resetSimulation()
    #p.setGravity(0,0,-10) # NOTE
    planeID = p.loadURDF("plane.urdf")

    blockStartPos = [0.1, -0.50, 0.87]
    blockStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    boxId = p.loadURDF("./urdf/objects/block.urdf", blockStartPos, blockStartOrientation)

    tableStartPos = [0.0, -0.9, 0.8]
    tableStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    boxId1 = p.loadURDF("./urdf/objects/table.urdf", tableStartPos, tableStartOrientation,useFixedBase = True)

    ur5standStartPos = [-0.7, -0.36, 0.0]
    ur5standStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    boxId2 = p.loadURDF("./urdf/objects/ur5_stand.urdf", ur5standStartPos, ur5standStartOrientation,useFixedBase = True)

    robotStartPos = [0,0,0.0]
    robotStartOrn = p.getQuaternionFromEuler([0,0,0])
    print("----------------------------------------")
    print("Loading robot from {}".format(sisbotUrdfPath))
    robotID = p.loadURDF(sisbotUrdfPath, robotStartPos, robotStartOrn,useFixedBase = True,
                         flags=p.URDF_USE_INERTIA_FROM_FILE)
    joints, controlRobotiqC2, controlJoints, mimicParentName = utils_robotiq_140.setup_sisbot(p, robotID)
    init_cubePos, init_cubeOrn = p.getBasePositionAndOrientation(boxId)


    return robotID,joints,controlRobotiqC2,controlJoints,mimicParentName,boxId,init_cubePos

width = 128
height = 128


eefID = 7 # ee_link


# setup pybullet
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        #print(" self.position",  self.position)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size)) // stride  + 1
        #print("w:", w)
        #print("h:", h)

        #print ("convw1 ", conv2d_size_out(w))
        #print ("convw2 ", conv2d_size_out(conv2d_size_out(w)))
        #print ("convw3 ",conv2d_size_out(conv2d_size_out(conv2d_size_out(w))))
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        #print("x1", x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        #print("x2", x.size())
        x = F.relu(self.bn3(self.conv3(x)))
        #print("x3", x.size())
        #print ("self.head(x.view(x.size(0), -1))", self.head(x.view(x.size(0), -1)))
        return self.head(x.view(x.size(0), -1))


def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb )

fov = 40
aspect = width / height
near = 0.2
far = 2
view_matrix = p.computeViewMatrix([0.132, -1.524, 1.205], [0.132, -0.539, 1.116], [0, 1, 0])
# Get depth values using the OpenGL renderer
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
def get_screen():
    images = p.getCameraImage(width,height,view_matrix,projection_matrix,shadow=True,lightDirection=[1,1,1],renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_image = rgba2rgb(images[2])
    rgb_opengl = np.reshape(rgb_image, (3, height, width,)) * 1. / 255.
    #print(rgb_opengl.shape[0])
    #print(rgb_opengl.shape[1])
    #print(rgb_opengl.shape[2])
    #print(rgb_opengl.shape[0])
    #print(rgb_opengl.shape[1])
    screen = torch.from_numpy(rgb_opengl)
    return screen.unsqueeze(0).to(device).float()

#plt.figure()
#plt.imshow(get_screen(width,height),interpolation='none')
#plt.title('Example extracted screen')
#plt.show()
#plt.pause(0.0001)
#time.sleep(1000)


# dqn parameter
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20000
TARGET_UPDATE = 10

n_actions = 4 #z_up, z_down, open, close

policy_net = DQN(height, width, n_actions).to(device)
target_net = DQN(height, width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(50000)

steps_done = 0

# 0:z_up, 1:z_down, 2:close 3:open
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    #print("eps_threshold", eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []
total_reward = []


def plot_durations(reward):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    total_reward = torch.tensor(reward, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration & total_reward')
    plt.plot(durations_t.numpy())
    plt.hold(True)    
    plt.plot(total_reward.numpy()*-1)

    # Take 100 episode averages and plot them too
    avg_ep = 100
    if len(durations_t) >= avg_ep:
        means = durations_t.unfold(0, avg_ep, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(avg_ep-1), means))
        total_reward_means = total_reward.unfold(0, avg_ep, 1).mean(1).view(-1)
        total_reward_means = torch.cat((torch.zeros(avg_ep-1), total_reward_means))
        plt.plot(means.numpy())
        plt.hold(True)    
        plt.plot(total_reward_means.numpy()*-1)

    plt.pause(0.000001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    #print("batch.action", batch.action)

    action_batch = torch.cat(batch.action)
    #print("action_batch", action_batch)
    reward_batch = torch.cat(batch.reward)
    #print("hi5")

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    #print("hi6")
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #print("hi7")

    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    #print(type(non_final_mask))
    #print(non_final_mask)

    #print("hi8")

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    #print("hi9")

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #print("hi10")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def initial_position(robotID,joints,controlRobotiqC2,controlJoints,mimicParentName):
    userParams = dict()

    x = 0.11
    y = -0.49
    z = 1.29
    roll = 0
    pitch = 1.57
    yaw = 1.57
    orn = p.getQuaternionFromEuler([roll, pitch, yaw])
    gripper_opening_length = 0.085

    userParams[0] = -1.57
    userParams[1] = -1.57
    userParams[2] = 1.57
    userParams[3] = -1.57
    userParams[4] = -1.57
    userParams[5] = 0

    gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)    # angle calculation

    jointPose = p.calculateInverseKinematics(robotID, eefID, [x,y,z],orn)
    for i, name in enumerate(controlJoints):
        joint = joints[name]
        pose = jointPose[i]
        if i != 6:
            pose1 = userParams[i]
        if name==mimicParentName:
            controlRobotiqC2(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
        else:         
            p.setJointMotorControl2(robotID, joint.id, p.POSITION_CONTROL,targetPosition=pose1, force=joint.maxForce,maxVelocity=joint.maxVelocity)
    for i in range (200):
        p.stepSimulation()
        rXYZ = p.getLinkState(robotID, eefID)[0] # real XYZ
        if(abs((rXYZ[0]-x)<0.009)):
          if(abs((rXYZ[1]-y)<0.009)):
            if(abs((rXYZ[2]-z)<0.009)):
                #print("i", i)
                break


def get_reward_from_action(robotID,boxId,init_cubePos,done,reward):
    rXYZ = p.getLinkState(robotID, eefID)[0] # real XYZ
    current_cubePos, current_cubeOrn = p.getBasePositionAndOrientation(boxId)
    #r_dist = abs(rXYZ[2] - cubePos[2])
    #r_block = abs(1.00 - cubePos[2])
    #r_block_x = abs(init_cubePos[0] - cubePos[0])
    #r_block_y = abs(init_cubePos[1] - cubePos[1])
    #r_total = 1/r_dist + 1/r_block - r_block_x*200 - r_block_y*200

    
    if (current_cubePos[2] > 1.00 and rXYZ[2]> 1.15):
        reward = reward + 100
        done = 1 
    if(abs(current_cubePos[0] - init_cubePos[0] > 0.01 ) or abs(current_cubePos[1] - init_cubePos[1])> 0.01):
        print("Robot cannot grasp block anymore!!!")
        reward = reward - 10;            
        done = 1
    reward = reward - 1;
    return reward, done

# 0:z_up, 1:z_down, 2:close 3:open
def action_to_move_arm(action,robotID,joints,controlRobotiqC2,controlJoints,mimicParentName):
    rXYZ = p.getLinkState(robotID, eefID)[0] # real XYZ
    roll = 0
    pitch = 1.57
    yaw = -1.57
    orn = p.getQuaternionFromEuler([roll, pitch, yaw])
    gripper_state = p.getJointState(robotID, 12) # 12 is joint[name].id
    gripper_opening_length = 0.085 # initial grippoer pose
    goal = rXYZ[2]
    if(action == 0):
        #print("enter action0")
        goal = rXYZ[2] + 0.02
        #print("goal", goal)
        if goal > 1.3:
            goal = rXYZ[2]
    if(action == 1):
        #print("enter action1")
        goal = rXYZ[2] - 0.02
        #print("goal", goal)
        if goal < 1.075:
            goal = rXYZ[2]
    if(action == 2):
        #print("enter action2")
        gripper_opening_length = 0.048

    if(action == 3):
        #print("enter action3")
        gripper_opening_length = 0.085

    #print("final_goal", goal)
    gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)    # angle calculation
    jointPose = p.calculateInverseKinematics(robotID, eefID, [0.11,-0.49,goal],orn)

    for i, name in enumerate(controlJoints):
        joint = joints[name]
        pose = jointPose[i]
        if name==mimicParentName:
            controlRobotiqC2(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
        else:         
            p.setJointMotorControl2(robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=pose, force=joint.maxForce, 
                                        maxVelocity=joint.maxVelocity)
    for i in range (50):
        p.stepSimulation()
        #time.sleep(0.01)
        rXYZ = p.getLinkState(robotID, eefID)[0] # real XYZ
        gripper_state = p.getJointState(robotID, 12) # 12 is joint[name].id

        if(action == 0 or action == 1):
            #print("different arm movement", abs((rXYZ[2]-goal)<0.009))
            if(abs((rXYZ[2]-goal)<0.009)):
                #print("rXYZ[2]", rXYZ[2])
                #print("goal", goal)
                #print("i_1_2", i)
                break
        if(action == 2 and action == 3):
            #print("different gripper movement", abs((gripper_state-gripper_opening_angle)<0.009))
            if(abs((gripper_state-gripper_opening_angle)<0.009)):
                #print("gripper_state", gripper_state)
                #print("gripper_opening_angle", gripper_opening_angle)
                #print("i_3_4", i)
                break
def save_model(episode):
    print("save_the_model")
    torch.save(target_net.state_dict(),'./train/DQN_model' + str(episode) + '.pt')

def main():
    num_episodes = 100000
    for i_episode in range(num_episodes):
        robotID,joints,controlRobotiqC2,controlJoints,mimicParentName,boxId,init_cubePos = reset_env()
        initial_position(robotID,joints,controlRobotiqC2,controlJoints,mimicParentName)
        last_screen = get_screen()
        current_screen = get_screen()
        done = 0
        reward = 0
        state = current_screen - last_screen
        for t in range(1000):
            action = select_action(state)
            #print("action", action)
            action_to_move_arm(action,robotID,joints,controlRobotiqC2,controlJoints,mimicParentName)
            #current_cubePos, current_cubeOrn = p.getBasePositionAndOrientation(boxId)
            reward,done = get_reward_from_action(robotID,boxId,init_cubePos,done,reward)
            reward = torch.tensor([reward], dtype=torch.float, device=device)
            #print("reward", reward)


            #print("init_cubePos[0]", init_cubePos[0])   
            #print("init_cubePos[1]", init_cubePos[1])        
         
            #print("current_cubePos[0]", current_cubePos[0])   
            #print("current_cubePos[1]", current_cubePos[1]) 


            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            #print("hi2")
            memory.push(state, action, next_state, reward)

            # Perform one step of the optimization (on the target network)
            #print("hi3")
            optimize_model()
            #print("hi4")
            
            if done:
                episode_durations.append(t + 1)
                total_reward.append(reward)
                plot_durations(total_reward)
                print("total_reward", total_reward)
                break
        print(i_episode)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if i_episode % 100 == 0:
            print("i_episode", i_episode)
            save_model(i_episode)

    p.disconnect()

 
if __name__== "__main__":
  main()

