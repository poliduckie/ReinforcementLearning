import os

import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt
import gym_duckietown

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gym.spaces import Box

log_dir = "./ppo_duckieloop/"
os.makedirs(log_dir, exist_ok=True)

map_name = "Duckietown-zigzag_dists-v0" #@param ['Duckietown-straight_road-v0','Duckietown-4way-v0','Duckietown-udem1-v0','Duckietown-small_loop-v0','Duckietown-small_loop_cw-v0','Duckietown-zigzag_dists-v0','Duckietown-loop_obstacles-v0','Duckietown-loop_pedestrians-v0']
cutted_img_height = 350 #@param {type: "slider", min: 0, max: 480, step:1}
resize_ratio = 0.35 #@param {type: "slider", min: 0.0, max: 1.0, step:0.01}

img_height = 480
top_crop = img_height - cutted_img_height

img_final_height = int(cutted_img_height * resize_ratio)
img_final_width = int(640 * resize_ratio)

# Wrapper used to reduce the duckie's action space to permit it from going backward(to avoid stuttering in place)
class ActionWrapper(gym.ActionWrapper):
    """
    Wrapper to change the range of possible actions.
    """
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)
        # This line stops the car from moving backwards
        self.action_space = Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        # Default:
        # self.action_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.env = env
    
    def action(self, act):
        return act

class NormalizeWrapper(gym.ObservationWrapper):
    """
    Normalize image to have values between 0 and 1
    """
    def __init__(self, env):
        super(NormalizeWrapper, self).__init__(env)
        self.observation_space = Box(0, 1, self.observation_space.shape, dtype=self.observation_space.dtype)
        self.env = env

    def observation(self, obs):
        obs = obs/255
        return obs

class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)
        self.observation_space = Box(0, 255, (img_final_height, img_final_width, 3), dtype=self.observation_space.dtype)
        self.accept_start_angle_deg = 4
        self.env = env

    def observation(self, obs):
        cropped = cropimg(obs)
        resized = resizeimg(cropped, resize_ratio)
        balanced = white_balance(resized)
        img = takewhiteyellow(balanced)
        return img

def cropimg(img):
    """
    Crop top of image top_crop px, they are noise most of the time

    :param img: (RGB image as np array) Image to be cropped
    """
    return img[top_crop:,:]

def houghtransform(img):
    """
    Apply Hough Line transform, for theory see:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

    :param img: (RGB image as np array)
    """
    frame_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY, 3)
    edges = cv2.Canny(frame_BGR,50,150,apertureSize = 3)
    #minLineLength = 100
    #maxLineGap = 10
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    #for x1,y1,x2,y2 in lines[0]:
    #    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    imgRGB = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)
    return imgRGB

def resizeimg(img, ratio):
    """
    Resize image
    :param img: (np array)
    :param ratio: (float) 0<ratio<1
    """
    return cv2.resize(img, (0,0), fx=ratio, fy=ratio) 
  
def takeyellow(img):
    """
    Extract yellow lines, for color ranges see:
    https://stackoverflow.com/questions/48109650/how-to-detect-two-different-colors-using-cv2-inrange-in-python-opencv

    :param img: (RGB image as np array)
    """
    frame_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (20,100,100), (50, 255, 255))
    imgRGB = cv2.cvtColor(frame_threshold, cv2.COLOR_GRAY2RGB)
    return imgRGB

def takewhiteyellow(img):
    """
    Extract white and yellow lines

    :param img: (RGB image as np array)
    """
    #white
    sensitivity = 100
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])
    frame_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    maskwhite = cv2.inRange(frame_HSV, lower_white, upper_white)
    img[maskwhite > 0] = (255, 0, 0)
    img[maskwhite == 0] = (0,0,0)
    #yellow
    maskyellow = cv2.inRange(frame_HSV, (15,70,70), (50, 255, 255))
    img[maskyellow > 0] = (0, 255, 0)
    return img

def white_balance(img):
    """
    Grayworld assumption:
    https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption/46391574

    :param img: (RGB image as np array)
    """
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result

if __name__ == '__main__':
    """
    Run this file to test the wrapper
    """
    env = gym.make(map_name)
    env = ObsWrapper(env)
    env = ActionWrapper(env)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=False,
        tensorboard_log="./ppo_duckieloop/",
        gamma=0.99,
        # https://arxiv.org/abs/2005.05719
        use_sde=True,
        # https://www.researchgate.net/figure/TD3-Hyperparameters_tbl2_341341608
        sde_sample_freq=64, # def: -1
        # https://github.com/hill-a/stable-baselines/issues/213
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        # target kl sets an early stopping limit if the kl diverges too much
        # target_kl
    )

    # env_new = gym.make(map_names[3])
    # env_new = ObsWrapper(env_new)
    # env_new = ActionWrapper(env_new)
    # env_new = Monitor(env_new, log_dir)
    # model.set_env = env_new

    # number of interations the for loop is going to make
    N_ITERATIONS = 10
    ## maps without obstacles, to be used for circular training later
    map_no_obstacles_names = ["Duckietown-straight_road-v0","Duckietown-small_loop-v0","Duckietown-small_loop_cw-v0","Duckietown-zigzag_dists-v0"]

    # number of timesteps every loop is going to take
    total_timesteps = int(1e5)
    # number of episodes the model will be evaluated in
    n_eval_episodes = 10

    for time in range(N_ITERATIONS):
        current_map = map_no_obstacles_names[3]
        model.learn(total_timesteps=total_timesteps, tb_log_name="run_"+str(time))
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
        model.save("ppo_multiple_maps"+current_map+str(time))
        if time%5==0:
            print(f"#{time} Trained {str(total_timesteps*time)} timesteps, mean_reward: {mean_reward}, std_reward: {std_reward}, current map: {current_map}")
        else:
            break

