import pygame                     # PYGAME package
from pygame.locals import *       # PYGAME constant & functions
from sys import exit              # exit script
from PIL import Image             # bilbiothèque pour traitement d'image
from class_Terrain import Terrain  # classe Terrain() du fichier tank_Terrain.py
from class_Tank import Tank        # classe Tank() du fichier tank_Tank.py              

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque

class Game():
    """
    classe principale du jeux
    """

    def __init__(self, size_spriteX=64, size_spriteY=64, nb_spritesX=18, nb_spritesY=12, fps=60):
        """
        constructeur de la classe
            size_spriteX=64, size_spriteY=64 représentent la taille d'une tuile (64*64 pixels par défaut)
            nb_spritesX=18, nb_spritesY=12 représentent la taille du plateau de jeux en nombre de tuiles (18*12 tuiles par défaut)
            fps: nombre d'images max par secondes
        """
        print('démarrage jeux Tank')
        self.terrain = Terrain(map_filenames=['Map/tank_background.map', # terrain de jeux avec maps
                                              'Map/tank_vegetaux.map',
                                              'Map/tank_fondations.map',
                                              'Map/tank_fondations2.map',
                                              'Map/tank_flags1.map',
                                              'Map/tank_flags2.map',
                                              'Map/tank_bords.map',
                                              ])  
        self.tanks = []  # liste des tanks
        self.objectivPos = (2, 1)
        self.time = 0

        self.state_dim = 4 #1769472
        self.action_dim = 4
        self.max_action = 1

        self._max_episode_steps = 500

       # tank joué au clavier
        # self.tanks.append(Tank(self.terrain, 'Tank_human',
        #                       l_img_name=['Media/Tank/Tank2/Markup_128x128.png',
        #                         'Media/Tank/Tank2/Hull_02_128x128.png',
        #                         'Media/Tank/Tank2/Gun_03_128x128.png'],
        #                       human=True,
        #                       pos=( (self.terrain.nb_spritesX-3.5) * self.terrain.size_spriteX,
        #                             (self.terrain.nb_spritesY-3.5) * self.terrain.size_spriteY)
        #                     ))
        # self.humanTank = self.tanks[0] #tank joué au clavier par un humain

        #autres tank joué par l'ordi
        
        for i in range(1):
            self.tanks.append(Tank(self.terrain, 'Tank_pi',
                                    l_img_name=['Media/Tank/Tank1/Markup_128x128.png',
                                    'Media/Tank/Tank1/Hull_02_128x128.png',
                                    'Media/Tank/Tank1/Gun_03_128x128.png'],
                                    human=False,
                                    pos=( (self.terrain.nb_spritesX-3.5) * self.terrain.size_spriteX,
                                          (self.terrain.nb_spritesY-3.5) * self.terrain.size_spriteY)
                                ))
        for tank in self.tanks:
            tank.changes_shield()


        self.timer = pygame.time.Clock()  # timer pour contrôler le FPS
        self.fps = fps  
 
        fps = 200 #images par seconde 
        
    # def loop(self):
    #     """
    #     boucle infinie  du jeux: lecture des événements et dessin
    #     """
    #     for i in range(2):
    #         #lecture des événements Pygame 
    #         for event in pygame.event.get():  
    #             if event.type == QUIT:  # evènement click sur fermeture de fenêtre
    #                 self.destroy()      # dans ce cas on appelle le destructeur de la classe
    #             elif event.type == KEYDOWN: 
    #                 if (event.key==K_SPACE):  # activation du bouclier
    #                     self.humanTank.changes_shield()
    #                 elif (event.key==K_f):    # tir d'un obus
    #                     self.humanTank.fire()
                        
    #         self.timer.tick(self.fps)   #limite le fps
            
    #         self.terrain.dessine()      #dessin du terrain
            
    #         for i in range(len(self.tanks)):        # dessin des tanks et obus tirés
    #             self.tanks[i].dessine()             # dessine le tank
    #             if self.tanks[i].shell.countdown>0: # dessine l'obus s'il a été tiré
    #                 self.tanks[i].shell.dessine()
    #             if self.tanks[i].shell.explosion.booom: # dessine l'explosion de l'obus si activée
    #                 self.tanks[i].shell.explosion.dessine()
    #             if self.tanks[i].human == True:
    #                 self.tanks[i].bouge()            # fait bouger le tank et l'obus s'il est tiré
                    
    #             if self.tanks[i].pos_to_grid() == (2, 1):
    #                 self.tanks[i].win = True
    #                 self.tanks[i].alive = False

                    
    #         print(self.timer.get_fps())
    #         pygame.display.update()     # rafraîchi l'écran
    #         pygame.image.save(pygame.display.get_surface(),'test.jpg')
    #         from PIL import Image
    #         img = Image.open('test.jpg').convert('LA')
    #         img.save('greyscale.png')
            
            
    def sample(self):
        x = random.randint(0,7)
        if x == 0:
            l = [1,0,0,0]
        if x == 1:
            l = [1,1,0,0]
        if x == 2: 
            l = [1,0,0,1]
        if x == 3:
            l = [0,1,0,0]
        if x == 4:
            l = [0,0,0,1]
        if x == 5:
            l = [0,0,1,0]
        if x == 6:
            l = [0,1,1,0]
        if x == 7:
            l = [0,0,1,1]
        return l
            
            
        return [random.randint(-1,1) for i in range(4)]
    
    def rotateAngle(self,pts,center,angle):
        angle = -angle
        angle *= np.pi / 180
        Xpts = pts[0] - center[0]
        Ypts = pts[1] - center[1]
        x = Xpts * np.cos(angle) - Ypts * np.sin(angle) + center[0]
        y = Xpts * np.sin(angle) + Ypts * np.cos(angle) + center[1]

        return (x,y)
    
    def test36(self,x):
        return x * 360 / 64
    
    def foundDistance(self):
        debut =  (self.tanks[0].pos[0] + 65, self.tanks[0].pos[1] + 65)
        step = 5
        up = (0,step)
        upRight = (0,-step)
        right = (step,step)
        rightDown = (-step,step)
        down = (-step,-step)
        downLeft = (step,-step)
        left = (step,0)
        leftUp = (-step,0)
        
        directions = [up,upRight,right,rightDown,down,downLeft,left,leftUp]
        
        for i in directions:
            for j in np.linspace(0, 5, 50):
                step = (step[0] + j, step[1] + j)
                pts = (dubut[0] + i[0], debut[1] + i[1])
                gfxdraw.pixel(self.terrain.screen, (0,255,0),self.rotateAngle(pts,debut,self.test36(self.tanks[0].l_rotation[0])))

        
    def step(self, action):
        for event in pygame.event.get():  
            if event.type == QUIT:  # evènement click sur fermeture de fenêtre
                self.destroy()      # dans ce cas on appelle le destructeur de la classe
                
        self.timer.tick(self.fps)   #limite le fps      
        self.terrain.dessine()      #dessin du terrain
        
        debut =  (self.tanks[0].pos[0] + 65, self.tanks[0].pos[1] + 65)
        pygame.draw.line(self.terrain.screen, (0,255,0), debut,self.rotateAngle((debut[0],debut[1] + 50),debut,self.test36(self.tanks[0].l_rotation[0])))
        pygame.draw.line(self.terrain.screen, (0,255,0), debut, self.rotateAngle((debut[0],debut[1] - 50),debut,self.test36(self.tanks[0].l_rotation[0])))
        pygame.draw.line(self.terrain.screen, (0,255,0), debut, self.rotateAngle((debut[0] +50,debut[1] + 50),debut,self.test36(self.tanks[0].l_rotation[0])))
        pygame.draw.line(self.terrain.screen, (0,255,0), debut, self.rotateAngle((debut[0] -50 ,debut[1] + 50),debut,self.test36(self.tanks[0].l_rotation[0])))
        pygame.draw.line(self.terrain.screen, (0,255,0), debut, self.rotateAngle((debut[0] -50,debut[1] - 50),debut,self.test36(self.tanks[0].l_rotation[0])))
        pygame.draw.line(self.terrain.screen, (0,255,0), debut, self.rotateAngle((debut[0] + 50,debut[1] - 50),debut,self.test36(self.tanks[0].l_rotation[0])))
        pygame.draw.line(self.terrain.screen, (0,255,0), debut, self.rotateAngle((debut[0] + 50,debut[1]),debut,self.test36(self.tanks[0].l_rotation[0]))) 
        pygame.draw.line(self.terrain.screen, (0,255,0), debut, self.rotateAngle((debut[0] - 50,debut[1]),debut,self.test36(self.tanks[0].l_rotation[0])))
        
        
        
        u, r, d ,l = action
        for tank in self.tanks:
            tank.dessine()
            tank.bouge(u, r, d ,l)
            if tank.pos_to_grid() == (2, 1):
                tank.win = True
                tank.alive = False 
        pygame.display.update()     # rafraîchi l'écran
        self.time += 1
        
        # pygame.image.save(pygame.display.get_surface(),'test.jpg')
        # obs = Image.open('test.jpg').convert('LA').resize((50,50),PIL.Image.ANTIALIAS)
        
        # obs = [i[0] for i in list(obs.getdata())]
        
        obs =  [self.tanks[0].l_rotation[0],self.tanks[0].pos_to_grid()[0],self.tanks[0].pos_to_grid()[1],self.tanks[0].v]
        reward = self.costFunction(self.tanks[0])
        done = False
        if self.tanks[0].alive == False or self.tanks[0].win == True:
            done = True
        info = {}
        
        return obs, -reward, done, info
        
    def reset(self):
        self.tanks[0].reset()
        # pygame.image.save(pygame.display.get_surface(),'test.jpg')
        # obs = Image.open('test.jpg').convert('LA').resize((50,50),PIL.Image.ANTIALIAS)
        
        # obs = [i[0] for i in list(obs.getdata())]
        obs = (self.tanks[0].l_rotation[0],self.tanks[0].pos_to_grid()[0],self.tanks[0].pos_to_grid()[1],self.tanks[0].v)
        
        self.time = 0
        return obs
        
    def costFunction(self,tank):
        c = 0
        posTank = tank.pos
        c += ( 1 / abs(abs(self.objectivPos[0]) - abs(posTank[0]) 
        + abs(self.objectivPos[1]) - abs(posTank[1])) ) * 1000
        # print(c)
        # print(tank.pos_to_grid())
        # if tank.win != True:
        #     c -= self.time / 100
        #     if tank.alive != True:
        #         c -= 30
        return c

    def destroy(self):
        """
        destructeur de la classe
        """
        print('Bye!')
        pygame.quit() # ferme la fenêtre principale
        exit()        # termine tous les process en cours

####################################################################





class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
     
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x))
    return x


class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u):
    xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2

  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1


# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state):
    state = torch.Tensor(state.reshape(1, -1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
   
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))



def evaluate_policy(policy, eval_episodes=10):
  avg_reward = 0.
  for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward



env_name = "AntBulletEnv-v0" # Name of a environment (set it to any Continous environment you want)
seed = 0 # Random seed number
start_timesteps = 10000#1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 10000 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 200 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.0005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 10 # Number of iterations to wait before the policy network (Actor model) is updated


file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")


if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")

#env = gym.make(env_name)
env = Game()


# env.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# max_action = float(env.action_space.high[0])
# print(state_dim)

state_dim = env.state_dim
action_dim = env.action_dim
max_action = env.max_action

resetPolicy = True

policy = TD3(state_dim, action_dim, max_action)
if not resetPolicy:
    policy.load('TD3_AntBulletEnv-v0_0',directory="./pytorch_models")



replay_buffer = ReplayBuffer()



#evaluations = [evaluate_policy(policy)]



def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')
max_episode_steps = env._max_episode_steps
save_env_vid = False
if save_env_vid:
  env = wrappers.Monitor(env, monitor_dir, force = True)
  env.reset()


firstI = True

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()


if resetPolicy:
        total_timesteps = 0
else:
    with open(r'./save/total_timesteps.txt') as f:
        total_timesteps = int(f.read()) 

# We start the main loop over 500,000 timesteps
while 1:
    
  # If the episode is done
  if done:

    # If we are not at the very beginning, we start the training process of the model
    if total_timesteps != 0 and not firstI :
      print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
      policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

    # We evaluate the episode and we save the policy
    if timesteps_since_eval >= eval_freq:
      timesteps_since_eval %= eval_freq
      #evaluations.append(evaluate_policy(policy))
      policy.save(file_name, directory="./pytorch_models")
      #np.save("./results/%s" % (file_name), evaluations)
      #with open(r'./save/total_timesteps.txt','w') as f:
      #    f.write(str(total_timesteps))
          
    # When the training step is done, we reset the state of the environment
    obs = env.reset()
    
    # Set the Done to False
    done = False
    
    # Set rewards and episode timesteps to zero
    episode_reward = 0
    episode_timesteps = 0
    episode_num += 1
  
  # Before 10000 timesteps, we play random actions
  if total_timesteps < start_timesteps:
    action = env.sample()
  else: # After 10000 timesteps, we switch to the model
    action = policy.select_action(np.array(obs))
    # If the explore_noise parameter is not 0, we add noise to the action and we clip it
    if expl_noise != 0:
      action = (action + np.random.normal(0, expl_noise, size=env.action_dim)).clip(-1, 1)
      action = [int(i) for i in action]
  
  # The agent performs the action in the environment, then reaches the next state and receives the reward
  new_obs, reward, done, _ = env.step(action)

  # We check if the episode is done
  done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
  
  if episode_timesteps == env._max_episode_steps:
      done = True

  # We increase the total reward
  episode_reward += reward
  
  # We store the new transition into the Experience Replay memory (ReplayBuffer)
  replay_buffer.add((obs, new_obs, action, reward, done_bool))

  # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
  obs = new_obs
  episode_timesteps += 1
  total_timesteps += 1
  timesteps_since_eval += 1
  
  firstI = False
      

# We add the last policy evaluation to our list of evaluations and we save our model
evaluations.append(evaluate_policy(policy))
if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
np.save("./results/%s" % (file_name), evaluations)
