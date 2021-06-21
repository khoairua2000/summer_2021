""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle as pickle
import gym
import matplotlib.pyplot as plt


def displayImage(image):
    #to check prepro
    plt.imshow(image)
    plt.show()

# env = gym.make("Pong-v0")
# #env = gym.wrappers.Monitor(env, '.', force=True) # visualize
# I = env.reset()
# I = I[35:195] # crop top and bottom
# I = I[::2,::2,0] # downsample by factor of 2, only need black white color
# displayImage(I)
def sigmoid(x):
    """
    Parameters
    ----------
    x : float

    Returns
    -------
    real number in range [0,1].

    """
    return 1.0 / (1.0 + np.exp(-x))

def discount_rewards(r, gamma):
    """
    Parameters
    ----------
    r : 1D float array reward.

    Returns
    -------
    discounted_r : 1D float array discount rewards

    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
      if r[t] != 0: 
          running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
      running_add = running_add * gamma + r[t]
      discounted_r[t] = running_add
    return discounted_r

def prepro(I):
        """
        Parameters
        ----------
        I : RGB image 210x160x3.
        Returns
        -------
        6400 (80x80) 1D float vector
    
        """
        I = I[35:195] # crop top and bottom
        I = I[::2,::2,0] # downsample by factor of 2, only need black white color
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()     #copy the array, make it to float, ravel to flatten
class Pong:
    def __init__(self, neurons = 200, batch_size = 10, learning_rate = 1e-3, gamma = 0.99,
                 decay_rate = 0.99, resume = True, render = True):
        self.H = neurons        
        self.batch_size = batch_size # every how many episodes to do a param update?
        self.learning_rate = learning_rate
        self.gamma = gamma # discount factor for reward
        self.decay_rate = decay_rate # decay factor for RMSProp leaky sum of grad^2
        self.resume = resume # resume from previous checkpoint?
        self.render = render
        
        # model initialization
        self.D = 80 * 80 # input dimensionality: 80x80 grid
        if self.resume:
          self.model = pickle.load(open('save.p', 'rb'))
        else:
          self.model = {}
          self.model['W1'] = np.random.randn(self.H,self.D) / np.sqrt(self.D) # "Xavier" initialization
          self.model['W2'] = np.random.randn(self.H) / np.sqrt(self.H)
          
        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() } # update buffers that add up gradients over a batch
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.items() } # rmsprop memory
    


    
    def policy_forward(self,x):
        """
        Parameters
        ----------
        x : input - different of observation.
    
        Returns
        -------
        p : probability of taking action 2.
        h : hidden state
    
        """
        h = np.dot(self.model['W1'], x) # (H x D) (D x 1) = (Hx1) 
        #print(model['W1'].shape)
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h) # (H) x (H) = number
        p = sigmoid(logp)
        return p, h # return probability of taking action 2, and hidden state
    
    def policy_backward(self,eph, epdlogp, epx):
        """
        Parameters
        ----------
        eph : array of intermediate hidden states. (H x episode)
        epdlogp : array of intermediate Derivative of log p - direction for update (episode x 1)
    
        Returns
        -------
        dict : direct for update for weight.
    
        """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}
    
    def play(self):
        env = gym.make("Pong-v0")
        #env = gym.wrappers.Monitor(env, '.', force=True) # visualize
        observation = env.reset()
        prev_x = None # used in computing the difference frame
        xs,hs,dlogps,drs = [],[],[],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0
        while True:
          if self.render: env.render()
        
          # preprocess the observation, set input to network to be difference image
          cur_x = prepro(observation)
          x = cur_x - prev_x if prev_x is not None else np.zeros(self.D)
          prev_x = cur_x
        
          # forward the policy network and sample an action from the returned probability
          aprob, h = self.policy_forward(x)
          action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
        
          # record various intermediates (needed later for backprop)
          xs.append(x) # observation
          hs.append(h) # hidden state
          y = 1 if action == 2 else 0 # a "fake label"
          dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
                                      #calculus from https://www.youtube.com/watch?v=tqrcjHuNdmQ&t=1704s     27:24
          # step the environment and get new measurements
          observation, reward, done, info = env.step(action)
          reward_sum += reward
        
          drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
        
          if done: # an episode finished
            episode_number += 1
        
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs,hs,dlogps,drs = [],[],[],[] # reset array memory
        
            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr, self.gamma)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
        
            epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
            grad = self.policy_backward(eph, epdlogp, epx)
            for k in self.model: self.grad_buffer[k] += grad[k] # accumulate grad over batch
        
            # perform rmsprop parameter update every batch_size episodes
            if episode_number % self.batch_size == 0:
              for k,v in self.model.items():
                g = self.grad_buffer[k] # gradient
                self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
        
            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            if episode_number % 100 == 0: pickle.dump(self.model, open('save.p', 'wb'))
            reward_sum = 0
            observation = env.reset() # reset env
            prev_x = None
        
          if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            print (('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
