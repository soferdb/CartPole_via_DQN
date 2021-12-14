# CartPole_via_DQN

The environment is the CartPole environment. The goal of CartPole is to balance a pole connected with one joint on top of a moving cart. Each state can be defined as 4 dimension vector of rational numbers, which contains the following information
s = [cart position, cart velocity, pole angle, pole angular velocity].

![Pole](https://user-images.githubusercontent.com/49614331/145956577-0a45f1d9-1211-4e0b-bfed-754c264d9389.png)

 For further information, refer to the pdf provided.
 
 This repository contains tensorflow2 implementation of DQN algorithm, with DoubleDQN and Prioritized Replay-Buffer (PER) extentions.

Instructions for training a DQN model with different setups.
In all of the following, we used hard-update for the target update method.

    DQN with uniform replay buffer:
      run the following command -
      python DQN.py --Q 2 --exp [exp_name]
      to receive results under exp_name folder in the log-directory.
    
    DQN with additional improvements: 
    
    For DQN with PER and state-dependent reward shaping, run the following command -
    python DQN.py --Q 3 --exp [exp_name]
    
    Extra arguments:
    run the following command -
    python DQN.py --Q 2
    with the following optional arguments to change configurations. 
        1. "--model_layers 5" for 5 layers model
        2. "--model_type DDQN" to use Double DQN model
        3. "--buffer_type priority" to use Prioritized Experience Replay 
        4. "--exp [exp_name]" to set experiment name
        5. "--n_episodes [integer]" to set the number of max episodes
        6. "--no_breaks" to force running until max number of episodes
        7. "--reward_shaping"  without reward-shaping.
        8. "--reward_shaping 1" to add state-dependent reward shaping
        9. "--reward_shaping 2" to add fail-dependent reward shaping
        10. "--reward_shaping 1 2" to add both reward shaping methods above.
        
Results: 
The different methods are compared using the plots of the average score over the last 100 episodes.

![reward_shaping](https://user-images.githubusercontent.com/49614331/145958925-d38f7007-f4e1-4913-92e4-30cc349d7837.png)

Although using only state-dependent reward-shaping (RS1) leads our agent towards a slower convergence. We claim that this method improves the
training stability since it increase the average score curve smoothness.

![dqn_ddqn_per_rs](https://user-images.githubusercontent.com/49614331/145959016-44e4ce03-62a5-4624-bcca-66e44c096d69.png)

Comparison of different combinations of methods, for further information please refer to the PDF.
