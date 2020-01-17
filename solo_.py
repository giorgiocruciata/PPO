import gym
import tensorflow as tf
import numpy as np
from random import choices
from sklearn.utils import shuffle

### test LUNAR LANDER ########
env = gym.make("LunarLander-v2")
state = env.reset()
num_state=env.observation_space.shape[0]
num_action=env.action_space.n
##### CREATE THE ACTOR CRITC MODEL ############
input_state=tf.keras.layers.Input(shape=(num_state,))
x=tf.keras.layers.Dense(32, activation='relu')(input_state)
out_act=tf.keras.layers.Dense(num_action, activation = "softmax")(x)
out_value=tf.keras.layers.Dense(1, activation = "tanh")(x)
model = tf.keras.models.Model(inputs=[input_state], outputs=[out_act,out_value])
model_new = tf.keras.models.Model(inputs=[input_state], outputs=[out_act,out_value])
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
#critic_loss = """MEAN SQUARED ERROR BETWEEN REWARD-VALUE"""
critic_loss = tf.keras.losses.MSE
############## --------------- ###################
def get_advantages(values, masks, rewards):
    returns = []
    gamma=0.99
    lmbda=0.95
    gae = 0
    gamma_lmbda=gamma*lmbda
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma_lmbda * gae
        returns.insert(0, gae)
    return returns
###############  Discount reward  ###########################
def discount_rewards(r, gamma = 0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
###############  PPO LOSS ###########################
def proximal_policy_optimization_loss(old_onehot,old_prediction, advantage, new_predictio,new_onehot):
    LOSS_CLIPPING=0.2
    ENTROPY_LOSS=0.001
    prob = tf.math.reduce_sum(new_onehot * new_predictio, axis=-1)
    old_prob = tf.math.reduce_sum(old_onehot * old_prediction, axis=-1)
    ratio=  prob/(old_prob + 1e-10)
    return tf.math.minimum(ratio * advantage,tf.clip_by_value(ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING) * advantage)  + ENTROPY_LOSS * -prob * tf.math.log(prob + 1e-10)
###############-----###########################
def restituisci_azione(prob_action,num_act=num_action,prob_eps=0.3):
    choice=choices(range(2),weights=[1-prob_eps,prob_eps])
    #print("choice",choice)
    if choice[0]==0:
        action = np.argmax(prob_action)
        #print("max",action)
    else:
        action = np.random.choice(len(prob_action[0]))
        #print("casual",action)
    return action
########## INPUT: INITIAL POLICY PARAMETERS TETA_0, AND VALUE PARAMETERS PHI_0
model_new.set_weights(model.get_weights())
gradBuffer = model.trainable_variables
for ix, grad in enumerate(gradBuffer):
    gradBuffer[ix] = grad * 0
################### FOR K= 0,1,2
episodes=1000
reset=True

################## COLLECT SET OF TRAJECTORIES D_k
for e in range(episodes):
    cumulate_reward=0
    if reset is True:
        states = []
        actions = []
        values = []
        masks = []
        rewards = []
        actions_probs = []
        actions_onehot = []
        reset = False
    state = env.reset()
    state_input = tf.keras.backend.expand_dims(state, 0)
    _, q_value_uno = model(state_input)
    values.append(q_value_uno)
    states.append(state)
    done = False
    while not done:
        ## calcola il gradiente in modo automatico
        state_input = tf.keras.backend.expand_dims(state, 0)
        prob_action, v_value = model(state_input)
        action=restituisci_azione(prob_action)
        action_onehot = np.zeros(num_action)
        action_onehot[action] = 1
        state, reward, done, info = env.step(action)
        cumulate_reward+=reward
        #print(state, reward, done, info)
        env.render()
        actions_probs.append(prob_action)
        states.append(state)
        actions.append(action)
        values.append(v_value[0][0])
        rewards.append(reward)
        actions_onehot.append(action_onehot)
        if not done:
            mask=1
        else:
            mask=0
        masks.append(mask)
    print("episode :",e ,"cumulate_reward", cumulate_reward )

        # Discound the rewards
    ##if ###e 5
    if e % 1==0:
        actions =np.asarray(actions)
        values =np.asarray(values)
        masks = np.asarray(masks)
        rewards = np.asarray(rewards)
        actions_onehot=np.asarray(actions_onehot,dtype="float32")
        disc_re=discount_rewards(rewards)
        actions_probs = np.asarray(actions_probs)
        advantages=np.asarray(get_advantages(values,masks,rewards))
    ###########calcolo advanytage attraverso delta vedi art ppo
    ###########calcola loss v e funz obiettivo
        for k in range(4):#epoche
            model_new.set_weights(model.get_weights())
            for itera in range(len(rewards)):
                with tf.GradientTape() as tape:
                    # forward pass
                    state_input = tf.keras.backend.expand_dims(states[itera], 0)
                    new_prob_action, v_value = model_new(state_input)
                    action = actions[itera]
                    action_onehot = np.zeros(num_action)
                    action_onehot[action] = 1
                    loss = proximal_policy_optimization_loss(actions_onehot[itera],actions_probs[itera],advantages[itera],new_prob_action,actions_onehot[itera]) - critic_loss(disc_re[itera],v_value)
                    gradiente = tape.gradient(loss, model_new.trainable_variables)
                for ix, grad in enumerate(gradiente):
                    gradBuffer[ix]-=grad*(1/len(rewards))
            optimizer.apply_gradients(zip(gradBuffer, model_new.trainable_variables))
            for ix, grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0
            model.set_weights(model_new.get_weights())
        reset = True
