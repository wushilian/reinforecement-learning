import tensorflow as tf
from keras.layers import Dense,Activation
from keras.layers.convolutional import Conv2D
import numpy as np

IMG_SIZE=84#图像大小
CHANNEL=3#图像通道
NUM_ACT=4#动作数目

class DQN():
    def __init__(self,env,names):
        self.env=env
        self.memory=[]
        self.gamma=0.9  #decay rate
        self.epsilon=1   #exploration
        self.epsilon_decay=0.995
        self.epsilon_min=0.1
        self.learning_rate=0.001
        self.build_model()
        optimizer=tf.train.AdagradOptimizer(self.learning_rate)
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        
    def build_model(self):
        self.inputs=tf.placeholder(tf.float32,shape=[None,IMG_SIZEIMG_SIZECHANNEL,])
        self.conv1=Conv2D(32,[8,8],strides=4,activation='relu')(self.inputs)
        self.conv2=Conv2D(64,[4,4],strides=2,activation='relu')(self.conv1)
        self.conv3=Conv2D(64,[3,3],strides=1,activation='relu')(self.conv2)
        
        conv3_dim=self.conv3[1]*self.conv3[2]*self.conv3[3]#flat
        self.conv3_flat=tf.reshape(self.conv3,shape=[-1,conv3_dim])

        self.fc4=Dense(units=512,activation='relu',kernel_initializer='random_uniform')(self.conv3_flat)
        self.fc5=Dense(NUM_ACT,activation='linear')
    def predict(state):
        return sess.run(self.fc5,feed_dict={self.inputs:state})

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def replay(self,batch_size):
        batches=min(batch_size,len(self.memory))
        batches=np.random.choice(len(self.memory),batches)
        for i in batches:
            state, action, reward, next_state, done = self.memory[i]
            target=reward
            if not done:
                target=reward+self.gamma*np.amax(self.predict(next_state)[0])
            target_f=self.predict(state)
            target_f[0][action]=target
            























            
        
        
