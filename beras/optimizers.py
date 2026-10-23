from collections import defaultdict
import numpy as np

class BasicOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def apply_gradients(self, trainable_params, grads):
        for param, grad in zip(trainable_params, grads):
            param.assign(param - (grad)*self.learning_rate)

class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, trainable_params, grads):
        B = self.beta
        for param, grad, v in zip(trainable_params, grads,self.v):
            v.assign((0.9*B)+(1-B)*np.square(grad))
            param.assign(param-(self.learning_rate/(np.square(v))+self.epsilon)*(grad))

class Adam:
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):


        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.t = 0                              # Time counter

    def apply_gradients(self, trainable_params, grads):
        for m, param,grad, v in zip(self.m, trainable_params, grads, self.v):
            m.assign((m*self.beta_1)+(1-self.beta_1)*(grad))
            v.assign((v*self.beta_2)+(1-self.beta_2)*np.square(grad))
            m_hat = m/(1-np.power(self.beta_1,self.t)) #doublecheck if power works
            v_hat = v/(1-np.power(self.beta_2,self.t))
            param.assign(param-((self.learning_rate*m_hat)/(np.square(v_hat))+self.epsilon))