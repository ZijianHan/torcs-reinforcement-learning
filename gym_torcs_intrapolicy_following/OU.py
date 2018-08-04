import random
import numpy as np

class OU(object):

    def function1(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * (np.random.randn(1))

    def function2(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * (2*np.random.randn(1)-1)
