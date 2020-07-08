import numpy as np
from maxlikespy.model import Model
import autograd.numpy as np
import autograd.scipy.special as sse
import matplotlib.pyplot as plt
import math


class ConstVariable(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["a_0"]

    def model(self, x, plot=False):
        o = x
        return o

    def objective(self, x):

        fun = self.model(x)

        total = 0
        for ind, trial in enumerate(self.spikes):
                total+= np.sum(trial[self.window[ind,0]:self.window[ind, 1]] * (-np.log(fun)) +
                            (1 - trial[self.window[ind,0]:self.window[ind, 1]]) * (-np.log(1 - (fun))))

        return total

    def plot_model(self, x):

        o = x 
        return o

class SigmaMuTauVariableLength(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu", "tau", "a_1", "a_0"]

    def info_callback(self):
        self.trial_lengths = self.info["trial_length"]
        for ind, trial in enumerate(self.trial_lengths):
            self.spikes[ind][trial:] = np.nan

    def objective(self, x):

        fun = self.model(x)
        total = 0
        for ind, trial in enumerate(self.spikes):
                if self.window[ind, 0] < 0:
                    min_ind = 0
                else:
                    min_ind = self.window[ind, 0]
                
                total+= np.sum(trial[min_ind:self.window[ind, 1]] * (-np.log(fun[min_ind:self.window[ind, 1]])) +
                            (1 - trial[min_ind:self.window[ind, 1]]) * (-np.log(1 - (fun[min_ind:self.window[ind, 1]]))))
        return total

    def model(self, x):

        s, mu, tau, a_1, a_0 = x
        l = 1/tau
        '''old method'''
        # fun = a_1*np.exp(-0.5*(np.power((self.t-m)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-m)/s))))
        # ) + a_0

        self.function = (a_1*(np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))) + a_0

        return self.function
    
    def plot_model(self, x):

        return self.model(x)


class InhibitSigmaMuTauJayLEC(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["sigma", "mu", "tau", "a_1", "a_0"]

    def info_callback(self):
        self.trial_lengths = self.info["trial_length"]
        for ind, trial in enumerate(self.trial_lengths):
            self.spikes[ind][trial:] = np.nan

    def objective(self, x):

        fun = self.model(x)
        total = 0
        for ind, trial in enumerate(self.spikes):
                if self.window[ind, 0] < 0:
                    min_ind = 0
                else:
                    min_ind = self.window[ind, 0]
                
                total+= np.sum(trial[min_ind:self.window[ind, 1]] * (-np.log(fun[min_ind:self.window[ind, 1]])) +
                            (1 - trial[min_ind:self.window[ind, 1]]) * (-np.log(1 - (fun[min_ind:self.window[ind, 1]]))))
        return total

    def model(self, x):

        s, mu, tau, a_1, a_0 = x
        l = 1/tau
        '''old method'''
        # fun = a_1*np.exp(-0.5*(np.power((self.t-m)/s,2)))*(s/tau)*np.sqrt(np.pi/2)*(
        #     np.array(list(map(self.erfcx, (1/np.sqrt(2))*((s/tau)- (self.t-m)/s))))
        # ) + a_0

        self.function = (-a_1*(np.exp((l/2)*(2*mu+l*s**2-2*self.t))*sse.erfc((mu+l*s**2-self.t)/(np.sqrt(2)*s)))) + a_0

        return self.function
    
    def plot_model(self, x):

        return self.model(x)


class TimeVariableLength(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["a_1", "ut", "st", "a_0"]

    def info_callback(self):
        self.trial_lengths = self.info["trial_length"]
        for ind, trial in enumerate(self.trial_lengths):
            self.spikes[ind][trial:] = np.nan

    def objective(self, x):

        fun = self.model(x)
        total = 0
        for ind, trial in enumerate(self.spikes):
                if self.window[ind, 0] < 0:
                    min_ind = 0
                else:
                    min_ind = self.window[ind, 0]
                
                total+= np.sum(trial[min_ind:self.window[ind, 1]] * (-np.log(fun[min_ind:self.window[ind, 1]])) +
                            (1 - trial[min_ind:self.window[ind, 1]]) * (-np.log(1 - (fun[min_ind:self.window[ind, 1]]))))
        return total

    def model(self, x):
        a, ut, st, o = x

        self.function = (
            (a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        return self.function 
    
    def plot_model(self, x):
     
        return self.model(x)

class TimeInhibitVariableLength(Model):

    def __init__(self, data):
        super().__init__(data)
        self.param_names = ["a_1", "ut", "st", "a_0"]

    def info_callback(self):
        self.trial_lengths = self.info["trial_length"]
        for ind, trial in enumerate(self.trial_lengths):
            self.spikes[ind][trial:] = np.nan

    def objective(self, x):

        fun = self.model(x)
        total = 0
        for ind, trial in enumerate(self.spikes):
                if self.window[ind, 0] < 0:
                    min_ind = 0
                else:
                    min_ind = self.window[ind, 0]
                
                total+= np.sum(trial[min_ind:self.window[ind, 1]] * (-np.log(fun[min_ind:self.window[ind, 1]])) +
                            (1 - trial[min_ind:self.window[ind, 1]]) * (-np.log(1 - (fun[min_ind:self.window[ind, 1]]))))
        return total

    def model(self, x):
        a, ut, st, o = x

        self.function = (
            (-a * np.exp(-np.power(self.t - ut, 2.) / (2 * np.power(st, 2.)))) + o)
        return self.function 
    
    def plot_model(self, x):

        return self.model(x)