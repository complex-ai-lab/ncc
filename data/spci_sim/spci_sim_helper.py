import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import random


class simulate_data_loader():
    def __init__(self):
        pass

    def get_simul_data(self, simul_type):
        if simul_type == 1:
            Data_dict = self.simulation_state_space(
                num_pts=2000, alpha=0.9, beta=0.9)
        if simul_type == 2:
            Data_dict = self.simulation_non_stationary()
            Data_dict['X'] = torch.from_numpy(Data_dict['X']).float()
            Data_dict['Y'] = torch.from_numpy(Data_dict['Y']).float()
        if simul_type == 3:
            # NOTE: somehow for this case, currently RF quantile regression does not yield shorter interval. We may tune past window to get different results (like decrease it to 250) if need
            Data_dict = self.simultaion_heteroskedastic()
        return Data_dict

    def simulation_state_space(self, num_pts, alpha, beta):
        '''
            Y_t = alpha*Y_{t-1}+\eps_t
            \eps_t = beta*\eps_{t-1}+v_t
            v_t ~ N(0,1)
            So X_t = Y_{t-1}, f(X_t) = alpha*X_t
            If t = 0:
                X_t = 0, Y_t=\eps_t = v_t
        '''
        v0 = torch.randn(1)
        Y, X, fX, eps = [v0], [torch.zeros(1)], [torch.zeros(1)], [v0]
        scale = torch.sqrt(torch.ones(1)*0.1)
        for _ in range(num_pts-1):
            vt = torch.randn(1)*scale
            X.append(Y[-1])
            fX.append(alpha*Y[-1])
            eps.append(beta*eps[-1]+vt)
            Y.append(fX[-1]+eps[-1])
        Y, X, fX, eps = torch.hstack(Y), torch.vstack(
            X), torch.vstack(fX), torch.hstack(eps)
        return {'Y': Y.float(), 'X': X.float(), 'f(X)': fX, 'Eps': eps}

    def simulation_non_stationary(self):
        with open(f'Data_nochangepts_nonlinear.p', 'rb') as fp:
            Data_dc_old = pickle.load(fp)
        fXold = np.array(Data_dc_old['f(X)'])
        gX = non_stationarity(len(fXold))
        fXnew = gX*fXold
        # for _ in ['quick_plot']:
        #     fig, ax = plt.subplots(figsize=(12, 3))
        #     ax.plot(fXold, label='old f(X)')
        #     ax.plot(fXnew, label='new f(X)')
        #     ax.legend()
        Data_dc_new = {}
        for key in Data_dc_old.keys():
            if key == 'Y':
                continue
            if key == 'X':
                Data_dc_new[key] = np.c_[
                    np.arange(Data_dc_old[key].shape[0]) % 12, Data_dc_old[key]]
            elif key == 'f(X)':
                Data_dc_new[key] = fXnew
            else:
                Data_dc_new[key] = Data_dc_old[key]
        Data_dc_new['Y'] = np.array(Data_dc_new['f(X)'])+np.array(Data_dc_new['Eps'])
        Data_dc_old['Y'] = Data_dc_new['Y']
        Data_dc_old['f(X)'] = Data_dc_new['f(X)']
        # return Data_dc_old, Data_dc_new
        return Data_dc_new

    def simultaion_heteroskedastic(self):
        ''' Note, the difference from earlier case 3 in paper is that
            1) I reduce d from 100 to 20,
            2) I let X to be different, so sigmaX differs
                The sigmaX is a linear model so this effect in X is immediate
            I keep the same AR(1) eps & everything else.'''
        def True_mod_nonlinear_pre(feature):
            '''
            Input:
            Output:
            Description:
                f(feature): R^d -> R
            '''
            # Attempt 3 Nonlinear model:
            # f(X)=sqrt(1+(beta^TX)+(beta^TX)^2+(beta^TX)^3), where 1 is added in case beta^TX is zero
            d = len(feature)
            np.random.seed(0)
            # e.g. 20% of the entries are NON-missing
            beta1 = random(1, d, density=0.2).A
            betaX = np.abs(beta1.dot(feature))
            return (betaX + betaX**2 + betaX**3)**(1/4)
        Tot, d = 1000, 20
        Fmap = True_mod_nonlinear_pre
        # Multiply each random feature by exponential component, which is repeated every Tot/365 elements
        mult = np.exp(0.01*np.mod(np.arange(Tot), 100))
        X = np.random.rand(Tot, d)*mult.reshape(-1, 1)
        fX = np.array([Fmap(x) for x in X]).flatten()
        beta_Sigma = np.ones(d)
        sigmaX = np.maximum(X.dot(beta_Sigma).T, 0)
        with open(f'Data_nochangepts_nonlinear.p', 'rb') as fp:
            Data_dc = pickle.load(fp)
        eps = np.array(Data_dc['Eps'])
        Y = fX + sigmaX*eps[:Tot]
        np.random.seed(1103)
        idx = np.random.choice(Tot, Tot, replace=False)
        Y, X, fX, sigmaX, eps = Y[idx], X[idx], fX[idx], sigmaX[idx], eps[idx]
        return {'Y': torch.from_numpy(Y).float(), 'X': torch.from_numpy(X).float(), 'f(X)': fX, 'sigma(X)': sigmaX, 'Eps': eps}


''' Data Helpers '''


def non_stationarity(N):
    '''
        Compute g(t)=t'*sin(2*pi*t'/12), which is multiplicative on top of f(X), where
        t' = t mod 12 (for seaonality)
    '''
    cycle = 12
    trange = np.arange(N)
    tprime = trange % cycle
    term2 = np.sin(2*np.pi*tprime/cycle)
    return tprime*term2


def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def mySim():
    total_steps = 10000
    window_size = 5
    seq = np.zeros(total_steps+window_size)
    for i in range(total_steps):
        t = i+window_size
        t_ = np.mod(t, 12) + 1
        gt = np.log(t_) * np.sin(2*np.pi*t_ / 12)
        
        