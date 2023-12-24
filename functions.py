import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import csv


# --------------------------------------------------------------------------------------------- #

def read_data(data):
    pos = []
    vel = []

    run = False
    data = data.split("\n")
    for row in data:
        if row == "$$SOE": 
            run=True
            continue
        if row == "$$EOE": run=False
        if run:
            pos.append(row.split(",")[2:5])
            vel.append(row.split(",")[5:8])

    return np.asarray(pos).astype(float), np.asarray(vel).astype(float)

# --------------------------------------------------------------------------------------------- #

def prep_data(data):
    q = np.array([])
    dq = np.array([])
    p = np.array([])
    dp = np.array([])

    for j in range(len(data[0][0])-1):
        for i in range(len(data)):

            q = np.append(q, data[i][0][j])
            dq = np.append(dq, data[i][0][j+1]-data[i][0][j])

            p = np.append(p, data[i][1][j])
            dp = np.append(dp, data[i][1][j+1]-data[i][1][j])

    shape = [len(data[i][0])-1, 3*len(data)]

    return q.reshape(shape), dq.reshape(shape), p.reshape(shape), dp.reshape(shape)

# --------------------------------------------------------------------------------------------- #

def HNN(x, model):
    H = model(x)

    # this is where the HNN magic happens!!
    x_field = torch.autograd.grad(H.sum(), x, create_graph=True, retain_graph=True)[0] # gradients for field
    dH = torch.zeros_like(x_field)

    # Hamiltonian equations
    # dq = dH/dp
    dH.T[0] = x_field.T[3]
    dH.T[1] = x_field.T[4]
    dH.T[2] = x_field.T[5]

    # dp = -dH/dq
    dH.T[3] = -x_field.T[0]
    dH.T[4] = -x_field.T[1]
    dH.T[5] = -x_field.T[2]

    return dH

# --------------------------------------------------------------------------------------------- #

def integrate_model(model, t_span, y0, t_eval, baseline=True):
    
    def fun(t, np_x, model=model, baseline=baseline):
        #x = torch.tensor( np_x, requires_grad=True).view(1,2)
        x = torch.tensor( np_x, requires_grad=True)
        if baseline:
            pred = model(x)
        else:
            pred = HNN(x, model)
            
        return pred.detach().numpy()

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, t_eval=t_eval)

# --------------------------------------------------------------------------------------------- #

def reshape_data(data):
    targets = int(data.y.shape[0]/(3*2))
    steps = data.y.shape[1]

    data_out = np.zeros([targets, 2, steps, 3])

    for i in range(steps):
        for j in range(targets):
            data_out[j][0][i][0] = data.y[3*j][i]
            data_out[j][0][i][1] = data.y[3*j+1][i]
            data_out[j][0][i][2] = data.y[3*j+2][i]

            data_out[j][1][i][0] = data.y[3*j+targets][i]
            data_out[j][1][i][1] = data.y[3*j+targets][i]
            data_out[j][1][i][2] = data.y[3*j+targets][i]

    return data_out

# --------------------------------------------------------------------------------------------- #

def plot_phase_space(data, target_names, title):
    fig, axs = plt.subplots(len(target_names),3, figsize=(15,5*len(target_names)))
    coords = ["x", "y", "z"]
    c=np.linspace(0,1,len(data[0][0]))

    if len(target_names) == 1:
        for j in range(0, 3):
            axs[j].scatter(data[0][0].T[j], data[0][1].T[j], s=2, c=c)
            axs[j].set_title(target_names[0] + " " + coords[j])
    else:
        for i in range(len(target_names)):
            for j in range(0, 3):
                axs[i, j].scatter(data[i][0].T[j], data[i][1].T[j], s=2, c=c)
                axs[i, j].set_title(target_names[i] + " " + coords[j])

    fig.suptitle(title)
    plt.show()

# --------------------------------------------------------------------------------------------- #
    
def plot_space(data, target_names, title, color=False):
    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(12)

    ax = fig.add_subplot(projection='3d')

    if color:
        c=np.linspace(0,1,len(data[0][0]))
        for i in range(len(target_names)):
            ax.scatter(data[i][0].T[0], data[i][0].T[1], data[i][0].T[2], s=1, label=target_names[i], c=c)

    else:
        for i in range(len(target_names)):
            ax.scatter(data[i][0].T[0], data[i][0].T[1], data[i][0].T[2], s=1, label=target_names[i])

    
    #ax.view_init(elev=0, roll=0, azim=300)
    fig.suptitle(title)
    plt.legend()
    plt.show()

# --------------------------------------------------------------------------------------------- #
    