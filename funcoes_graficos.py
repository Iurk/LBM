#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:31:44 2019

@author: iurk
"""
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def grafico(u, step):
    
    file = 'vel.' + str(step) + '.png'
    path = 'Simulacao' + '/%s' % file
    plt.clf()
    plt.imshow(u, cmap=cm.RdBu, interpolation='nearest')
    plt.colorbar(cmap=cm.RdBu)
    plt.savefig(path, dpi=250)
    
def images(u, ims):
    ims.append([plt.imshow(u, cmap='RdBu', animated=True, interpolation='nearest')])
    return ims

def create_animation():
    fig = plt.figure()
    ims = []
    return fig, ims

def save_animation(fig, ims):
    writer = animation.writers['ffmpeg'](fps=60)
    ani = animation.ArtistAnimation(fig, ims)
    ani.save('simulation.mp4', writer=writer, dpi=500)