#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:31:44 2019

@author: iurk
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def grafico(u, step):
    
    file = 'vel.' + str(step) + '.png'
    path = 'Simulacao' + '/%s' % file
    image = plot(u)
    image.savefig(path, dpi=250)
    
def plot(u):
    graph = plt.figure()
    plt.imshow(u, cmap='RdBu', animated=True, interpolation='nearest')
    plt.colorbar(cmap='RdBu')
    return graph
    
def images(u, ims):
    im = plot(u)
    ims.append([im])
    return ims

def create_animation():
    fig = plt.figure()
    ims = []
    return fig, ims

def save_animation(fig, ims):
    writer = animation.writers['ffmpeg'](fps=30)
    ani = animation.ArtistAnimation(fig, ims)
    ani.save('simulation.mp4', writer=writer, dpi=500)