#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:31:44 2019

@author: iurk
"""
from images2gif import writeGif
from matplotlib import cm
import matplotlib.pyplot as plt
from PIL import Image



def grafico(u, step, pasta_imagens):
    file = 'vel.' + str(step) + '.png'
    path = pasta_imagens + '/%s' % file
    plt.clf()
    plt.imshow(u, cmap=cm.RdBu, interpolation='nearest')
    plt.colorbar(cmap=cm.RdBu)
    plt.savefig(path, dpi=250)
    
def plot(u):
    return plt.imshow(u, cmap='RdBu', animated=True, interpolation='nearest')
    
def images(u, ims):
    plt.clf()
    ims.append([plot(u)])
    return ims

def create_animation():
    fig = plt.figure()
    ims = []
    return fig, ims

def animation(fig, ims, pasta, pasta_imagens):
    from os import listdir
    
    size = (1008, 1504)
    file = 'simulation.GIF'
    path = pasta + '/%s' % file
    
    files_imgs = [im for im in listdir(pasta_imagens) if im.endswith('.png')]
    files_imgs.sort(key=__ordenar)
    
    images = [Image.open(pasta_imagens + '/%s' % file) for file in files_imgs]
    writeGif(path, images, duration=0.2)
    
    
def __ordenar(item):
    return int(item.split('.')[1])