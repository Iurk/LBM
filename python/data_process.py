#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:27:54 2020

@author: iurk
"""
import yaml
import numpy as np
from os import walk
import utilidades as util
import funcoes_graficos as fg

main = "./bin"
fileyaml = "./bin/dados.yml"
velocity = "Velocity"


datafile = open(fileyaml)
data = yaml.load(datafile, Loader=yaml.FullLoader)
datafile.close()

Nx = data['domain']['Nx']
Ny = data['domain']['Ny']

Steps = data['simulation']['NSTEPS']
Saves = data['simulation']['NSAVE']
digitos = len(str(Steps))

idx_files = ["%0{}d".format(digitos) % i for i in range(0, Steps+Saves, Saves)]

variables = ["rho", "ux", "uy"]
results = "./bin/Results/"

rho_files = []
ux_files = []
uy_files = []
dic = {"rho": rho_files, "ux":ux_files, "uy":uy_files}

for var in variables:
    path = results + var
    
    for root, dirs, files in walk(path):
        for file in sorted(files):
            path_full = path + "/%s" % file
            dic[var].append(path_full)
    
pasta_img = util.criar_pasta('Images', folder=velocity, main_root=main)
pasta_stream = util.criar_pasta('Stream', folder=velocity, main_root=main)

x = np.arange(1, Nx+1, 1)
y = np.arange(1, Ny+1, 1)

rho = np.empty((Ny, Nx))
ux = np.empty_like(rho)
uy = np.empty_like(rho)

print("Plotting...")
for i in range(len(idx_files)):
    rho = np.fromfile(rho_files[i]).reshape(Ny, Nx)
    ux = np.fromfile(ux_files[i]).reshape(Ny, Nx)
    uy = np.fromfile(uy_files[i]).reshape(Ny, Nx)
    
    u_mod = np.sqrt(ux**2 + uy**2)
    
    fg.grafico(u_mod, idx_files[i], pasta_img)
    fg.stream(x, y, ux, uy, u_mod, idx_files[i], pasta_stream)

print('Animating...')
fg.animation('Velocidade', './', pasta_img)
fg.animation('Stream', './', pasta_stream)
print('Done!')
