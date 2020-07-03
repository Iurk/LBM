#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:49:54 2020

@author: iurk
"""
import yaml
import numpy as np

def generate_solid(Nx, Ny, Cx, Cy, D):
    solid = np.zeros(Nx*Ny, dtype=bool)
    for y in range(Ny):
        for x in range(Nx):
            if (x - Cx)**2 + (y - Cy)**2 <= (D/2)**2:
                solid[Nx*y + x] = True
    return solid

def fluid_wall(Nx, Ny, q, ex, ey, solid):
    fluid = np.zeros(Nx*Ny, dtype=bool)
    for y in range(Ny):
        for x in range(Nx):
            if (solid[Nx*y + x]):
                for n in range(q):
                    x_next = x + ex[n]
                    y_next = y + ey[n]
                    if not solid[Nx*y_next + x_next]:
                        fluid[Nx*y_next + x_next] = True
    return fluid
    

Sim_yaml = "./bin/dados.yml"
Lattices_yaml = "./bin/lattices.yml"

simulation_data = open(Sim_yaml)
simulation = yaml.load(simulation_data, Loader=yaml.FullLoader)

lattices_data = open(Lattices_yaml)
lattices = yaml.load(lattices_data, Loader=yaml.FullLoader)

Nx = simulation['domain']['Nx']
Ny = simulation['domain']['Ny']

Cx = simulation['cylinder']['Cx']
Cy = simulation['cylinder']['Cy']
D = simulation['cylinder']['D']

lattice = simulation['simulation']['lattice']

q = lattices[lattice]['q']
ex = lattices[lattice]['ex']
ey = lattices[lattice]['ey']

Cx = eval(Cx)
Cy = eval(Cy)
           
cylinder = generate_solid(Nx, Ny, Cx, Cy, D)
fluid = fluid_wall(Nx, Ny, q, ex, ey, cylinder)
        
solid_file = open('./bin/Mesh/mesh.bin', 'wb')
fluid_file = open('./bin/Mesh/fluid_wall.bin', 'wb')

solid_file.write(bytearray(cylinder))
fluid_file.write(bytearray(fluid))

simulation_data.close()
lattices_data.close()
solid_file.close()
fluid_file.close()
