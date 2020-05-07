#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:54:48 2020

@author: iurk
"""
from numpy import zeros
class Simulation():
        
    def __init__(self, Nx, Ny, Cx, Cy, D):
        self.nx = Nx
        self.ny = Ny
        self.cx = Cx
        self.cy = Cy
        self.d = D
        
    def gpu_parameter(self):
        return self.nx*self.ny
    
    def initialize_u(self, uini):
        self.u = zeros((2, self.nx, self.ny))
        self.u[0, 0, :] = uini
        
        return self.u
        
    def solid_create(self):
        self.solid = zeros((self.nx, self.ny), dtype=bool)
        for y in range(self.ny):
            for x in range(self.nx):
                if (x - self.cx)**2 + (y - self.cy)**2 <= (self.d/2)**2:
                    self.solid[x, y] = True
                    
    def get_solid(self):
        return self.solid
            