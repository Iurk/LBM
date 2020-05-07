#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:41:55 2020

@author: iurk
"""

class Lattice():
    
    def __init__(self, lattice):
        from numpy import array, empty, float32, int32, stack, sqrt
        
        if lattice == 'D2Q9':
            self.q = 9
            self.cs = float32(1/sqrt(3))
            self.W = empty(self.q, dtype=float32)
            
            self.W[0] = 16/36
            self.W[1:5] = 4/36
            self.W[5:self.q] = 1/36
            
            ex = array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=int32)
            ey = array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=int32)
            self.e = stack((ex, ey), axis=1)        # Maybe axis=0 !!!
            
            
    def get_attributes(self):
        return self.q, self.cs, self.W, self.e

