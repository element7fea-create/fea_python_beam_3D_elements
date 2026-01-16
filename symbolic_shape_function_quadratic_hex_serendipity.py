#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sympy
import numpy as np

xi, eta, phi = sympy.symbols(['xi','eta','phi'])

coords = sympy.Array([[-1,-1,-1],
                      [1,-1,-1],
                      [1,1,-1],
                      [-1,1,-1],
                      [-1,-1,1],
                    [1,-1,1],
                    [1,1,1],
                    [-1,1,1],
                    [0,-1,-1],
                    [1,0,-1],
                    [0,1,-1],
                    [-1,0,-1],
                    [0,-1,1],
                    [1,0,1],
                    [0,1,1],
                    [-1,0,1],
                    [-1,-1,0],
                    [1,-1,0],
                    [1,1,0],
                    [-1,1,0]
                    ])

def val(x,y,z):
    return sympy.Array([1, 
              x, y, z, 
              x*y, y*z, z*x, 
              x**2, y**2, z**2,
              x**2*y, y**2*z, z**2*x, 
              y**2*x,  y*z**2, x**2*z,
              x**2*y*z, y**2*x*z, z**2*x*y, 
              x*y*z])

coeffs = sympy.zeros(20,20)

for i in range(0,20):
    coeffs[:,i] = val(coords[i,0], coords[i,1], coords[i,2])
    
vector = sympy.Matrix([1, 
          xi, eta , phi , 
          xi*eta , eta *phi , phi *xi, 
          xi**2, eta**2, phi **2,
          xi**2*eta , eta**2*phi , phi **2*xi, 
          eta**2*xi, eta*phi**2, xi**2*phi,
          xi**2*eta *phi , eta **2*xi*phi , phi **2*xi*eta ,
          xi*eta*phi ])

N = (coeffs.inv()*vector).transpose()

dN_dxi = N.diff(xi)
dN_deta = N.diff(eta)
dN_dphi = N.diff(phi)

dN = np.vstack((dN_dxi, dN_deta, dN_dphi))
