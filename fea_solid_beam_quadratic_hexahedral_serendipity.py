#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-


import numpy as np
import scipy as sp
import pyvista as pv
import matplotlib.pyplot as plt

# geometry
xmin, xmax = (0, 5)
ymin, ymax = (0, 0.2)
zmin, zmax = (0, 0.3)

# material properties
E = 2e9
nu = 0.3

C = E/((1+nu)*(1-2*nu))*np.array([[1-nu,nu,nu,0,0,0],
                                  [nu,1-nu,nu,0,0,0],
                                  [nu,nu,1-nu,0,0,0],
                                  [0,0,0,0.5-nu,0,0],
                                  [0,0,0,0,0.5-nu,0],
                                  [0,0,0,0,0,0.5-nu]])

# generate mesh
dx = 2.5
dy = 0.1
dz = 0.15

nx = int(np.floor((xmax-xmin)/dx))+1
ny = int(np.floor((ymax-ymin)/dy))+1
nz = int(np.floor((zmax-zmin)/dz))+1

xpts = np.linspace(xmin, xmax, nx)
ypts = np.linspace(ymin, ymax, ny)
zpts = np.linspace(zmin, zmax, nz)

nodesmeshgrid = np.meshgrid(ypts, zpts, xpts)
nodes = np.zeros((nx*ny*nz,3))
nodes[:,0] = nodesmeshgrid[2].ravel()
nodes[:,1] = nodesmeshgrid[0].ravel()
nodes[:,2] = nodesmeshgrid[1].ravel()

elements = np.zeros(((nx-1)*(ny-1)*(nz-1),8), dtype = int)
nels = 0

# i = z-index j = y-index, k = x-index
for i in range(0, nz-1):
    for j in range(0, ny-1):
        for k in range(0,nx-1):
            n0 = k+j*nx+i*nx*ny
            n1 = n0+1
            n2 = n1+nx
            n3 = n0 + nx
            n4 = n0+nx*ny
            n5 = n4+1
            n6 = n5+nx
            n7 = n4+nx
            
            elements[nels,:] = [n0, n1,n2,n3,n4,n5,n6,n7]
            nels += 1


# convert to quadratic
midpoint_sum = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],dtype=int)
nodes2 = np.zeros((nels*20,3))
for i in range(0, nels):
    nds = elements[i,:]
    coords = nodes[nds,:]
    nodes2[i*20:i*20+8,:] = coords
    for j in range(0,12):
        nodes2[i*20+8+j,:] = 1/2*(coords[midpoint_sum[j,0],:]+coords[midpoint_sum[j,1],:])
 
nodes2_rounded = np.round(nodes2, decimals=6)
_, unique_indices, atob = np.unique(nodes2_rounded, axis=0, return_index=True, return_inverse=True)    
nodes = nodes2[unique_indices]  
elements = atob.reshape(nels, 20)   

# integration points (underintegration with 2x2x2 Gauss points instead of full integration with 3x3x3 Gauss points)
v = 1.0/np.sqrt(3)
ips = np.array([[-v,1],[v,1]])

# element and index matrices
dof_per_element = elements.shape[1]*3
Kel = np.zeros((dof_per_element, dof_per_element, nels))
rowid = np.zeros((dof_per_element,dof_per_element,nels),dtype=int)
colid = np.zeros((dof_per_element,dof_per_element,nels),dtype=int)
    
for i in range(0,nels):
    nds = elements[i,:]
    dofs = np.zeros((dof_per_element,))
    dofs[0::3] = nds*3
    dofs[1::3] = nds*3+1
    dofs[2::3] = nds*3+2
    
    rowid[:,:,i]= np.tile(dofs,(dof_per_element,1))
    colid[:,:,i] = np.transpose(rowid[:,:,i])
    
    coords = nodes[nds,:]
    for j in range(0,ips.shape[0]):
        xi = ips[j,0]
        w1 = ips[j,1]
        for k in range(0,ips.shape[0]):
            eta = ips[k,0]
            w2 = ips[k,1]
            for l in range(0,ips.shape[0]):
                phi = ips[l,0]
                w3 = ips[l,1]
                w = w1*w2*w3
                
                dN = [[eta**2*phi/8 - eta**2/8 + eta*phi**2/8 + eta*phi*xi/4 - eta*phi/8 - eta*xi/4 - phi**2/8 - phi*xi/4 + xi/4 + 1/8,
                        -eta**2*phi/8 + eta**2/8 - eta*phi**2/8 + eta*phi*xi/4 + eta*phi/8 - eta*xi/4 + phi**2/8 - phi*xi/4 + xi/4 - 1/8,
                        -eta**2*phi/8 + eta**2/8 + eta*phi**2/8 - eta*phi*xi/4 - eta*phi/8 + eta*xi/4 + phi**2/8 - phi*xi/4 + xi/4 - 1/8,
                        eta**2*phi/8 - eta**2/8 - eta*phi**2/8 - eta*phi*xi/4 + eta*phi/8 + eta*xi/4 - phi**2/8 - phi*xi/4 + xi/4 + 1/8,
                        -eta**2*phi/8 - eta**2/8 + eta*phi**2/8 - eta*phi*xi/4 + eta*phi/8 - eta*xi/4 - phi**2/8 + phi*xi/4 + xi/4 + 1/8,
                        eta**2*phi/8 + eta**2/8 - eta*phi**2/8 - eta*phi*xi/4 - eta*phi/8 - eta*xi/4 + phi**2/8 + phi*xi/4 + xi/4 - 1/8,
                        eta**2*phi/8 + eta**2/8 + eta*phi**2/8 + eta*phi*xi/4 + eta*phi/8 + eta*xi/4 + phi**2/8 + phi*xi/4 + xi/4 - 1/8,
                        -eta**2*phi/8 - eta**2/8 - eta*phi**2/8 + eta*phi*xi/4 - eta*phi/8 + eta*xi/4 - phi**2/8 + phi*xi/4 + xi/4 + 1/8,
                        -eta*phi*xi/2 + eta*xi/2 + phi*xi/2 - xi/2,
                        eta**2*phi/4 - eta**2/4 - phi/4 + 1/4,
                        eta*phi*xi/2 - eta*xi/2 + phi*xi/2 - xi/2,
                        -eta**2*phi/4 + eta**2/4 + phi/4 - 1/4,
                        eta*phi*xi/2 + eta*xi/2 - phi*xi/2 - xi/2,
                        -eta**2*phi/4 - eta**2/4 + phi/4 + 1/4,
                        -eta*phi*xi/2 - eta*xi/2 - phi*xi/2 - xi/2,
                        eta**2*phi/4 + eta**2/4 - phi/4 - 1/4,
                        -eta*phi**2/4 + eta/4 + phi**2/4 - 1/4,
                        eta*phi**2/4 - eta/4 - phi**2/4 + 1/4,
                        -eta*phi**2/4 + eta/4 - phi**2/4 + 1/4,
                        eta*phi**2/4 - eta/4 + phi**2/4 - 1/4],
                       [eta*phi*xi/4 - eta*phi/4 - eta*xi/4 + eta/4 + phi**2*xi/8 - phi**2/8 + phi*xi**2/8 - phi*xi/8 - xi**2/8 + 1/8,
                        -eta*phi*xi/4 - eta*phi/4 + eta*xi/4 + eta/4 - phi**2*xi/8 - phi**2/8 + phi*xi**2/8 + phi*xi/8 - xi**2/8 + 1/8,
                        -eta*phi*xi/4 - eta*phi/4 + eta*xi/4 + eta/4 + phi**2*xi/8 + phi**2/8 - phi*xi**2/8 - phi*xi/8 + xi**2/8 - 1/8,
                        eta*phi*xi/4 - eta*phi/4 - eta*xi/4 + eta/4 - phi**2*xi/8 + phi**2/8 - phi*xi**2/8 + phi*xi/8 + xi**2/8 - 1/8,
                        -eta*phi*xi/4 + eta*phi/4 - eta*xi/4 + eta/4 + phi**2*xi/8 - phi**2/8 - phi*xi**2/8 + phi*xi/8 - xi**2/8 + 1/8,
                        eta*phi*xi/4 + eta*phi/4 + eta*xi/4 + eta/4 - phi**2*xi/8 - phi**2/8 - phi*xi**2/8 - phi*xi/8 - xi**2/8 + 1/8,
                        eta*phi*xi/4 + eta*phi/4 + eta*xi/4 + eta/4 + phi**2*xi/8 + phi**2/8 + phi*xi**2/8 + phi*xi/8 + xi**2/8 - 1/8,
                        -eta*phi*xi/4 + eta*phi/4 - eta*xi/4 + eta/4 - phi**2*xi/8 + phi**2/8 + phi*xi**2/8 - phi*xi/8 + xi**2/8 - 1/8,
                        -phi*xi**2/4 + phi/4 + xi**2/4 - 1/4,
                        eta*phi*xi/2 + eta*phi/2 - eta*xi/2 - eta/2,
                        phi*xi**2/4 - phi/4 - xi**2/4 + 1/4,
                        -eta*phi*xi/2 + eta*phi/2 + eta*xi/2 - eta/2,
                        phi*xi**2/4 - phi/4 + xi**2/4 - 1/4,
                        -eta*phi*xi/2 - eta*phi/2 - eta*xi/2 - eta/2,
                        -phi*xi**2/4 + phi/4 - xi**2/4 + 1/4,
                        eta*phi*xi/2 - eta*phi/2 + eta*xi/2 - eta/2,
                        -phi**2*xi/4 + phi**2/4 + xi/4 - 1/4,
                        phi**2*xi/4 + phi**2/4 - xi/4 - 1/4,
                        -phi**2*xi/4 - phi**2/4 + xi/4 + 1/4,
                        phi**2*xi/4 - phi**2/4 - xi/4 + 1/4],
                       [eta**2*xi/8 - eta**2/8 + eta*phi*xi/4 - eta*phi/4 + eta*xi**2/8 - eta*xi/8 - phi*xi/4 + phi/4 - xi**2/8 + 1/8,
                        -eta**2*xi/8 - eta**2/8 - eta*phi*xi/4 - eta*phi/4 + eta*xi**2/8 + eta*xi/8 + phi*xi/4 + phi/4 - xi**2/8 + 1/8,
                        -eta**2*xi/8 - eta**2/8 + eta*phi*xi/4 + eta*phi/4 - eta*xi**2/8 - eta*xi/8 + phi*xi/4 + phi/4 - xi**2/8 + 1/8,
                        eta**2*xi/8 - eta**2/8 - eta*phi*xi/4 + eta*phi/4 - eta*xi**2/8 + eta*xi/8 - phi*xi/4 + phi/4 - xi**2/8 + 1/8,
                        -eta**2*xi/8 + eta**2/8 + eta*phi*xi/4 - eta*phi/4 - eta*xi**2/8 + eta*xi/8 - phi*xi/4 + phi/4 + xi**2/8 - 1/8,
                        eta**2*xi/8 + eta**2/8 - eta*phi*xi/4 - eta*phi/4 - eta*xi**2/8 - eta*xi/8 + phi*xi/4 + phi/4 + xi**2/8 - 1/8,
                        eta**2*xi/8 + eta**2/8 + eta*phi*xi/4 + eta*phi/4 + eta*xi**2/8 + eta*xi/8 + phi*xi/4 + phi/4 + xi**2/8 - 1/8,
                        -eta**2*xi/8 + eta**2/8 - eta*phi*xi/4 + eta*phi/4 + eta*xi**2/8 - eta*xi/8 - phi*xi/4 + phi/4 + xi**2/8 - 1/8,
                        -eta*xi**2/4 + eta/4 + xi**2/4 - 1/4,
                        eta**2*xi/4 + eta**2/4 - xi/4 - 1/4,
                        eta*xi**2/4 - eta/4 + xi**2/4 - 1/4,
                        -eta**2*xi/4 + eta**2/4 + xi/4 - 1/4,
                        eta*xi**2/4 - eta/4 - xi**2/4 + 1/4,
                        -eta**2*xi/4 - eta**2/4 + xi/4 + 1/4,
                        -eta*xi**2/4 + eta/4 - xi**2/4 + 1/4,
                        eta**2*xi/4 - eta**2/4 - xi/4 + 1/4,
                        -eta*phi*xi/2 + eta*phi/2 + phi*xi/2 - phi/2,
                        eta*phi*xi/2 + eta*phi/2 - phi*xi/2 - phi/2,
                        -eta*phi*xi/2 - eta*phi/2 - phi*xi/2 - phi/2,
                        eta*phi*xi/2 - eta*phi/2 + phi*xi/2 - phi/2]]
                
                matjac = dN@coords
                dV = np.linalg.det(matjac)*w
                
                dcoords = np.linalg.solve(matjac, dN)
                
                B = np.zeros((6,dof_per_element))
                
                B[0,0::3] = dcoords[0,:]
                B[1,1::3] = dcoords[1,:]
                B[2,2::3] = dcoords[2,:]
                
                B[3,0::3] = dcoords[1,:]
                B[3,1::3] = dcoords[0,:]
                
                B[4,1::3] = dcoords[2,:]
                B[4,2::3] = dcoords[1,:]
                
                B[5,0::3] = dcoords[2,:]
                B[5,2::3] = dcoords[0,:]
                
                Kel[:,:,i] += B.transpose()@C@B*dV
        
# assemble to global
numentries = Kel.shape[0]*Kel.shape[1] *Kel.shape[2]
data = Kel.reshape(numentries,)
row = rowid.reshape(numentries,)
col = colid.reshape(numentries,)

Kgcoo = sp.sparse.coo_matrix((data,(row,col)), (nodes.shape[0]*3, nodes.shape[0]*3))

Kg = Kgcoo.tocsr()

# boundary conditions
bcs = np.zeros((nodes.shape[0]*3,), dtype = bool)
bcnds = np.where(nodes[:,0] < 1e-6)[0]
bcs[bcnds*3] = True
bcs[bcnds*3+1] = True
bcs[bcnds*3+2] = True
#bcs[bcnds[-1]*3+2] = True

# loads
P = -100
loads = np.zeros((nodes.shape[0]*3,))
ldnds = np.where(np.abs(nodes[:,0]-xmax)<1e-6)[0]
loads[ldnds*3+2] = P/ldnds.size

# solve Ku = F
Kgbc = Kg[~bcs,:][:,~bcs]
u = np.zeros((nodes.shape[0]*3,))
u[~bcs] = sp.sparse.linalg.spsolve(Kgbc, loads[~bcs])

displacement_vectors = u.reshape((nodes.shape[0], 3))

# theoretical deflection

L = xmax - xmin
b = ymax - ymin
h = zmax - zmin

I = b*h**3/12
uth = P*L**3/3/E/I

ufea = np.min(displacement_vectors[:,2])

percentage_error = (ufea-uth)/uth*100

print(f'displacement based on Euler Bernoulli beam is, {uth:.7f} m')
print(f'displacement based on fea is, {uth:.7f}m')

print(f'percentage error is, {percentage_error:.7f}%')

# stress calculation
stresses = np.zeros((6,8,nels))
nodal_stresses = np.zeros((nodes.shape[0],6))
nodal_stresses_count = np.zeros((nodes.shape[0],))

    
for i in range(0,nels):
    nds = elements[i,:]
    dofs = np.zeros((dof_per_element,),dtype = int)
    dofs[0::3] = nds*3
    dofs[1::3] = nds*3+1
    dofs[2::3] = nds*3+2
    
    rowid[:,:,i]= np.tile(dofs,(dof_per_element,1))
    colid[:,:,i] = np.transpose(rowid[:,:,i])
    
    coords = nodes[nds,:]
    
    udofs = u[dofs].reshape((dof_per_element,1))
    
    ipcount = 0
    #nodal_stresses_count[nds] += 1
    for j in range(0,ips.shape[0]):
        xi = ips[j,0]
        w1 = ips[j,1]
        xip = 1/xi
        for k in range(0,ips.shape[0]):
            eta = ips[k,0]
            w2 = ips[k,1]
            etap = 1/eta
            for l in range(0,ips.shape[0]):
                phi = ips[l,0]
                phip = 1/phi
                w3 = ips[l,1]
                w = w1*w2*w3
                
                dN = [[eta**2*phi/8 - eta**2/8 + eta*phi**2/8 + eta*phi*xi/4 - eta*phi/8 - eta*xi/4 - phi**2/8 - phi*xi/4 + xi/4 + 1/8,
                        -eta**2*phi/8 + eta**2/8 - eta*phi**2/8 + eta*phi*xi/4 + eta*phi/8 - eta*xi/4 + phi**2/8 - phi*xi/4 + xi/4 - 1/8,
                        -eta**2*phi/8 + eta**2/8 + eta*phi**2/8 - eta*phi*xi/4 - eta*phi/8 + eta*xi/4 + phi**2/8 - phi*xi/4 + xi/4 - 1/8,
                        eta**2*phi/8 - eta**2/8 - eta*phi**2/8 - eta*phi*xi/4 + eta*phi/8 + eta*xi/4 - phi**2/8 - phi*xi/4 + xi/4 + 1/8,
                        -eta**2*phi/8 - eta**2/8 + eta*phi**2/8 - eta*phi*xi/4 + eta*phi/8 - eta*xi/4 - phi**2/8 + phi*xi/4 + xi/4 + 1/8,
                        eta**2*phi/8 + eta**2/8 - eta*phi**2/8 - eta*phi*xi/4 - eta*phi/8 - eta*xi/4 + phi**2/8 + phi*xi/4 + xi/4 - 1/8,
                        eta**2*phi/8 + eta**2/8 + eta*phi**2/8 + eta*phi*xi/4 + eta*phi/8 + eta*xi/4 + phi**2/8 + phi*xi/4 + xi/4 - 1/8,
                        -eta**2*phi/8 - eta**2/8 - eta*phi**2/8 + eta*phi*xi/4 - eta*phi/8 + eta*xi/4 - phi**2/8 + phi*xi/4 + xi/4 + 1/8,
                        -eta*phi*xi/2 + eta*xi/2 + phi*xi/2 - xi/2,
                        eta**2*phi/4 - eta**2/4 - phi/4 + 1/4,
                        eta*phi*xi/2 - eta*xi/2 + phi*xi/2 - xi/2,
                        -eta**2*phi/4 + eta**2/4 + phi/4 - 1/4,
                        eta*phi*xi/2 + eta*xi/2 - phi*xi/2 - xi/2,
                        -eta**2*phi/4 - eta**2/4 + phi/4 + 1/4,
                        -eta*phi*xi/2 - eta*xi/2 - phi*xi/2 - xi/2,
                        eta**2*phi/4 + eta**2/4 - phi/4 - 1/4,
                        -eta*phi**2/4 + eta/4 + phi**2/4 - 1/4,
                        eta*phi**2/4 - eta/4 - phi**2/4 + 1/4,
                        -eta*phi**2/4 + eta/4 - phi**2/4 + 1/4,
                        eta*phi**2/4 - eta/4 + phi**2/4 - 1/4],
                       [eta*phi*xi/4 - eta*phi/4 - eta*xi/4 + eta/4 + phi**2*xi/8 - phi**2/8 + phi*xi**2/8 - phi*xi/8 - xi**2/8 + 1/8,
                        -eta*phi*xi/4 - eta*phi/4 + eta*xi/4 + eta/4 - phi**2*xi/8 - phi**2/8 + phi*xi**2/8 + phi*xi/8 - xi**2/8 + 1/8,
                        -eta*phi*xi/4 - eta*phi/4 + eta*xi/4 + eta/4 + phi**2*xi/8 + phi**2/8 - phi*xi**2/8 - phi*xi/8 + xi**2/8 - 1/8,
                        eta*phi*xi/4 - eta*phi/4 - eta*xi/4 + eta/4 - phi**2*xi/8 + phi**2/8 - phi*xi**2/8 + phi*xi/8 + xi**2/8 - 1/8,
                        -eta*phi*xi/4 + eta*phi/4 - eta*xi/4 + eta/4 + phi**2*xi/8 - phi**2/8 - phi*xi**2/8 + phi*xi/8 - xi**2/8 + 1/8,
                        eta*phi*xi/4 + eta*phi/4 + eta*xi/4 + eta/4 - phi**2*xi/8 - phi**2/8 - phi*xi**2/8 - phi*xi/8 - xi**2/8 + 1/8,
                        eta*phi*xi/4 + eta*phi/4 + eta*xi/4 + eta/4 + phi**2*xi/8 + phi**2/8 + phi*xi**2/8 + phi*xi/8 + xi**2/8 - 1/8,
                        -eta*phi*xi/4 + eta*phi/4 - eta*xi/4 + eta/4 - phi**2*xi/8 + phi**2/8 + phi*xi**2/8 - phi*xi/8 + xi**2/8 - 1/8,
                        -phi*xi**2/4 + phi/4 + xi**2/4 - 1/4,
                        eta*phi*xi/2 + eta*phi/2 - eta*xi/2 - eta/2,
                        phi*xi**2/4 - phi/4 - xi**2/4 + 1/4,
                        -eta*phi*xi/2 + eta*phi/2 + eta*xi/2 - eta/2,
                        phi*xi**2/4 - phi/4 + xi**2/4 - 1/4,
                        -eta*phi*xi/2 - eta*phi/2 - eta*xi/2 - eta/2,
                        -phi*xi**2/4 + phi/4 - xi**2/4 + 1/4,
                        eta*phi*xi/2 - eta*phi/2 + eta*xi/2 - eta/2,
                        -phi**2*xi/4 + phi**2/4 + xi/4 - 1/4,
                        phi**2*xi/4 + phi**2/4 - xi/4 - 1/4,
                        -phi**2*xi/4 - phi**2/4 + xi/4 + 1/4,
                        phi**2*xi/4 - phi**2/4 - xi/4 + 1/4],
                       [eta**2*xi/8 - eta**2/8 + eta*phi*xi/4 - eta*phi/4 + eta*xi**2/8 - eta*xi/8 - phi*xi/4 + phi/4 - xi**2/8 + 1/8,
                        -eta**2*xi/8 - eta**2/8 - eta*phi*xi/4 - eta*phi/4 + eta*xi**2/8 + eta*xi/8 + phi*xi/4 + phi/4 - xi**2/8 + 1/8,
                        -eta**2*xi/8 - eta**2/8 + eta*phi*xi/4 + eta*phi/4 - eta*xi**2/8 - eta*xi/8 + phi*xi/4 + phi/4 - xi**2/8 + 1/8,
                        eta**2*xi/8 - eta**2/8 - eta*phi*xi/4 + eta*phi/4 - eta*xi**2/8 + eta*xi/8 - phi*xi/4 + phi/4 - xi**2/8 + 1/8,
                        -eta**2*xi/8 + eta**2/8 + eta*phi*xi/4 - eta*phi/4 - eta*xi**2/8 + eta*xi/8 - phi*xi/4 + phi/4 + xi**2/8 - 1/8,
                        eta**2*xi/8 + eta**2/8 - eta*phi*xi/4 - eta*phi/4 - eta*xi**2/8 - eta*xi/8 + phi*xi/4 + phi/4 + xi**2/8 - 1/8,
                        eta**2*xi/8 + eta**2/8 + eta*phi*xi/4 + eta*phi/4 + eta*xi**2/8 + eta*xi/8 + phi*xi/4 + phi/4 + xi**2/8 - 1/8,
                        -eta**2*xi/8 + eta**2/8 - eta*phi*xi/4 + eta*phi/4 + eta*xi**2/8 - eta*xi/8 - phi*xi/4 + phi/4 + xi**2/8 - 1/8,
                        -eta*xi**2/4 + eta/4 + xi**2/4 - 1/4,
                        eta**2*xi/4 + eta**2/4 - xi/4 - 1/4,
                        eta*xi**2/4 - eta/4 + xi**2/4 - 1/4,
                        -eta**2*xi/4 + eta**2/4 + xi/4 - 1/4,
                        eta*xi**2/4 - eta/4 - xi**2/4 + 1/4,
                        -eta**2*xi/4 - eta**2/4 + xi/4 + 1/4,
                        -eta*xi**2/4 + eta/4 - xi**2/4 + 1/4,
                        eta**2*xi/4 - eta**2/4 - xi/4 + 1/4,
                        -eta*phi*xi/2 + eta*phi/2 + phi*xi/2 - phi/2,
                        eta*phi*xi/2 + eta*phi/2 - phi*xi/2 - phi/2,
                        -eta*phi*xi/2 - eta*phi/2 - phi*xi/2 - phi/2,
                        eta*phi*xi/2 - eta*phi/2 + phi*xi/2 - phi/2]]
                
               # matjac = dN@coords
                #dV = np.linalg.det(matjac)*w
                
                dcoords = np.linalg.solve(matjac, dN)
                
                B = np.zeros((6,dof_per_element))
                
                B[0,0::3] = dcoords[0,:]
                B[1,1::3] = dcoords[1,:]
                B[2,2::3] = dcoords[2,:]
                
                B[3,0::3] = dcoords[1,:]
                B[3,1::3] = dcoords[0,:]
                
                B[4,1::3] = dcoords[2,:]
                B[4,2::3] = dcoords[1,:]
                
                B[5,0::3] = dcoords[2,:]
                B[5,2::3] = dcoords[0,:]
                
                stress = (C@B@udofs).flatten()
                stresses[:,ipcount,i] = stress
                ipcount += 1
                
                N = np.array([etap**2*phip*xip/8 - etap**2*phip/8 - etap**2*xip/8 + etap**2/8 + etap*phip**2*xip/8 - etap*phip**2/8 + etap*phip*xip**2/8 - etap*phip*xip/8 - etap*xip**2/8 + etap/8 - phip**2*xip/8 + phip**2/8 - phip*xip**2/8 + phip/8 + xip**2/8 + xip/8 - 1/4, -etap**2*phip*xip/8 - etap**2*phip/8 + etap**2*xip/8 + etap**2/8 - etap*phip**2*xip/8 - etap*phip**2/8 + etap*phip*xip**2/8 + etap*phip*xip/8 - etap*xip**2/8 + etap/8 + phip**2*xip/8 + phip**2/8 - phip*xip**2/8 + phip/8 + xip**2/8 - xip/8 - 1/4, -etap**2*phip*xip/8 - etap**2*phip/8 + etap**2*xip/8 + etap**2/8 + etap*phip**2*xip/8 + etap*phip**2/8 - etap*phip*xip**2/8 - etap*phip*xip/8 + etap*xip**2/8 - etap/8 + phip**2*xip/8 + phip**2/8 - phip*xip**2/8 + phip/8 + xip**2/8 - xip/8 - 1/4, etap**2*phip*xip/8 - etap**2*phip/8 - etap**2*xip/8 + etap**2/8 - etap*phip**2*xip/8 + etap*phip**2/8 - etap*phip*xip**2/8 + etap*phip*xip/8 + etap*xip**2/8 - etap/8 - phip**2*xip/8 + phip**2/8 - phip*xip**2/8 + phip/8 + xip**2/8 + xip/8 - 1/4, -etap**2*phip*xip/8 + etap**2*phip/8 - etap**2*xip/8 + etap**2/8 + etap*phip**2*xip/8 - etap*phip**2/8 - etap*phip*xip**2/8 + etap*phip*xip/8 - etap*xip**2/8 + etap/8 - phip**2*xip/8 + phip**2/8 + phip*xip**2/8 - phip/8 + xip**2/8 + xip/8 - 1/4, etap**2*phip*xip/8 + etap**2*phip/8 + etap**2*xip/8 + etap**2/8 - etap*phip**2*xip/8 - etap*phip**2/8 - etap*phip*xip**2/8 - etap*phip*xip/8 - etap*xip**2/8 + etap/8 + phip**2*xip/8 + phip**2/8 + phip*xip**2/8 - phip/8 + xip**2/8 - xip/8 - 1/4, etap**2*phip*xip/8 + etap**2*phip/8 + etap**2*xip/8 + etap**2/8 + etap*phip**2*xip/8 + etap*phip**2/8 + etap*phip*xip**2/8 + etap*phip*xip/8 + etap*xip**2/8 - etap/8 + phip**2*xip/8 + phip**2/8 + phip*xip**2/8 - phip/8 + xip**2/8 - xip/8 - 1/4, -etap**2*phip*xip/8 + etap**2*phip/8 - etap**2*xip/8 + etap**2/8 - etap*phip**2*xip/8 + etap*phip**2/8 + etap*phip*xip**2/8 - etap*phip*xip/8 + etap*xip**2/8 - etap/8 - phip**2*xip/8 + phip**2/8 + phip*xip**2/8 - phip/8 + xip**2/8 + xip/8 - 1/4, -etap*phip*xip**2/4 + etap*phip/4 + etap*xip**2/4 - etap/4 + phip*xip**2/4 - phip/4 - xip**2/4 + 1/4, etap**2*phip*xip/4 + etap**2*phip/4 - etap**2*xip/4 - etap**2/4 - phip*xip/4 - phip/4 + xip/4 + 1/4, etap*phip*xip**2/4 - etap*phip/4 - etap*xip**2/4 + etap/4 + phip*xip**2/4 - phip/4 - xip**2/4 + 1/4, -etap**2*phip*xip/4 + etap**2*phip/4 + etap**2*xip/4 - etap**2/4 + phip*xip/4 - phip/4 - xip/4 + 1/4, etap*phip*xip**2/4 - etap*phip/4 + etap*xip**2/4 - etap/4 - phip*xip**2/4 + phip/4 - xip**2/4 + 1/4, -etap**2*phip*xip/4 - etap**2*phip/4 - etap**2*xip/4 - etap**2/4 + phip*xip/4 + phip/4 + xip/4 + 1/4, -etap*phip*xip**2/4 + etap*phip/4 - etap*xip**2/4 + etap/4 - phip*xip**2/4 + phip/4 - xip**2/4 + 1/4, etap**2*phip*xip/4 - etap**2*phip/4 + etap**2*xip/4 - etap**2/4 - phip*xip/4 + phip/4 - xip/4 + 1/4, -etap*phip**2*xip/4 + etap*phip**2/4 + etap*xip/4 - etap/4 + phip**2*xip/4 - phip**2/4 - xip/4 + 1/4, etap*phip**2*xip/4 + etap*phip**2/4 - etap*xip/4 - etap/4 - phip**2*xip/4 - phip**2/4 + xip/4 + 1/4, -etap*phip**2*xip/4 - etap*phip**2/4 + etap*xip/4 + etap/4 - phip**2*xip/4 - phip**2/4 + xip/4 + 1/4, etap*phip**2*xip/4 - etap*phip**2/4 - etap*xip/4 + etap/4 + phip**2*xip/4 - phip**2/4 - xip/4 + 1/4])
                
                nodal_stresses[nds,0] += N*stress[0]
                nodal_stresses[nds,1] += N*stress[1]
                nodal_stresses[nds,2] += N*stress[2]
                nodal_stresses[nds,3] += N*stress[3]
                nodal_stresses[nds,4] += N*stress[4]
                nodal_stresses[nds,5] += N*stress[5]
                
                nodal_stresses_count[nds] += N
                
                
                

for i in range(0,nodes.shape[0]):
    for j in range(0,6):
        nodal_stresses[i,j] = nodal_stresses[i,j]/nodal_stresses_count[i]


# percentage error in nodal stress
sigma11_max_th = P*L/I*h/2
sigma11_max_fea = np.min(nodal_stresses[:,0])
percentage_error = (sigma11_max_fea-sigma11_max_th)/sigma11_max_th*100

print(f'Theoretical stress at fixed end, {sigma11_max_th:0.0f} Pa')
print(f'FEA stress (Avg) at fixed end, {sigma11_max_fea:0.0f} Pa')
print(f'Percentage error at fixed end, {percentage_error:0.3f}%')
    

pl = pv.Plotter()
pvelements = np.zeros((nels,21),dtype = int)
pvelements[:,0] = 20
pvelements[:,1:21] = elements

element_types = np.full(nels, pv.CellType.QUADRATIC_HEXAHEDRON, dtype = np.uint8)
grid = pv.UnstructuredGrid(pvelements.flatten(), element_types, nodes)

grid.point_data["Displacement"] = displacement_vectors
grid.point_data["Magnitude"] = np.linalg.norm(displacement_vectors, axis = 1)
grid.point_data["sigma_11"] = nodal_stresses[:,0]
grid.point_data["Displacement_X"] = displacement_vectors[:,0]
warp_factor = 50
warped_grid = grid.warp_by_vector("Displacement", factor = warp_factor)


#pl.add_mesh(grid, opacity = 0.8, show_edges=True, label = 'original', style = 'wireframe')
pl.add_mesh(warped_grid, scalars = 'sigma_11', cmap='jet', opacity = 0.8, label='deformed', show_edges=True)
pl.view_xz()
pl.show()


# Validation of stress at x = L/4

# Define the slice location
L = xmax - xmin
x_check = xmin + L / 4.0  # 1.25 m
tolerance = 1e-4          # Tolerance to find nodes at this x-coordinate

# Find nodes at x = 1.25
slice_indices = np.where(np.abs(nodes[:, 0] - x_check) < tolerance)[0]


# 3. Filter for Top Surface Nodes (z = zmax) to get max tensile stress
# In your geometry, zmax = 0.3. Neutral axis is at 0.15.
top_surface_indices = slice_indices[np.abs(nodes[slice_indices, 2] - zmax) < tolerance]

# 4. Extract Stress (Sigma_11 is index 0)
# Using the 'nodal_stresses' array calculated in the previous step
fea_stresses_at_slice = nodal_stresses[top_surface_indices, 0]
    
sigma_fea_slice_avg = np.mean(fea_stresses_at_slice)

# 5. Theoretical Calculation
P_val = 100 # Magnitude of load
moment_arm = (xmax - x_check)
M = P_val * moment_arm

b_beam = ymax - ymin
h_beam = zmax - zmin
I_beam = b_beam * h_beam**3 / 12
c_dist = h_beam / 2

sigma_th_slice = M * c_dist / I_beam

# 6. Comparison
error_slice = (sigma_fea_slice_avg - sigma_th_slice) / sigma_th_slice * 100


print(f"Theoretical Stress: {sigma_th_slice:.2f} Pa")
print(f"FEA Stress (Avg):   {sigma_fea_slice_avg:.2f} Pa")
print(f"Percentage error at L/4:       {error_slice:.4f} %")

    
    
