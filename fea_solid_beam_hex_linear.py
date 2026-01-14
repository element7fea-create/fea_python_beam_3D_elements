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
dx = 0.05
dy = 0.05
dz = 0.05

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


# integration points
v = 1.0/np.sqrt(3)
ips = np.array([[-v,-v,-v,1],
    [v,-v,-v,1],
    [v,v,-v,1],
    [-v,v,-v,1],
    [-v,-v,v,1],
        [v,-v,v,1],
        [v,v,v,1],
        [-v,v,v,1]])

# element and index matrices
Kel = np.zeros((24,24,nels))
rowid = np.zeros((24,24,nels),dtype=int)
colid = np.zeros((24,24,nels),dtype=int)
    
for i in range(0,nels):
    nds = elements[i,:]
    dofs = np.zeros((24,))
    dofs[0::3] = nds*3
    dofs[1::3] = nds*3+1
    dofs[2::3] = nds*3+2
    
    rowid[:,:,i]= np.tile(dofs,(24,1))
    colid[:,:,i] = np.transpose(rowid[:,:,i])
    
    coords = nodes[nds,:]
    for j in range(0,ips.shape[0]):
        xi = ips[j,0]
        eta = ips[j,1]
        phi =ips[j,2]
        w = ips[j,3]
        dN_dxi = np.array([-(1 - eta)*(1 - phi), (1 - eta)*(1 - phi), (1 - phi)*(eta + 1), -(1 - phi)*(eta + 1), -(1 - eta)*(phi + 1), (1 - eta)*(phi + 1), (eta + 1)*(phi + 1), -(eta + 1)*(phi + 1)])
        dN_deta = [-(1 - phi)*(1 - xi), -(1 - phi)*(xi + 1), (1 - phi)*(xi + 1), (1 - phi)*(1 - xi), -(1 - xi)*(phi + 1), -(phi + 1)*(xi + 1), (phi + 1)*(xi + 1), (1 - xi)*(phi + 1)]
        dN_dphi = [-(1 - eta)*(1 - xi), -(1 - eta)*(xi + 1), -(eta + 1)*(xi + 1), -(1 - xi)*(eta + 1), (1 - eta)*(1 - xi), (1 - eta)*(xi + 1), (eta + 1)*(xi + 1), (1 - xi)*(eta + 1)]

        dN = 1/8*np.vstack((dN_dxi, dN_deta, dN_dphi))        
        
        matjac = dN@coords
        dV = np.linalg.det(matjac)*w
        
        dcoords = np.linalg.solve(matjac, dN)
        
        B = np.zeros((6,24))
        
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


pl = pv.Plotter()
pvelements = np.zeros((nels, 9),dtype=int)
pvelements[:,0] = 8
pvelements[:,1::] = elements
element_types = np.full(nels, pv.CellType.HEXAHEDRON, dtype = np.uint8)
grid = pv.UnstructuredGrid(pvelements.flatten(), element_types, nodes)
pl.add_mesh(grid,  opacity = 0.8,label='Original', style="wireframe")


grid.point_data["Displacement"] = displacement_vectors
grid.point_data["Magnitude"] = np.linalg.norm(displacement_vectors, axis = 1)
warp_factor = 50
warped_grid = grid.warp_by_vector("Displacement", factor = warp_factor)

pl.add_mesh(warped_grid, scalars="Magnitude", cmap="jet", show_edges = True, label ="Deformed")

pl.add_axes()

pl.add_text(f"Deformation scaled by {warp_factor:.1f}x", position = "upper_left")
pl.add_legend()

pl.view_xz()
pl.show()


