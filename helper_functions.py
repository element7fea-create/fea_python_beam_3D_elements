import numpy as np
from numba import njit

@njit(cache = True, fastmath = True)
def calculate_element_stiffness_matrices(nels, elements, nodes, ips, C):
    Kel = np.zeros((24,24,nels))
    rowid = np.zeros((24,24,nels),dtype=np.int64)
    colid = np.zeros((24,24,nels),dtype=np.int64)
        
    for i in range(0,nels):
        nds = elements[i,:]
        dofs = np.zeros((24,))
        dofs[0::3] = nds*3
        dofs[1::3] = nds*3+1
        dofs[2::3] = nds*3+2
        
        for j in range(0,24):
            rowid[:,j,i] = dofs
        
        colid[:,:,i] = np.transpose(rowid[:,:,i])
        
        coords = nodes[nds,:]
        
        dN = np.zeros((3,8))
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
                    dN_dxi = np.array([-(1 - eta)*(1 - phi), (1 - eta)*(1 - phi), (1 - phi)*(eta + 1), -(1 - phi)*(eta + 1), -(1 - eta)*(phi + 1), (1 - eta)*(phi + 1), (eta + 1)*(phi + 1), -(eta + 1)*(phi + 1)])
                    dN_deta = [-(1 - phi)*(1 - xi), -(1 - phi)*(xi + 1), (1 - phi)*(xi + 1), (1 - phi)*(1 - xi), -(1 - xi)*(phi + 1), -(phi + 1)*(xi + 1), (phi + 1)*(xi + 1), (1 - xi)*(phi + 1)]
                    dN_dphi = [-(1 - eta)*(1 - xi), -(1 - eta)*(xi + 1), -(eta + 1)*(xi + 1), -(1 - xi)*(eta + 1), (1 - eta)*(1 - xi), (1 - eta)*(xi + 1), (eta + 1)*(xi + 1), (1 - xi)*(eta + 1)]
            
                    #dN = 1/8*np.vstack((dN_dxi, dN_deta, dN_dphi)) 
                    dN[0,:] = dN_dxi
                    dN[1,:] = dN_deta
                    dN[2,:] = dN_dphi
                    
                    dN = dN/8.0
                    
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
                    
    return(Kel, rowid, colid)


@njit(cache = True, fastmath = True)
def calculate_stresses(nels, nodes, elements, u, ips, C):
    stresses = np.zeros((6,8,nels))
    nodal_stresses = np.zeros((nodes.shape[0],6))
    nodal_stresses_count = np.zeros((nodes.shape[0],))

    for i in range(0,nels):
        nds = elements[i,:]
        dofs = np.zeros((24,),dtype=np.int64)
        dofs[0::3] = nds*3
        dofs[1::3] = nds*3+1
        dofs[2::3] = nds*3+2
        
        unds = u[dofs]
        
        coords = nodes[nds,:]
        
        stressloc = 0
        dN = np.zeros((3,8))
        for j in range(0,ips.shape[0]):
            xi = ips[j,0]
            
            xip = 1/xi
            for k in range(0,ips.shape[0]):  
                eta = ips[k,0]
                
                etap = 1/eta
                for l in range(0,ips.shape[0]):
                    phi = ips[l,0]
                  
                    phip = 1/phi
            
                    dN_dxi = np.array([-(1 - eta)*(1 - phi), (1 - eta)*(1 - phi), (1 - phi)*(eta + 1), -(1 - phi)*(eta + 1), -(1 - eta)*(phi + 1), (1 - eta)*(phi + 1), (eta + 1)*(phi + 1), -(eta + 1)*(phi + 1)])
                    dN_deta = [-(1 - phi)*(1 - xi), -(1 - phi)*(xi + 1), (1 - phi)*(xi + 1), (1 - phi)*(1 - xi), -(1 - xi)*(phi + 1), -(phi + 1)*(xi + 1), (phi + 1)*(xi + 1), (1 - xi)*(phi + 1)]
                    dN_dphi = [-(1 - eta)*(1 - xi), -(1 - eta)*(xi + 1), -(eta + 1)*(xi + 1), -(1 - xi)*(eta + 1), (1 - eta)*(1 - xi), (1 - eta)*(xi + 1), (eta + 1)*(xi + 1), (1 - xi)*(eta + 1)]
            
                    #dN = 1/8*np.vstack((dN_dxi, dN_deta, dN_dphi))  
                    dN[0,:] = dN_dxi
                    dN[1,:] = dN_deta
                    dN[2,:] = dN_dphi
                    
                    dN = dN/8.0
                    
                    matjac = dN@coords
                    
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
                    
                    stress = C@B@unds
                    stresses[:,stressloc, i] = stress
                    stressloc += 1
                    
                    N = 1/8*np.array([(1-xip)*(1-etap)*(1-phip),(1+xip)*(1-etap)*(1-phip),(1+xip)*(1+etap)*(1-phip),(1-xip)*(1+etap)*(1-phip),
                                      (1-xip)*(1-etap)*(1+phip),(1+xip)*(1-etap)*(1+phip),(1+xip)*(1+etap)*(1+phip),(1-xip)*(1+etap)*(1+phip)])
                    
                    nodal_stresses[nds,0] += N*stress[0]
                    nodal_stresses[nds,1] += N*stress[1]
                    nodal_stresses[nds,2] += N*stress[2]
                    nodal_stresses[nds,3] += N*stress[3]
                    nodal_stresses[nds,4] += N*stress[4]
                    nodal_stresses[nds,5] += N*stress[5]
                    
                    nodal_stresses_count[nds] += N
                    
    for i in range(0, nodes.shape[0]):
        for j in range(0,6):
            nodal_stresses[i,j] = nodal_stresses[i,j]/nodal_stresses_count[i]
            
    return (stresses, nodal_stresses)
