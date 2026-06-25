import os as os
import sys as sys
import time
import numpy as np
from numpy import genfromtxt
import taichi as ti
from pyevtk.hl import pointsToVTK


@ti.kernel
def Substep():
    
    for p in x:                                                          # Cycle through each particle
        CellBase = (x[p] * InvNodeSpacing).cast(int)                     # Define the corner of the cell that contains the current particle
        SupportBase = (x[p] * InvNodeSpacing - Shift).cast(int)          # Define the corner of the grid of nodes that contain the current particle within their support
        
        F[p] = (ti.Matrix.identity(float, Dim) + dt * C[p]) @ F[p]       # Update the deformation gradient
        U, Sigma, V = ti.svd(F[p])                                       # Singular value decomposition (SVD) of the deformation gradient

        ParticleVolume[p] = InitialParticleVolume[p] * J[p]              # Update the particle volume
        Normal[p] = (U @ V.transpose()) @ NormalInitial[p]               # Update the boundary normal vectors
        
        DetF = 1.0
        for d in ti.static(range(Dim)):
            DetF *= Sigma[d, d]
        J[p] = DetF                                                      # Define the determinant of the deformation gradient

        if MaterialModel[BodyID[p]] == 1:                                # Linear elastic material model
            D = 0.5 * (C[p] + C[p].transpose())                          # Define the particle strain rate
            StrainInc = dt * D                                           # Define the particle strain increment
            ParticleStrain[p] += StrainInc                               # Update the particle strain using the strain increment

            if Switch_PressureStab:                                      # Strain projection technique
                StabilizeNum[CellBase] += ParticleMass[p] * ParticleStrain[p].trace()  
                StabilizeDen[CellBase] += ParticleMass[p]
        
        elif MaterialModel[BodyID[p]] == 2 and Switch_PressureStab:      # Ogden neo-Hookean material model, F-Bar projection technique
            StabilizeNum[CellBase] += InitialParticleVolume[p] * J[p]
            StabilizeDen[CellBase] += InitialParticleVolume[p]
        
        elif MaterialModel[BodyID[p]] == 4:                              # Elasto-plastic material model, strain projection technique
            D = 0.5 * (C[p] + C[p].transpose())                          # Define the particle strain rate    
            StrainInc = dt * D                                           # Define the particle strain increment
            ParticleStrain[p] += StrainInc                               # Update the particle strain using the strain increment

            StabilizeNum[CellBase] += ParticleMass[p] * ParticleStrain[p].trace()     
            StabilizeDen[CellBase] += ParticleMass[p]

        phiStep, phi_dxStep, phi_dyStep, phi_dzStep = GetRK(x[p], SupportBase, a)      # Get the reproducing kernel (RK) approximation for all surrounding grid nodes evaluated at the current particle
        for i, j, k in ti.static(ti.ndrange(NodeNum, NodeNum, NodeNum)):               # Cycle through each surrounding grid node that contains the current particle within its support
            SupportBaseToNode = ti.Vector([i, j, k])                                   # Define the vector from the support base to the current grid node
            
            phi[p][i + Dim * j + Dim ** 2 * k] = phiStep[i + Dim * j + Dim ** 2 * k]   # Save the RK approximation and derivative values (for computational efficiency)
            dphi_dx[p][i + Dim * j + Dim ** 2 * k] = phi_dxStep[i + Dim * j + Dim ** 2 * k]
            dphi_dy[p][i + Dim * j + Dim ** 2 * k] = phi_dyStep[i + Dim * j + Dim ** 2 * k]
            dphi_dz[p][i + Dim * j + Dim ** 2 * k] = phi_dzStep[i + Dim * j + Dim ** 2 * k]

            phiGrad = ti.Vector([dphi_dx[p][i + Dim * j + Dim ** 2 * k], dphi_dy[p][i + Dim * j + Dim ** 2 * k], dphi_dz[p][i + Dim * j + Dim ** 2 * k]])    # Define the shape function gradient vector


    for p in x:                                                   # Cycle through each particle
        CellBase = (x[p] * InvNodeSpacing).cast(int)              # Define the corner of the cell that contains the current particle
        SupportBase = (x[p] * InvNodeSpacing - Shift).cast(int)   # Define the corner of the grid of nodes that contain the current particle within their support
        SupportBaseToParticle = ( x[p] * InvNodeSpacing - SupportBase.cast(float) ) * NodeSpacing  # Define the vector from the support base to the current particle 
        
        if MaterialModel[BodyID[p]] == 1:    # Linear elastic material model
            if Switch_PressureStab:          # Strain projection technique
                devStrain = ParticleStrain[p] - (1 / 3) * ParticleStrain[p].trace() * ti.Matrix.identity(float, Dim)    # Define the particle deviatoric strain
                ParticleStrain[p] = devStrain + (1 / 3) * (StabilizeNum[CellBase] / StabilizeDen[CellBase]) * ti.Matrix.identity(float, Dim)    # Replace the particle volumetric strain with the cell-averaged volumetric strain

            ParticleStress[p] = 2 * mu[p] * ParticleStrain[p] + la[p] * ParticleStrain[p].trace() * ti.Matrix.identity(float, Dim)  # Define the particle stress from particle strain
        
        elif MaterialModel[BodyID[p]] == 2:  # Ogden neo-Hookean material model
            if Switch_PressureStab:          # F-Bar projection technique
                FBar = (((StabilizeNum[CellBase] / StabilizeDen[CellBase]) / J[p]) ** (1 / Dim)) * F[p]  # Define the new deformation gradient with cell-averaged volumetric deformation
                J[p] = StabilizeNum[CellBase] / StabilizeDen[CellBase]                                   # Define the new determinant of the deformation gradient
                F[p] = FBar                                                                              # Assign the new F-Bar deformation gradient
            
            U, _ , V = ti.svd(F[p])          # Singular value decomposition (SVD) of the deformation gradient
            ParticleStress[p] = 2 * mu[p] * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, Dim) * la[p] * J[p] * (J[p] - 1)  # Define the Ogden neo-Hookean Cauchy stress 
        
        elif MaterialModel[BodyID[p]] == 4:       # Elasto-plastic material model
            D = 0.5 * (C[p] + C[p].transpose())   # Define the particle strain rate 
            StrainInc = dt * D                    # Define the particle strain increment
            
            ParticleStressDev[p] = 2 * mu[p] * (ParticleStrain[p] - ParticleStrainPlastic[p] - (1 / 3) * (ParticleStrain[p] - ParticleStrainPlastic[p]).trace() * ti.Matrix.identity(float, Dim))   # Define the particle deviatoric stress
            if Switch_PressureStab:
                ParticleStressVol[p] = (la[p] + (2 / 3) * mu[p]) * (StabilizeNum[CellBase] / StabilizeDen[CellBase]) * ti.Matrix.identity(float, Dim)
            else:
                ParticleStressVol[p] = (la[p] + (2 / 3) * mu[p]) * (ParticleStrain[p] - ParticleStrainPlastic[p]).trace() * ti.Matrix.identity(float, Dim)
            
            H = LinearHardening[BodyID[p]]                                                     # Linear hardening modulus
            vonMisesFlowStress = YieldStress[BodyID[p]] + H * P[p]                             # Calculate the von mises yield stress, including linear hardening (J2 plasticity)
            vonMisesTrialStress = ti.sqrt((3 / 2) * (ParticleStressDev[p].norm() ** 2))        # Calculate the plastic trial stress
            
            YieldFunction = vonMisesTrialStress - vonMisesFlowStress                           # Plastic yield function
            if YieldFunction > eps:                                                            # If the trial stress is greater than the yield stress (there is plastic deformation)
                DeltaDelta_P[p] = (YieldFunction - 3 * mu[p] * Delta_P[p]) / (3 * mu[p] + H)   # Change in delta_p (youtube video)
                Delta_P[p] += DeltaDelta_P[p]                                                  # Update delta_p    (youtube video)

                P[p] += Delta_P[p]                                                                                # Update p (youtube video)
                ParticleStrainPlastic[p] += Delta_P[p] * (3 / 2) * ParticleStressDev[p] / vonMisesTrialStress     # Update the particle plastic strain
                ParticleStressDev[p] = 2 * mu[p] * (ParticleStrain[p] - ParticleStrainPlastic[p] - (1 / 3) * (ParticleStrain[p] - ParticleStrainPlastic[p]).trace() * ti.Matrix.identity(float, Dim))   # Update the particle deviatoric stress (not including plastic deformation)
                ParticleStress[p] = ParticleStressVol[p] + ParticleStressDev[p]    # Update the particle stress
            else:
                ParticleStress[p] = ParticleStressVol[p] + ParticleStressDev[p]    # Update particle stress (no plastic deformation)

        PartitionOfUnity[p] = 0.0  # Initialize "partition of unity" at zero for each particle
        Consistency[p] = 0.0       # Initialize "consistency" at zero for each particle
        for i, j, k in ti.static(ti.ndrange(NodeNum, NodeNum, NodeNum)):  # Cycle through all grid nodes with the current particle within their shape function support
            SupportBaseToNode = ti.Vector([i, j, k])   # Define the vector from the support base to the current grid node
            ParticleToNode = (SupportBaseToNode.cast(float) * NodeSpacing - SupportBaseToParticle)  # Define the vector from the particle to the current grid node

            PartitionOfUnity[p] += phi[p][i + Dim * j + Dim ** 2 * k]     # Caculate the sum of each node's shape function evaluated at the current particle
            Consistency[p] += phi[p][i + Dim * j + Dim ** 2 * k] * (SupportBase[0] + SupportBaseToNode[0]) * NodeSpacing   # Calculate the shape function's ability to reproduce the function f(x,y,z) = x

            phiGrad = ti.Vector([dphi_dx[p][i + Dim * j + Dim ** 2 * k], dphi_dy[p][i + Dim * j + Dim ** 2 * k], dphi_dz[p][i + Dim * j + Dim ** 2 * k]])   # Define the shape function gradient vector

            APIC = ParticleToNode * 0
            if ParticleUpdate == 'APIC':        # If APIC is used
                APIC = C[p] @ ParticleToNode    # Add the velocity gradient term
            

            ## The 5 lines below are evaluating each integral of the weak form ##
            GridMass[SupportBase + SupportBaseToNode] += phi[p][i + Dim * j + Dim ** 2 * k] * ParticleMass[p]                                   # Mass matrix
            GridVelocity[SupportBase + SupportBaseToNode] += phi[p][i + Dim * j + Dim ** 2 * k] * (ParticleMass[p]) * (v[p] + APIC)             # Momentum projection
            GridVelocityInitial[SupportBase + SupportBaseToNode] += phi[p][i + Dim * j + Dim ** 2 * k] * (ParticleMass[p]) * (v[p] + APIC)      # Momentum projection (save for FLIP)
            GridVelocity[SupportBase + SupportBaseToNode] += dt * ParticleVolume[p] * phi[p][i + Dim * j + Dim ** 2 * k] * Gravity[BodyID[p]]   # External force vector (body force)
            GridVelocity[SupportBase + SupportBaseToNode] -= dt * ParticleVolume[p] * ParticleStress[p] @ phiGrad                               # Internal force vector 

            if MaterialModel[BodyID[p]] == 3:                                           
                RigidBodyGridIndex[SupportBase + SupportBaseToNode] = BodyID[p] + 1       # Indentify the grid nodes around the current particle of a rigid body

            if DisplacementParticleIndex[p] > eps:
                DisplacementGridIndex[SupportBase + SupportBaseToNode] = SurfaceID[p]     # Indentify the grid nodes around the current particle on a displacement boundary

            elif FixedParticleIndex[p] == 1: 
                FixedGridIndex[SupportBase + SupportBaseToNode] = 1                       # Indentify the grid nodes around the current particle on a fixed boundary
        
        Consistency[p] -= x[p][0]    # Find the error of the shape function's ability to reproduce the function f(x,y,z) = x 
        PartitionOfUnity[p] -= 1     # Find the error of the shape function's ability to satisfy partion of unity 


    for i, j, k in GridMass:          # Cycle through each grid node
        if GridMass[i, j, k] > eps:   # If the current grid node has at least one particle within its shape function support

            GridVelocity[i, j, k] = (1 / GridMass[i, j, k]) * GridVelocity[i, j, k]                     # Define the grid node velocity as momentum divided by mass
            GridVelocityInitial[i, j, k] = (1 / GridMass[i, j, k]) * GridVelocityInitial[i, j, k]       # Define the initial grid node velocity (before the momentum equation is solved) as momentum divided by mass

            if FixedGridIndex[i, j, k] == 1:               # If the current grid node is marked as a fixed boundary
                GridVelocity[i, j, k] = [0.0, 0.0, 0.0]    # Set velocity to zero 
            
            elif DisplacementGridIndex[i, j, k] > eps:                                                      # If the current grid node is marked as a displacement boundary
                for s in ti.static(ti.ndrange(DisplacementSurfaceNumber)):                                  # Cycle through each surface marked as a displacement surface
                    if DisplacementGridIndex[i, j, k] == DisplacementSurfaceIndex[s]:                       # If the current node corresponds to the current displacement surface
                        GridVelocity[i, j, k] = [TaichiSetVx[0][s], TaichiSetVy[0][s], TaichiSetVz[0][s]]   # Set the designated input velocity to the corresponding displacement surface

            if RigidBodyGridIndex[i, j, k] > eps:                                            # If the current grid node is marked as belonging to a rigid body
                GridVelocity[i, j, k] = InitialVelocity[RigidBodyGridIndex[i, j, k] - 1]     # Set the corresponding rigid body velocity

            ## The 6 lines below are applying boundary conditions to the edges of the background grid (so particles cannot escape the grid) ##
            if i < NodeNum and GridVelocity[i, j, k][0] < 0: GridVelocity[i, j, k][0] = 0                        # Prevent the particle from leaving the lower y - z plane boundary
            if i > GridNodeNumber - NodeNum - 1 and GridVelocity[i, j, k][0] > 0: GridVelocity[i, j, k][0] = 0   # Prevent the particle from leaving the upper y - z plane boundary
            if j < NodeNum and GridVelocity[i, j, k][1] < 0: GridVelocity[i, j, k][1] = 0                        # Prevent the particle from leaving the lower x - z plane boundary
            if j > GridNodeNumber - NodeNum - 1 and GridVelocity[i, j, k][1] > 0: GridVelocity[i, j, k][1] = 0   # Prevent the particle from leaving the upper x - z plane boundary
            if k < NodeNum and GridVelocity[i, j, k][2] < 0: GridVelocity[i, j, k][2] = 0                        # Prevent the particle from leaving the lower x - y plane boundary
            if k > GridNodeNumber - NodeNum - 1 and GridVelocity[i, j, k][2] > 0: GridVelocity[i, j, k][2] = 0   # Prevent the particle from leaving the upper x - y plane boundary
    
    
    for p in x:                                                   # Cycle through each particle
        SupportBase = (x[p] * InvNodeSpacing - Shift).cast(int)   # Define the corner of the grid of nodes that contain the current particle within their support
        
        VstepG2P = ti.Vector.zero(float, Dim)    # Initialize the "normal" particle velocity after the G2P step 
        VstepFLIP = ti.Vector.zero(float, Dim)   # Initialize the FLIP particle velocity after the G2P step 
        Cstep = ti.Matrix.zero(float, Dim, Dim)  # Initialize the particle velocity gradient after the G2P step 

        for i, j, k in ti.static(ti.ndrange(NodeNum, NodeNum, NodeNum)):   # Cycle through all grid nodes with the current particle within their shape function support          
            SupportBaseToNode = ti.Vector([i, j, k])   # Define the vector from the support base to the current grid node

            phiGrad = ti.Vector([dphi_dx[p][i + Dim * j + Dim ** 2 * k], dphi_dy[p][i + Dim * j + Dim ** 2 * k], dphi_dz[p][i + Dim * j + Dim ** 2 * k]])  # Define the shape function gradient vector
            
            VstepG2P += phi[p][i + Dim * j + Dim ** 2 * k] * GridVelocity[SupportBase + SupportBaseToNode]  # Interpolate the grid velocity to particles
            if ParticleUpdate == 'FLIP':   # If FLIP is used
                VstepFLIP += phi[p][i + Dim * j + Dim ** 2 * k] * (GridVelocity[SupportBase + SupportBaseToNode] - GridVelocityInitial[SupportBase + SupportBaseToNode])  # Interpolate the grid velocity increment to particles (FLIP)
                
            Cstep += GridVelocity[SupportBase + SupportBaseToNode].outer_product(phiGrad)  # Interpolate the grid velocity gradient to particles

            x[p] += dt * phi[p][i + Dim * j + Dim ** 2 * k] * GridVelocity[SupportBase + SupportBaseToNode]  # Update particle position using the grid node velocity

            x2D[p][0] = x[p][0]  # Save the 'x' position for the 2D snapshots saved in the "movie" folder
            x2D[p][1] = x[p][2]  # Save the 'z' position for the 2D snapshots saved in the "movie" folder

        if ParticleUpdate == 'FLIP':   # If FLIP is used
            v[p] = alpha * (v[p] + VstepFLIP) + (1.0 - alpha) * VstepG2P   # Update particle velocity using the FLIP blending technique
        else:                 # If FLIP is not used
            v[p] = VstepG2P   # Update particle velocity using the standard technique

        C[p] = Cstep          # Update particle velocity gradient
    

    for i, j, k in GridMass:     # Cycle through each grid node

        ## Grid resetting step ##
        GridVelocity[i, j, k] = [0, 0, 0]
        GridVelocityInitial[i, j, k] = [0, 0, 0]
        GridMass[i, j, k] = 0
        DisplacementGridIndex[i, j, k] = 0
        FixedGridIndex[i, j, k] = 0
        RigidBodyGridIndex[i, j, k] = 0
        StabilizeDen[i, j, k] = 0
        StabilizeNum[i, j, k] = 0


@ti.func 
def GetRK(xp, base, a):    # Function tht defines the RK approximation for the grid nodes surrounding a given particle
    phi = ti.Vector.zero(float,NodeNum ** Dim)
    dphi_dx = ti.Vector.zero(float,NodeNum ** Dim)
    dphi_dy = ti.Vector.zero(float,NodeNum ** Dim)
    dphi_dz = ti.Vector.zero(float,NodeNum ** Dim)
    M = ti.Matrix.zero(float,Dim + 1, Dim + 1)
    weight1D = ti.Matrix.zero(float, NodeNum,Dim)
    weight2D = ti.Vector.zero(float, NodeNum ** Dim)
    for i, d in ti.static(ti.ndrange(NodeNum,Dim)):
        gridNode = float( i + base[d] ) * NodeSpacing
        z = abs( xp[d] - float(gridNode)) / a 
        if 0 <= z and z < 0.5:
            weight1D[i,d] = 2/3 - 4*z**2 + 4*z**3 
        elif 1/2 <= z and z < 1:
            weight1D[i,d] = 4/3 - 4*z + 4*z**2 - (4/3)*z**3
        elif 1 <= z: 
            weight1D[i,d] = 0 
    for i, j, k in ti.static(ti.ndrange(NodeNum, NodeNum, NodeNum)): 
        gridNode = [float(i+base[0]) * NodeSpacing, float(j+base[1]) * NodeSpacing, float(k+base[2]) * NodeSpacing]
        weight2D[i + Dim * j + Dim ** 2 * k] = weight1D[i,0] * weight1D[j,1] * weight1D[k,2]
        H = (ti.Vector([1.0, xp[0] - gridNode[0], xp[1] - gridNode[1], xp[2] - gridNode[2]]))
        if weight2D[i + Dim * j + Dim ** 2 * k] != 0:
            M = M + weight2D[i + Dim * j + Dim ** 2 * k] * H @ H.transpose()     
    M_inverse = M.inverse()
    for i, j, k in ti.static(ti.ndrange(NodeNum, NodeNum, NodeNum)):                 
        gridNode = [float(i+base[0]) * NodeSpacing, float(j+base[1]) * NodeSpacing, float(k+base[2]) * NodeSpacing]
        if weight2D[i + Dim * j + Dim ** 2 * k] != 0:
            H = (ti.Vector([1.0, xp[0] - gridNode[0], xp[1] - gridNode[1], xp[2] - gridNode[2]]))
            H0 = (ti.Vector([1.0, 0, 0, 0]))
            dH0_dx = (ti.Vector([0.0, -1.0, 0.0, 0.0]))
            dH0_dy = (ti.Vector([0.0, 0.0, -1.0, 0.0]))
            dH0_dz = (ti.Vector([0.0, 0.0, 0.0, -1.0]))

            phiStep = weight2D[i + Dim * j + Dim ** 2 * k] * H0.transpose() @ (M_inverse @ H)
            phi[i + Dim * j + Dim ** 2 * k] = phiStep[0]
            dphi_dx_Step = weight2D[i + Dim * j + Dim ** 2 * k] * dH0_dx.transpose() @ (M_inverse @ H)
            dphi_dx[i + Dim * j + Dim ** 2 * k] = dphi_dx_Step[0]
            dphi_dy_Step = weight2D[i + Dim * j + Dim ** 2 * k] * dH0_dy.transpose() @ (M_inverse @ H)
            dphi_dy[i + Dim * j + Dim ** 2 * k] = dphi_dy_Step[0]
            dphi_dz_Step = weight2D[i + Dim * j + Dim ** 2 * k] * dH0_dz.transpose() @ (M_inverse @ H)
            dphi_dz[i + Dim * j + Dim ** 2 * k] = dphi_dz_Step[0]

    return phi, dphi_dx, dphi_dy, dphi_dz 


time0 = time.time()   # Mark time 0
ti.init(arch=ti.gpu)  # Initialize Taichi
# ti.init(arch=ti.gpu, default_ip = ti.i32, default_fp = ti.f64)

FixedSurfaceIndex = []            # Initialize list of surfaces to fix
DisplacementSurfaceIndex = []     # Initialize a list of surfaces to apply displacement boundary conditions to
PassUpdateSchemeCheck = False     # Check if we have defined the update scheme
FLIPSwitch = False                # Are we using FLIP or no
alpha = 0.0                       # Initialize FLIP alpha parameter as 0.0 (which means we assume PIC / APIC)
SetVx = [0.0]                       
SetVy = [0.0]
SetVz = [0.0]

with open(f'Input_0.txt') as Input:         # Read the input file
    SectionNum = -1                            # Initialize the "Section Number" as 0
    for lineNum, line in enumerate(Input):    # Read each line of the input file

        if 'Section' in line:                 # If we have reached a new "Section Number"
            index = line.find('.')            
            SectionNum = line[index - 1]      # Define which "Section Number" we are currently working with        
        
        if '--' in line:                      # "--" means that there is data to be read
            match int(SectionNum):            # Cycle through each "Section Number"
                case 0:                                      # Section 0
                    index = line.find('.')                   # "." means that there is data to be read
                    variableNum = line[index - 1]
                    index = line.find('--')                  # "--" means that there is data to be read
                    match int(variableNum):
                        case 1:
                            indexLeft = line.find('[')
                            indexRightEnd = line.find(']')
                            indexComma = line.find(',')   
                            enterWhile = False
                            if indexComma == -1:
                                MeshFileIndex = [line[indexLeft + 1:indexRightEnd].replace(" ", "")]
                            else:
                                indexRight = indexComma
                                MeshFileIndex = [line[indexLeft + 1:indexRight].replace(" ", "")]
                                indexLeft = indexRight
                                enterWhile = True
                            while enterWhile:
                                indexComma = line.find(',', indexLeft + 1, indexRightEnd) 
                                if indexComma == -1:
                                    MeshFileIndex.append(line[indexLeft + 1:indexRightEnd].replace(" ", ""))
                                    enterWhile = False
                                else:
                                    indexRight = indexComma
                                    MeshFileIndex.append(line[indexLeft + 1:indexRight].replace(" ", ""))
                                    indexLeft = indexRight
                        case 2:
                            FramesPerSecond = int(line[index + 2:len(line)])
                        case 3:
                            AnimationTime = int(line[index + 2:len(line)])


MeshNum = len(MeshFileIndex)  # Determine the number of meshes used
for m in range(MeshNum):      # Cycle through each mesh
    StartSkipping = False
    with open(f'Input_{m}.txt') as Input:         # Read the input file
        SectionNum = 0                            # Initialize the "Section Number" as 0
        for lineNum, line in enumerate(Input):    # Read each line of the input file

            if 'Section' in line:                 # If we have reached a new "Section Number"
                index = line.find('.')            
                SectionNum = line[index - 1]      # Define which "Section Number" we are currently working with
                if '~' in line:
                    StartSkipping = True
                    continue
                if StartSkipping:
                    StartSkipping = False
            else:
                if StartSkipping:
                    continue
            
            
            if '--' in line:                      # "--" means that there is data to be read
                match int(SectionNum):            # Cycle through each "Section Number"
                    case 1:                                      # Section 1
                        index = line.find('.')                   # "." means that there is data to be read
                        variableNum = line[index - 1]
                        index = line.find('--')                  # "--" means that there is data to be read
                        match int(variableNum):
                            case 1:
                                GridNodeNumber = int(line[index + 2:len(line)])
                            case 2:
                                GridDimension = float(line[index + 2:len(line)])
                            case 3:
                                dt = float(line[index + 2:len(line)])
                            case 4:
                                SimulationTime = float(line[index + 2:len(line)])

                    case 2:                                               # Section 2
                        index = line.find('--')                           # Find the data to be read (material model)
                        variableNum = line[index + 2:len(line)]
                        match int(variableNum):
                            case 1:
                                exec(f"MaterialModel_Body{m} = {variableNum}")
                            case 2:
                                exec(f"MaterialModel_Body{m} = {variableNum}")
                            case 3:
                                exec(f"MaterialModel_Body{m} = {variableNum}")
                            case 4: 
                                exec(f"MaterialModel_Body{m} = {variableNum}")

                    case 3:                                      # Section 3
                        index = line.find('.')                   # "." means that there is data to be read
                        variableNum = line[index - 1]
                        index = line.find('--')                  # "--" means that there is data to be read
                        match int(variableNum):
                            case 1:
                                exec(f"E_Body{m} = {float(line[index + 2:len(line)])}")
                            case 2:
                                exec(f"nu_Body{m} = {float(line[index + 2:len(line)])}")
                            case 3:
                                exec(f"rho_Body{m} = {float(line[index + 2:len(line)])}")
                            case 4:
                                exec(f"YieldStress_Body{m} = {float(line[index + 2:len(line)])}")
                            case 5:
                                exec(f"H_Body{m} = {float(line[index + 2:len(line)])}")

                    case 4:                                      # Section 4
                        index = line.find('.')                   # "." means that there is data to be read
                        variableNum = line[index - 1]
                        index = line.find('--')                  # "--" means that there is data to be read
                        match int(variableNum):
                            case 1:
                                exec(f"Gravity_Body{m} = {-1 * float(line[index + 2:len(line)])}")
                            case 2:
                                exec(f"Vx0_Body{m} = {float(line[index + 2:len(line)])}")
                            case 3:
                                exec(f"Vy0_Body{m} = {float(line[index + 2:len(line)])}")
                            case 4:
                                exec(f"Vz0_Body{m} = {float(line[index + 2:len(line)])}")

                    case 5:                                           # Section 5
                        index = line.find('--')                       # "--" means that there is data to be read
                        variableNum = line[index + 2:len(line)]
                        if not FLIPSwitch:                            # If either 1. we are not using FLIP or 2. we have not defined the update scheme yet
                            if not PassUpdateSchemeCheck:             # Check to see if we have defined the update scheme
                                match int(variableNum):
                                    case 1:
                                        ParticleUpdate = 'APIC'
                                    case 2:
                                        ParticleUpdate = 'PIC'
                                    case 3:
                                        ParticleUpdate = 'FLIP'
                                        FLIPSwitch = True             # We are using FLIP, now we need to define alpha
                        else:
                            alpha = float(line[index + 2:len(line)])  # Define the FLIP parameter alpha
                        PassUpdateSchemeCheck = True                  # We have defined the update scheme

                    case 6:                                           # Section 6
                        index = line.find('.')                        # "." means that there is data to be read
                        variableNum = line[index - 1]
                        index = line.find('--')                       # "--" means that there is data to be read
                        match int(variableNum):
                            case 1:
                                Switch_PressureStab = int(line[index + 2:len(line)])

                    case 7:                                           # Section 7
                        indexLeft = line.find('[')
                        indexRightEnd = line.find(']')
                        indexComma = line.find(',')   
                        enterWhile = False
                        if indexComma == -1:
                            FixedSurfaceIndex = [int(line[indexLeft + 1:indexRightEnd])]
                        else:
                            indexRight = indexComma
                            FixedSurfaceIndex = [int(line[indexLeft + 1:indexRight])]
                            indexLeft = indexRight
                            enterWhile = True
                        while enterWhile:
                            indexComma = line.find(',', indexLeft + 1, indexRightEnd) 
                            if indexComma == -1:
                                FixedSurfaceIndex.append(int(line[indexLeft + 1:indexRightEnd]))
                                enterWhile = False
                            else:
                                indexRight = indexComma
                                FixedSurfaceIndex.append(int(line[indexLeft + 1:indexRight]))
                                indexLeft = indexRight             

                    case 8:                                           # Section 8
                        indexLeftVelocity = line.find('{')
                        if indexLeftVelocity == -1:
                            indexLeft = line.find('[')
                            indexRightEnd = line.find(']')
                            indexComma = line.find(',')   
                            enterWhile = False
                            if indexComma == -1:
                                DisplacementSurfaceIndex = [int(line[indexLeft + 1:indexRightEnd])]
                            else:
                                indexRight = indexComma
                                DisplacementSurfaceIndex = [int(line[indexLeft + 1:indexRight])]
                                indexLeft = indexRight
                                enterWhile = True
                            while enterWhile:
                                indexComma = line.find(',', indexLeft + 1, indexRightEnd) 
                                if indexComma == -1:
                                    DisplacementSurfaceIndex.append(int(line[indexLeft + 1:indexRightEnd]))
                                    enterWhile = False
                                else:
                                    indexRight = indexComma
                                    DisplacementSurfaceIndex.append(int(line[indexLeft + 1:indexRight]))
                                    indexLeft = indexRight   
                        else:
                            indexRightVelocity = line.find('}')
                            indexSemiColon = line.find(';')   
                            indexLeft = line.find('[')
                            indexRight = line.find(']')
                            enterWhile = False
                            indexComma1 = line.find(',', indexLeft, indexRight)
                            SetVx = [float(line[indexLeft + 1:indexComma1])]
                            indexComma2 = line.find(',', indexComma1 + 1, indexRight)
                            SetVy = [float(line[indexComma1 + 1:indexComma2])]
                            SetVz = [float(line[indexComma2 + 1:indexRight])]
                            if indexSemiColon > -1:
                                indexLeft = line.find('[', indexSemiColon, indexRightVelocity)
                                indexRight = line.find(']', indexSemiColon, indexRightVelocity)
                                enterWhile = True
                            while enterWhile:
                                indexSemiColon = line.find(';', indexLeft + 1, indexRightVelocity) 
                                indexComma1 = line.find(',', indexLeft, indexRight)
                                SetVx.append(float(line[indexLeft + 1:indexComma1]))
                                indexComma2 = line.find(',', indexComma1 + 1, indexRight)
                                SetVy.append(float(line[indexComma1 + 1:indexComma2]))
                                SetVz.append(float(line[indexComma2 + 1:indexRight]))
                                if indexSemiColon == -1:
                                    enterWhile = False
                                else:
                                    indexRight = indexSemiColon
                                    indexLeft = indexRight 

  
DisplacementSurfaceNumber = len(DisplacementSurfaceIndex)   # Define the number of displacement surfaces
NodeSpacing = GridDimension / float(GridNodeNumber - 1)     # Node Spacing 
InvNodeSpacing = float(GridNodeNumber - 1) / GridDimension  # 1 / dx              

Dim = 3      # 3D problem
eps = 1e-15  # Machine epsilon
NormalizedSupportSize = 1.5               # RK normalized support size (can change, but I haven't tried for this code)
a = NormalizedSupportSize * NodeSpacing   # RK non-normalized support size
NodeNum = int(a * InvNodeSpacing * 2 + eps)  # Number of nodes (1D) that contain a particle in its shape function support
Shift = float(a * InvNodeSpacing - 1.0)      # Used to find the "SupportBase"


## Inport the geometric parameters from the "Initialization" file ##
xNumpy = genfromtxt('Initialize/Particle_Positions.csv', delimiter=',')
normalNumpy = genfromtxt('Initialize/Particle_Normals.csv', delimiter=',')
SurfaceIDNumpy = genfromtxt('Initialize/Particle_SurfaceID.csv', delimiter=',')
BodyIDNumpy = genfromtxt('Initialize/Particle_BodyID.csv', delimiter=',')
NodeIDNumpy = genfromtxt('Initialize/Particle_NodeID.csv', delimiter=',')
ParticleVolumeNumpy = genfromtxt('Initialize/Particle_Volume.csv', delimiter=',')
ParticleSurfaceAreaNumpy = genfromtxt('Initialize/Particle_SurfaceArea.csv', delimiter=',')

ParticleNum = len(xNumpy[:,0])  # Define the total number of particles

## Initialize problem parameters that possibly vary for each mesh used ##
Gravity = ti.Vector.field(Dim, dtype=float, shape=(MeshNum))
InitialVelocity = ti.Vector.field(Dim, dtype=float, shape=(MeshNum))
MaterialModel = ti.field(dtype=int, shape=(MeshNum))
YieldStress = ti.field(dtype=float, shape=(MeshNum))
LinearHardening = ti.field(dtype=float, shape=(MeshNum))
for m in range(MeshNum):
    Current_E = f"E_Body{m}"
    E = globals()[Current_E]
    
    Current_nu = f"nu_Body{m}"
    nu = globals()[Current_nu]

    Current_rho = f"rho_Body{m}"

    Current_YieldStress = f"YieldStress_Body{m}"
    YieldStress[m] = globals()[Current_YieldStress]

    Current_LinearHardening = f"H_Body{m}"
    LinearHardening[m] = globals()[Current_LinearHardening]

    Current_Gravity = f"Gravity_Body{m}"
    GravityMag = globals()[Current_Gravity]
    Gravity[m] = [0, 0, GravityMag]

    Current_MaterialModel = f"MaterialModel_Body{m}"
    MaterialModel[m] = globals()[Current_MaterialModel]

    Current_Vx0 = f"Vx0_Body{m}"
    Current_Vy0 = f"Vy0_Body{m}"
    Current_Vz0 = f"Vz0_Body{m}"
    InitialVelocity[m] = [globals()[Current_Vx0], globals()[Current_Vy0], globals()[Current_Vz0]]

    exec(f"mu_Body{m} = {E / (2 * (1 + nu))}")
    exec(f"la_Body{m} = {E * nu / ((1 + nu) * (1 - 2 * nu))}")

if DisplacementSurfaceNumber > 0:
    TaichiSetVx = ti.Vector.field(DisplacementSurfaceNumber, dtype=float, shape=1)
    TaichiSetVy = ti.Vector.field(DisplacementSurfaceNumber, dtype=float, shape=1)
    TaichiSetVz = ti.Vector.field(DisplacementSurfaceNumber, dtype=float, shape=1)

    for s in range(DisplacementSurfaceNumber):
        TaichiSetVx[0][s] = SetVx[s]
        TaichiSetVy[0][s] = SetVy[s]
        TaichiSetVz[0][s] = SetVz[s]


## Initialize all Taichi fields ##
x = ti.Vector.field(Dim, dtype=float, shape=ParticleNum)
x2D = ti.Vector.field(Dim - 1, dtype=float, shape=ParticleNum)
v = ti.Vector.field(Dim, dtype=float, shape=ParticleNum)
C = ti.Matrix.field(Dim, Dim, dtype=float, shape=ParticleNum)
F = ti.Matrix.field(Dim, Dim, dtype=float, shape=ParticleNum)
ParticleStress = ti.Matrix.field(Dim, Dim, dtype=float, shape=ParticleNum)
ParticleStressDev = ti.Matrix.field(Dim, Dim, dtype=float, shape=ParticleNum)
ParticleStressVol = ti.Matrix.field(Dim, Dim, dtype=float, shape=ParticleNum)
ParticleStrain = ti.Matrix.field(Dim, Dim, dtype=float, shape=ParticleNum)
ParticleStrainPlastic = ti.Matrix.field(Dim, Dim, dtype=float, shape=ParticleNum)
mu = ti.field(dtype=float, shape=ParticleNum)
la = ti.field(dtype=float, shape=ParticleNum)
rho = ti.field(dtype=float, shape=ParticleNum)
SurfaceID = ti.field(dtype=int, shape=ParticleNum)
BodyID = ti.field(dtype=int, shape=ParticleNum)
InitialParticleVolume = ti.field(dtype=float, shape=ParticleNum)
ParticleVolume = ti.field(dtype=float, shape=ParticleNum)
ParticleSurfaceArea = ti.field(dtype=float, shape=ParticleNum)
ParticleMass = ti.field(dtype=float, shape=ParticleNum)
GridVelocity = ti.Vector.field(Dim, dtype=float, shape=(GridNodeNumber, GridNodeNumber, GridNodeNumber))
GridVelocityInitial = ti.Vector.field(Dim, dtype=float, shape=(GridNodeNumber, GridNodeNumber, GridNodeNumber))
GridMass = ti.field(dtype=float, shape=(GridNodeNumber, GridNodeNumber, GridNodeNumber))
J = ti.field(dtype=float, shape=ParticleNum)
PartitionOfUnity = ti.field(dtype=float, shape=ParticleNum)
Consistency = ti.field(dtype=float, shape=ParticleNum)
phi = ti.Vector.field(NodeNum ** Dim, dtype=float, shape=ParticleNum)
dphi_dx = ti.Vector.field(NodeNum ** Dim, dtype=float, shape=ParticleNum)
dphi_dy = ti.Vector.field(NodeNum ** Dim, dtype=float, shape=ParticleNum)
dphi_dz = ti.Vector.field(NodeNum ** Dim, dtype=float, shape=ParticleNum)
Normal = ti.Vector.field(Dim, dtype=float, shape=ParticleNum)
NormalInitial = ti.Vector.field(Dim, dtype=float, shape=ParticleNum)

Delta_P = ti.field(dtype=float, shape=ParticleNum)
DeltaDelta_P = ti.field(dtype=float, shape=ParticleNum)
P = ti.field(dtype=float, shape=ParticleNum)
PlasticIteration = ti.field(dtype=int, shape=ParticleNum)

StabilizeNum = ti.field(dtype=float, shape=(GridNodeNumber - 1, GridNodeNumber - 1, GridNodeNumber - 1))
StabilizeDen = ti.field(dtype=float, shape=(GridNodeNumber - 1, GridNodeNumber - 1, GridNodeNumber - 1))

FixedParticleIndex = ti.field(dtype=int, shape=ParticleNum)
DisplacementParticleIndex = ti.field(dtype=int, shape=ParticleNum)
FixedGridIndex = ti.field(dtype=int, shape=(GridNodeNumber, GridNodeNumber, GridNodeNumber))
DisplacementGridIndex = ti.field(dtype=int, shape=(GridNodeNumber, GridNodeNumber, GridNodeNumber))
RigidBodyGridIndex = ti.field(dtype=int, shape=(GridNodeNumber, GridNodeNumber, GridNodeNumber))

x.from_numpy(xNumpy)
SurfaceID.from_numpy(SurfaceIDNumpy)
Normal.from_numpy(normalNumpy)
ParticleVolume.from_numpy(ParticleVolumeNumpy)
ParticleSurfaceArea.from_numpy(ParticleSurfaceAreaNumpy)
BodyID.from_numpy(BodyIDNumpy)


## Initialize initial conditions and material parameters ##
count = 0
for p in SurfaceIDNumpy:

    Current_mu = f"mu_Body{BodyID[count]}"
    mu[count] = locals()[Current_mu]

    Current_la = f"la_Body{BodyID[count]}"
    la[count] = locals()[Current_la]

    Current_rho = f"rho_Body{BodyID[count]}"
    rho[count] = locals()[Current_rho]

    Current_Vx0 = f"Vx0_Body{BodyID[count]}"
    Current_Vy0 = f"Vy0_Body{BodyID[count]}"
    Current_Vz0 = f"Vz0_Body{BodyID[count]}"
    v[count] = [locals()[Current_Vx0], locals()[Current_Vy0], locals()[Current_Vz0]]

    F[count] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    J[count] = 1.0
    
    InitialParticleVolume[count] = ParticleVolume[count]
    ParticleMass[count] = ParticleVolume[count] * rho[count]

    NormalInitial[count] = Normal[count]
    if SurfaceID[count] in FixedSurfaceIndex:
        FixedParticleIndex[count] = 1
    elif SurfaceID[count] in DisplacementSurfaceIndex:
        DisplacementParticleIndex[count] = SurfaceID[count]
    count += 1


gui = ti.GUI('Window Title', res=512, show_gui=False)
filepath = "movie"
vtkpath = "vtk"
os.getcwd()
if not os.path.exists(filepath):
    os.makedirs(filepath)
if not os.path.exists(vtkpath):
    os.makedirs(vtkpath)

TotalFrameNum = SimulationTime / dt            # Total number of frames 
VTKFileNum = AnimationTime * FramesPerSecond   # Total number of vtk files exported
Group = TotalFrameNum / VTKFileNum             # How many times to run "Substep" before one vtk file is saved

currentTime = 0e-15
while (currentTime < SimulationTime):         # While the simulation hasn't hit the maximum time

    xCoordinate = np.zeros(ParticleNum)
    yCoordinate = np.zeros(ParticleNum)
    zCoordinate = np.zeros(ParticleNum)
    Vx = np.zeros(ParticleNum)
    Vy = np.zeros(ParticleNum)
    Vz = np.zeros(ParticleNum)
    pressure = np.zeros(ParticleNum)
    sigma11 = np.zeros(ParticleNum)
    sigma22 = np.zeros(ParticleNum)
    sigma33 = np.zeros(ParticleNum)
    sigma12 = np.zeros(ParticleNum)
    deformation = np.zeros(ParticleNum)
    PoU = np.zeros(ParticleNum)
    Con = np.zeros(ParticleNum)
    PlasticStrain = np.zeros(ParticleNum)
    ParticleIterationDisplay = np.zeros(ParticleNum)
    DeltaDelta_P_Display = np.zeros(ParticleNum)

    ## Save all parameters to visualize in ParaView ##
    xCoordinate[:] = x.to_numpy()[:,0]
    yCoordinate[:] = x.to_numpy()[:,1]
    zCoordinate[:] = x.to_numpy()[:,2]
    Vx[:] = v.to_numpy()[:,0]
    Vy[:] = v.to_numpy()[:,1]
    Vz[:] = v.to_numpy()[:,2]
    pressure[:] = - ( ParticleStress.to_numpy()[:,0,0] + ParticleStress.to_numpy()[:,1,1] + ParticleStress.to_numpy()[:,2,2] ) / 3
    sigma11[:] = ParticleStress.to_numpy()[:,0,0]
    sigma22[:] = ParticleStress.to_numpy()[:,1,1]
    sigma33[:] = ParticleStress.to_numpy()[:,2,2]
    sigma12[:] = ParticleStress.to_numpy()[:,0,1]
    deformation[:] = J.to_numpy()[:]
    PoU[:] = PartitionOfUnity.to_numpy()[:]
    Con[:] = Consistency.to_numpy()[:]
    PlasticStrain[:] = P.to_numpy()[:]
    ParticleIterationDisplay[:] = PlasticIteration.to_numpy()[:]
    DeltaDelta_P_Display[:] = DeltaDelta_P.to_numpy()[:]
    
    ## Save parameters to vtk file ##
    pointsToVTK("./vtk/points"f'{gui.frame:06d}', xCoordinate, yCoordinate, zCoordinate, data = {
    "Surface ID" : SurfaceID.to_numpy(),
    "Body ID" : BodyIDNumpy,
    "Node ID" : NodeIDNumpy,
    "simgaxx" : sigma11, 
    "simgayy" : sigma22, 
    "simgazz" : sigma33, 
    "simgaxy" : sigma12, 
    "Vx" : Vx, 
    "Vy" : Vy, 
    "Vz" : Vz, 
    "Pressure" : pressure, 
    "Deformation": deformation, 
    "Partition of Unity": PoU, 
    "Consistency": Con,
    "Plastic Strain": PlasticStrain,
    "Plastic Iteration": ParticleIterationDisplay,
    "Delta Delta P": DeltaDelta_P_Display},
    fieldData = {"Velocity": np.concatenate((Vx,Vy,Vz),axis = 0)})
    
    ## Generate the snapshots for the "movie" folder ##
    colors = np.array([0x068587, 0xEEEEF0, 0xED553B], dtype=np.uint32) 
    gui.circles(x2D.to_numpy(), radius=2.0, color=colors[SurfaceID.to_numpy() * 0])
    gui.show('movie/'f'{gui.frame:06d}.png')
    
    for s in range(int(Group)):
        Substep()   # Run one timestep of the simulation 
        currentTime = currentTime + dt
    print("Current Time: ",currentTime)  # Print current time

time1 = time.time()
print('Run Time:', time1 - time0)   # Print total run time