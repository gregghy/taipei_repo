import sys as sys
import gmsh
import numpy as np
import matplotlib.pyplot as plt
import os as os


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


MeshNum = len(MeshFileIndex)           # Define the number of meshes used 

Dim = 3                                # Problem Dimension
DimVol = 3                             # Volume Dimension
DimSurf = 2                            # Surface Dimension
SurfaceCount = 0                       # Surface counter index
NodeCount = 0                          # Node counter index

gmsh.initialize(sys.argv)
for m in range(MeshNum):   
    gmsh.open(MeshFileIndex[m]) 
    gmsh.model.mesh.createGeometry()

    BoundaryElementNumber = 0              # Initialize Number of Mesh Elements on the Surface Boundary
    VolumeElementNumber = 0                # Initialize Number of Mesh Elements within the Volume
    for e in gmsh.model.getEntities():     
        dimEntity = e[0]
        tagEntity = e[1]
        if dimEntity == DimVol:
            _, VolumeElementTag, _ = gmsh.model.mesh.getElements(dimEntity, tagEntity)
            VolumeElementNumber += len(VolumeElementTag[0])
        elif dimEntity == DimSurf:
            _, boundaryElemTag, _ = gmsh.model.mesh.getElements(dimEntity, tagEntity)
            BoundaryElementNumber += len(boundaryElemTag[0])
    ParticleNum = VolumeElementNumber + BoundaryElementNumber

    xTemp = np.zeros((ParticleNum, Dim))
    normalTemp = np.zeros((ParticleNum, Dim))
    BodyIDTemp = np.zeros(ParticleNum)
    NodeIDTemp = np.zeros(ParticleNum)
    SurfaceIDTemp = np.zeros(ParticleNum)
    ParticleVolumeTemp = np.zeros(ParticleNum)
    ParticleSurfaceAreaTemp = np.zeros(ParticleNum)

    BoundaryCount = 0
    newSurface = True
    newBody = False
    for e in gmsh.model.getEntities():
        dimEntity = e[0]
        tagEntity = e[1] 
        ElemTypes, ElemTags, ElemNodeTags = gmsh.model.mesh.getElements(dimEntity, tagEntity) 

        if len(ElemTags):
            for i in range(len(ElemTags[0])):
                if dimEntity == DimVol:
                    vertexNum = 4
                    SurfaceIDTemp[BoundaryCount] = 0
                elif dimEntity == DimSurf:
                    vertexNum = 3
                    if newSurface:
                        SurfaceCount += 1
                        newSurface = False
                    SurfaceIDTemp[BoundaryCount] = SurfaceCount
                else:
                    continue
                nodeCoordAvg = np.zeros(Dim)
                nodeCoordTot = np.zeros((vertexNum, Dim))
                for j in range(vertexNum):
                    nodeCoord, _, _, _ = gmsh.model.mesh.getNode(ElemNodeTags[0][vertexNum * i + j])
                    nodeCoordAvg += nodeCoord
                    nodeCoordTot[j,:] = nodeCoord
                
                nodeCoordAvg = nodeCoordAvg / vertexNum
                xTemp[BoundaryCount,:] = nodeCoordAvg
                BodyIDTemp[BoundaryCount] = m

                if dimEntity == DimSurf:
                    surfCenterCoords = gmsh.model.getParametrization(dimEntity, tagEntity, nodeCoordAvg)
                    norm = gmsh.model.getNormal(tagEntity, surfCenterCoords)
                    normalTemp[BoundaryCount] = norm
                    ParticleSurfaceAreaTemp[BoundaryCount] = 0.5 * np.linalg.norm(np.cross((nodeCoordTot[1,:] - nodeCoordTot[0,:]), (nodeCoordTot[2,:],nodeCoordTot[0,:])))
                    ParticleVolumeTemp[BoundaryCount] =  (1 / 6) * np.linalg.norm(np.cross(nodeCoordTot[1,:] - nodeCoordTot[0,:], nodeCoordTot[2,:] - nodeCoordTot[0,:])) * (1 / 2) * np.linalg.norm((nodeCoordTot[1,:] - nodeCoordTot[0,:]) + (nodeCoordTot[2,:] - nodeCoordTot[0,:]))
                else:
                    ParticleVolumeTemp[BoundaryCount] = (1 / 6) * np.linalg.norm(np.dot(np.cross(nodeCoordTot[1,:] - nodeCoordTot[0,:], nodeCoordTot[2,:] - nodeCoordTot[0,:]), nodeCoordTot[3,:] - nodeCoordTot[0,:]))  
                NodeIDTemp[BoundaryCount] = NodeCount
                BoundaryCount += 1
                NodeCount += 1
            newSurface = True

    if m == 0:
        x = xTemp
        normal = normalTemp
        BodyID = BodyIDTemp
        NodeID = NodeIDTemp
        SurfaceID = SurfaceIDTemp
        ParticleVolume = ParticleVolumeTemp
        ParticleSurfaceArea = ParticleSurfaceAreaTemp
    else:
        x = np.append(x, xTemp, axis=0)
        normal = np.append(normal, normalTemp, axis=0)
        BodyID = np.append(BodyID, BodyIDTemp)
        NodeID = np.append(NodeID, NodeIDTemp)
        SurfaceID = np.append(SurfaceID, SurfaceIDTemp)
        ParticleVolume = np.append(ParticleVolume, ParticleVolumeTemp)
        ParticleSurfaceArea = np.append(ParticleSurfaceArea, ParticleSurfaceAreaTemp)


fig = plt.figure(dpi=1000)
ax = fig.add_subplot(projection='3d')
ax.scatter(x[:,0], x[:,1], x[:,2], c='k')
arrow_color = plt.cm.Reds(0.75)
ax.quiver(x[:,0], x[:,1], x[:,2], normal[:,0], normal[:,1], normal[:,2], length=0.06, colors=arrow_color, arrow_length_ratio=0.05)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0.0,1.0)
ax.set_ylim(0.0,1.0)
ax.set_zlim(0.0,1.0)

filepath = 'Initialize'
if not os.path.exists(filepath):
    os.makedirs(filepath)

plt.savefig('Initialize/Plot_ParticleDistribution.png')
np.savetxt('Initialize/Particle_Positions.csv', x, delimiter=',')
np.savetxt('Initialize/Particle_Normals.csv', normal, delimiter=',')
np.savetxt('Initialize/Particle_SurfaceID.csv', SurfaceID, delimiter=',')
np.savetxt('Initialize/Particle_Volume.csv', ParticleVolume, delimiter=',')
np.savetxt('Initialize/Particle_SurfaceArea.csv', ParticleSurfaceArea, delimiter=',')
np.savetxt('Initialize/Particle_BodyID.csv', BodyID, delimiter=',')
np.savetxt('Initialize/Particle_NodeID.csv', NodeID, delimiter=',')