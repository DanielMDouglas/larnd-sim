import numpy as np

# a bit wider than the generation volume
# xMin, xMax, xWidth = 410.0, 920.0, 0.38
# yMin, yMax, yWidth = -225.0, 85.0, 0.38
# zMin, zMax, zWidth = -305.0, 405.0, 0.38
xMin, xMax, xWidth = 400.0, 950.0, 0.38
yMin, yMax, yWidth = -250.0, 100.0, 0.38
zMin, zMax, zWidth = -350.0, 450.0, 0.38

nVoxX = int((xMax - xMin)/xWidth)
nVoxY = int((yMax - yMin)/yWidth)
nVoxZ = int((zMax - zMin)/zWidth)

minVox = np.array([xMin, yMin, zMin])
maxVox = np.array([xMax, yMax, zMax])

spacing = np.array([xWidth, yWidth, zWidth])

trackVoxelEdges = (np.linspace(xMin, xMax, nVoxX + 1),
                   np.linspace(yMin, yMax, nVoxY + 1),
                   np.linspace(zMin, zMax, nVoxZ + 1))

# idea #1
# probably less efficient, but more accurate
# for each voxel:
#     dE = 0
#     for each track:
#         is the track intersecting?
#             is it through-going?
#                 dx = np.linalg.mag(pointOfEntry - pointOfExit)
#             is it starting/ending?
#                 dx = np.linalg.mag(pointOfEntry/Exit - end/startPoint)
#             is it starting + stopping?
#                 dx = track['length']
#          dE += track['dEdx']*dx

# idea #2
# much more efficient, but also less accurate
# set sampleDensity
# create sample points along the track [0, 1] with that density
# (nSamples = sampleDensity*track['lenght'])
# do the 3D histogram with the voxel scheme, weight by track['dE']/nSamples
# take only non-zero bin values
# error goes like voxelDensity/sampleDensity

# this is idea #2
from collections import defaultdict

def voxelize(tracks):
    sampleDensity = 100000 # samples per unit (mm) length

    # print ('voxelizing tracks')
    # for track in tqdm.tqdm(tracks):
    for track in tracks:
        start = np.array([track['x_start'],
                          track['y_start'],
                          track['z_start']])
        end = np.array([track['x_end'],
                        track['y_end'],
                        track['z_end']])

        nSamples = int(track['dx']*sampleDensity)

        trackSampleSpace = np.expand_dims(np.linspace(0, 1, nSamples), -1)
        # sample with a fixed spacing along the track trajectory
        trackSamplePoints = start + trackSampleSpace*(end - start)
        # give each sample the approprate amount of dE
        trackSampleWeights = np.ones(trackSamplePoints.shape[0])*track['dE']/nSamples

        if not 'samplePoints' in dir():
            samplePoints = trackSamplePoints
            sampleWeights = trackSampleWeights
        else:
            samplePoints = np.concatenate([samplePoints,
                                           trackSamplePoints])
            sampleWeights = np.concatenate([sampleWeights,
                                            trackSampleWeights])

    ind = np.cast[int]((samplePoints - minVox)//spacing)

    # get the bin centers that correspond to each track sample
    print (samplePoints)
    print (ind)
    binCenters = np.array([trackVoxelEdges[i][ind[:,i]] + 0.5*spacing[i]
                           for i in range(3)])

    # sum the energy in voxels that are occupied
    voxelContent = defaultdict(int)
    for coord, w in zip(binCenters.T, sampleWeights):
        voxelContent[tuple(coord)] += w

    return list(voxelContent.keys()), list(voxelContent.values())
