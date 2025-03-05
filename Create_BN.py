# create monolayer BN sheet with triangular holes

import numpy as np
import random
from matplotlib import pyplot as plt
import time
import warnings

random.seed(0)
# np.set_printoptions(threshold=np.inf)  # No ignored info in screen
start = time.perf_counter()
warnings.filterwarnings("ignore")

def Build_BN(X, Y, BL):
    # create perfect BN sheet
    # X, Y: boundary
    # BL: bond length

    # structure info
    uL = 2 * BL * np.cos(np.pi / 6)  # unit length along zigzag
    xn = int(np.floor(X / uL) + 1)  # unit number along x
    yn = int(np.floor(Y / (1.5 * BL)))  # unit number along y

    # create unit cell
    unit_B = np.zeros((2, 2))  # each unit cell contain two B atoms
    unit_N = np.zeros((2, 2))  # each unit cell contain two N atoms

    # coordinate of 4 atom (2 B, 2 N) in unit cell
    unit_B[0, :] = [0.5 * uL, 0]
    unit_B[1, :] = [0, BL * np.sin(np.pi / 6) + BL]
    unit_N[0, :] = [0, BL * np.sin(np.pi / 6)]
    unit_N[1, :] = [0.5 * uL, 2 * BL]

    # -------------------------------------------------create super cell
    cordata_1 = unit_B
    cordata_2 = unit_N

    for ix in range(0, xn):  # copy unit cell along x direction
        
        cordata_1 = np.append(cordata_1, np.zeros((2, 2)), axis=0)
        cordata_2 = np.append(cordata_2, np.zeros((2, 2)), axis=0)
        cordata_1[2 * (ix + 1):, 0] = cordata_1[2 * ix:2 * ix + 2, 0] + uL
        cordata_1[2 * (ix + 1):, 1] = cordata_1[2 * ix:2 * ix + 2, 1]
        cordata_2[2 * (ix + 1):, 0] = cordata_2[2 * ix:2 * ix + 2, 0] + uL
        cordata_2[2 * (ix + 1):, 1] = cordata_2[2 * ix:2 * ix + 2, 1]

    # yn must be an odd number to satisfy periodic boundary conditions
    if yn % 2 == 0:
        yn = yn + 1
    iyn = int(np.floor((yn - 1) / 2))  # once copy equal three units along y direction

    xatoms = 2 * (xn + 1)  # the number of atoms in each row along x direction
    for iy in range(0, iyn):  # copy unit cell along y direction
        cordata_1 = np.append(cordata_1, np.zeros((xatoms, 2)), axis=0)  # create empty saving space
        cordata_2 = np.append(cordata_2, np.zeros((xatoms, 2)), axis=0)
        cordata_1[(iy + 1) * xatoms:, 0] = cordata_1[iy * xatoms:(iy + 1) * xatoms, 0]
        cordata_1[(iy + 1) * xatoms:, 1] = cordata_1[iy * xatoms:(iy + 1) * xatoms, 1] + 3 * BL
        cordata_2[(iy + 1) * xatoms:, 0] = cordata_2[iy * xatoms:(iy + 1) * xatoms, 0]
        cordata_2[(iy + 1) * xatoms:, 1] = cordata_2[iy * xatoms:(iy + 1) * xatoms, 1] + 3 * BL

    # rotation the box and atoms, make the armchair direction as x direction
    cordata_1_build = np.zeros((len(cordata_1), 2))
    cordata_2_build = np.zeros((len(cordata_2), 2))
    cordata_1_build[:, 0] = cordata_1[:, 1]
    cordata_1_build[:, 1] = cordata_1[:, 0]
    cordata_2_build[:, 0] = cordata_2[:, 1]
    cordata_2_build[:, 1] = cordata_2[:, 0]
    cordata_1 = cordata_1_build
    cordata_2 = cordata_2_build

    # move to center
    cordata_1[:, 0] = cordata_1[:, 0] + 0.5 * BL
    cordata_2[:, 0] = cordata_2[:, 0] + 0.5 * BL

    return cordata_1, cordata_2  # atom coordinate:[[x0, y0], [x1, y1] ... ]






def SelectRandomTriangles_IDAndDirection(boundary, MeshSize, N_tri, pattern):
    # create candidate triangle center points by mesh box
    # gap: gap between the box and mesh boundaries
    # MeshSize: the number of node in each row
    # N_tri: the number of triangles
    # pattern : the number of patterns

    # determine the boundary of mesh
    boundary_xmin = boundary[0]
    boundary_xmax = boundary[1]
    boundary_ymin = boundary[2]
    boundary_ymax = boundary[3]
    disp_x = 1.5 * bL  # the displacement of unit in x direction       ***此处bl是键长
    disp_y = 2 * bL * np.cos(np.pi / 6)  # the displacement of unit in y direction

    # determine the initial mesh node
    mesh_unit_x = (boundary_xmax - boundary_xmin) / (MeshSize + 1)
    mesh_unit_y = (boundary_ymax - boundary_ymin) / (MeshSize + 1)
    mesh_unit_xn = np.floor(mesh_unit_x / disp_x)
    mesh_unit_yn = np.floor(mesh_unit_y / disp_y)
    if mesh_unit_xn % 2 == 1:  # the number of unit in x direction must be even
        mesh_unit_xn = mesh_unit_xn + 1
    mesh_unit_x = mesh_unit_xn * disp_x
    mesh_unit_y = mesh_unit_yn * disp_y

    # determine the first node cordata [x,y]
    node1 = [0, 0]
    node1[0] = boundary_xmin + mesh_unit_x
    node1[1] = boundary_ymin + mesh_unit_y
    move_yn = np.floor(0.5 * Tri_L * np.tan(np.pi / 6) / disp_y)
    if move_yn % 2 == 1:  # the number of unit in y direction must be even
        move_yn = move_yn + 1
    node1[1] = node1[1] + move_yn * disp_y  # the move length is multiple of disp_x/y

    node1[0] = node1[0] - 2 * disp_x

    # determine all mesh nodes' position
    MeshXY = np.zeros((MeshSize * MeshSize, 2))  # save coordinate of nodes: [[x0, y0], [x1, y1] ... ]
    for icr in range(MeshSize):
        MeshXY[icr * MeshSize:((icr + 1) * MeshSize), 1] = node1[1] + icr * mesh_unit_y
        for icc in range(MeshSize):
            MeshXY[icc + icr * MeshSize, 0] = node1[0] + icc * mesh_unit_x               #此处确定所有网格点的坐标位置

    # # determine the method of random selection
    # if Method == 1:
    #     # Be suitable for the volume of all possibilities is relatively small
    #     # list all the possibilities: select "N_tri" numbers from the range of "0-MeshSize*MeshSize"
    #     all_poss_id = list(combinations(np.linspace(0, MeshSize * MeshSize - 1, MeshSize * MeshSize), N_tri))
    #     all_poss_id = np.array(all_poss_id)  # change the type LIST to ARRAY
    #     # randomly select numbers(id)
    #     random_id = random.sample(range(0, len(all_poss_id)), pattern)
    #     TriCp_mesh_id = all_poss_id[random_id, :]  # each patter corresponds triangles id: [[id1, id2], [id1, id2] ... ]
    #
    # elif Method == 2:
    #     # Be suitable for the volume of all possibilities is relatively large
    #     TriCp_mesh_id = np.zeros((pattern, N_tri))
    #     for iTri in range(pattern):
    #         random_i = np.zeros((1, N_tri))
    #         while random_i in TriCp_mesh_id:
    #             random_i = random.sample(range(MeshSize * MeshSize), N_tri)
    #         TriCp_mesh_id[iTri, :] = random_i

    TriCp_mesh_id = np.zeros((pattern, N_tri))
    TriCp_mesh_dir = np.zeros((pattern, N_tri))
    for iPat in range(pattern):
        random_iid = np.ones((1, N_tri))
        random_idir = np.ones((1, N_tri))
        id_random_iid = []
        id_random_idir = []
        while id_random_iid == id_random_idir:
            random_iid = random.sample(range(MeshSize * MeshSize), N_tri)
            random_idir = [random.randint(0, 1) for _ in range(N_tri)]
            id_random_iid = np.where(np.all(TriCp_mesh_id == random_iid, axis=1))
            id_random_idir = np.where(np.all(TriCp_mesh_dir == random_idir, axis=1))
        TriCp_mesh_id[iPat, :] = random_iid
        TriCp_mesh_dir[iPat, :] = random_idir


    return MeshXY, TriCp_mesh_id, TriCp_mesh_dir


def CreateTriangle(center, L, direction):      #center为竖边中点，L为三角形边长，direction是三角顶角的朝向
    # node coordinate of triangle: [[x0, y0], [x1, y1] ... [x2, y2], [x0, x0]] MUST BE COUNTERCLOCKWISE !!!!
    TriXY = np.zeros((4, 2))
    if direction == 'right':
        TriXY[0, 0] = center[0]
        TriXY[0, 1] = center[1] - L * np.cos(np.pi / 3)
        TriXY[1, 0] = TriXY[0, 0] + L * np.cos(np.pi / 6)
        TriXY[1, 1] = TriXY[0, 1] + L * np.cos(np.pi / 3)
        TriXY[2, 0] = TriXY[0, 0]
        TriXY[2, 1] = TriXY[0, 1] + L
        TriXY[3, :] = TriXY[0, :]
    elif direction == 'left':
        TriXY[0, 0] = center[0]
        TriXY[0, 1] = center[1]
        TriXY[1, 0] = TriXY[0, 0] + L * np.cos(np.pi / 6)
        TriXY[1, 1] = TriXY[0, 1] + L * np.cos(np.pi / 3)
        TriXY[2, 0] = TriXY[0, 0] + L * np.cos(np.pi / 6)
        TriXY[2, 1] = TriXY[0, 1] - L * np.cos(np.pi / 3)
        TriXY[3, :] = TriXY[0, :]
    return TriXY


def CreateHoles(cordata_1, cordata_2, CenterXY, dir):
    # create triangular holes according to their center points
    # CenterXY: center point coordinate: [[x0, y0], [x1, y1] ... ]
    # triangle_L: edge length of triangle (unit: A)

    def isRayIntersectsSegment(poi, s_poi, e_poi):
        # 射线与边是否有交点 :[x,y] [lng,lat]
        if s_poi[1] == e_poi[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
            return False
        if s_poi[1] > poi[1] and e_poi[1] > poi[1]:
            return False
        if s_poi[1] < poi[1] and e_poi[1] < poi[1]:
            return False
        if s_poi[1] == poi[1] and e_poi[1] > poi[1]:
            return False
        if e_poi[1] == poi[1] and s_poi[1] > poi[1]:
            return False
        if s_poi[0] < poi[0] and e_poi[1] < poi[1]:
            return False
        xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])  # 求交
        if xseg < poi[0]:
            return False
        return True

    def isPoiWithinSimplePoly(poi, simPoly):
        # #只适用简单多边形: 点；多边形数组
        # simPoly=[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]], , MUST BE COUNTERCLOCKWISE
        polylen = len(simPoly)
        sinsc = 0  # 交点个数
        for i in range(polylen - 1):
            s_poi = simPoly[i]
            e_poi = simPoly[i + 1]
            if isRayIntersectsSegment(poi, s_poi, e_poi):
                sinsc += 1
        return True if sinsc % 2 == 1 else False

    def DeleteAtomsWithinPoly(cordata, poly):     #删除相应原子
        for ip in range(len(cordata)):
            if isPoiWithinSimplePoly(cordata[ip, :], poly):
                cordata[ip, 1] = -10
        idp = np.where(cordata[:, 1] == -10)
        cordata_new = np.delete(cordata, idp, axis=0)
        return cordata_new

    cordata_triangle = CreateTriangle(CenterXY, Tri_L, dir)  # give node coordinate of triangle
    cordata_1_new = DeleteAtomsWithinPoly(cordata_1, cordata_triangle)  # delete type 1 atoms with in triangle
    cordata_2_new = DeleteAtomsWithinPoly(cordata_2, cordata_triangle)  # delete type 2 atoms with in triangle
    return cordata_1_new, cordata_2_new, cordata_triangle


def Write_LmpData(cordata_1, cordata_2, Path):  # create lammps data
    # # rotation the box and atoms, make the zigzag direction as x direction
    # cordata_1_build = np.zeros((len(cordata_1), 2))
    # cordata_2_build = np.zeros((len(cordata_2), 2))
    # cordata_1_build[:, 0] = cordata_1[:, 1]
    # cordata_1_build[:, 1] = cordata_1[:, 0]
    # cordata_2_build[:, 0] = cordata_2[:, 1]
    # cordata_2_build[:, 1] = cordata_2[:, 0]

    # move to center
    # cordata_1_build[:, 0] = cordata_1_build[:, 0] + 0.5 * BL
    # cordata_2_build[:, 0] = cordata_2_build[:, 0] + 0.5 * BL
    # cordata_1 = cordata_1_build
    # cordata_2 = cordata_2_build

    # cordata_1[:, 0] = cordata_1[:, 0] + 0.5 * BL
    # cordata_2[:, 0] = cordata_2[:, 0] + 0.5 * BL

    # reform cordata
    # finaldata = [id type x y z]
    atoms = len(cordata_1) + len(cordata_2)
    finaldata = np.zeros((atoms, 5))
    finaldata[0:len(cordata_1), 1] = 1
    finaldata[0:len(cordata_1), 2:4] = cordata_1
    finaldata[len(cordata_1):, 1] = 2
    finaldata[len(cordata_1):, 2:4] = cordata_2
    finaldata[:, 4] = 0.5 * Lz
    finaldata[:, 3] = finaldata[:, 3] + 20

    # idd = np.where(finaldata[:, 3] > (Y - 0.5))
    # finaldata = np.delete(finaldata, idd, axis=0)

    # box size
    X = np.max(cordata_2[:, 0]) + 40
    Y = np.max(cordata_2[:, 1]) + 40
    Z = Lz + 20
    # Y = np.max(cordata_2[:, 1]) + BL * np.cos(np.pi/6)

    finaldata[:, 0] = np.arange(1, len(finaldata) + 1)
    Natoms = len(finaldata)

    # write and save data
    f = open(Path, "w")
    f.write("# BN sheet\n\n")
    f.write(str(Natoms) + " atoms\n\n")
    f.write("2 atom types\n\n")
    f.write(format(-20, '.5f') + " " + format(X-20, '.5f') + " xlo xhi\n")
    f.write(format(0, '.5f') + " " + format(Y, '.5f') + " ylo yhi\n")
    f.write(format(0, '.5f') + " " + format(Z, '.5f') + " zlo zhi\n\n")
    f.write("Masses\n\n")
    f.write("1 10.81\n")
    f.write("2 14.0067\n\n")
    f.write("Atoms\n\n")
    for i in range(finaldata.shape[0]):
        f.write(format(finaldata[i, 0], '.0f') + " ")
        f.write(format(finaldata[i, 1], '.0f') + " ")
        f.write(format(finaldata[i, 2], '.5f') + " ")
        f.write(format(finaldata[i, 3], '.5f') + " ")
        f.write(format(finaldata[i, 4], '.5f') + " ")
        f.write('\n')


def Create_Png(Path, i, meshxy, tri_id_dir):
    # save to grayscale figures
    # plt.figure(i, facecolor='gray', figsize=(4, 4))
    plt.figure(i, facecolor='black', figsize=(4, 4))
    for iT in range(N_triangle):
        TriCp = meshxy[int(tri_id_dir[0, iT]), :]
        if tri_id_dir[1, iT] == 1:
            Dir = 'right'
        else:
            Dir = 'left'
        TriNp = CreateTriangle(TriCp, Tri_L, Dir)
        plt.fill(TriNp[:, 0], TriNp[:, 1], fc='white')
    plt.xlim((0, Boundary[1]))
    plt.ylim((0, Boundary[3]))
    plt.axis('off')
    plt.savefig(Path, bbox_inches='tight', dpi=200, pad_inches=0.0)
    plt.close('all')


if __name__ == '__main__':

    print("CODE RUNNING".center(40, "*"))

    # the geometric parameters of BN sheet (unit: A)
    Lx = 400
    Ly = 400
    Lz = 40
    bL = 1.45  # B-N bond length
    uL = 2 * bL * np.cos(np.pi / 6)  # unit length in zigzag direction

    N_pattern = 100  # the number of random generated patterns
    N_triangle = 10  # the number of triangular holes in hBN
    N_mesh = 5  # node number

    tricenter=np.zeros((20,2))
    trixy=np.zeros((60,2))

    Tri_unit_N = 13  # The number of atoms in hole edge, MUST BE EVEN（13）
    Tri_L = (Tri_unit_N + 1) * uL  # triangle edge length        

    (cordata_B, cordata_N) = Build_BN(Lx, Ly, bL)  # create the initial BN sheet without defects
    xmax = np.max(cordata_N[:, 0]) + 0.5 * bL  # the box size in x direction
    ymax = np.max(cordata_N[:, 1])  # the box size in y direction
    Boundary = [0.0, xmax, 0.0, ymax]  # the box edge

    BNSheet_total = np.zeros(shape=(N_pattern, N_mesh, N_mesh))  # save the ML input data

    if N_triangle == 0:

        LmpData_SavePath = 'LmpData_SavePath'
        Write_LmpData(cordata_B, cordata_N, LmpData_SavePath)  # save as Lammps data file

    else:

        # select 'N_pattern' patterns with 'N_triangle' random triangles
        (cordata_mesh, trianglescp_mesh_id, trianglescp_mesh_dir) = SelectRandomTriangles_IDAndDirection(Boundary,
                                                                                                         N_mesh,
                                                                                                         N_triangle,
                                                                                                         N_pattern)
        for iN in range(N_pattern):
            png_savepath = 'png_savepath' + str(iN + 1) + '.png'
            lmpdata_savepath = 'lmpdata_savepath' + 'BN_' + str(iN + 1) + '.data'

            (cordata_B, cordata_N) = Build_BN(Lx, Ly, bL)  # create the initial BN sheet without defects


            triangles_id = trianglescp_mesh_id[iN, :]  # current triangle center point ID
            triangles_dir = trianglescp_mesh_dir[iN, :]  # current each triangle direction
            triangles_id_dir = np.append([triangles_id], [triangles_dir], axis=0)
            # triangles_id_dir = np.array([[0, 0], [1, 1], [2, 1], [3, 0]])  # current triangle center point ID

            # generate ML input data
            BNSheet = np.zeros(N_mesh*N_mesh)

            # create holes
            for iP in range(N_triangle):
                # the center coordinate of current triangular hole
                cordata_trianglescp = cordata_mesh[int(triangles_id_dir[0, iP]), :]
                if triangles_id_dir[1, iP] == 1:
                    DirectionType = 'right'
                    BNSheet[int(triangles_id_dir[0, iP])] = 0.5
                else:
                    DirectionType = 'left'
                    BNSheet[int(triangles_id_dir[0, iP])] = 1
                (cordata_B, cordata_N, cordata_triangle) = CreateHoles(cordata_B, cordata_N, cordata_trianglescp, DirectionType)
                tricenter[iP,:]=cordata_triangle[0,:]
                trixy[3*iP:3*(iP+1),:]=cordata_triangle[1:4,:]
         
            Write_LmpData(cordata_B, cordata_N, lmpdata_savepath)  # create lammps data
            Create_Png(png_savepath, iN, cordata_mesh, triangles_id_dir)
            BNSheet = np.reshape(BNSheet, (N_mesh, N_mesh))
            BNSheet = BNSheet[::-1]
            BNSheet_total[iN] = BNSheet

            # Time bar
            dur = time.perf_counter() - start
            print("{}/{} patterns have been constructed.  Run time: {:.2f}s\n".format(iN+1, N_pattern, dur), end="")
            time.sleep(0.1)

            # # visualization
            # plt.figure(iN)
            # plt.scatter(cordata_B[:, 0], cordata_B[:, 1], marker='o', c='red', alpha=0.5, s=10)
            # plt.scatter(cordata_N[:, 0], cordata_N[:, 1], marker='o', c='blue', alpha=0.5, s=10)
            # plt.axis('equal')
            # # plt.savefig(str(iN) + '.png')
            # plt.show()
            # plt.close('all')
            # plt.show()
            # plt.close('all')
        np.save('./pattern.npy', BNSheet_total)
    print("ALL DONE".center(40, "*"))