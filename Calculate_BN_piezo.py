from mpi4py import MPI
from lammps import lammps
import numpy as np
from matplotlib import pyplot as plt

block_1 = """
# calculate piezoelectric coefficient of BN sheet

##---------------------------------------------metal units（energy:eV  length:A  time:ps）

units       metal
dimension   3
boundary    p p f
atom_style  atomic

##------------------------------------------------------------------read data & potential

"""
block_2 = """
pair_style  tersoff
pair_coeff  * * BNC.tersoff B N

##-------------------------------------------------- --------------------variable setting

variable T  equal 300
variable DT equal 0.001
timestep ${DT}

# store initial cell length
variable mylx equal "lx"
variable lx0 equal ${mylx}

##--------------------------------------------------------------------------- group set





group      B type 1
group      N type 2

variable   BNL        equal 400   # the length of BN sheet
variable   BL         equal 1.45
variable   LeftBound  equal 2.5*${BL}
variable   RightBound equal ${lx0}-2.5*${BL}

region     LeftFixed    block      INF             ${LeftBound}   INF INF INF INF
region     RightFixed   block      ${RightBound}   INF            INF INF INF INF
group      Left         region     LeftFixed
group      Right        region     RightFixed
group      Boundary     union      Left  Right
group      Mobile       subtract   all   Boundary

##----------------------------------------------------------------------------- relaxing

print "*************MINIMIZATION***************"

fix             1 all box/relax x 0.0 y 0.0
min_style       cg
minimize        1e-12  1e-12  1000  10000
unfix           1

print "*************RELAXING***************"

reset_timestep 	0
fix             f1 Boundary setforce 0.0 0.0 0.0
velocity		Mobile create ${T} 1234 mom yes rot yes dist gaussian
fix             NPT all npt temp ${T} ${T} 1 x 0 0 1 y 0 0 1
thermo			10000
thermo_style	custom step temp pxx pyy pzz etotal atoms
dump			xyz_1  all custom 10000 relax.lammpstrj id type x y z
run				100000
undump			xyz_1
unfix           NPT

##----------------------------------------------------------------------------- tensile

change_box all boundary s p f

print "*************TENSILE***************"

# peizoelectric parameters setting
compute     corB    B property/atom x y z
compute     corN    N property/atom x y z

variable    e       equal   -1.6e-19                    # quantity of electric charge (unit:C)
variable    Area    equal   ${BNL}*${BNL}               # area of BN (unit:A^2)
variable    PBxa    atom    c_corB[1]*${e}*3            # B polarization along x direction P1
variable    PNxa    atom    c_corN[1]*${e}*(-3)         # N polarization along x direction P1
compute     Px      Mobile  reduce sum v_PBxa v_PNxa    # summation of polarization
variable    Px      equal   "(c_Px[1]+c_Px[2])/(v_Area*10e-10)"

# tensile parameters setting
variable    preS       equal 0.05                       # pre-tensile strain
variable    v          equal 1.0                        # velocity (unit:1 A/ps = 0.001 A/step)
variable    run_preT   equal ${BNL}*${preS}/0.001       # run to strian = 0.05
variable    mysnxx      equal "(0.001/v_BNL)*step"      # x-strain

# stress parameters setting
compute     perss       all     stress/atom NULL virial
variable	perssxx     atom	c_perss[1]/10000
variable	perssyy     atom	c_perss[2]/10000
variable	perssxy     atom	c_perss[4]/10000
compute     SSxx        all     reduce sum v_perssxx
variable    SSxx        equal   c_SSxx/(${Area}*3.3)

fix         NVT     all nvt temp ${T} ${T} 0.01
velocity    Right   set ${v} 0.0 0.0  sum yes units box
velocity    Mobile ramp vx 0.0 ${v} x ${LeftBound} ${RightBound} sum yes

# Pre-tensile
reset_timestep 	0
thermo			1000
thermo_style	custom step temp lx v_mysnxx v_Px etotal atoms
thermo_modify   lost ignore
run				2000从                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

# tensile
reset_timestep 	0
thermo			1000
thermo_style	custom step temp lx v_mysnxx v_Px etotal atoms

"""
block_3 = """
run				${run_preT}
undump			xyz_2
unfix           NVT

print "*************ALL DONE***************"
"""

plt.figure()
# plt.xlim((0, 0.05))
# plt.ylim((-0.005, 0.1))
N_data = 1000
E_value = np.zeros((N_data, 1))
array_E = []


for iL in range(int(N_data)):
    lmp = lammps()
    lmpdata_savepath = 'lmpdata_savepath/' + 'BN_' + str(iL + 1) + '.data'
    lmp.command('print "******************* BN_' + str(iL + 1) + '.data ******************"')
    lmp.commands_string(block_1)
    lmp.command('read_data ' + lmpdata_savepath)
    lmp.commands_string(block_2)
    lmp.command('dump xyz_2   all custom 1000 def_' + str(iL + 1) + '.lammpstrj id type x y z v_perssxx')
    lmp.command('fix output  all print 500 "${mysnxx} ${Px}" file P_' + str(iL + 1) + '.txt screen no')
    lmp.commands_string(block_3)
    lmp.close()

    # linear fitting
    data = ('P_' + str(iL + 1) + '.txt')
    PValue_input = np.loadtxt(data, dtype=np.float32, delimiter=' ', skiprows=1)
    PValue = PValue_input
    PValue[:, 1] = PValue[:, 1] * 10 ** 10
    PBasic = PValue[0, 1]
    PValue[:, 1] = PValue[:, 1] - PBasic
    LinearFit = np.polyfit(PValue[:, 0], PValue[:, 1], deg=1)
    PLinear = np.polyval(LinearFit, PValue[:, 0])
    # print(LinearFit)
    E_value[iL] = LinearFit[0]

    # visualization
    line = plt.scatter(PValue[:, 0], PValue[:, 1], marker='o')
    # plt.legend(line, str(iL+1))
    plt.plot(PValue[:, 0], PLinear)
    plt.savefig('linear.png')
    # plt.show()

    array_E = np.append(array_E, LinearFit[0])
    np.save('E.npy', array_E)

    # # save piezoelectric coefficient
    np.savetxt('E.txt', E_value, fmt='%.5f')
