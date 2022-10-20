import time
import numpy as np
from sclmd.negf import bpt
from matplotlib import pyplot as plt
import netCDF4 as nc

ds = nc.Dataset('grapheneLambda-r-0.3-ver2.nc', 'r')
lammpsinfile = [
    #"log none",
    "units metal ",
    "dimension 3 ",
    "boundary p p f",
    "atom_style full",
    "read_data GC4e-full.data",
    "pair_style deepmd CH-yh-0.5.pb",
    "pair_coeff * * ",
    #"region reg_0   block   0 4.5 INF INF INF INF units   box",
    #"region reg_1   block   61 65.5 INF INF INF INF units   box",
    #"group  g_0 region  reg_0",
    #"group  g_1 region  reg_1",
    #"fix    g_0 g_0 setforce    0   0   0",
    #"fix    g_1 g_1 setforce    0   0   0",
    "min_style  cg",
    "minimize   1e-25   1e-25   10000    10000",
    #"unfix  g_0",
    #"unfix  g_1",
    #"dump   1   all     xyz     1   C10-minimize.xyz",
    "run    0",
]
time_start = time.time()
atomfixed = [range(0*3, (7+1)*3), range(226*3, (241+1)*3)]
atomofbath = [range(8*3, (47+1)*3), range(186*3, (225+1)*3)]
atomofcenter = range(111*3, (122+1)*3)

mybpt = bpt(infile=lammpsinfile, maxomega=0.5, damp=0.1, dofatomofbath=atomofbath,
            dofatomfixed=atomfixed, dynmatfile=None, num=10000)

centerdamp = ds['eta_r'][:]/0.658211814201041*1000
chiplus = ds['xip_r'][:]/0.658211814201041*1000
chiminus = ds['xim_r'][:]/0.658211814201041*1000

delta = 0.0001
omegas = [0-delta, 0.364603, 0.366568, 0.368462, 0.370482,
          0.37195, 0.375304, 0.376315, 0.378235, 0.5-delta]
omegalist = []
omegalist = np.array(omegalist)
for i in range(len(omegas)-1):
    omegalist = np.append(omegalist, np.linspace(
        omegas[i]+delta, omegas[i+1]-delta, 3000))
    omegalist = np.append(omegalist, np.linspace(
        omegas[i+1]-delta, omegas[i+1]+delta, 10000))

mybpt.setbias(0, bdamp=centerdamp, chiplus=chiplus,
              chiminus=chiminus, dofatomofbias=atomofcenter)
mybpt.getps(300, 0.5, 10000, atomofcenter,
            filename='biascenter0', omegalist=omegalist)

mybpt.setbias(0.6, bdamp=centerdamp, chiplus=chiplus,
              chiminus=chiminus, dofatomofbias=atomofcenter)
mybpt.getps(300, 0.5, 10000, atomofcenter,
            filename='biascenter6', omegalist=omegalist)

mybpt.setbias(1.0, bdamp=centerdamp, chiplus=chiplus,
              chiminus=chiminus, dofatomofbias=atomofcenter)
mybpt.getps(300, 0.5, 10000, atomofcenter,
            filename='biascenter10', omegalist=omegalist)

time_end = time.time()
print('time cost', time_end-time_start, 's.')