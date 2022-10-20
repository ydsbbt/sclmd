import time

import netCDF4 as nc
import numpy as np
from sclmd.baths import ebath
from sclmd.lammpsdriver import lammpsdriver
from sclmd.md import md
from sclmd.tools import calHF

ds = nc.Dataset('grapheneLambda-r-0.3-ver2.nc', 'r')

lammpsinfile = [
    #"log none",
    #"atom_modify map yes",
    "units metal ",
    "dimension 3 ",
    "boundary p p f",
    "atom_style full",
    "read_data GC4e-full.data",
    "pair_style deepmd CH-yh-0.5.pb",
    "pair_coeff * * ",
    #"region reg_0   block   0 6 INF INF INF INF units   box",
    #"region reg_1   block   89 93.2 INF INF INF INF units   box",
    #"group  g_0 region  reg_0",
    #"group  g_1 region  reg_1",
    #"fix    g_0 g_0 setforce    0   0   0",
    #"fix    g_1 g_1 setforce    0   0   0",
    "min_style  cg",
    "minimize   1e-25   1e-25   5000    10000",
    #"unfix  g_0",
    #"unfix  g_1",
    "dump   1   all     xyz     1   GC4-minimize.xyz",
    "run    0",
]
# temperature
T = 300
delta = 0
nstart = 0
nstop = 40
# time = 0.658fs #time unit
dt = 0.5/0.658
# number of md steps
nmd = 2*10**5
# initialise lammps run
time_start = time.time()

#dp = deepmddriver('GC4e-atomic.data', 'lammps/lmp','CH-yh-0.5.pb', 'System', ['C', 'H'])
lmp = lammpsdriver(infile=lammpsinfile)

print("initialise md")
fixatoms = [range(0*3, (7+1)*3), range(226*3, (241+1)*3)]
# Molecular Junction atom indices
slist = list(range(111*3, (122+1)*3))

# atom indices that are connecting to debyge bath
ecatsl = list(range(8*3, (47+1)*3))
ecatsr = list(range(186*3, (225+1)*3))

# if slist is not given, md will initialize it using xyz
mdrun = md(dt, nmd, T, syslist=None, axyz=lmp.axyz, dyn=None,
           nstart=nstart, nstop=nstop)
# attache lammps driver to md
mdrun.AddPotential(lmp)

# unit in 0.658211814201041 fs
damp = 100/0.658211814201041
etal = (1.0/damp)*np.identity(len(ecatsl))
etar = (1.0/damp)*np.identity(len(ecatsr))
# atom indices that are connecting to bath
ebltemp = ebath(ecatsl, T*(1-delta/2), mdrun.dt, mdrun.nmd,
                wmax=2., nw=1000, bias=0, efric=etal, classical=False, zpmotion=False)
mdrun.AddBath(ebltemp)
ebrtemp = ebath(ecatsr, T*(1-delta/2), mdrun.dt, mdrun.nmd,
                wmax=2., nw=1000, bias=0, efric=etar, classical=False, zpmotion=False)
mdrun.AddBath(ebrtemp)

ebcbias = ebath(slist, T*(1+delta/2), mdrun.dt, mdrun.nmd,
                wmax=2., nw=1000, bias=1.0, efric=ds['eta_r'][:], exim=ds['xim_r'][:], exip=ds['xip_r'][:], zeta1=None, zeta2=None, classical=False, zpmotion=False)
mdrun.AddBath(ebcbias)

mdrun.AddConstr(fixatoms)
mdrun.noranvel()
mdrun.CalPowerSpec()
mdrun.AddPowerSection([ecatsl, slist, ecatsr])
mdrun.SaveTraj(1000)
# mdrun.CalAveStruct()
# mdrun.RemoveNC()
mdrun.Run()

calHF(dlist=1, bathnum=3)
time_end = time.time()
print('time cost', time_end-time_start, 's.')
