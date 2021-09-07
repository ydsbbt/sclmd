import time
from sclmd.selfenergy import sig
infile = [
    'atom_style full',
    'units metal',
    'boundary p p p',
    'read_data sig.data',
    'pair_style rebo',
    'pair_coeff * * CH.rebo C',
    'min_style  cg',
    'minimize   1e-25   1e-25   5000    10000',
    'dump 1 all xyz 1 dump.xyz',
    'run 0',
]
time_start = time.time()
dofatomK10 = range(204*3, 306*3)
dofatomK00 = range(306*3, 408*3)
dofatomK01 = range(408*3, 510*3)
#atomfixed = [range(0*3, (19+1)*3), range(181*3, (200+1)*3)]
mode = sig(infile, dofatomK00, dofatomK01, dofatomK10)
mode.K00()
mode.K01()
mode.K10()
mode.sgf()
mode.selfenergy()
print('time cost', time.time()-time_start, 's')
