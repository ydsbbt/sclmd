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
atomgroup0 = range(24*3, 32*3)
atomgroup1 = range(32*3, 40*3)
mode = sig(infile, 0.12, atomgroup0, atomgroup1,
           dofatomfixed=[[], []], dynmatfile=None, num=2000, eta=0.164e-3)
mode.getse()
mode.gettm()
# mode.plotresult()
print('time cost', time.time()-time_start, 's')
