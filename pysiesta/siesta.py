#!/usr/bin/env python

import numpy as np
import pysiesta as PS

print(PS.__doc__)
print(PS.pysiestaforce.__doc__)
print(PS.pysiestalaunch.__doc__)
print(PS.pysiestaunits.__doc__)

#1D array gives the coordinates of each atom
#[xi,yi,zi],i=1,na
x = [0.0, 0.0, 0.0,0.7, 0.7, 0.0,-0.7, 0.7, 0.0]
xa = np.array(x,dtype='d')
#1D array gives the unit cell
#x1,y1,z1,x2,y2,z2,x3,y3,z3
c = [10.0,0.0,0.0,0.0,10.0,0.0,0.0,0.0,10.0]
cell = np.array(c,dtype='d')

label = "h2o.fast"
ll = len(label)

#set units to : Angstrom, eV
PS.pysiestaunits("Ang", "eV")

#PS.pysiestalaunch(label, 8, 'mpirun -np')	#parallel siesta
PS.pysiestalaunch(label,2)	#serial siesta
print("siesta launched!")
energy,force = PS.pysiestaforce( label , xa, cell )

PS.pysiestaquit(label)
print("siesta quit!")
print("Siesta Energy: ", energy)
print("Siesta force: ", force)
