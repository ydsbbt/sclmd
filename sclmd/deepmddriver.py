from lammps import lammps
from sclmd.tools import get_atomname
import sys

import numpy as np
from deepmd.infer import DeepPot


class deepmddriver():
    def __init__(self, pbinfile, lmpinfile, cell=np.diag(10 * np.ones(3)).reshape([1, -1]),
                 atype=[1, 0, 1], label="",
                 constraints=[], tdir="./", lunit="Ang", eunit="eV", md2ang=0.06466,
                 ):
        self.pbinfile = pbinfile
        self.lmpinfile = lmpinfile
        self.cell = cell
        self.atype = atype
        self.md2ang = md2ang
        self.constraints = constraints
        self.label = label
        self.lunit = lunit
        self.eunit = eunit
        if self.eunit == "eV":
            self.para = 1.0
        elif self.eunit == "Kcal/mole":
            self.para = 0.04336344201657351
        else:
            print("Wrong vaule in eunit")
            sys.exit(0)
        self.runlmp()
        self.dpstart()

    def runlmp(self):
        self.lmp = structure(self.lmpinfile)
        self.lmp.quit()

    def dpstart(self):
        self.dp = DeepPot(self.pbinfile)
        print("deepmdkit launched")
        # print(self.conv.shape)
        self.initforce()

    def newx(self, q):
        newxyz = self.lmp.xyz + self.lmp.conv*q
        return newxyz.ctypes

    def absforce(self, q):
        self.e, self.f, self.v = self.dp.eval(q, self.cell, self.atype)
        return self.para*self.lmp.conv*np.array(self.f)

    def initforce(self):
        print("Calculate zero displacement force")
        self.f0 = self.absforce(np.zeros(3*self.lmp.number))

    def force(self, q):
        return self.absforce(q) - self.f0

    def energy(self):
        return self.e


class structure():
    def __init__(self, lmpinfile, label="", constraints=[],
                 tdir="./", lunit="Ang", eunit="eV", md2ang=0.06466,
                 name="", cmdargs="-screen none".split(), ptr=None, comm=None
                 ):
        lammps.__init__(self, name=name, cmdargs=cmdargs, ptr=None, comm=None)
        self.infile = lmpinfile
        self.md2ang = md2ang
        self.constraints = constraints
        self.label = label
        self.lunit = lunit
        self.eunit = eunit
        if self.eunit == "eV":
            self.para = 1.0
        elif self.eunit == "Kcal/mole":
            self.para = 0.04336344201657351
        else:
            print("Wrong vaule in eunit")
            sys.exit(0)
        # start lammps
        self.start()

    def start(self):
        print("LAMMPS launched")
        #for line in lines: self.command(line)
        self.type = np.array(self.gather_atoms("type", 0, 1))
        #self.mass = np.array(self.gather_atoms("mass",1,1))
        self.mass = self.extract_atom("mass", 2)
        self.number = self.get_natoms()
        self.els = []
        for type in self.type:
            self.els.append(self.mass[type])
        self.xyz = self.gather_atoms("x", 1, 3)
        self.conv = self.md2ang*np.array([3*[1.0/np.sqrt(mass)]
                                          for mass in self.els]).flatten()
        self.type = np.array(self.gather_atoms("type", 0, 1))
        #self.mass = np.array(self.gather_atoms("mass",1,1))
        self.axyz = []
        for i, a in enumerate(self.els):
            self.axyz.append([get_atomname(a), self.xyz[i*3],
                              self.xyz[i*3+1], self.xyz[i*3+2]])

    def quit(self):
        self.close()
        print("LAMMPS closed.")
