#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys

import numpy as np


class sig:
    # This class is used to create a selfenergy object.
    def __init__(self, infile, dofatomK00, dofatomK01, dofatomK10=[], dofatomfixed=[[], []], dynmatfile=None):
        # Initializes the selfenergy object.
        # reduced Planck constant unit in: eV*ps
        self.rpc = 6.582119569e-4
        # Boltzmann constant unit in: eV/K
        self.bc = 8.617333262e-5
        #self.infile = infile
        #self.damp = damp
        #self.maxomega = maxomega/self.rpc
        #self.intnum = num
        self.dofatomK00 = dofatomK00
        self.dofatomK01 = dofatomK01
        self.dofatomK10 = dofatomK10
        self.dofatomfixed = dofatomfixed
        self.dynmatfile = dynmatfile
        # self.extendcell()
        self.getdynmat(infile)

    def getdynmat(self, infile):
        from lammps import lammps
        lmp = lammps()
        #lmp = lammps(cmdargs=['-screen', 'none', '-log', 'none'])
        lmp.commands_list(infile)
        self.natoms = lmp.get_natoms()
        box = lmp.extract_box()
        self.boxlo = np.array(box[0])
        self.boxhi = np.array(box[1])
        systype = np.array(lmp.gather_atoms("type", 0, 1))
        mass = lmp.extract_atom("mass", 2)
        self.els = []
        for type in systype:
            self.els.append([mass[type]]*3)
        self.els = np.array(self.els).flatten()
        self.xyz = lmp.gather_atoms("x", 1, 3)
        self.els = np.delete(self.els, self.dofatomfixed[0])
        self.els = np.delete(self.els, [
            dof-len(self.dofatomfixed[0]) for dof in self.dofatomfixed[1]])
        self.xyz = np.delete(self.xyz, self.dofatomfixed[0])
        self.xyz = np.delete(self.xyz, [
            dof-len(self.dofatomfixed[0]) for dof in self.dofatomfixed[1]])
        if self.dynmatfile is None:
            print('Calculate dynamical matrix')
            lmp.command('dynamical_matrix all eskm 0.000001 file dynmat.dat')
            dynmatdat = np.loadtxt('dynmat.dat')
        else:
            print('Load dynamical matrix from '+str(self.dynmatfile))
            dynmatdat = np.loadtxt(self.dynmatfile)
        lmp.close()
        self.dynmat = []
        omegas = []
        self.doffreeatom = 0
        dynlen = int(3*np.sqrt(len(dynmatdat)/3))
        if dynlen != self.natoms*3:
            print('System DOF test failed after load dynmat, check again')
            sys.exit()
        self.dynmat = dynmatdat.reshape((dynlen, dynlen))
        dynmatexcpfixed = np.delete(self.dynmat, self.dofatomfixed[0], axis=0)
        dynmatexcpfixed = np.delete(
            dynmatexcpfixed, self.dofatomfixed[0], axis=1)
        dynmatexcpfixed = np.delete(dynmatexcpfixed, [
            dof-len(self.dofatomfixed[0]) for dof in self.dofatomfixed[1]], axis=0)
        dynmatexcpfixed = np.delete(dynmatexcpfixed, [
            dof-len(self.dofatomfixed[0]) for dof in self.dofatomfixed[1]], axis=1)
        if len(self.xyz) != len(dynmatexcpfixed):
            print('System DOF test failed after atoms reduced, check again')
            sys.exit()
        print('Calculate angular frequency')
        eigvals, eigvecs = np.linalg.eigh(dynmatexcpfixed)
        for i, val in enumerate(eigvals):
            if val > 0:
                omegas.append(np.sqrt(val)*self.rpc)
            else:
                print('False frequency exists in system DOF %i ' %
                      (i+len(self.dofatomfixed[0])))
                omegas.append(-np.sqrt(-val)*self.rpc)
        np.savetxt('omegas.dat', omegas)
        np.savetxt('eigvecs.dat', eigvecs)

    def K00(self):
        return self.dynmat[self.dofatomK00, :][:, self.dofatomK00]

    def K01(self):
        return self.dynmat[self.dofatomK00, :][:, self.dofatomK01]

    def K10(self):
        return self.dynmat[self.dofatomK00, :][:, self.dofatomK10]

    def sgf(self):
        # Algorithms for surface Green’s functions
        s = self.K00()
        e = self.K00()
        alpha = self.K01()
        epsilon = 1e-6
        eta = 1e-9
        iter = 0
        omega = 1
        while np.linalg.det(alpha) > epsilon:
            g = np.linalg.inv((omega+1j*eta) ** 2*np.identity(len(e))-e)
            beta = np.transpose(alpha)
            s += np.dot(np.dot(alpha, g), beta)
            e += np.dot(np.dot(alpha, g), beta) + \
                np.dot(np.dot(beta, g), alpha)
            alpha = np.dot(np.dot(alpha, g), alpha)
            iter += 1
            if iter > 10000:
                print('Iteration number exceeded')
                return -1
        return np.linalg.inv((omega+1j*eta) ** 2*np.identity(len(s))-s)

    def selfenergy(self):
        # self energy
        return np.dot(np.dot(self.K01(), self.sgf()), self.K10())


if __name__ == '__main__':
    import time

    #from sclmd.selfenergy import sig
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
