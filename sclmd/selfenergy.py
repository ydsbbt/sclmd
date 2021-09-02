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

#    def extendcell(lammpsdata, trajectoriesfiles, position_only=False, outputname="avestructure.data"):
#        from ovito.io import export_file, import_file
#        # generate large structure file
#        print("import LAMMPS data file %s" % lammpsdata)
#        data = import_file(lammpsdata).source.compute(0)
#        aveposition = np.zeros(
#            [len(trajectoriesfiles), data.number_of_particles, 3])
#        if position_only:
#            for i, trajfile in enumerate(trajectoriesfiles):
#                print("import trajectorie file %s" % trajfile)
#                traj = import_file(trajfile, columns=[
#                    "Particle Type", "Position.X", "Position.Y", "Position.Z"])
#                for frame_index in range(traj.source.num_frames):
#                    position = np.array(traj.source.compute(
#                        frame_index).particles.positions)
#                    aveposition[i] = (aveposition[i]*frame_index +
#                                      position)/(frame_index+1)
#        else:
#            for i, trajfile in enumerate(trajectoriesfiles):
#                print("import trajectorie file %s" % trajfile)
#                traj = import_file(trajfile, columns=[
#                    "Particle Type", "Position.X", "Position.Y", "Position.Z", "Force.X", "Force.Y", "Force.Z"])
#                for frame_index in range(traj.source.num_frames):
#                    position = np.array(traj.source.compute(
#                        frame_index).particles.positions)
#                    aveposition[i] = (aveposition[i]*frame_index +
#                                      position)/(frame_index+1)
#        data.particles_.positions_[:] = np.mean(aveposition, axis=0)
#        print("export LAMMPS data file %s" % outputname)
#        export_file(data, outputname, "lammps/data", atom_style="full")

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


if __name__ == '__main__':
    '''
    Units
    Time: ps
    Frequence: eV
    Temperture: K
    Heat Current: nW
    '''
    import time

    import numpy as np

    from sclmd.selfenergy import sig
    infile = [
        'atom_style full',
        'units metal',
        'boundary p p p',
        'read_data structure.data',
        'pair_style rebo',
        'pair_coeff * * CH.rebo C H',
        'min_style  cg',
        'minimize   1e-25   1e-25   5000    10000',
    ]
    time_start = time.time()
    dofatomK10 = range(20*3, (69+1)*3)
    dofatomK00 = range(70*3, 131*3)
    dofatomK01 = range(131*3, (180+1)*3)
    atomfixed = [range(0*3, (19+1)*3), range(181*3, (200+1)*3)]
    mode = sig(infile, dofatomK00, dofatomK01, dofatomK10, atomfixed)
    mode.K00()
    mode.K01()
    mode.K10()
    print('time cost', time.time()-time_start, 's')
