#!/usr/bin/env python
import os
import sys
import time

import Inelastica.MakeGeom as MG
import numpy as np
import sisl as SI
from netCDF4 import Dataset

try:
    RunName = sys.argv[1:]
    print("Loading from ", RunName)
except:
    print('Missing RunName input')
    #RunName = "Graphene"
    sys.exit(1)

# Misc. defs.
eV = 13.6058
mm = np.dot


def dagger(M):
    return np.conjugate(np.transpose(M))


def abs2(z):
    return (z*np.conjugate(z)).real
####


# geometry
geom = MG.Geom("./"+RunName+".XV")
snr = geom.snr
anr = np.array(geom.anr)
xyz = np.array(geom.xyz)
na = len(snr)

# full hamiltonian L+D+R
# Hfile=SI.get_sile("./siesta.TSHS").read_es()
Hfile = SI.Hamiltonian.read("./"+RunName+".TSHS")
Hobj = Hfile.H
Sobj = Hfile.S
# g=SI.get_sile("./RUN.fdf").read_geom()

# sigmas projected into D (tbtrans-style)
sigma = Dataset("./"+RunName+".TBT.SE.nc")
# Left self-energy
pvl = sigma.groups["Left"].variables["pivot"][:] - 1
rSL = sigma.groups["Left"].variables["ReSelfEnergy"]
iSL = sigma.groups["Left"].variables["ImSelfEnergy"]

# Right self-energy
pvr = sigma.groups["Right"].variables["pivot"][:] - 1
rSR = sigma.groups["Right"].variables["ReSelfEnergy"]
iSR = sigma.groups["Right"].variables["ImSelfEnergy"]
print("Self-energies on states:")
print("left:", min(pvl), max(pvl))
print("right:", min(pvr), max(pvr))


start_time = time.time()

lasto = sigma.variables['lasto'][:]
na = len(lasto)  # total no. atoms
no = lasto[-1]   # lasto for full system incl. electrodes

a_dev = sigma.variables['a_dev'][:]  # index of device atoms
adev = sorted(a_dev)
a_dev = adev
# print adev
print(len(a_dev))

print("Device atoms: ", min(a_dev), max(a_dev))

# adev gives atoms in 1-index, lasto gives orbitals in 1-index

iad1 = a_dev[0]-1  # first atom in device region in 0-index
iad2 = a_dev[-1]-1  # last atom in device region in 0-index
nad = iad2-iad1+1  # no. atoms in device region

iod1 = lasto[iad1-1]  # first orbital of device using 0-index
# 1+last orbital of device using 0-index so we incl. iod2 in [iod1:iod2]
iod2 = lasto[iad2]

# make orbital list norb:
norb = lasto.copy()
l0 = 0
for ia in range(na):
    norb[ia] = lasto[ia]-l0
    l0 = lasto[ia]

print("Device orbitals: (0-index)", iod1, " to ", iod2-1, " nos=", iod2-iod1)


iasel = []
iosel = []
nos = 0
nas = 0

for ia1 in a_dev:
    ia = ia1-1
    iasel.append(ia)
    nas = nas+1
    nos = nos+norb[ia]
    if ia == 0:
        io1 = 0
    else:
        io1 = lasto[ia-1]
        io2 = lasto[ia]
    for io in range(io1, io2):
        iosel.append(io)
# print iasel
print("Selected no. atoms, orbitals=", nas, nos)

kpts = sigma.variables["kpt"][:]
wkpts = sigma.variables["wkpt"][:]

nkpts = len(kpts)
ens = sigma.variables["E"][:]
ne = len(ens)
print("No. energy points, kpoints: ", ne, nkpts)

eta = 0.0001  # In eV! Note Im-energy is not included in SFE file!

FirstTimek = True
FirstTimeMean = True

Trans = np.zeros((nkpts, ne), np.float)
ft = open("Trans.realspace.dat", "w")
ft.write("# Transmission using real-space self-energy\n")
ft.write("# energy, gammaoverlap, sigmaoverlap T(SIGMATOT), T(SIGMAL+SIGMAR)\n")
for ien in range(ne):
    en = ens[ien]*eV

    print("Doing ien=", ien, ", Energy(eV)=", en)

################## k-loop begin  ####################
    for ikpt in range(nkpts):

        print("Doing kpt=", ikpt, " out of ", nkpts)
        kpt = kpts[ikpt]
        wkpt = wkpts[ikpt]
        ncfile = 'HSSigma_'+str(ikpt)+'.nc'

        hssetup_time = time.time()
        Hfullsparse = Hobj.Hk(kpt)
        Sfullsparse = Sobj.Sk(kpt)
        Hfull = Hfullsparse.toarray()
        Sfull = Sfullsparse.toarray()
        # device orbitals out of full made from 3 equal sizes.
        # H is already in eV!!
        H = np.array(Hfull)[iod1:iod2, iod1:iod2]
        S = np.array(Sfull)[iod1:iod2, iod1:iod2]
        hssetup_time = time.time() - hssetup_time
        print('hssetup time: %.3f' % hssetup_time)
        print("shape H:", np.shape(H))

        if(FirstTimek):  # set up HSSigma for each k-point

            if os.path.exists(ncfile):
                print("Removing ", ncfile)
                os.remove(ncfile)
            nc = Dataset(ncfile, 'w')

            nc.createDimension('na', nad)
            nc.createDimension('xyz', 3)
            xa_save = nc.createVariable('xa', np.float, ('na', 'xyz'))
            xa_save.unit = "Ang."
            xa_save.long_name = "Atomic coordinates"
            anr_save = nc.createVariable('anr', np.int, ('na',))
            norb_save = nc.createVariable('norb', np.int, ('na',))

            iad = 0
            for ia1 in a_dev:
                ia = ia1-1
                xa_save[iad, :] = xyz[ia, :]
                anr_save[iad] = anr[ia]
                norb_save[iad] = norb[ia]
                iad = iad+1

            nc.createDimension('ne', ne)
            ReEn = nc.createVariable('ReE', np.float, ('ne',))
            ReEn.unit = "eV"
            ImEn = nc.createVariable('ImE', np.float, ('ne',))
            ImEn.unit = "eV"

            nc.createDimension('nkpts', nkpts)
            kpt_save = nc.createVariable('kpt', np.float, ('xyz',))
            kpt_save[:] = kpt[:]

            nc.createDimension('nos', nos)
            ReH_save = nc.createVariable('ReH', np.float, ('nos', 'nos'))
            ImH_save = nc.createVariable('ImH', np.float, ('nos', 'nos'))
            ReS_save = nc.createVariable('ReS', np.float, ('nos', 'nos'))
            ImS_save = nc.createVariable('ImS', np.float, ('nos', 'nos'))
            ReH_save[:, :] = H.real[:, :]
            ImH_save[:, :] = H.imag[:, :]
            ReS_save[:, :] = S.real[:, :]
            ImS_save[:, :] = S.imag[:, :]

            ReSigmaL = nc.createVariable(
                'ReSigmaL', np.float, ('ne', 'nos', 'nos'))
            ImSigmaL = nc.createVariable(
                'ImSigmaL', np.float, ('ne', 'nos', 'nos'))
            ReSigmaL.unit = "eV"
            ImSigmaL.unit = "eV"
            ReSigmaL.long_name = "Real part of left self-energy"
            ImSigmaL.long_name = "Imag. part of left self-energy"

            ReSigmaR = nc.createVariable(
                'ReSigmaR', np.float, ('ne', 'nos', 'nos'))
            ImSigmaR = nc.createVariable(
                'ImSigmaR', np.float, ('ne', 'nos', 'nos'))
            ReSigmaR.unit = "eV"
            ImSigmaR.unit = "eV"
            ReSigmaR.long_name = "Real part of right self-energy"
            ImSigmaR.long_name = "Imag. part of right self-energy"

            # calculate eigenvalues
            # eig=sli.eigh(h.todense(),s.todense(),eigvals_only=True,overwrite_a=True,overwrite_b=True)
            # print eig
        # append energy point to HSSigma for each k-point
        else:
            nc = Dataset(ncfile, 'a')

        ReEn[ien] = ens[ien]
        ImEn[ien] = eta

        # put self-energies into hm
        SL = np.zeros(np.shape(Hfull), np.complex)
        SR = np.zeros(np.shape(Hfull), np.complex)
        SFEL = rSL[ikpt, ien, :, :] + 1j*iSL[ikpt, ien, :, :]
        SFER = rSR[ikpt, ien, :, :] + 1j*iSR[ikpt, ien, :, :]

        for i, idx in enumerate(pvl):
            SL[idx, pvl] += SFEL[i, :]

        for i, idx in enumerate(pvr):
            SR[idx, pvr] += SFER[i, :]

        SL1 = np.array(SL)[iod1:iod2, iod1:iod2]*eV
        SR1 = np.array(SR)[iod1:iod2, iod1:iod2]*eV

        ReSigmaL[ien, :, :] = SL1.real[:, :]
        ImSigmaL[ien, :, :] = SL1.imag[:, :]

        ReSigmaR[ien, :, :] = SR1.real[:, :]
        ImSigmaR[ien, :, :] = SR1.imag[:, :]

        invG = S*(en + 1j*eta) - H - SL1 - SR1
        invGL = S*(en + 1j*eta) - H - SL1
        invGR = S*(en + 1j*eta) - H - SR1

        inv_time = time.time()

        G = np.linalg.inv(invG)
        GL = np.linalg.inv(invGL)
        GR = np.linalg.inv(invGR)

        inv_time = time.time() - inv_time

        if ikpt == 0:
            print(ikpt, " -- Zero sums")
            Gsum = np.zeros(np.shape(G), np.complex)
            GLsum = np.zeros(np.shape(G), np.complex)
            GRsum = np.zeros(np.shape(G), np.complex)
            Hsum = np.zeros(np.shape(G), np.complex)
            Ssum = np.zeros(np.shape(G), np.complex)

    # Note we use time-reversal symmetry here, G(-k)=transpose(G(k))!!
        Gsum = Gsum + (0.5*wkpt)*(G + np.transpose(G))
        GLsum = GLsum + (0.5*wkpt)*(GL + np.transpose(GL))
        GRsum = GRsum + (0.5*wkpt)*(GR + np.transpose(GR))
        Hsum = Hsum + (0.5*wkpt)*(H + np.transpose(H))
        Ssum = Ssum + (0.5*wkpt)*(S + np.transpose(S))

    # Calculate transmission
        GamL = 1.0j*(SL1-dagger(SL1))
        GamR = 1.0j*(SR1-dagger(SR1))

        AL = mm(mm(G, GamL), dagger(G))
        T = np.trace(mm(GamR, AL))
        print("en,kp,T", en, kpt, T)
        Trans[ikpt, ien] = T.real

        print("closing ", ncfile)
        nc.close()
    ################## k-loop end ####################
    FirstTimek = False

    ncfile = 'HSSigmaMEAN.nc'
    if(FirstTimeMean):
        if os.path.exists(ncfile):
            print("Removing ", ncfile)
            os.remove(ncfile)
        nc = Dataset(ncfile, 'w')
        nc.createDimension('na', nad)
        nc.createDimension('xyz', 3)
        xa_save = nc.createVariable('xa', np.float, ('na', 'xyz'))
        xa_save.unit = "Ang."
        xa_save.long_name = "Atomic coordinates"
        anr_save = nc.createVariable('anr', np.int, ('na',))
        norb_save = nc.createVariable('norb', np.int, ('na',))

        iad = 0
        for ia1 in a_dev:
            ia = ia1-1
            xa_save[iad, :] = xyz[ia, :]
            anr_save[iad] = anr[ia]
            norb_save[iad] = norb[ia]
            iad = iad+1

        nc.createDimension('ne', ne)
        ReEn = nc.createVariable('ReE', np.float, ('ne',))
        ReEn.unit = "eV"
        ImEn = nc.createVariable('ImE', np.float, ('ne',))
        ImEn.unit = "eV"

        nc.createDimension('nkpts', nkpts)
        kpts_save = nc.createVariable('kpts', np.float, ('nkpts', 'xyz'))
        kpts_save[:] = kpts[:]

        nc.createDimension('nos', nos)

        ReH_save = nc.createVariable('ReH', np.float, ('nos', 'nos'))
        ImH_save = nc.createVariable('ImH', np.float, ('nos', 'nos'))
        ReS_save = nc.createVariable('ReS', np.float, ('nos', 'nos'))
        ImS_save = nc.createVariable('ImS', np.float, ('nos', 'nos'))
        ReH_save[:, :] = Hsum.real[:, :]
        ImH_save[:, :] = Hsum.imag[:, :]
        ReS_save[:, :] = Ssum.real[:, :]
        ImS_save[:, :] = Ssum.imag[:, :]

        # Write real-space self-energy
        ReSIGMAL = nc.createVariable(
            'ReSigmaL', np.float, ('ne', 'nos', 'nos'))
        ImSIGMAL = nc.createVariable(
            'ImSigmaL', np.float, ('ne', 'nos', 'nos'))
        ReSIGMAL.unit = "eV"
        ImSIGMAL.unit = "eV"
        ReSIGMAL.long_name = "Real part of real-space left self-energy"
        ImSIGMAL.long_name = "Imag. part of real-space left self-energy"

        ReSIGMAR = nc.createVariable(
            'ReSigmaR', np.float, ('ne', 'nos', 'nos'))
        ImSIGMAR = nc.createVariable(
            'ImSigmaR', np.float, ('ne', 'nos', 'nos'))
        ReSIGMAR.unit = "eV"
        ImSIGMAR.unit = "eV"
        ReSIGMAR.long_name = "Real part of real-space right self-energy"
        ImSIGMAR.long_name = "Imag. part of real-space right self-energy"

        ReSIGMATOT = nc.createVariable(
            'ReSigmaTOT', np.float, ('ne', 'nos', 'nos'))
        ImSIGMATOT = nc.createVariable(
            'ImSigmaTOT', np.float, ('ne', 'nos', 'nos'))
        ReSIGMATOT.unit = "eV"
        ImSIGMATOT.unit = "eV"
        ReSIGMATOT.long_name = "Real part of real-space total self-energy"
        ImSIGMATOT.long_name = "Imag. part of real-space total self-energy"
        FirstTimeMean = False
    else:
        nc = Dataset(ncfile, 'a')

    ReEn[ien] = en
    ImEn[ien] = eta

    # Get real-space total self-energy:
    invGsum = np.linalg.inv(Gsum)

    # Get real-space left self-energy:
    invGLsum = np.linalg.inv(GLsum)

    # Get real-space right self-energy:
    invGRsum = np.linalg.inv(GRsum)

    HSsum = (en + 1j*eta)*Ssum - Hsum

    SIGMATOT = (HSsum - invGsum)
    SIGMAL = (HSsum - invGLsum)
    SIGMAR = (HSsum - invGRsum)

    ReSIGMAL[ien, :, :] = SIGMAL.real[:, :]
    ImSIGMAL[ien, :, :] = SIGMAL.imag[:, :]

    ReSIGMAR[ien, :, :] = SIGMAR.real[:, :]
    ImSIGMAR[ien, :, :] = SIGMAR.imag[:, :]

    ReSIGMATOT[ien, :, :] = SIGMATOT.real[:, :]
    ImSIGMATOT[ien, :, :] = SIGMATOT.imag[:, :]

    # Analyze/Calculate mean transmission

    GamL = 1.0j*(SIGMAL-dagger(SIGMAL))
    GamR = 1.0j*(SIGMAR-dagger(SIGMAR))

    sigmaoverlap = max(abs(np.ravel(SIGMAL*SIGMAR)))
    gammaoverlap = max(abs(np.ravel(GamL*GamR)))

    # calculation using full averaged G
    AL = mm(mm(G, GamL), dagger(G))
    T1 = np.trace(mm(GamR, AL))

    # calculation using G with SIGMAL, SIGMAR
    invG = HSsum - SIGMAL - SIGMAR
    G = np.linalg.inv(invG)
    AL = mm(mm(G, GamL), dagger(G))
    T2 = np.trace(mm(GamR, AL))

    ft.write("%f   %f %f      %f   %f\n" %
             (en, gammaoverlap, sigmaoverlap, T1.real, T2.real))

    print("en,T1, T2", en, gammaoverlap, sigmaoverlap, T1, T2)

    print("closing ", ncfile)
    nc.close()

##############################
# End of energy loop, ien
##############################
ft.close()

ft = open("Trans.dat", "w")
ft.write("# Trans (kpt)\n")
for ikpt in range(nkpts):
    kpt = kpts[ikpt]
    ft.write("# ikpt = %d, kpt = %f  %f  %f\n" %
             (ikpt, kpt[0], kpt[1], kpt[2]))
    for ien in range(ne):
        ft.write("%f  %f\n" % (ens[ien]*eV, Trans[ikpt, ien]))
ft.close()

ft = open("Trans.kav.dat", "w")
for ien in range(ne):
    Transav = 0.0
    for ikpt in range(nkpts):
        Transav = Transav + Trans[ikpt, ien]*wkpts[ikpt]
    ft.write("%f  %f\n" % (ens[ien]*eV, Transav))
ft.close()


eps_time = time.time() - start_time
print('inv time: %.3f' % inv_time)
print('Elapsed time: %.3f' % eps_time)
