# Semi-classical Langevin Molecular Dynamics

This is a set of scripts to run semi-classical Langevin molecular dynamics on junctions connecting to several electron or phonon baths. The details of the method are presented in Prog. Surf. Sci. [https://doi.org/10.1016/j.progsurf.2018.07.002].

To do the molecular dynamics, we need a force driver. Currently, we implemented Siesta, Brenner and LAMMPS potentials.

The important feature of this script is that, the statistics of the bath degrees of freeom is quantum. For example, the zero point fluctuations are included; they fulfill the quantum-mechanical Bose-Einstein distribution. Moreover, the electron bath is allowed to be in a nonequilibrium steady state (non-thermal).

## Install

```bash
pip install sclmd
```

Install LAMMPS python package as force driver

```bash
# Compile & install LAMMPS shared library with Intel® oneAPI HPC Toolkit https://software.intel.com/content/www/us/en/develop/tools/oneapi.html
git clone https://github.com/lammps/lammps.git -b stable
# Activate Intel compilation environment
source /opt/intel/oneapi/setvars.sh
cd lammps/src
# Select LAMMPS packages
make yes-body yes-class2 yes-manybody yes-molecule yes-kspace yes-reaxff yes-phonon
# Compile LAMMPS shared library
make -j 8 intel_cpu_intelmpi mode=shlib
# Install LAMMPS shared library in python
make install-python
```

## Examples

* **examples/runmd.py**: Calculate thermal conductance of single molecular junction with quantum thermal bath molecular dynmics.

* **examples/runnegf.py**: Calculate thermal conductance of single molecular junction with none equilibrium Green' s function.
