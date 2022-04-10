## Modified Lee More Model (MLM) for Electronic Transport

A library is provided that evaluates the Lee-More (LM) electrical conductivity model. Currently (as of April 2022) the MLM only calculates the electrical conductivity for an unmagnetized plasma. MLM is written in Python. 

The original LM paper is [here](https://aip.scitation.org/doi/10.1063/1.864744).

### What is it? 

The MLM is a variant of the original LM model that makes slightly different choices for parameters.  Briefly, choices for undefined parameters (e.g., mean distance between particles) and inconsistent parameters (e.g., using an effective temperature) were made. 

Although the MLM is a standalone library, it was originally developed in the context of discrepancy learning in which the MLM is the base model and conductivity data (originally from Katsouros and DeSilva) was used to learn the discrepancy using radial basis function neural networks. That original work is cited below.

### How do I use it? 

The first step is to insall the libray using
```
pip install MLM_Transport
```

Next, import the library into your Python code with
```
import MLM
```

To compute an electrical conductivty with a known element with nuclear charge $Z$, mass density $\rho$ in g/cc and temperature $T$ in eV, use
```
elect_cond = MLM(z, rho, T).
```

### What's next?

The next version of MLM will include thermal conductivty. The next phase will be focussed on magnetized plasmas. Suggestions and collaborations are welcome.

### How do I cite this? 

The origin version of this work appeared in:
* Data-driven Electrical Conductivities of Dense Plasmas, Michael S. Murillo, Frontiers in Physics, 2022

