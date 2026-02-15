# Quantum Teleportation Simulator with Noise and Tomography

---

## Overview

This project implements a fully self-contained **Quantum Teleportation Simulator** using NumPy and Matplotlib.

Unlike basic teleportation demos that directly access the final statevector, this implementation:

- Accepts user-defined qubit state to teleport
- Injects stochastic noise
- Verifies teleportation using **quantum state tomography**
- Computes fidelity from reconstructed density matrices
- Visualizes results on the Bloch sphere

---

## What This Project Demonstrates

This project demonstrates:

- Step-by-step quantum teleportation
- How noise degrades quantum information
- How tomography reconstructs quantum states from measurement data
- How fidelity is computed experimentally
- How Bloch sphere geometry reflects state degradation

---

## Teleportation Protocol Implemented

The simulator follows the standard teleportation procedure:

1. **Alice prepares an unknown qubit**
   
   \[
   |\psi\rangle = \alpha |0\rangle + \beta |1\rangle
   \]

2. **Alice and Bob share a Bell pair**

3. **Alice performs a Bell measurement**

4. **Alice sends two classical bits to Bob**

5. **Bob applies conditional corrections (X and/or Z)**

6. **Bob obtains the teleported state**

7. **Teleported state then goes through quantum tomography to determine the state obtained at Bob's side**

8. **Fidelity is calculated using input state and obtained state**

---

## Noise Model

The simulator includes a **depolarizing noise channel**.

After each gate layer, for each qubit:

- With probability \(1 - p\): nothing happens
- With probability \(p/3\): apply X
- With probability \(p/3\): apply Y
- With probability \(p/3\): apply Z

The noise parameter `p` is entered by the user.

## No-Cheat Philosophy

This project does **not** extract Bob's stored statevector to compute fidelity like most simulations do, instead it uses quantum tomography to compute the received qubit.

- Teleportation is re-run for each measurement shot
- Bob is measured only in X, Y, Z bases
- Expectation values are estimated statistically
- Bob's density matrix is reconstructed

---
 
## Bloch Sphere Visualization

The simulator plots:

- Alice's theoretical Bloch vector
- Bob's reconstructed Bloch vector

## 📦 Requirements

- Python 3.8+
- NumPy
- Matplotlib
