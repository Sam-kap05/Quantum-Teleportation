import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------- Bloch Plot --------------------
def plot_bloch_vectors(vec_in, vec_out):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(xs, ys, zs, alpha=0.10)

    # Axes
    ax.quiver(0,0,0,1,0,0,length=1,color='black')
    ax.quiver(0,0,0,0,1,0,length=1,color='black')
    ax.quiver(0,0,0,0,0,1,length=1,color='black')

    # Alice vector
    ax.quiver(0,0,0, vec_in[0], vec_in[1], vec_in[2],
              linewidth=3, label='Alice (theory)', color='blue')

    # Bob vector
    ax.quiver(0,0,0, vec_out[0], vec_out[1], vec_out[2],
              linewidth=3, label='Bob (tomography)', color='red')
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("Quantum Teleportation: Bloch Sphere Verification")
    ax.legend()
    plt.show()


# -------------------- Basics --------------------
ket0 = np.array([[1], [0]], dtype=complex)
ket1 = np.array([[0], [1]], dtype=complex)

I = np.array([[1, 0],
              [0, 1]], dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Y = np.array([[0, -1j],
              [1j,  0]], dtype=complex)

Z = np.array([[1,  0],
              [0, -1]], dtype=complex)

H = (1/np.sqrt(2)) * np.array([[1,  1],
                               [1, -1]], dtype=complex)

Sdg = np.array([[1, 0],
                [0, -1j]], dtype=complex)  # S†


def kron(*args):
    out = args[0]
    for a in args[1:]:
        out = np.kron(out, a)
    return out

def apply_1q(state, gate, qubit, n=3):
    ops = [I]*n
    ops[qubit] = gate
    return kron(*ops) @ state

def apply_cnot(state, control, target, n=3):
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        bits = [(i >> (n-1-q)) & 1 for q in range(n)]
        jbits = bits.copy()
        if bits[control] == 1:
            jbits[target] ^= 1
        j = 0
        for b in jbits:
            j = (j << 1) | b
        U[j, i] = 1.0
    return U @ state

def measure_qubits(state, measure_qubits_list, n=3):
    """
    Measures specified qubits in Z basis, returns outcomes dict and collapsed state.
    """
    measure_qubits_list = list(measure_qubits_list)
    k = len(measure_qubits_list)
    dim = 2**n

    amps = state.flatten()
    probs = np.zeros(2**k, dtype=float)

    for i in range(dim):
        p_i = abs(amps[i])**2
        bits = [(i >> (n-1-q)) & 1 for q in range(n)]
        m = 0
        for q in measure_qubits_list:
            m = (m << 1) | bits[q]
        probs[m] += p_i

    outcome_int = np.random.choice(np.arange(2**k), p=probs)
    outcome_bits = [(outcome_int >> (k-1-t)) & 1 for t in range(k)]
    outcomes = {measure_qubits_list[t]: outcome_bits[t] for t in range(k)}

    # collapse
    new = state.copy().flatten()
    for i in range(dim):
        bits = [(i >> (n-1-q)) & 1 for q in range(n)]
        ok = True
        for q in measure_qubits_list:
            if bits[q] != outcomes[q]:
                ok = False
                break
        if not ok:
            new[i] = 0.0

    new = new.reshape((-1, 1))
    new /= np.linalg.norm(new)
    return outcomes, new

def fmt_complex(z):
    return f"{z.real:+.4f}{z.imag:+.4f}j"


# -------------------- Bloch + density matrix + fidelity (no cheating) --------------------
def bloch_vector_from_state(psi):
    """
    Alice's theoretical Bloch vector (we know the input state because user provided it).
    """
    x = float((psi.conj().T @ X @ psi)[0,0].real)
    y = float((psi.conj().T @ Y @ psi)[0,0].real)
    z = float((psi.conj().T @ Z @ psi)[0,0].real)
    return np.array([x,y,z], dtype=float)

def rho_from_bloch(bloch):
    """
    rho = 1/2 (I + xX + yY + zZ)
    """
    x, y, z = bloch
    r = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(r)
    if norm > 1.0:
        r = r / norm  # project back onto Bloch sphere
    x, y, z = r
    return 0.5 * (I + x*X + y*Y + z*Z)

def fidelity_from_tomography(psi_in, rho_bob):
    """
    F = <psi_in| rho_bob |psi_in>   (no access to Bob's statevector)
    """
    return float((psi_in.conj().T @ rho_bob @ psi_in)[0,0].real)


# -------------------- NEW: Noise model (stochastic depolarizing) --------------------
def apply_depolarizing_noise(state, p, n=3):
    """
    For each qubit independently:
      with prob (1-p): do nothing
      with prob p/3: apply X
      with prob p/3: apply Y
      with prob p/3: apply Z
    """
    if p <= 0:
        return state
    p = float(np.clip(p, 0.0, 1.0))

    for q in range(n):
        r = np.random.random()
        if r < (1 - p):
            continue
        elif r < (1 - p) + p/3:
            state = apply_1q(state, X, qubit=q, n=n)
        elif r < (1 - p) + 2*p/3:
            state = apply_1q(state, Y, qubit=q, n=n)
        else:
            state = apply_1q(state, Z, qubit=q, n=n)
    return state


# -------------------- User input for alpha, beta --------------------
def parse_complex(s: str) -> complex:
    s = s.strip().lower().replace(" ", "")
    s = s.replace("i", "j")
    return complex(s)

def input_qubit_from_user():
    print("\nEnter the qubit as |ψ> = α|0> + β|1>")
    print("Examples: 0.54-0.32j , 0.8 , -0.1+0.5j\n")

    while True:
        try:
            alpha = parse_complex(input("Enter α: "))
            beta  = parse_complex(input("Enter β: "))

            norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
            if norm < 1e-12:
                print("❌ α and β cannot both be 0. Try again.\n")
                continue

            alpha /= norm
            beta  /= norm
            psi = np.array([[alpha], [beta]], dtype=complex)

            print("\n✅ Normalized input:")
            print(f"α = {fmt_complex(alpha)}")
            print(f"β = {fmt_complex(beta)}")
            print(f"Check |α|^2+|β|^2 = {(abs(alpha)**2 + abs(beta)**2):.6f}\n")

            return psi
        except ValueError:
            print("❌ Invalid complex format. Try again.\n")


# -------------------- One experimental shot: teleport + measure Bob once --------------------
def teleport_and_measure_bob_once(psi_in, p, basis="Z"):
    """
    One full experimental run:
      - Prepare |psi_in> on q0
      - Teleport with noisy gates
      - Apply basis rotation on Bob
      - Measure Bob in Z
    Returns: measured bit (0/1)

    IMPORTANT: We never extract Bob's statevector as output.
    We only return a classical measurement bit.
    """
    n = 3
    state = kron(psi_in, ket0, ket0)

    # Bell pair: H(q1), CNOT(q1->q2)
    state = apply_1q(state, H, qubit=1, n=n)
    state = apply_depolarizing_noise(state, p, n=n)

    state = apply_cnot(state, control=1, target=2, n=n)
    state = apply_depolarizing_noise(state, p, n=n)

    # Bell measurement prep: CNOT(q0->q1), H(q0)
    state = apply_cnot(state, control=0, target=1, n=n)
    state = apply_depolarizing_noise(state, p, n=n)

    state = apply_1q(state, H, qubit=0, n=n)
    state = apply_depolarizing_noise(state, p, n=n)

    # Measure Alice qubits q0,q1 -> (m0,m1)
    outcomes, state = measure_qubits(state, [0, 1], n=n)
    m0, m1 = outcomes[0], outcomes[1]

    # Bob corrections
    if m1 == 1:
        state = apply_1q(state, X, qubit=2, n=n)
    if m0 == 1:
        state = apply_1q(state, Z, qubit=2, n=n)

    # Basis rotation on Bob BEFORE measuring in Z
    if basis.upper() == "Z":
        pass
    elif basis.upper() == "X":
        state = apply_1q(state, H, qubit=2, n=n)          # X-basis -> Z via H
    elif basis.upper() == "Y":
        state = apply_1q(state, Sdg, qubit=2, n=n)        # Y-basis -> Z via S† then H
        state = apply_1q(state, H, qubit=2, n=n)
    else:
        raise ValueError("basis must be 'X', 'Y', or 'Z'")

    # Measure Bob qubit q2 in Z
    out_bob, _ = measure_qubits(state, [2], n=n)
    return out_bob[2]


# -------------------- Tomography (Bob defined only by measurement stats) --------------------
def tomography_verification(psi_in, p, shots=2000):
    print("\n--- Quantum State Tomography (No Statevector Access to Bob) ---")
    print(f"Noise p = {p}, shots per basis = {shots}")
    print("We re-run teleportation each shot and measure Bob once per shot.\n")

    # Alice theoretical Bloch vector (we know input)
    bloch_in = bloch_vector_from_state(psi_in)

    # Collect measurement outcomes for Bob in each basis
    z_bits = [teleport_and_measure_bob_once(psi_in, p, basis="Z") for _ in range(shots)]
    x_bits = [teleport_and_measure_bob_once(psi_in, p, basis="X") for _ in range(shots)]
    y_bits = [teleport_and_measure_bob_once(psi_in, p, basis="Y") for _ in range(shots)]

    # Convert bit counts -> expectation values
    # <Z> = P(0) - P(1) = 1 - 2P(1)
    p1z = np.mean(np.array(z_bits) == 1)
    p1x = np.mean(np.array(x_bits) == 1)
    p1y = np.mean(np.array(y_bits) == 1)

    z_exp = 1 - 2*p1z
    x_exp = 1 - 2*p1x
    y_exp = 1 - 2*p1y

    bloch_out = np.array([x_exp, y_exp, z_exp], dtype=float)

    # Reconstruct density matrix and fidelity from tomography
    rho_bob = rho_from_bloch(bloch_out)
    F_tomo = fidelity_from_tomography(psi_in, rho_bob)

    diff = float(np.linalg.norm(bloch_in - bloch_out))

    print(f"Alice Bloch vector (theory):       {bloch_in}")
    print(f"Bob Bloch vector (tomography):      {bloch_out}")
    print(f"Bloch difference norm:              {diff:.6f}")
    print(f"Fidelity from reconstructed rho:    {F_tomo:.6f}")

    plot_bloch_vectors(bloch_in, bloch_out)

    return bloch_in, bloch_out, F_tomo, diff


# -------------------- Narrated (single run story, still no Bob amplitude printing) --------------------
def teleport_narrated_single_run(psi_in, p, pause=True):
    def step(msg):
        print("\n" + "─"*70)
        print(msg)
        if pause:
            input("Press Enter to continue...")

    print("\nQUANTUM TELEPORTATION (Terminal Demo)")
    print("Qubit layout: q0=Alice(unknown), q1=Alice(entangled), q2=Bob(entangled)")
    print(f"Noise model: stochastic depolarizing, p={p}\n")

    step("Step 1) Alice prepares an UNKNOWN qubit |ψ> on q0")
    print(f"α = {fmt_complex(psi_in[0,0])}")
    print(f"β = {fmt_complex(psi_in[1,0])}")

    step("Step 2) Create Bell pair (q1-q2) with noise after layers")
    step("Step 3) Alice Bell-prep + measures q0,q1 to get (m0,m1)")
    step("Step 4) Bob applies corrections from (m0,m1)")
    print("✅ (Single run complete — Bob’s state is not printed; only measurable stats define it.)")


# -------------------- Run --------------------
if __name__ == "__main__":
    np.random.seed()  # remove for full randomness

    psi_in = input_qubit_from_user()

    # User input noise p
    while True:
        try:
            p = float(input("Enter depolarizing noise p (0 to 1): ").strip())
            if not (0.0 <= p <= 1.0):
                print("❌ p must be between 0 and 1.\n")
                continue
            break
        except ValueError:
            print("❌ Enter a valid number like 0.02 or 0.1\n")

    # Narrated single run (story)
    teleport_narrated_single_run(psi_in, p=p, pause=True)

    # Tomography defines Bob (no statevector access)
    shots = 2000
    tomography_verification(psi_in, p=p, shots=shots)
