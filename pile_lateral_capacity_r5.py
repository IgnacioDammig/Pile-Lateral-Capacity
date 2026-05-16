import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(page_title="Pile Lateral Analysis", layout="wide")
st.title("Pile Lateral Capacity & Deflection Analysis")

st.markdown(
    """
Nonlinear lateral pile analysis using **p-y curves** and an **Euler-Bernoulli FEM beam**
with lumped Winkler springs.
"""
)

with st.expander("Methods used in this app", expanded=True):
    st.markdown(
        """
### Structural model
- The pile is modeled as an **Euler-Bernoulli beam**.
- The response is solved with the **Finite Element Method (FEM)** using **Hermitian beam elements**.
- Soil-pile interaction is represented with **lumped Winkler springs** acting in the lateral direction.
- The nonlinear problem is solved iteratively using **secant spring stiffness updates** with relaxation.

### Sand model
For **sand**, the app uses the **API RP 2GEO / API RP 2A-WSD p-y formulation**:
- Ultimate soil resistance is calculated using the **API coefficients C1, C2 and C3**
- The nonlinear p-y curve is represented with the **tanh formulation**
- The initial modulus of subgrade reaction is taken as **k · z**, increasing with depth below ground level

### Clay model
For **clay**, the app uses the **Matlock (1970) static p-y formulation for soft clay**:
- Ultimate lateral resistance is limited using the standard Matlock expression
- The characteristic displacement is defined with **y50 = 2.5 · eps50 · D**
- The p-y curve follows the typical **power law formulation**
- A small numerical regularization is introduced near **y = 0** to improve solver stability

### Ground level and load application
- The app models the **total pile length**
- The pile head may be located **above ground level**
- The **lateral load is always applied at the pile head**
- **Soil springs are applied only below ground level**
- Depth used in the p-y curves is measured **from ground level**, not from the pile head

### Output
The app provides:
- load-deflection curve
- deflection profile
- bending moment profile
- shear profile
- p-y curves at selected depths
"""
    )


# =============================================================================
# FUNCTIONS
# =============================================================================

def api_coeffs(phi_deg):
    phi = np.radians(phi_deg)
    alpha = phi / 2.0
    beta = np.radians(45.0) + phi / 2.0
    K0 = 0.4
    Ka = (1.0 - np.sin(phi)) / (1.0 + np.sin(phi))

    C1 = (
        (np.tan(beta) ** 2 * np.tan(alpha)) / np.tan(beta - phi)
        + K0 * (
            (np.tan(phi) * np.sin(beta)) / (np.cos(alpha) * np.tan(beta - phi))
            + np.tan(beta) * (np.tan(phi) * np.sin(beta) - np.tan(alpha))
        )
    )
    C2 = np.tan(beta) / np.tan(beta - phi) - Ka
    C3 = Ka * (np.tan(beta) ** 8 - 1.0) + K0 * np.tan(phi) * np.tan(beta) ** 4
    return C1, C2, C3


def pu_sand(z_soil, D, gamma, C1_API, C2_API, C3_API):
    if z_soil <= 0.0:
        return 1e-9

    A = max(3.0 - 0.8 * z_soil / D, 0.9)
    pus = (C1_API * z_soil + C2_API * D) * gamma * z_soil
    pud = C3_API * D * gamma * z_soil
    return A * min(pus, pud)


def py_sand(z_soil, y, D, gamma, k_sand, C1_API, C2_API, C3_API):
    pu = pu_sand(z_soil, D, gamma, C1_API, C2_API, C3_API)
    kz = k_sand * max(z_soil, 1e-4)  # kN/m²
    return pu * np.tanh(kz * y / pu)


def pu_clay(z_soil, D, Su, gamma_c):
    if z_soil <= 0.0:
        return 0.0

    J = 0.5
    return min(
        (3.0 + gamma_c * z_soil / Su + J * z_soil / D) * Su * D,
        9.0 * Su * D
    )


def py_clay(z_soil, y, D, Su, eps50, gamma_c):
    if z_soil <= 0.0:
        return 0.0

    pu = pu_clay(z_soil, D, Su, gamma_c)
    y50 = 2.5 * eps50 * D

    if abs(y) < 1e-16:
        return 0.0

    p = 0.5 * pu * (abs(y) / y50) ** (1.0 / 3.0)
    p = min(p, pu)
    return np.sign(y) * p


def p_y(z_soil, y, soil, D, gamma, k_sand, phi_deg, Su, eps50, gamma_c):
    if soil == "sand":
        C1_API, C2_API, C3_API = api_coeffs(phi_deg)
        return py_sand(z_soil, y, D, gamma, k_sand, C1_API, C2_API, C3_API)
    elif soil == "clay":
        return py_clay(z_soil, y, D, Su, eps50, gamma_c)
    else:
        raise ValueError("soil must be 'sand' or 'clay'")


def k_secant_soil(z_soil, y, soil, D, gamma, k_sand, phi_deg, Su, eps50, gamma_c, y_reg):
    if z_soil <= 0.0:
        return 0.0

    if soil == "sand":
        if abs(y) < y_reg:
            return k_sand * max(z_soil, 1e-4)
        return p_y(z_soil, y, soil, D, gamma, k_sand, phi_deg, Su, eps50, gamma_c) / y

    if soil == "clay":
        y_eff = max(abs(y), y_reg)
        pu = pu_clay(z_soil, D, Su, gamma_c)
        y50 = 2.5 * eps50 * D
        p_eff = 0.5 * pu * (y_eff / y50) ** (1.0 / 3.0)
        p_eff = min(p_eff, pu)
        return p_eff / y_eff

    raise ValueError("soil must be 'sand' or 'clay'")


def hermitian_stiffness(EI, Le, N_elem):
    n_dof = 2 * (N_elem + 1)
    Kb = np.zeros((n_dof, n_dof), dtype=float)

    ke = EI / Le**3 * np.array([
        [12.0,       6.0 * Le,   -12.0,       6.0 * Le],
        [6.0 * Le,   4.0 * Le**2, -6.0 * Le,  2.0 * Le**2],
        [-12.0,     -6.0 * Le,    12.0,      -6.0 * Le],
        [6.0 * Le,   2.0 * Le**2, -6.0 * Le,  4.0 * Le**2],
    ])

    for e in range(N_elem):
        d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        for a in range(4):
            for b in range(4):
                Kb[d[a], d[b]] += ke[a, b]

    return Kb


def apply_boundary_conditions(K, f, free_head=True):
    K_mod = K.copy()
    f_mod = f.copy()

    if not free_head:
        dof = 1  # rotation at head
        K_mod[dof, :] = 0.0
        K_mod[:, dof] = 0.0
        K_mod[dof, dof] = 1.0
        f_mod[dof] = 0.0

    return K_mod, f_mod


def compute_moment_profile(y_nd, EI, Le, N_elem, free_head=True):
    M = np.zeros_like(y_nd)

    for i in range(1, N_elem):
        M[i] = EI * (y_nd[i - 1] - 2.0 * y_nd[i] + y_nd[i + 1]) / Le**2

    if free_head:
        M[0] = 0.0
    else:
        M[0] = EI * (2.0 * y_nd[0] - 5.0 * y_nd[1] + 4.0 * y_nd[2] - y_nd[3]) / Le**2

    M[-1] = 0.0
    return M


def compute_shear_profile(M, H, Le, N_elem):
    V = np.zeros_like(M)
    V[0] = H

    for i in range(1, N_elem):
        V[i] = (M[i - 1] - M[i + 1]) / (2.0 * Le)

    V[-1] = 0.0
    return V


def solve_pile(
    H,
    L_total,
    h_above_ground,
    D,
    EI,
    soil,
    phi_deg,
    gamma,
    k_sand,
    Su,
    eps50,
    gamma_c,
    free_head,
    n_elem,
    tol_disp,
    tol_res,
    max_iter,
    relaxation,
    y_reg,
):
    if h_above_ground < 0:
        raise ValueError("h_above_ground must be >= 0")

    if h_above_ground >= L_total:
        raise ValueError("h_above_ground must be smaller than L_total")

    Le = L_total / n_elem
    z_node = np.linspace(0.0, L_total, n_elem + 1)
    z_ground = h_above_ground
    z_soil_node = np.maximum(z_node - z_ground, 0.0)

    trib = np.full(n_elem + 1, Le)
    trib[0] = Le / 2.0
    trib[-1] = Le / 2.0

    Kb_global = hermitian_stiffness(EI, Le, n_elem)

    n_dof = 2 * (n_elem + 1)
    y_nd = np.zeros(n_elem + 1, dtype=float)
    th_nd = np.zeros(n_elem + 1, dtype=float)

    converged = False

    for _ in range(max_iter):
        y_old = y_nd.copy()
        th_old = th_nd.copy()

        ks = np.zeros(n_elem + 1)
        for i in range(n_elem + 1):
            if z_node[i] >= z_ground:
                ks[i] = k_secant_soil(
                    z_soil_node[i],
                    y_nd[i],
                    soil,
                    D,
                    gamma,
                    k_sand,
                    phi_deg,
                    Su,
                    eps50,
                    gamma_c,
                    y_reg,
                )
            else:
                ks[i] = 0.0

        K = Kb_global.copy()
        for i in range(n_elem + 1):
            K[2 * i, 2 * i] += ks[i] * trib[i]

        f = np.zeros(n_dof, dtype=float)
        f[0] = H  # load always at pile head

        K_mod, f_mod = apply_boundary_conditions(K, f, free_head=free_head)

        sol = np.linalg.solve(K_mod, f_mod)
        y_new = sol[0::2]
        th_new = sol[1::2]

        y_nd = relaxation * y_new + (1.0 - relaxation) * y_old
        th_nd = relaxation * th_new + (1.0 - relaxation) * th_old

        u = np.zeros(n_dof, dtype=float)
        u[0::2] = y_nd
        u[1::2] = th_nd

        res = np.linalg.norm(K_mod @ u - f_mod)
        err_disp = np.max(np.abs(y_nd - y_old))

        if err_disp < tol_disp and res < tol_res:
            converged = True
            break

    M = compute_moment_profile(y_nd, EI, Le, n_elem, free_head=free_head)
    V = compute_shear_profile(M, H, Le, n_elem)

    return {
        "y": y_nd,
        "theta": th_nd,
        "M": M,
        "V": V,
        "ks": ks,
        "z_node": z_node,
        "z_ground": z_ground,
        "z_soil_node": z_soil_node,
        "embedded_length": L_total - h_above_ground,
        "converged": converged,
    }


# =============================================================================
# SIDEBAR INPUTS
# =============================================================================

st.sidebar.header("Input Parameters")

st.sidebar.subheader("Geometry")
L_total = st.sidebar.number_input("Total pile length L_total (m)", min_value=0.1, value=12.0, step=0.1)
h_above_ground = st.sidebar.number_input("Pile head above ground (m)", min_value=0.0, value=1.0, step=0.1)
D = st.sidebar.number_input("Pile diameter D (m)", min_value=0.01, value=0.610, step=0.01)
EI = st.sidebar.number_input("Flexural stiffness EI (kN·m²)", min_value=1.0, value=108200.0, step=1000.0)

st.sidebar.subheader("Soil")
soil = st.sidebar.selectbox("Soil type", ["clay", "sand"])

if soil == "sand":
    phi_deg = st.sidebar.number_input("Friction angle phi (deg)", min_value=1.0, max_value=89.0, value=35.0, step=1.0)
    gamma = st.sidebar.number_input("Unit weight gamma (kN/m³)", min_value=1.0, value=18.0, step=0.5)
    k_sand = st.sidebar.number_input("Initial subgrade modulus k (kN/m³)", min_value=1.0, value=20000.0, step=100.0)

    Su = 30.85
    eps50 = 0.02
    gamma_c = 6.3
else:
    Su = st.sidebar.number_input("Undrained shear strength Su (kPa)", min_value=0.1, value=30.85, step=1.0)
    eps50 = st.sidebar.number_input("eps50", min_value=0.001, value=0.02, step=0.001, format="%.3f")
    gamma_c = st.sidebar.number_input("Effective unit weight gamma' (kN/m³)", min_value=0.1, value=6.3, step=0.1)

    phi_deg = 35.0
    gamma = 18.0
    k_sand = 20000.0

st.sidebar.subheader("Loading")
H_max = st.sidebar.number_input("Maximum lateral load H_max (kN)", min_value=0.0, value=300.0, step=10.0)
n_load = st.sidebar.number_input("Number of load steps", min_value=1, value=40, step=1)

st.sidebar.subheader("Head Condition")
free_head = st.sidebar.checkbox("Free head", value=True)

st.sidebar.subheader("Numerical Controls")
n_elem = st.sidebar.number_input("Number of elements", min_value=10, value=120, step=10)
tol_disp = st.sidebar.number_input("Displacement tolerance", min_value=1e-12, value=1e-8, format="%.1e")
tol_res = st.sidebar.number_input("Residual tolerance", min_value=1e-12, value=1e-6, format="%.1e")
max_iter = st.sidebar.number_input("Max iterations", min_value=1, value=100, step=1)
relaxation = st.sidebar.slider("Relaxation factor", min_value=0.1, max_value=1.0, value=0.65, step=0.05)
y_reg = st.sidebar.number_input("Clay regularization Y_REG (m)", min_value=1e-12, value=1e-6, format="%.1e")


# =============================================================================
# RUN
# =============================================================================

run = st.sidebar.button("Run Analysis", type="primary")

if run:
    if h_above_ground >= L_total:
        st.error("Pile head above ground must be smaller than total pile length.")
        st.stop()

    loads = np.linspace(0.0, H_max, int(n_load) + 1)
    y_heads = []
    M_maxes = []

    final_result = None

    for H in loads:
        result = solve_pile(
            H=H,
            L_total=L_total,
            h_above_ground=h_above_ground,
            D=D,
            EI=EI,
            soil=soil,
            phi_deg=phi_deg,
            gamma=gamma,
            k_sand=k_sand,
            Su=Su,
            eps50=eps50,
            gamma_c=gamma_c,
            free_head=free_head,
            n_elem=int(n_elem),
            tol_disp=tol_disp,
            tol_res=tol_res,
            max_iter=int(max_iter),
            relaxation=relaxation,
            y_reg=y_reg,
        )

        y_head_mm = result["y"][0] * 1000.0
        M_max = float(np.nanmax(np.abs(result["M"])))

        y_heads.append(y_head_mm)
        M_maxes.append(M_max)

        if np.isclose(H, H_max):
            final_result = result

    if final_result is None:
        st.error("Analysis failed.")
        st.stop()

    z_node = final_result["z_node"]
    z_ground = final_result["z_ground"]
    y_full = final_result["y"]
    M_full = final_result["M"]
    V_full = final_result["V"]
    embedded_length = final_result["embedded_length"]
    converged = final_result["converged"]

    if converged:
        st.success("Analysis completed.")
    else:
        st.warning("Analysis completed, but the final load step did not fully converge.")

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Head deflection at H_max", f"{y_heads[-1]:.2f} mm")
    c2.metric("Maximum moment at H_max", f"{np.nanmax(np.abs(M_full)):.2f} kN·m")
    c3.metric("Embedded length", f"{embedded_length:.2f} m")
    c4.metric("Head condition", "Free" if free_head else "Fixed")

    st.subheader("Load Step Results")
    table_data = {
        "H (kN)": loads,
        "Head deflection (mm)": y_heads,
        "M_max (kN·m)": M_maxes,
    }
    st.dataframe(table_data, use_container_width=True)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, wspace=0.40, hspace=0.45)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(y_heads, loads, lw=2)
    ax1.fill_betweenx(loads, y_heads, alpha=0.10)
    d100 = D / 100.0 * 1000.0
    ax1.axvline(d100, lw=1.2, ls="--")
    ax1.set_xlabel("Head deflection (mm)")
    ax1.set_ylabel("Lateral load (kN)")
    ax1.set_title("Load - Deflection")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(y_full * 1000.0, -z_node, lw=2)
    ax2.axvline(0.0, lw=0.8, ls=":")
    ax2.axhline(-z_ground, lw=1.2, ls="--")
    ax2.set_xlabel("Deflection y (mm)")
    ax2.set_ylabel("Depth from head z (m)")
    ax2.set_title("Deflection Profile")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-L_total, 0.5)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(M_full, -z_node, lw=2)
    ax3.axvline(0.0, lw=0.8, ls=":")
    ax3.axhline(-z_ground, lw=1.2, ls="--")
    ax3.set_xlabel("Moment M (kN·m)")
    ax3.set_ylabel("Depth from head z (m)")
    ax3.set_title("Moment Profile")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-L_total, 0.5)

    ax4 = fig.add_subplot(gs[1, 1])
    y_rng = np.linspace(0.0, 0.20 * D, 300)
    depths_py = [1 * D, 3 * D, 6 * D, 10 * D]
    depths_py = [z for z in depths_py if z <= embedded_length]
    if len(depths_py) == 0:
        depths_py = [max(0.1, 0.5 * embedded_length)]

    for dz in depths_py:
        p_vals = [
            p_y(dz, yi, soil, D, gamma, k_sand, phi_deg, Su, eps50, gamma_c)
            for yi in y_rng
        ]
        ax4.plot(y_rng * 1000.0, p_vals, lw=1.8, label=f"z_soil={dz:.2f} m")

    ax4.set_xlabel("Deflection y (mm)")
    ax4.set_ylabel("Soil resistance p (kN/m)")
    ax4.set_title("p-y Curves")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(V_full, -z_node, lw=2)
    ax5.axvline(0.0, lw=0.8, ls=":")
    ax5.axhline(-z_ground, lw=1.2, ls="--")
    ax5.set_xlabel("Shear V (kN)")
    ax5.set_ylabel("Depth from head z (m)")
    ax5.set_title("Shear Profile")
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-L_total, 0.5)

    st.pyplot(fig, clear_figure=True)

    st.subheader("Notes")
    st.markdown(
        """
- The lateral load is always applied at the **pile head**.
- Soil springs are applied only **below ground level**.
- p-y depth is measured from **ground level**, not from the pile head.
- Moment and shear are postprocessed from the deflection profile.
"""
    )
else:
    st.info("Set the parameters in the sidebar and click **Run Analysis**.")