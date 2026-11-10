"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Engineering Calculator Component
=============================================================================
Module: src.ui.components.calculator
Purpose: Interactive quick-calculator for hydropower sizing, turbine selection,
         power generation potential, and annual energy estimations.
=============================================================================
"""

import streamlit as st


def recommend_turbine(head_m: float, flow_m3s: float) -> tuple[str, str]:
    """
    Determines the optimal turbine type and reason based on IEC/ESHA envelope guidelines.
    """
    if head_m > 250:
        return (
            "Pelton Impulse Turbine",
            "High net head (>250m) favors Pelton impulse runners to avoid excessive casing pressures and cavitation.",
        )
    elif 50 <= head_m <= 250:
        if flow_m3s < 1.0:
            return (
                "Pelton or Turgo Impulse Turbine",
                "Medium-high head with low discharge favors Pelton or Turgo runners for high part-load efficiency.",
            )
        else:
            return (
                "Francis Reaction Turbine",
                "Medium head (50–250m) with moderate-to-high discharge is the classic operational envelope for Francis runners.",
            )
    elif 15 <= head_m < 50:
        if flow_m3s > 5.0:
            return (
                "Kaplan / Propeller Reaction Turbine",
                "Low head (15–50m) with high flow favors Kaplan axial reaction turbines with adjustable blades.",
            )
        elif flow_m3s < 2.0:
            return (
                "Cross-Flow (Banki-Michell) or Francis Turbine",
                "Low-to-medium head with lower discharge favors robust Cross-Flow or compact Francis units.",
            )
        else:
            return (
                "Francis or Kaplan Turbine",
                "Transition zone (15–50m): Francis and Kaplan overlap; selection depends on seasonal discharge variability.",
            )
    else:  # head_m < 15
        return (
            "Kaplan, Tubular, or Archimedes Screw Turbine",
            "Very low head (<15m) requires high-flow axial Kaplan, Bulb/Tubular, or Archimedes Screw installations.",
        )


def render_calculator():
    """Renders the Small Hydropower Plant engineering sizing calculator."""
    st.markdown("### ⚙️ **SHP Sizing & Turbine Selector**")
    st.caption("Perform rapid feasibility checks and theoretical power estimations based on standard fluid mechanics.")

    # Engineering Equation Callout
    st.latex(r"P = \rho \cdot g \cdot Q \cdot H_{\text{net}} \cdot \eta_{\text{overall}}")

    c1, c2 = st.columns(2)
    with c1:
        head = st.number_input(
            "Net Head H (meters)",
            min_value=1.0,
            max_value=1500.0,
            value=45.0,
            step=1.0,
            help="Effective water fall height after subtracting hydraulic head losses in penstocks.",
        )
        flow = st.number_input(
            "Design Discharge Q (m³/s)",
            min_value=0.01,
            max_value=250.0,
            value=2.5,
            step=0.1,
            help="Rated volumetric flow rate through turbine penstocks.",
        )

    with c2:
        eff_percent = st.slider(
            "Overall System Efficiency η (%)",
            min_value=50,
            max_value=96,
            value=88,
            step=1,
            help="Combined product of turbine (η_t ~ 88-93%), generator (η_g ~ 96%), and transformer/drive efficiencies.",
        )
        clf_percent = st.slider(
            "Plant Capacity Factor / PLF (%)",
            min_value=20,
            max_value=95,
            value=60,
            step=5,
            help="Expected annual operating load factor considering seasonal river hydrology.",
        )

    # Computations
    rho = 1000.0  # kg/m^3
    g = 9.81      # m/s^2
    eta = eff_percent / 100.0
    plf = clf_percent / 100.0

    power_watts = rho * g * flow * head * eta
    power_kw = power_watts / 1000.0
    power_mw = power_kw / 1000.0

    # Annual Energy: P (kW) * 8760 hours * PLF
    annual_energy_mwh = (power_kw * 8760.0 * plf) / 1000.0

    # Recommendation
    turbine_type, reason = recommend_turbine(head, flow)

    st.markdown("---")
    st.markdown("#### ⚡ **Theoretical Power Output**")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Rated Capacity (kW)", f"{power_kw:,.1f} kW")
    with m2:
        st.metric("Rated Capacity (MW)", f"{power_mw:,.3f} MW")
    with m3:
        st.metric("Annual Generation", f"{annual_energy_mwh:,.1f} MWh/yr")

    st.success(f"**Recommended Turbine:** {turbine_type}")
    st.caption(f"**Rationale:** {reason}")

    # Helpful prompt button
    suggested_prompt = f"For an SHP site with {head}m net head and {flow} m3/s flow rate, compare Francis and Pelton turbine suitability and cost considerations."
    if st.button("💬 Ask Agent About This Sizing", use_container_width=True, type="secondary"):
        st.session_state.pending_query = suggested_prompt
        st.rerun()
