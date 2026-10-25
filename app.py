"""Streamlit UI: predict protein-protein interactions and compute dipole moments.

    streamlit run app.py
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ppi_ai.dipole import charge_sites, compute_dipole
from ppi_ai.predict import load_metrics, predict_interaction
from ppi_ai.resolve import Protein, resolve_pdb_file, resolve_text

st.set_page_config(page_title="PPI + Dipole AI", layout="wide")

EXAMPLES = {
    "Hemoglobin alpha + beta (known interaction)": ("P69905", "P68871"),
    "Insulin + insulin receptor (known interaction)": ("P01308", "P06213"),
    "PDB complex: barnase + barstar (1BRS)": ("1BRS:A", "1BRS:D"),
    "Hemoglobin alpha + insulin (no known interaction)": ("P69905", "P01308"),
}


@st.cache_data(show_spinner=False)
def cached_resolve(text: str) -> Protein:
    return resolve_text(text)


def protein_input(col, key: str, default: str) -> tuple:
    with col:
        st.subheader(f"Protein {key}")
        mode = st.radio("Input type", ["ID or sequence", "Upload PDB/mmCIF"], key=f"mode_{key}", horizontal=True)
        if mode == "ID or sequence":
            text = st.text_area(
                "UniProt ID, PDB ID[:chain], or amino-acid sequence",
                value=default,
                key=f"text_{key}",
                height=90,
            )
            return ("text", text)
        upload = st.file_uploader("Structure file", type=["pdb", "cif", "mmcif"], key=f"file_{key}")
        chain = st.text_input("Chain (optional)", key=f"chain_{key}").strip() or None
        return ("file", upload, chain)


def resolve(spec) -> Protein:
    if spec[0] == "text":
        return cached_resolve(spec[1])
    _, upload, chain = spec
    if upload is None:
        raise ValueError("Upload a structure file.")
    text = upload.getvalue().decode("utf-8", errors="ignore")
    return resolve_pdb_file(text, upload.name, chain)


def dipole_figure(protein: Protein, result, ph: float) -> go.Figure:
    model = next(iter(protein.structure))
    chains = [protein.chain] if protein.chain else None
    ca = np.array([r["CA"].coord for c in model if not chains or c.id in chains for r in c if r.id[0] == " " and "CA" in r])
    sites = charge_sites(protein.structure, ph, chains)
    pos = np.array([s[0] for s in sites]) if sites else np.zeros((0, 3))
    q = np.array([s[1] for s in sites]) if sites else np.zeros(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=ca[:, 0], y=ca[:, 1], z=ca[:, 2], mode="lines",
                               line=dict(color="lightgray", width=4), name="Backbone (CA)"))
    for mask, color, name in [(q > 0.05, "blue", "Positive"), (q < -0.05, "red", "Negative")]:
        fig.add_trace(go.Scatter3d(x=pos[mask, 0], y=pos[mask, 1], z=pos[mask, 2], mode="markers",
                                   marker=dict(size=4, color=color), name=f"{name} charge",
                                   text=[sites[i][2] for i in np.where(mask)[0]], hoverinfo="text"))
    com = result.center_of_mass
    if result.magnitude_debye > 0:
        span = np.ptp(ca, axis=0).max() * 0.6
        tip = com + result.vector_debye / result.magnitude_debye * span
        fig.add_trace(go.Scatter3d(x=[com[0], tip[0]], y=[com[1], tip[1]], z=[com[2], tip[2]], mode="lines",
                                   line=dict(color="gold", width=10), name="Dipole direction"))
        fig.add_trace(go.Cone(x=[tip[0]], y=[tip[1]], z=[tip[2]],
                              u=[tip[0] - com[0]], v=[tip[1] - com[1]], w=[tip[2] - com[2]],
                              sizemode="absolute", sizeref=span * 0.15, anchor="tip",
                              colorscale=[[0, "gold"], [1, "gold"]], showscale=False, name="Dipole"))
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0), scene=dict(aspectmode="data"),
                      legend=dict(orientation="h"))
    return fig


# ---------------- UI ----------------
st.title("Protein-Protein Interaction & Dipole Moment AI")
st.caption("Predict whether two proteins physically interact (machine learning on STRING data) "
           "and compute each protein's electric dipole moment from its 3D structure.")

with st.sidebar:
    st.header("Settings")
    ph = st.slider("pH (for charges)", 2.0, 12.0, 7.0, 0.1)
    example = st.selectbox("Load an example", list(EXAMPLES))
    if st.button("Use example"):
        st.session_state["text_A"], st.session_state["text_B"] = EXAMPLES[example]
    metrics = load_metrics()
    if metrics:
        st.header("Model performance")
        st.caption(metrics["data"])
        r, c = metrics["random_pair_split"], metrics["unseen_protein_split"]
        st.markdown(
            f"| Test set | Accuracy | ROC-AUC |\n|---|---|---|\n"
            f"| Random pairs | {r['accuracy']:.3f} | {r['roc_auc']:.3f} |\n"
            f"| Unseen proteins | {c['accuracy']:.3f} | {c['roc_auc']:.3f} |"
        )

col_a, col_b = st.columns(2)
spec_a = protein_input(col_a, "A", "P69905")
spec_b = protein_input(col_b, "B", "P68871")

if st.button("Analyze", type="primary"):
    try:
        with st.spinner("Fetching proteins..."):
            prot_a, prot_b = resolve(spec_a), resolve(spec_b)
    except Exception as e:
        st.error(f"Could not load protein: {e}")
        st.stop()

    st.header("Interaction prediction")
    prob = predict_interaction(prot_a.sequence, prot_b.sequence)
    verdict = "Likely to interact" if prob >= 0.5 else "Unlikely to interact"
    m1, m2 = st.columns([1, 3])
    m1.metric("Interaction probability", f"{prob:.1%}")
    m2.progress(prob, text=f"**{verdict}**: {prot_a.label} + {prot_b.label}")
    st.caption("Prediction uses sequence features only (amino-acid composition, conjoint triad, "
               "physicochemical properties). Treat it as a screening score, not experimental proof.")

    st.header(f"Dipole moments (pH {ph:.1f})")
    for col, prot in zip(st.columns(2), (prot_a, prot_b)):
        with col:
            st.subheader(prot.label)
            st.caption(f"{prot.source} | {len(prot.sequence)} residues")
            if prot.structure is None:
                st.info("No 3D structure available, so the dipole moment cannot be computed. "
                        "Use a UniProt ID with an AlphaFold model, a PDB ID, or upload a structure.")
                continue
            res = compute_dipole(prot.structure, ph, [prot.chain] if prot.chain else None)
            d1, d2, d3 = st.columns(3)
            d1.metric("Dipole |mu|", f"{res.magnitude_debye:,.0f} D")
            d2.metric("Per residue", f"{res.debye_per_residue:.2f} D")
            d3.metric("Net charge", f"{res.net_charge:+.1f} e")
            vx, vy, vz = res.vector_debye
            st.caption(f"Vector (D): ({vx:,.0f}, {vy:,.0f}, {vz:,.0f}) | {res.n_charged_sites} charged sites")
            st.plotly_chart(dipole_figure(prot, res, ph), use_container_width=True)
