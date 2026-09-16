import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Getaround – délai minimum entre deux locations", layout="wide")


@st.cache_data
def load_data():
    delays = pd.read_csv("data/get_around_delay_analysis.csv")
    pairs = delays.dropna(subset=["previous_ended_rental_id"]).copy()
    pairs["previous_ended_rental_id"] = pairs["previous_ended_rental_id"].astype(int)
    previous = delays[["rental_id", "delay_at_checkout_in_minutes"]].rename(
        columns={"rental_id": "previous_ended_rental_id", "delay_at_checkout_in_minutes": "previous_delay"}
    )
    pairs = pairs.merge(previous, on="previous_ended_rental_id")
    pairs["delay_for_next_driver"] = pairs["previous_delay"] - pairs["time_delta_with_previous_rental_in_minutes"]
    return delays, pairs


delays, pairs = load_data()

st.title("Délai minimum entre deux locations")
st.caption("Aide à la décision pour le Product Manager : quel seuil, quel périmètre ?")

st.sidebar.header("Réglages")
threshold = st.sidebar.slider("Seuil (minutes)", min_value=0, max_value=720, value=120, step=15)
scope = st.sidebar.radio("Périmètre", ["Toutes les réservations", "Connect seulement"])

scoped = pairs if scope == "Toutes les réservations" else pairs[pairs["checkin_type"] == "connect"]
blocked = scoped["time_delta_with_previous_rental_in_minutes"] < threshold
problematic = scoped["delay_for_next_driver"] > 0
ended = delays[delays["state"] == "ended"]

affected = int(blocked.sum())
solved = int((blocked & problematic).sum())
total_problematic = int(problematic.sum())
revenue_share = (blocked & (scoped["state"] == "ended")).sum() / len(ended) * 100
late_share = problematic[scoped["delay_for_next_driver"].notna()].mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Locations affectées", f"{affected}")
col2.metric("Cas problématiques résolus", f"{solved} / {total_problematic}")
col3.metric("Part du revenu en jeu", f"{revenue_share:.1f} %")
col4.metric("Conducteurs suivants ayant attendu", f"{late_share:.1f} %")

st.subheader("Effet du seuil sur le périmètre choisi")
delay_grid = range(0, 721, 30)
simulation = pd.DataFrame({
    "Seuil (minutes)": delay_grid,
    "Locations affectées": [(scoped["time_delta_with_previous_rental_in_minutes"] < t).sum() for t in delay_grid],
    "Cas résolus": [((scoped["time_delta_with_previous_rental_in_minutes"] < t) & problematic).sum() for t in delay_grid],
})
fig = px.line(
    simulation.melt(id_vars="Seuil (minutes)", var_name="Indicateur", value_name="Nombre de locations"),
    x="Seuil (minutes)",
    y="Nombre de locations",
    color="Indicateur",
)
fig.add_vline(x=threshold, line_dash="dash", line_color="orange", annotation_text=f"seuil choisi : {threshold} min")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Distribution des retards au checkout")
window = delays[delays["delay_at_checkout_in_minutes"].between(-720, 720)]
fig = px.histogram(
    window,
    x="delay_at_checkout_in_minutes",
    color="checkin_type",
    nbins=96,
    barmode="overlay",
    labels={"delay_at_checkout_in_minutes": "Retard au checkout (minutes)", "checkin_type": "Type de checkin"},
)
fig.add_vline(x=threshold, line_dash="dash", line_color="orange", annotation_text=f"seuil choisi : {threshold} min")
st.plotly_chart(fig, use_container_width=True)
