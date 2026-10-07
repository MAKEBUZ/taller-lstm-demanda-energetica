"""App web para predecir la demanda de la siguiente hora con historial observado."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from taller_lstm import NUMERIC, OUT, PERIODS, feature_frame, make_model, load_weights

st.set_page_config(page_title="Predicción de demanda LSTM", layout="wide")
st.title("Predicción de demanda eléctrica: próxima hora")
st.write("Ingrese horas históricas consecutivas para estimar la demanda de la hora siguiente.")

config_path = OUT / "normalizacion.json"
info_path = OUT / "mejor_modelo.json"
model_path = OUT / "mejor_modelo_pesos.npz"
if not all(p.exists() for p in [config_path, info_path, model_path]):
    st.error("Primero ejecute `python taller_lstm.py entrenar` para crear el modelo y los escaladores.")
    st.stop()

config = json.loads(config_path.read_text(encoding="utf-8"))
info = json.loads(info_path.read_text(encoding="utf-8"))
audit = json.loads((OUT / "auditoria.json").read_text(encoding="utf-8"))
size = int(info["ventana"])
st.info(f"Modelo seleccionado por validación: {info['modelo']} · ventana de {size} horas")
st.warning(
    f"**Aviso sobre el alcance:** la predicción usa las últimas {size} horas observadas y estima "
    "solo la demanda de la hora siguiente, en MW. No se ingresan datos futuros ni la columna "
    "`demanda_objetivo`. Es un resultado académico y orientativo, no una garantía para "
    "decisiones operativas."
)

required = ["timestamp", *NUMERIC, "festivo", "periodo_dia"]
uploaded = st.file_uploader("Cargue un CSV con al menos las últimas horas requeridas", type="csv")
if uploaded is None:
    source = pd.read_csv(OUT / "dataset_limpio_auditado.csv")
    source["timestamp"] = pd.to_datetime(source.timestamp)
    good = source.entrada_valida.to_numpy(dtype=bool)
    candidates = [i for i in range(size - 1, len(source)) if good[i - size + 1:i + 1].all()]
    if not candidates:
        st.error("No hay una ventana de ejemplo válida.")
        st.stop()
    end = candidates[-1]
    example = source.iloc[end - size + 1:end + 1][required].copy()
    example["periodo_dia"] = example.periodo_dia.astype(str).str.strip().str.lower()
    st.download_button("Descargar plantilla de ejemplo", example.to_csv(index=False).encode("utf-8"),
                       file_name="historial_ejemplo.csv", mime="text/csv")
    st.caption("Cargue su propio historial o edite la tabla de ejemplo antes de predecir.")
    data = example
else:
    data = pd.read_csv(uploaded)

missing = set(required) - set(data.columns)
if missing:
    st.error("Faltan columnas: " + ", ".join(sorted(missing)))
    st.stop()

data = data[required].tail(size).copy()
edited = st.data_editor(data, num_rows="fixed", width="stretch")

if st.button("Predecir siguiente hora", type="primary"):
    try:
        if len(edited) != size:
            raise ValueError(f"Se requieren exactamente {size} horas.")
        edited["timestamp"] = pd.to_datetime(edited["timestamp"], errors="raise")
        edited = edited.sort_values("timestamp").reset_index(drop=True)
        if edited.timestamp.duplicated().any() or not edited.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError("Las horas deben ser únicas y consecutivas.")
        for col in NUMERIC + ["festivo"]:
            edited[col] = pd.to_numeric(edited[col], errors="raise")
        if edited[NUMERIC + ["festivo"]].isna().any().any():
            raise ValueError("Hay valores faltantes en las variables de entrada.")
        if (edited.demanda_mw <= 0).any() or not edited.humedad_pct.between(0, 100).all() or \
           (edited.viento_kmh < 0).any() or (edited.radiacion_wm2 < 0).any() or \
           (edited.precipitacion_mm < 0).any():
            raise ValueError("Hay mediciones físicamente imposibles.")
        for col, (low, high) in audit["limites_iqr_entrenamiento"].items():
            if not edited[col].between(low, high).all():
                raise ValueError(f"{col} contiene valores fuera de los límites usados en entrenamiento.")
        edited["periodo_dia"] = edited.periodo_dia.astype(str).str.strip().str.lower()
        if not edited.periodo_dia.isin(PERIODS).all():
            raise ValueError("periodo_dia debe ser madrugada, mañana, tarde o noche.")
        frame = feature_frame(edited)[config["features"]]
        x = ((frame.to_numpy(dtype=float) - np.asarray(config["mean"])) /
             np.asarray(config["scale"])).astype(np.float32)[None, :, :]
        model = load_weights(make_model(info["modelo"], size, len(config["features"])), model_path)
        standardized = float(model.predict(x, verbose=0)[0, 0])
        predicted = standardized * config["target_scale"] + config["target_mean"]
        target_time = edited.timestamp.iloc[-1] + pd.Timedelta(hours=1)
        st.metric(f"Demanda estimada para {target_time:%Y-%m-%d %H:%M}", f"{predicted:,.2f} MW")
    except (ValueError, TypeError, KeyError) as exc:
        st.error(str(exc))
