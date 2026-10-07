"""Taller LSTM: limpieza, ventanas cronológicas, comparación y artefactos."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset_demanda_energia_LSTM_2025_2026.csv"
OUT = ROOT / "resultados"
TRAIN_END = pd.Timestamp("2026-04-01")
VAL_END = pd.Timestamp("2026-07-01")
NUMERIC = ["demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh",
           "radiacion_wm2", "precipitacion_mm", "precio_kwh"]
PERIODS = ["madrugada", "mañana", "tarde", "noche"]


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Misma codificación determinista para entrenamiento e inferencia."""
    t = pd.to_datetime(df["timestamp"])
    x = df[NUMERIC].astype(float).copy()
    hour = t.dt.hour.to_numpy()
    weekday = t.dt.dayofweek.to_numpy()
    month = t.dt.month.to_numpy()
    for name, val, cycle in [("hora", hour, 24), ("dia_semana", weekday, 7),
                             ("mes", month - 1, 12)]:
        x[name + "_sin"] = np.sin(2 * np.pi * val / cycle)
        x[name + "_cos"] = np.cos(2 * np.pi * val / cycle)
    x["fin_semana"] = (weekday >= 5).astype(float)
    x["festivo"] = pd.to_numeric(df["festivo"], errors="coerce")
    period = df["periodo_dia"].astype("string").str.strip().str.lower()
    for name in PERIODS:
        x["periodo_" + name] = (period == name).astype(float)
    return x


def clean() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(DATA)
    report = {"filas_originales": len(raw), "duplicados_exactos": int(raw.duplicated().sum()),
              "faltantes_originales": raw.isna().sum().astype(int).to_dict()}
    df = raw.drop_duplicates().copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["periodo_dia"] = df["periodo_dia"].astype("string").str.strip().str.lower()
    report["periodos_corregidos"] = int((raw["periodo_dia"].astype("string") !=
                                          raw["periodo_dia"].astype("string").str.strip().str.lower()).sum())
    report["horas_ausentes"] = int(len(pd.date_range(df.timestamp.min(), df.timestamp.max(), freq="h")
                                      .difference(df.timestamp)))
    # Reglas físicas: los valores imposibles se excluyen, nunca se sustituyen.
    physical = (df["timestamp"].notna() & (df["demanda_mw"] > 0) &
                df["humedad_pct"].between(0, 100) & (df["viento_kmh"] >= 0) &
                (df["radiacion_wm2"] >= 0) & (df["precipitacion_mm"] >= 0) &
                df["periodo_dia"].isin(PERIODS))
    # Los límites estadísticos se calculan solamente con el tramo de entrenamiento.
    bounds = {}
    for col in ["demanda_mw", "temperatura_c", "viento_kmh", "precio_kwh"]:
        base = df.loc[physical & (df.timestamp < TRAIN_END), col].dropna()
        q1, q3 = base.quantile([.25, .75])
        iqr = q3 - q1
        bounds[col] = [float(q1 - 4 * iqr), float(q3 + 4 * iqr)]
    statistical = pd.Series(True, index=df.index)
    for col, (lo, hi) in bounds.items():
        statistical &= df[col].between(lo, hi) | df[col].isna()
    df["entrada_valida"] = (physical & statistical & df[NUMERIC + ["festivo"]].notna().all(axis=1))
    df["objetivo_valido"] = df["demanda_objetivo"].notna() & (df["demanda_objetivo"] > 0)
    report.update({"filas_tras_duplicados": len(df), "fallos_fisicos": int((~physical).sum()),
                   "fallos_estadisticos": int((~statistical).sum()),
                   "filas_entrada_valida": int(df.entrada_valida.sum()),
                   "objetivos_validos": int(df.objetivo_valido.sum()),
                   "limites_iqr_entrenamiento": bounds,
                   "regla": "Excluir ventanas con faltantes o anomalías; no imputar mediciones."})
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "dataset_limpio_auditado.csv", index=False)
    (OUT / "auditoria.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, report


def windows(df: pd.DataFrame, size: int, x_all: np.ndarray, y_all: np.ndarray):
    valid = df.entrada_valida.to_numpy(dtype=bool)
    yvalid = df.objetivo_valido.to_numpy(dtype=bool)
    times = df.timestamp.to_numpy(dtype="datetime64[h]")
    xs, ys, splits, target_times, persistence = [], [], [], [], []
    for end in range(size - 1, len(df)):
        start = end - size + 1
        if not valid[start:end + 1].all() or not yvalid[end]:
            continue
        if times[end] - times[start] != np.timedelta64(size - 1, "h"):
            continue
        target_time = pd.Timestamp(times[end]) + pd.Timedelta(hours=1)
        split = "train" if target_time < TRAIN_END else "val" if target_time < VAL_END else "test"
        xs.append(x_all[start:end + 1]); ys.append(y_all[end]); splits.append(split)
        target_times.append(target_time); persistence.append(float(df.demanda_mw.iloc[end]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(splits), target_times, np.asarray(persistence)


def metrics(y, pred):
    err = y - pred
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"MAE": float(np.mean(np.abs(err))), "MSE": float(np.mean(err ** 2)),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE": float(np.mean(np.abs(err / y)) * 100),
            "R2": float(1 - np.sum(err ** 2) / ss)}


def make_model(kind, size, nfeatures):
    import tensorflow as tf
    L = tf.keras.layers
    inp = L.Input(shape=(size, nfeatures))
    if kind == "base":
        z = L.LSTM(32)(inp)
    elif kind == "profunda":
        z = L.LSTM(48, return_sequences=True)(inp)
        z = L.Dropout(.2)(z)
        z = L.LSTM(24)(z)
    else:
        z = L.LSTM(32, return_sequences=True)(inp)
        z = L.Attention()([z, z])
        z = L.GlobalAveragePooling1D()(z)
        z = L.Dense(16, activation="relu")(z)
    out = L.Dense(1)(z)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss="mse")
    return model


def save_weights(model, path):
    np.savez_compressed(path, **{f"w{i}": w for i, w in enumerate(model.get_weights())})


def load_weights(model, path):
    with np.load(path) as data:
        model.set_weights([data[f"w{i}"] for i in range(len(data.files))])
    return model


def normalize():
    df, _ = clean()
    frame = feature_frame(df)
    fit_mask = df.entrada_valida & (df.timestamp < TRAIN_END)
    mean = frame.loc[fit_mask].mean().to_numpy(dtype=float)
    scale = frame.loc[fit_mask].std(ddof=0).replace(0, 1).to_numpy(dtype=float)
    x_all = ((frame.to_numpy(dtype=float) - mean) / scale).astype(np.float32)
    # El escalado del objetivo también usa exclusivamente etiquetas de entrenamiento.
    train_target = df.loc[df.objetivo_valido & (df.timestamp + pd.Timedelta(hours=1) < TRAIN_END), "demanda_objetivo"]
    ymean, yscale = float(train_target.mean()), float(train_target.std(ddof=0))
    y_all = ((df.demanda_objetivo.to_numpy(dtype=float) - ymean) / yscale).astype(np.float32)
    normalized = pd.DataFrame(x_all, columns=frame.columns)
    normalized.insert(0, "timestamp", df.timestamp)
    normalized["demanda_objetivo_normalizada"] = y_all
    normalized["entrada_valida"] = df.entrada_valida
    normalized["objetivo_valido"] = df.objetivo_valido
    normalized["particion_objetivo"] = np.where(df.timestamp + pd.Timedelta(hours=1) < TRAIN_END,
                                               "entrenamiento", np.where(df.timestamp + pd.Timedelta(hours=1) < VAL_END,
                                                                         "validacion", "prueba"))
    normalized.to_csv(OUT / "dataset_normalizado.csv", index=False)
    config = {"features": list(frame.columns), "mean": mean.tolist(), "scale": scale.tolist(),
              "target_mean": ymean, "target_scale": yscale, "train_end": str(TRAIN_END),
              "val_end": str(VAL_END)}
    (OUT / "normalizacion.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, frame, x_all, y_all, ymean, yscale


def train(max_epochs: int = 15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    df, frame, x_all, y_all, ymean, yscale = normalize()
    _, _, split48, dates48, _ = windows(df, 48, x_all, y_all)
    common_val = {d for d, s in zip(dates48, split48) if s == "val"}
    common_test = {d for d, s in zip(dates48, split48) if s == "test"}
    records, curves, baseline_records = [], {}, []
    best_val = float("inf")
    best_info = None
    for size in [12, 24, 48]:
        X, y, split, dates, persistence = windows(df, size, x_all, y_all)
        masks = {s: split == s for s in ["train", "val", "test"]}
        eval_val = np.asarray([d in common_val for d in dates], dtype=bool)
        eval_test = np.asarray([d in common_test for d in dates], dtype=bool)
        print(f"Ventana {size}: " + ", ".join(f"{s}={m.sum()}" for s, m in masks.items()), flush=True)
        if not all(m.sum() for m in masks.values()):
            raise RuntimeError("Una partición no tiene ventanas válidas")
        baseline_records.append({"Ventana": size, **metrics(y[eval_test] * yscale + ymean,
                                                          persistence[eval_test])})
        for kind in ["base", "profunda", "atencion"]:
            tf.keras.backend.clear_session()
            model = make_model(kind, size, X.shape[-1])
            callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                        min_delta=.0005, restore_best_weights=True)
            hist = model.fit(X[masks["train"]], y[masks["train"]],
                             validation_data=(X[eval_val], y[eval_val]),
                             epochs=max_epochs, batch_size=256, shuffle=False,
                             callbacks=[callback], verbose=0)
            pred_val = model.predict(X[eval_val], verbose=0).ravel() * yscale + ymean
            pred_test = model.predict(X[eval_test], verbose=0).ravel() * yscale + ymean
            actual_val = y[eval_val] * yscale + ymean
            actual_test = y[eval_test] * yscale + ymean
            val_m = metrics(actual_val, pred_val)
            test_m = metrics(actual_test, pred_test)
            name = f"{kind}_{size}h"
            records.append({"Modelo": kind, "Ventana": size, **test_m,
                            "Epocas": len(hist.history["loss"]), "MAE_validacion": val_m["MAE"],
                            "n_entrenamiento": int(masks["train"].sum()),
                            "n_validacion": int(eval_val.sum()),
                            "n_prueba": int(eval_test.sum())})
            curves[name] = hist.history
            pd.DataFrame({"timestamp_objetivo": np.asarray(dates, dtype=object)[eval_test],
                          "real_mw": actual_test, "predicho_mw": pred_test,
                          "persistencia_mw": persistence[eval_test]}).to_csv(OUT / f"predicciones_{name}.csv", index=False)
            if val_m["MAE"] < best_val:
                best_val = val_m["MAE"]
                save_weights(model, OUT / "mejor_modelo_pesos.npz")
                best_info = {"modelo": kind, "ventana": size, "MAE_validacion": best_val,
                             "archivo_predicciones": f"predicciones_{name}.csv"}
            print(f"{name}: val MAE={val_m['MAE']:.2f}; test MAE={test_m['MAE']:.2f}; epocas={len(hist.history['loss'])}", flush=True)
    results = pd.DataFrame(records).sort_values(["Modelo", "Ventana"])
    results.to_csv(OUT / "tabla_resultados.csv", index=False)
    pd.DataFrame(baseline_records).to_csv(OUT / "baseline_persistencia.csv", index=False)
    (OUT / "mejor_modelo.json").write_text(json.dumps(best_info, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
    for ax, (name, hist) in zip(axes.flat, curves.items()):
        ax.plot(hist["loss"], label="entrenamiento"); ax.plot(hist["val_loss"], label="validación")
        ax.set_title(name); ax.set_xlabel("Época"); ax.set_ylabel("MSE normalizado"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "curvas_loss.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for kind, group in results.groupby("Modelo"):
        ax.plot(group.Ventana, group.MAE, marker="o", label=kind)
    ax.set(xlabel="Ventana (horas)", ylabel="MAE prueba (MW)", title="Comparación de arquitecturas")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "comparacion_modelos.png", dpi=150); plt.close(fig)
    pred = pd.read_csv(OUT / best_info["archivo_predicciones"])
    fig, ax = plt.subplots(figsize=(13, 4))
    sample = pred.iloc[:168]
    ax.plot(sample.real_mw.to_numpy(), label="Real"); ax.plot(sample.predicho_mw.to_numpy(), label="Predicha")
    ax.set(xlabel="Primeras 168 muestras de prueba", ylabel="Demanda (MW)", title="Demanda real vs. predicha")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "real_vs_predicha.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(pred.real_mw.to_numpy() - pred.predicho_mw.to_numpy(), linewidth=.6)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xlabel="Muestra de prueba", ylabel="Error real - predicho (MW)", title="Error de predicción")
    fig.tight_layout(); fig.savefig(OUT / "error_prediccion.png", dpi=150); plt.close(fig)
    best_model = load_weights(make_model(best_info["modelo"], best_info["ventana"], X.shape[-1]),
                              OUT / "mejor_modelo_pesos.npz")
    best_x, best_y, _, best_dates, _ = windows(df, best_info["ventana"], x_all, y_all)
    best_eval_val = np.asarray([d in common_val for d in best_dates], dtype=bool)
    val_x = best_x[best_eval_val].copy()
    val_y = best_y[best_eval_val] * yscale + ymean
    baseline = metrics(val_y, best_model.predict(val_x, verbose=0).ravel() * yscale + ymean)["MAE"]
    rng = np.random.default_rng(42)
    importances = []
    for j, name in enumerate(frame.columns):
        altered = val_x.copy()
        altered[:, :, j] = altered[rng.permutation(len(altered)), :, j]
        mae = metrics(val_y, best_model.predict(altered, verbose=0).ravel() * yscale + ymean)["MAE"]
        importances.append({"variable": name, "incremento_MAE_MW": mae - baseline})
    pd.DataFrame(importances).sort_values("incremento_MAE_MW", ascending=False).to_csv(
        OUT / "importancia_permutacion_validacion.csv", index=False)
    print("Modelo elegido por validación:", best_info, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("accion", choices=["limpiar", "normalizar", "entrenar"])
    parser.add_argument("--epocas", type=int, default=15)
    args = parser.parse_args()
    if args.accion == "limpiar":
        _, report = clean(); print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.accion == "normalizar":
        normalize(); print("Dataset normalizado y parámetros guardados en resultados/", flush=True)
    else:
        train(args.epocas)
