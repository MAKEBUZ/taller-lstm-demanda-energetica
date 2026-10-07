# Taller LSTM: demanda energética

Autor: Diego Alejandro Ocampo Madroñero.

El proyecto usa Python. El script `taller_lstm.py` limpia el CSV, ajusta la normalización solo con entrenamiento, prueba 3 arquitecturas × 3 ventanas, guarda resultados y genera las gráficas. `app.py` despliega la mejor configuración seleccionada por MAE de validación.

## Ejecución local

Con Python 3.12 instalado:

```powershell
pip install -r requirements.txt
python taller_lstm.py limpiar
python taller_lstm.py normalizar
python taller_lstm.py entrenar
streamlit run app.py
```

Los productos principales están en `resultados/`: `dataset_limpio_auditado.csv`, `dataset_normalizado.csv`, `auditoria.json`, `normalizacion.json`, `tabla_resultados.csv`, cuatro gráficas y `mejor_modelo_pesos.npz`. Al repetir el entrenamiento también se generan nueve CSV de predicciones; se excluyen de GitHub porque son archivos intermedios.

La división temporal usa como objetivo `timestamp + 1 hora`: entrenamiento hasta el 31 de marzo de 2026, validación del 1 de abril al 30 de junio de 2026, y prueba desde el 1 de julio de 2026. Una ventana de validación o prueba puede usar historia previa; su objetivo siempre pertenece a su partición.

La app solicita las últimas horas observadas, no variables futuras. Las columnas `hora`, `dia_semana`, `mes` y `fin_semana` se calculan desde `timestamp`, para evitar inconsistencias. `demanda_objetivo` no se utiliza como entrada.

## Publicar con Streamlit Community Cloud

1. En [share.streamlit.io](https://share.streamlit.io), conecte su cuenta de GitHub y elija **Create app**.
2. Seleccione `MAKEBUZ/taller-lstm-demanda-energetica`, rama `main` y `app.py` como archivo principal.
3. En **Advanced settings**, seleccione Python 3.12, la versión usada para probar el proyecto. Pulse **Deploy**.
4. Abra el enlace `*.streamlit.app` generado y compruebe una predicción con el ejemplo de la app. Ese es el enlace que puede compartir; `127.0.0.1` solo funciona en su computador.

No hace falta ejecutar el entrenamiento en la nube: el repositorio incluye los pesos y parámetros ya calculados. Se requieren una cuenta de GitHub y una de Streamlit Community Cloud para publicar el enlace.
