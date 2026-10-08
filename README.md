# Taller LSTM: demanda energética

Autor: Diego Alejandro Ocampo Madroñero.

El proyecto usa Python. El script `taller_lstm.py` limpia el CSV, ajusta la normalización solo con entrenamiento, prueba 3 arquitecturas × 3 ventanas, guarda resultados y genera las gráficas. `app.py` despliega la mejor configuración seleccionada por MAE de validación.

## Aviso sobre el alcance

La app usa las últimas **48 horas históricas consecutivas** de variables observadas y estima únicamente la demanda de la **hora siguiente en MW**. No recibe datos futuros ni `demanda_objetivo` como entrada. La predicción es un resultado académico y orientativo; debe evaluarse y supervisarse antes de usarla en decisiones operativas.

Este repositorio contiene el código y los resultados del taller. La app está disponible en [Streamlit Community Cloud](https://taller-lstm-demanda-energetica-makebuz.streamlit.app/).

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

Las columnas `hora`, `dia_semana`, `mes` y `fin_semana` se calculan desde `timestamp`, para evitar inconsistencias.

## Publicar con Streamlit Community Cloud

1. En [share.streamlit.io](https://share.streamlit.io), conecte su cuenta de GitHub y elija **Create app**.
2. Seleccione `MAKEBUZ/taller-lstm-demanda-energetica`, rama `main` y `app.py` como archivo principal.
3. En **Advanced settings**, seleccione Python 3.12, pulse **Save** y luego **Deploy**.
4. Abra el enlace `*.streamlit.app` generado y compruebe una predicción con el ejemplo de la app. Ese es el enlace que puede compartir; `127.0.0.1` solo funciona en su computador.

No hace falta ejecutar el entrenamiento en la nube: el repositorio incluye los pesos y parámetros ya calculados. Se requieren una cuenta de GitHub y una de Streamlit Community Cloud para publicar el enlace.

### Si aparece un error de TensorFlow al instalar

Compruebe en los logs la versión de Python. `tensorflow-cpu==2.21.0` tiene paquete para **Python 3.12**, pero no para **Python 3.14**. Streamlit Community Cloud no permite cambiar la versión de Python de una app ya creada. Guarde primero su URL y cualquier configuración o secreto; luego elimine esa app y créela de nuevo con el mismo repositorio, rama `main` y archivo `app.py`. Antes de pulsar **Deploy**, abra **Advanced settings** y seleccione **Python 3.12**. Cambiar solo `requirements.txt` o reiniciar la app no corrige un entorno que ya fue creado con Python 3.14.
