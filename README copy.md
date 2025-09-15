<<<<<<< HEAD
# Proyecto ML — Estructura base

Este repositorio sigue la estructura propuesta en el enunciado.

## Estructura

```
nombre_proyecto_final_ML/
├─ notebooks/
│  ├─ 01_Fuentes.ipynb            # (opcional, no generado ahora porque no se obtienen datos de diferentes fuentes)
│  ├─ 02_LimpiezaEDA.ipynb        # Copia de Preprocesado.ipynb
│  ├─ 03_Entrenamiento.ipynb      # Parte de entrenamiento (split de ML.ipynb)
│  └─ 04_Evaluacion.ipynb         # Parte de evaluación (split de ML.ipynb)
├─ src/
│  ├─ preprocessing.py
│  ├─ training.py
│  └─ evaluation.py
├─ models/                         # Artefactos del modelo (no incluidos)
├─ app/
│  ├─ app.py
│  └─ requirements.txt
├─ docs/
│  ├─ negocio.ppt
│  ├─ ds.ppt
│  └─ memoria.md
└─ README.md
```

## Notas

- Los notebooks 03 y 04 se han generado con una heurística: importaciones/datos → comunes; celdas con `.fit`/búsqueda → entrenamiento; celdas con métricas/predicción → evaluación.
- Revisa que los paths relativos sigan funcionando tras mover los notebooks a `notebooks/`.
- Las funciones detectadas se han exportado a `src/`. Puedes refactorizar y reemplazar llamadas en los notebooks más adelante.

=======
# Proyecto-final-v3
>>>>>>> d1d4f9c988a06e63ece24fd40d8e7065cb5ccba6
