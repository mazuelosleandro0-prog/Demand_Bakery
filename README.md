# 🥖 ModelBakery: Predicción de la demanda diaria de pan

![Caso real](https://img.shields.io/badge/Caso-REAL-success) ![Ubicación](https://img.shields.io/badge/Pisco-Perú-red) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-series%20de%20tiempo-blue)

Notebook de Machine Learning que predice la **cantidad diaria de panes vendidos** en una **panadería artesanal de Pisco, Perú**, a partir del historial de ventas y de variables de calendario (feriados, temporada escolar e invierno). Compara una regresión lineal contra dos modelos de ensamble (Random Forest y XGBoost) usando validación cruzada para series de tiempo y un baseline ingenuo.

> 📓 Todo el análisis está en [`ModelBakery.ipynb`](ModelBakery.ipynb).

---

## ✅ Es un caso real

> **Los datos no son simulados ni de ejemplo.** Corresponden al registro diario de ventas de una panadería artesanal real de Pisco, Perú, con sus feriados, su temporada escolar y su invierno.

Por eso el análisis enfrenta situaciones propias de un negocio pequeño: pocos feriados en el calendario, ventas que cambian según el día de la semana y la época del año, y la necesidad de que el modelo sea útil para decidir cuánto hornear.

## 📌 Objetivo

Estimar cuántos panes se venderán cada día para apoyar la planificación de la producción y reducir tanto el desabastecimiento como el desperdicio.

## 📊 Datos

| Característica | Detalle |
|---|---|
| Origen | Panadería artesanal de Pisco, Perú (datos reales) |
| Periodo | 02-01-2024 al 04-03-2026 |
| Registros | 681 (675 tras crear rezagos) |
| Variable objetivo | `Ventas en Panes` (unidades por día) |
| Promedio / mínimo | ≈ 5 796 / 4 475 panes por día |

**Variables originales**

- `Fecha`
- `Ventas en Panes`
- `Feriado` (1,6 % de los días)
- `Feriado_Largo` (0,7 %)
- `Temporada_Escolar` (62,7 %)
- `Invierno` (23,1 %)

## 🔧 Metodología

1. **Carga y calidad de datos**: sin valores faltantes y con tipos de dato correctos.
2. **Ingeniería de variables**
   - Rezagos de ventas: `lag_1`, `lag_2` y `lag_6`.
   - Media móvil de 3 días (`media_movil_3`), calculada sobre valores pasados para evitar fuga de información.
3. **Análisis exploratorio (EDA)**: efecto de las variables binarias, tendencia semanal, mensual y anual, y distribución de las ventas.
4. **Pipeline de preprocesamiento** (`scikit-learn`)
   - Mes con codificación cíclica (seno y coseno).
   - Día de la semana con One-Hot.
   - Semana del año.
   - Resto de variables en passthrough.
5. **Partición cronológica**: 80 % entrenamiento / 20 % prueba (sin mezclar el orden temporal).
6. **Modelos**: Regresión Lineal, Random Forest y XGBoost, evaluados con `TimeSeriesSplit` (10 particiones) y RMSE.
7. **Ajuste de hiperparámetros**: `RandomizedSearchCV` (10 iteraciones) para Random Forest y XGBoost.
8. **Importancia de variables** y prueba de selección automática con `SelectFromModel`.
9. **Evaluación final** en el conjunto de prueba, comparada con un baseline (`lag_1`, es decir, "hoy vendo lo mismo que ayer").
10. **Intervalo de confianza del 95 %** para el RMSE mediante *bootstrap* (10 000 remuestreos).

## 📈 Resultados

RMSE en unidades de pan (menor es mejor):

| Modelo | CV RMSE (train) | RMSE en test |
|---|---:|---:|
| Regresión Lineal | 206 | **151** |
| XGBoost (ajustado) | 240 | 161 |
| Random Forest (ajustado) | 253 | 188 |

- **Mejor desempeño en prueba: Regresión Lineal** (RMSE ≈ 151 panes), con un IC 95 % por bootstrap de **[137, 166]**, sobre ventas de alrededor de 5 800 panes diarios.
- Con hiperparámetros por defecto, Random Forest y XGBoost mostraron sobreajuste (RMSE de entrenamiento muy inferior al de validación cruzada); el ajuste con `RandomizedSearchCV` lo redujo.
- La selección automática de variables **no mejoró** el RMSE en ninguno de los dos modelos de ensamble.
- Los tres modelos se comparan contra el baseline `lag_1` en el gráfico de la sección 12 del notebook.

### Variables más importantes

- **Random Forest**: `lag_6` (≈ 56 %), lunes (≈ 10 %) y `lag_1` (≈ 8 %).
- **XGBoost**: lunes (≈ 28 %), invierno (≈ 24 %) y `lag_6` (≈ 11 %).

### Hallazgos del EDA

- Las ventas más altas se concentran en junio, julio y agosto; enero y febrero son los meses más bajos.
- La mayor frecuencia de ventas está entre 5 500 y 6 000 panes por día.
- Los feriados son pocos, pero el feriado largo muestra la tendencia más marcada.

## 🛠️ Tecnologías

- Python 3
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn
- XGBoost
- SciPy

## 📝 Notas

- La validación usa `TimeSeriesSplit` para respetar el orden temporal y evitar fuga de datos.
- Las asociaciones observadas son exploratorias y no implican causalidad.

## 👤 Autor

**Leandro Richard Mazuelos Pastrana**
Bachiller en Ingeniería Industrial

🔗 [LinkedIn](https://www.linkedin.com/in/leandro-mazuelos-3419612b1/)
