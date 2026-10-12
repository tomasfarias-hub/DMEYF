# Predicción de bajas — Competencia 01 (agosto 2021)

Modelo LightGBM que arma el listado de clientes a los que se les envía la campaña de retención en **agosto 2021 (202108)**.

**Resultado:** 11.972 clientes (7,27 % de la cartera de 164.647). Aplicado a junio, el mismo procedimiento da una ganancia de unos 356 M.

## Contenido

1. [El problema](#1-el-problema)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Cómo correrlo](#3-cómo-correrlo)
4. [Qué hace la notebook, sección por sección](#4-qué-hace-la-notebook-sección-por-sección)
5. [Resultados](#5-resultados)

---

## 1. El problema

Cada cliente puede tener tres clases (`clase_ternaria`): `CONTINUA`, `BAJA+1` y `BAJA+2`. Se contacta a un cliente con una campaña de retención, y la economía es:

| Situación | Resultado |
|---|---|
| Contactar a un BAJA+2 | **+ $1.072.500** |
| Contactar a cualquier otro cliente | **− $27.500** |
| No contactar | $0 |

El objetivo es entregar la lista de `numero_de_cliente` a contactar en 202108, un mes para el que todavía no se conoce la clase. La lista tiene que maximizar la ganancia esperada.

**Punto de equilibrio.** Contactar a un cliente con probabilidad *p* de ser BAJA+2 conviene si
`p · 1.072.500 − (1 − p) · 27.500 > 0`, es decir, si `p > 27.500 / 1.100.000 = 0,025`.
Por eso el umbral de probabilidad 0,025 aparece en varias partes del código.

---

## 2. Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── competencia_01_lgbm.ipynb     # todo el código, con las salidas de la corrida de la entrega
├── docs/
│   └── reproducibilidad.md           # entorno, semillas y el tema de Optuna, en detalle
├── resultados/
│   ├── hiperparametros.json          # hiperparámetros del modelo final
│   └── metricas.md                   # ganancias en junio y datos de la entrega
├── data/                             # acá va competencia_01.csv (no se sube al repo)
└── modelos/                          # acá se guarda el CSV de la entrega
```

---

## 3. Cómo correrlo

La notebook está escrita para **Google Colab**: monta Google Drive y descarga el archivo final con `google.colab.files`.

1. Subir esta carpeta a Google Drive y colocar el dataset en `data/competencia_01.csv`.
2. Abrir `notebooks/competencia_01_lgbm.ipynb` en Colab.
3. En la segunda celda, cambiar `base_path` por la ruta de esa carpeta en el Drive propio. Debe terminar en `/`:
   ```python
   base_path = '/content/drive/MyDrive/DMEyF/2026/notebooks/'
   ```
4. Ejecutar todas las celdas en orden.

La notebook trae `USAR_PARAMS_FIJOS = True`, así que **no re-optimiza**: entrena directamente con los hiperparámetros de la entrega.
Los detalles (semillas, versiones, qué pasa si se activa Optuna) están en [`docs/reproducibilidad.md`](docs/reproducibilidad.md).

---

## 4. Qué hace la notebook, sección por sección

### 4.1 Librerías

Instala Optuna (`!pip install -q optuna`) e importa:

- **pandas y numpy** para los datos; **matplotlib** para los gráficos.
- **scikit-learn**: `StratifiedShuffleSplit`, que se usa para armar las particiones de validación.
- **LightGBM**: el modelo.
- **Optuna**: la optimización de hiperparámetros.

También se importan plotly, `gaussian_kde`, `ks_2samp` y las funciones `plot_*` de Optuna, que esta versión de la notebook no llega a usar.

### 4.2 Configuración y carga de datos

- Monta Google Drive y define las rutas: `data/` (dataset), `modelos/` (entrega) y `db/` (base de Optuna).
- Define la economía del problema: `ganancia_acierto = 1.072.500` y `costo_estimulo = 27.500`.
- Define las **particiones temporales**:

  | Variable | Meses | Uso |
  |---|---|---|
  | `mes_train` | 202103, 202104 | entrenamiento y optimización de hiperparámetros |
  | `mes_test` | 202106 | evaluación, con clase conocida |
  | `mes_final_train` | 202103 a 202106 | entrenamiento del modelo final |
  | `mes_futuro` | 202108 | mes a predecir |

  Mayo solo se usa en el entrenamiento final y julio no se usa. Entre el último mes de entrenamiento (abril) y el de evaluación (junio) hay dos meses,
  la misma distancia que entre junio y agosto.
- Fija la **semilla** `semilla = 2323` y genera con ella una lista de 50 semillas distintas (`semillas`), que se usan en toda la notebook para que los resultados se puedan repetir.
- Lee `competencia_01.csv`.

### 4.3 Feature engineering

Se agregan variables nuevas que describen cómo cambió cada cliente respecto del mes anterior:

1. Ordena el dataset por `numero_de_cliente` y `foto_mes`.
2. Crea `tc_consumo_total`: consumo de la tarjeta Visa más el de la Master (`Visa_mconsumototal + Master_mconsumototal`, con los vacíos en 0).
3. Para 17 variables (`CAMPOS_DELTA`) calcula `delta_1_<variable>` = valor del mes menos valor del mes anterior **del mismo cliente**. Las variables son:
   - Sueldos: `mpayroll`, `cpayroll_trx`.
   - Tarjetas: consumo y transacciones Visa, débitos automáticos, `tc_consumo_total`.
   - Actividad: `cproductos`, `ctrx_quarter`, `active_quarter`, `internet`, `cdescubierto_preacordado`.
   - Canales digitales: `chomebanking_transacciones`, `cmobile_app_trx`.
   - Saldos y productos: `mcuentas_saldo`, `mcaja_ahorro`, `mprestamos_personales`, `mplazo_fijo_pesos`.
4. Verifica que ninguna columna falte e imprime el porcentaje de filas con delta calculado por mes. En 202103 es 0 % porque no existe febrero; en el resto, alrededor de 99 %.

El dataset pasa de 155 columnas originales a **173** (155 + `tc_consumo_total` + 17 deltas), con 983.061 filas.
Las nuevas columnas se guardan como `float32`.

### 4.4 Target binario

Se crea `clase_binaria`: **1** si `clase_ternaria == 'BAJA+2'`, **0** en cualquier otro caso (`BAJA+1` y `CONTINUA`). Es el único tipo de cliente que da ganancia al contactarlo.

### 4.5 Armado de los conjuntos de entrenamiento y prueba

- Excluye de las variables predictoras `clase_ternaria`, `clase_binaria`, `numero_de_cliente` y `foto_mes`. Quedan **170 variables**.
- `train_data`: filas de marzo y abril. `test_data`: filas de junio.
- Arma `X_train`, `y_train_binaria`, `X_test`, `y_test_binaria` y el `lgb.Dataset` de entrenamiento.

### 4.6 Optimización de hiperparámetros con Optuna

Busca la mejor combinación de hiperparámetros de LightGBM para el objetivo del negocio, que es la ganancia y no una métrica estadística.

**Validación cruzada.** `StratifiedShuffleSplit` con 5 particiones y 30 % de validación en cada una, estratificadas por clase. Se usa dentro de `lgb.cv`.

**Métrica personalizada (`ganancia_prob_lgm`).** Para cada partición de validación suma la ganancia de los clientes cuya probabilidad predicha es ≥ 0,025
y la divide por 0,3. Esa división escala la ganancia de una validación del 30 % de los datos a la del conjunto completo.

**Función objetivo.** En cada trial, Optuna propone valores para cinco hiperparámetros:

| Hiperparámetro | Rango |
|---|---|
| `num_leaves` | 8 a 100 |
| `learning_rate` | 0,005 a 0,3 |
| `min_data_in_leaf` | 1 a 1000 |
| `feature_fraction` | 0,1 a 1,0 |
| `bagging_fraction` | 0,1 a 1,0 |

Con ellos corre `lgb.cv` hasta 1000 árboles, con `max_bin = 31` (más rápido) y *early stopping*: corta si la ganancia deja de mejorar durante `50 + 5 / learning_rate` iteraciones.
Cuanto menor la tasa de aprendizaje, más paciencia. El trial devuelve la **máxima ganancia promedio** de la validación cruzada y guarda en `best_iter` en qué iteración se alcanzó.

**Persistencia.** El estudio (`exp_301_lgbm_sss`) se guarda en una base SQLite en `db/`, de modo que puede retomarse y acumular trials.

### 4.7 Hiperparámetros fijos (interruptor)

Para que la notebook se pueda reproducir sin repetir la optimización, hay un interruptor:

- `USAR_PARAMS_FIJOS = True`: usa `PARAMS_OPTIMOS` y `BEST_ITER` (los valores del mejor trial de la corrida de la entrega).
- `USAR_PARAMS_FIJOS = False`: se ejecuta `study.optimize(objective, n_trials=15)` y se usa el mejor trial del estudio.

Valores fijos: `num_leaves = 19`, `learning_rate = 0,0211277`, `min_data_in_leaf = 858`, `feature_fraction = 0,8257278`, `bagging_fraction = 0,2302572` y 418 árboles.

### 4.8 Entrenamiento del modelo

Elige los hiperparámetros según el interruptor, arma el diccionario `params` y entrena un LightGBM con `lgb.train` sobre marzo y abril, con 418 árboles y `max_bin = 31`.

### 4.9 Importancia de variables

Calcula cuántas veces usó el modelo cada variable para dividir (importancia por *split*) y la ordena de mayor a menor. De las 170 variables, **138** tienen importancia mayor que cero.
Las más usadas son `mrentabilidad_annual`, `mcuentas_saldo`, `mprestamos_personales`, `ctrx_quarter` y `mactivos_margen`.

### 4.10 Predicción en junio

Aplica el modelo a las filas de junio (`X_test`), un mes que no se usó para entrenar ni para optimizar.

### 4.11 Ganancia en junio

Con la función `gan_prob`, suma la ganancia real de junio de los clientes con probabilidad ≥ 0,025.
Un solo modelo da **341.825.000** (sin escalar, porque se evalúa el mes completo).

### 4.12 Análisis de las hojas de los árboles

Un modelo LightGBM es un conjunto de árboles, y cada observación cae en una hoja de cada árbol. Esta parte examina qué tipo de clientes cae en cada hoja:

1. `get_leaf_info_lgbm` obtiene, con `pred_leaf=True`, la hoja de cada observación de entrenamiento en cada árbol.
2. Cuenta cuántos `BAJA+1`, `BAJA+2` y `CONTINUA` hay en cada hoja y marca la clase mayoritaria. Devuelve una tabla con una fila por árbol y hoja.
3. **Ganancia por nodo**: `1.072.500 × BAJA+2 − 27.500 × (BAJA+1 + CONTINUA)`, ordenando las hojas de mayor a menor ganancia.
4. **Valor de la hoja según el modelo**: extrae con `trees_to_dataframe` el valor de cada hoja (en escala *log-odds*) y lo convierte en probabilidad con la función sigmoide, `1 / (1 + e^-valor)`.
   También calcula la **probabilidad empírica** de cada hoja, `BAJA+2 / Samples`.

### 4.13 Probabilidad empírica contra probabilidad del modelo

Regenera la tabla de hojas y, para cada una, calcula la probabilidad final media que el modelo completo asigna a las observaciones que caen ahí. Luego grafica:

- Un diagrama de dispersión de la probabilidad empírica de cada hoja contra la probabilidad final media (con el umbral 0,025 marcado).
- Un histograma, en escala logarítmica, de las probabilidades predichas sobre entrenamiento.

Sirve para ver si las hojas con muchos BAJA+2 reales son las que el modelo califica con probabilidad alta.

### 4.14 Ganancia acumulada y puntos de corte

Pregunta a cuántos clientes conviene enviar. Sobre junio:

1. Ordena a los clientes de mayor a menor probabilidad predicha.
2. Recorre la lista sumando +1.072.500 por cada BAJA+2 y −27.500 por cada otro cliente, para obtener la **ganancia acumulada** según la cantidad de envíos.
3. Identifica el corte óptimo (el máximo de la curva) y el corte por umbral 0,025.
4. Muestra una tabla de ganancia para cortes de 500 a 30.000 envíos y dos gráficos: la curva completa y un zoom alrededor del óptimo.
5. Traslada el corte óptimo a agosto **como proporción de la cartera**, porque agosto no tiene la misma cantidad de clientes que junio.

### 4.15 Creación del archivo de entrega

Es la sección que produce el resultado final. Parámetros configurables: `metodo`, `umbral`, `n_envios_manual`, `N_SEMILLAS = 5` y `VENTANA = 500`.

1. **Semillerío.** La función `entrenar_promedio` entrena 5 modelos idénticos, cada uno con una semilla distinta, y promedia sus probabilidades. Promediar reduce la dependencia de una semilla en particular.
2. **Validación.** Entrena el semillerío con marzo y abril (`max_bin = 31`), predice junio, y calcula la curva de ganancia acumulada.
3. **Suavizado.** Aplica a la curva una media móvil centrada de 500 clientes para evitar que el máximo dependa de un pico aislado, y toma el máximo de la curva suavizada.
   Eso da el corte óptimo como proporción de la cartera (`prop_opt`).
4. **Modelo final.** Entrena el semillerío de 5 modelos con marzo a junio y `max_bin = 255` (más resolución en las variables numéricas), y predice agosto.
5. **Selección de clientes**, según `metodo`:
   - `"optimo"`: toma el mismo porcentaje de la cartera que el óptimo de junio.
   - `"umbral"`: todos los clientes con probabilidad ≥ `umbral`.
   - `"cantidad"`: los primeros `n_envios_manual`.

   La entrega se hizo con `"optimo"`.
6. **Controles.** Verifica que no haya clientes repetidos y que todos pertenezcan a agosto.
7. **Guardado.** Escribe el archivo sin encabezado, un `numero_de_cliente` por línea, con el nombre `entrega_202108_optimo_11972.csv`, en `modelos/` (Drive) y en el entorno de Colab, y lo descarga.

---

## 5. Resultados

Evaluación en junio (202106): 164.114 clientes, 1.098 BAJA+2.

| Modelo | Criterio de corte | Envíos | % cartera | Ganancia |
|---|---|---|---|---|
| 1 semilla | probabilidad ≥ 0,025 | 8.770 | 5,34 % | 341,82 M |
| 1 semilla | óptimo (máximo de la curva) | 11.618 | 7,08 % | 350,40 M |
| Semillerío de 5 | óptimo (curva suavizada) | 11.933 | 7,27 % | 356,04 M |

Ganancia de junio según la cantidad de envíos (un modelo):

| Envíos | % cartera | Prob. de corte | Ganancia |
|---|---|---|---|
| 2.000 | 1,22 % | 0,0873 | 152,9 M |
| 5.000 | 3,05 % | 0,0434 | 282,7 M |
| 8.000 | 4,87 % | 0,0276 | 341,0 M |
| 10.000 | 6,09 % | 0,0216 | 344,3 M |
| 15.000 | 9,14 % | 0,0133 | 341,0 M |
| 20.000 | 12,19 % | 0,0092 | 317,9 M |
| 30.000 | 18,28 % | 0,0052 | 176,0 M |

La ganancia crece rápido hasta unos 8.000 envíos, se mantiene casi plana entre 8.000 y 15.000, y cae después.

**Entrega 202108:** 11.972 clientes, 7,27 % de los 164.647 de agosto. El último cliente incluido tiene probabilidad 0,01685.
El detalle y los hiperparámetros están en [`resultados/`](resultados/).
