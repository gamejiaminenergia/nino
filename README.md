# Predictor de Fenómenos El Niño / La Niña

## Descripción

Este proyecto implementa un sistema de predicción para los fenómenos climáticos **El Niño** y **La Niña** utilizando datos históricos de anomalías de temperatura superficial del mar en la región Niño 3.4 del Océano Pacífico. El modelo genera pronósticos a 3 años vista basándose en más de un siglo de datos observacionales.

## Metodología

### Fuente de Datos

Los datos provienen del **National Oceanic and Atmospheric Administration (NOAA)** Physical Sciences Laboratory, específicamente del índice de anomalías de temperatura superficial del mar en la región Niño 3.4. Esta región, ubicada entre 5°N-5°S y 120°-170°W en el Pacífico ecuatorial central, es considerada el indicador más confiable para la detección y seguimiento de los eventos ENOS (El Niño-Oscilación del Sur).

- **Período de datos**: 1900 - 2025
- **Resolución temporal**: Mensual
- **Unidad**: Anomalías de temperatura en grados Celsius respecto al promedio climatológico

### Clasificación de Fenómenos

El sistema clasifica los fenómenos basándose en los umbrales estándar utilizados por la comunidad científica internacional:

- **El Niño** (valor +1): Anomalías ≥ +0.5°C
- **La Niña** (valor -1): Anomalías ≤ -0.5°C  
- **Condiciones Neutrales** (valor 0): Anomalías entre -0.5°C y +0.5°C

### Procesamiento de Datos

#### Suavizado Temporal
Para reducir la variabilidad de corto plazo y capturar las tendencias de mediano plazo características de los fenómenos ENOS, se aplica una **media móvil de 12 meses** a la serie temporal. Este procesamiento:

- Elimina fluctuaciones mensuales de alta frecuencia
- Preserva los ciclos estacionales e interanuales relevantes
- Mejora la capacidad predictiva del modelo al enfocarse en patrones persistentes

#### Preparación para Modelado
- Eliminación de valores faltantes o anómalos
- Normalización temporal con frecuencia mensual
- Estructuración de datos en formato compatible con modelos de series temporales

### Modelo de Predicción

Se utiliza **Prophet de Meta** (anteriormente Facebook Prophet), un modelo de series temporales especialmente diseñado para:

- **Tendencias no lineales**: Captura cambios graduales en el comportamiento del sistema climático
- **Estacionalidad múltiple**: Modela tanto los ciclos anuales como los patrones de más largo plazo
- **Robustez ante datos faltantes**: Maneja automáticamente gaps en la serie temporal
- **Intervalos de confianza**: Proporciona estimaciones de incertidumbre en las predicciones

### Horizonte de Predicción

El modelo genera pronósticos para **36 meses** (3 años) hacia el futuro, permitiendo:

- Planificación climática de mediano plazo
- Evaluación de riesgos asociados a eventos extremos
- Apoyo a la toma de decisiones en sectores sensibles al clima

## Interpretación de Resultados

### Valores de Salida
- **+1 a +0.5**: Condiciones de El Niño (calentamiento anómalo)
- **+0.5 a -0.5**: Condiciones neutrales
- **-0.5 a -1**: Condiciones de La Niña (enfriamiento anómalo)

### Consideraciones Importantes

1. **Limitaciones de Predictibilidad**: Los fenómenos ENOS tienen una predictibilidad intrínseca limitada más allá de 6-12 meses debido a su naturaleza caótica.

2. **Variabilidad Climática**: El modelo captura patrones estadísticos pero no puede predecir eventos climáticos extremos no precedentados.

3. **Contexto Regional**: Los impactos de El Niño/La Niña varían significativamente según la región geográfica y la estación del año.

## Aplicaciones

Este sistema de predicción puede ser utilizado para:

- **Agricultura**: Planificación de cultivos y gestión de riesgos
- **Gestión hídrica**: Preparación ante sequías o excesos de precipitación
- **Energía**: Optimización de recursos hidroeléctricos y eólicos
- **Gestión de desastres**: Preparación ante eventos climáticos extremos
- **Investigación climática**: Análisis de variabilidad climática interanual

## Referencias Científicas

- NOAA Physical Sciences Laboratory: Niño 3.4 Index
- Trenberth, K. E. (1997). The definition of El Niño. Bulletin of the American Meteorological Society
- McPhaden, M. J., et al. (2006). ENSO as an integrating concept in earth science