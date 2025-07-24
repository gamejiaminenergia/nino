# %%
import pandas as pd
import numpy as np
import datetime
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Union, Dict, Any, Tuple
import os
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %%
class EnhancedDetector:
    def __init__(self):
        """Carga los datos una sola vez al crear la instancia."""
        self.data = {}
        self.raw_data = None
        self.get_data()

    def get_data(self):
        """Descarga y procesa los datos de NOAA."""
        url = "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data"
        column_names = ["YEAR", "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                       "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        
        df = pd.read_fwf(url, skiprows=1, names=column_names)
        df = df[pd.to_numeric(df["YEAR"], errors="coerce").notnull()]
        df["YEAR"] = df["YEAR"].astype(int)
        df = df[df["YEAR"] > 1800]
        
        # Convertir a formato largo
        df_long = df.melt(id_vars="YEAR", var_name="MONTH", value_name="ANOMALY")
        df_long["ANOMALY"] = pd.to_numeric(df_long["ANOMALY"], errors="coerce")
        df_long["DATE"] = pd.to_datetime(df_long["YEAR"].astype(str) + df_long["MONTH"], format="%Y%b")
        
        # Guardar datos raw para análisis posterior
        self.raw_data = df_long.copy()
        
        # Crear diccionario para búsqueda rápida
        for _, row in df_long.iterrows():
            anomalia = row["ANOMALY"]
            if pd.isna(anomalia):
                fenomeno = np.nan
            elif anomalia <= -0.5:
                fenomeno = -1
            elif anomalia >= 0.5:
                fenomeno = 1
            else:
                fenomeno = 0

            self.data[row["DATE"]] = fenomeno

    def consult(self, fecha: Union[datetime.datetime, str]) -> str:
        """Consulta el fenómeno para una fecha específica."""
        fecha_norm = pd.to_datetime(fecha).replace(day=1)
        return self.data.get(fecha_norm, np.nan)

# %%
class ENSOPredictor:
    def __init__(self, detector: EnhancedDetector):
        self.detector = detector
        self.model = None
        self.df = None
        self.forecast = None
        self.cv_results = None
        
    def prepare_data(self, start_year: int = 1900, end_year: int = 2025) -> pd.DataFrame:
        """Prepara los datos con características adicionales."""
        date_min = datetime.datetime(start_year, 1, 1)
        date_max = datetime.datetime(end_year, 7, 31)
        
        # Crear DataFrame base
        df = pd.DataFrame([
            {'ds': date, 'y': self.detector.consult(date)} 
            for date in pd.date_range(start=date_min, end=date_max, freq='ME')
        ])
        
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Aplicar ventana móvil de 3 meses (más sensible que 12)
        df['y'] = df['y'].rolling(window=3, min_periods=1).mean()
        df['y'] = df['y'].astype(float)
        df = df.dropna(subset=['y'])
        
        # Agregar regresores adicionales
        df = self._add_regressors(df)
        
        self.df = df
        return df
    
    def _add_regressors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega regresores adicionales al DataFrame."""
        # Tendencia cuadrática
        idx_mean = len(df) / 2
        df['trend_squared'] = (df.index - idx_mean) ** 2
        
        # Componente cíclico de largo plazo (ciclo ~3-7 años)
        df['long_cycle'] = np.sin(2 * np.pi * df.index / (5 * 12))  # 5 años
        
        # Componente estacional adicional
        df['month'] = df['ds'].dt.month
        df['season_strength'] = np.sin(2 * np.pi * df['month'] / 12)
        
        # Indicador de décadas (para capturar cambios de largo plazo)
        df['decade'] = (df['ds'].dt.year // 10) * 10
        df['decade_normalized'] = (df['decade'] - df['decade'].min()) / (df['decade'].max() - df['decade'].min())
        
        return df
    
    def create_model(self) -> Prophet:
        """Crea modelo Prophet optimizado para datos ENSO."""
        model = Prophet(
            # Parámetros optimizados para datos climáticos
            changepoint_prior_scale=0.1,  # Más conservativo para cambios de tendencia
            seasonality_prior_scale=0.8,   # Estacionalidad moderada
            holidays_prior_scale=0.5,      # No hay holidays pero por si acaso
            seasonality_mode='additive',    # Modo aditivo es mejor para anomalías
            interval_width=0.85,           # Intervalos de confianza del 85%
            
            # Estacionalidades personalizadas
            yearly_seasonality=True,
            weekly_seasonality=False,       # No relevante para datos mensuales
            daily_seasonality=False,        # No relevante para datos mensuales
            
            # Configuraciones adicionales
            mcmc_samples=300,              # Más muestras para mejor incertidumbre
            uncertainty_samples=1000,      # Más muestras para intervalos
        )
        
        # Agregar estacionalidad personalizada para ciclos ENSO
        model.add_seasonality(
            name='enso_cycle',
            period=3.5 * 365.25,  # Ciclo promedio de 3.5 años
            fourier_order=3,
            prior_scale=0.5
        )
        
        # Agregar estacionalidad decenal
        model.add_seasonality(
            name='decadal',
            period=10 * 365.25,   # Ciclo de 10 años
            fourier_order=2,
            prior_scale=0.3
        )
        
        return model
    
    def fit_model(self, df: pd.DataFrame = None) -> Prophet:
        """Entrena el modelo con los datos."""
        if df is None:
            df = self.df
            
        self.model = self.create_model()
        
        # Agregar regresores al modelo
        for col in ['trend_squared', 'long_cycle', 'season_strength', 'decade_normalized']:
            if col in df.columns:
                self.model.add_regressor(col, prior_scale=0.5)
        
        # Entrenar modelo
        self.model.fit(df)
        return self.model
    
    def make_forecast(self, periods: int = 36) -> pd.DataFrame:
        """Genera predicciones futuras."""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecuta fit_model() primero.")
        
        # Crear DataFrame futuro
        future = self.model.make_future_dataframe(periods=periods, freq='ME')
        
        # Agregar regresores para fechas futuras
        future = self._add_future_regressors(future)
        
        # Generar predicciones
        self.forecast = self.model.predict(future)
        return self.forecast
    
    def _add_future_regressors(self, future: pd.DataFrame) -> pd.DataFrame:
        """Agrega regresores para fechas futuras."""
        # Usar los mismos cálculos que en _add_regressors
        base_index = len(self.df)
        
        idx_mean = len(future) / 2
        future_indices = np.arange(len(future))
        future['trend_squared'] = (future_indices - idx_mean) ** 2
        future['long_cycle'] = np.sin(2 * np.pi * future_indices / (5 * 12))
        future['month'] = future['ds'].dt.month
        future['season_strength'] = np.sin(2 * np.pi * future['month'] / 12)
        future['decade'] = (future['ds'].dt.year // 10) * 10
        
        # Normalizar década basado en datos históricos
        decade_min = self.df['decade'].min()
        decade_max = max(self.df['decade'].max(), future['decade'].max())
        future['decade_normalized'] = (future['decade'] - decade_min) / (decade_max - decade_min)
        
        return future
    
    def cross_validate_model(self, initial: str = '3650 days', 
                           period: str = '180 days', horizon: str = '365 days') -> pd.DataFrame:
        """Realiza validación cruzada temporal."""
        if self.model is None:
            raise ValueError("Modelo no entrenado.")
        
        self.cv_results = cross_validation(
            self.model, 
            initial=initial,
            period=period, 
            horizon=horizon,
            parallel="processes"
        )
        return self.cv_results
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evalúa el rendimiento del modelo."""
        if self.cv_results is None:
            print("Ejecutando validación cruzada...")
            self.cross_validate_model()
        
        # Métricas de Prophet
        prophet_metrics = performance_metrics(self.cv_results)
        
        # Métricas adicionales en datos de entrenamiento
        train_pred = self.forecast[self.forecast['ds'].isin(self.df['ds'])]
        train_actual = self.df.merge(train_pred[['ds', 'yhat']], on='ds')
        
        metrics = {
            'MAE_CV': prophet_metrics['mae'].mean(),
            'RMSE_CV': prophet_metrics['rmse'].mean(),
            'MAPE_CV': prophet_metrics['mape'].mean(),
            'MAE_Train': mean_absolute_error(train_actual['y'], train_actual['yhat']),
            'RMSE_Train': np.sqrt(mean_squared_error(train_actual['y'], train_actual['yhat'])),
            'R2_Train': r2_score(train_actual['y'], train_actual['yhat'])
        }
        
        return metrics
    
    def plot_forecast(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Crea visualización completa de la predicción."""
        if self.forecast is None:
            raise ValueError("No hay predicciones. Ejecuta make_forecast() primero.")
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('Análisis Completo: Predicción El Niño/La Niña', fontsize=16, fontweight='bold')
        
        # 1. Predicción principal
        ax1 = axes[0, 0]
        self.model.plot(self.forecast, ax=ax1, xlabel='Fecha', 
                       ylabel='Índice ENSO (1: El Niño, -1: La Niña)')
        ax1.set_title('Predicción con Intervalos de Confianza')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Umbral El Niño')
        ax1.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.7, label='Umbral La Niña')
        ax1.legend()
        
        # 2. Componentes
        ax2 = axes[0, 1]
        components = self.model.plot_components(self.forecast, figsize=(8, 6))
        plt.close(components)  # Cerrar la figura de componentes separada
        
        # Recrear componentes en subplot
        ax2.plot(self.forecast['ds'], self.forecast['trend'], label='Tendencia')
        ax2.plot(self.forecast['ds'], self.forecast['yearly'], label='Estacional Anual')
        if 'enso_cycle' in self.forecast.columns:
            ax2.plot(self.forecast['ds'], self.forecast['enso_cycle'], label='Ciclo ENSO')
        ax2.set_title('Componentes del Modelo')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Residuos
        ax3 = axes[1, 0]
        train_data = self.forecast[self.forecast['ds'].isin(self.df['ds'])]
        merged = self.df.merge(train_data[['ds', 'yhat']], on='ds')
        residuals = merged['y'] - merged['yhat']
        
        ax3.scatter(merged['yhat'], residuals, alpha=0.6)
        ax3.axhline(y=0, color='red', linestyle='--')
        ax3.set_xlabel('Valores Predichos')
        ax3.set_ylabel('Residuos')
        ax3.set_title('Análisis de Residuos')
        
        # 4. Distribución de residuos
        ax4 = axes[1, 1]
        ax4.hist(residuals, bins=30, alpha=0.7, density=True)
        ax4.axvline(residuals.mean(), color='red', linestyle='--', label=f'Media: {residuals.mean():.3f}')
        ax4.set_xlabel('Residuos')
        ax4.set_ylabel('Densidad')
        ax4.set_title('Distribución de Residuos')
        ax4.legend()
        
        # 5. Valores reales vs predichos
        ax5 = axes[2, 0]
        ax5.scatter(merged['y'], merged['yhat'], alpha=0.6)
        min_val, max_val = min(merged['y'].min(), merged['yhat'].min()), max(merged['y'].max(), merged['yhat'].max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', label='Línea perfecta')
        ax5.set_xlabel('Valores Reales')
        ax5.set_ylabel('Valores Predichos')
        ax5.set_title('Valores Reales vs Predichos')
        ax5.legend()
        
        # 6. Últimas predicciones destacadas
        ax6 = axes[2, 1]
        recent_data = self.forecast.tail(60)  # Últimos 5 años
        ax6.plot(recent_data['ds'], recent_data['yhat'], 'b-', linewidth=2, label='Predicción')
        ax6.fill_between(recent_data['ds'], recent_data['yhat_lower'], 
                        recent_data['yhat_upper'], alpha=0.3, label='Intervalo 85%')
        
        # Destacar datos reales recientes
        recent_actual = self.df.tail(24)  # Últimos 2 años reales
        ax6.plot(recent_actual['ds'], recent_actual['y'], 'ro-', label='Datos Reales')
        
        ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        ax6.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.7)
        ax6.set_title('Predicciones Recientes (5 años)')
        ax6.legend()
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def get_latest_predictions(self, months: int = 12) -> pd.DataFrame:
        """Obtiene las últimas predicciones con interpretación."""
        if self.forecast is None:
            raise ValueError("No hay predicciones disponibles.")
        
        latest = self.forecast.tail(months).copy()
        
        # Agregar interpretación
        latest['fenomeno'] = latest['yhat'].apply(lambda x: 
            'El Niño Fuerte' if x > 1.0 else
            'El Niño Moderado' if x > 0.5 else
            'La Niña Fuerte' if x < -1.0 else
            'La Niña Moderada' if x < -0.5 else
            'Neutral'
        )
        
        # Agregar probabilidades
        latest['prob_nino'] = np.where(latest['yhat'] > 0.5, 
                                     np.minimum((latest['yhat'] - 0.5) * 2, 1.0), 0)
        latest['prob_nina'] = np.where(latest['yhat'] < -0.5, 
                                     np.minimum((-latest['yhat'] - 0.5) * 2, 1.0), 0)
        latest['prob_neutral'] = 1 - latest['prob_nino'] - latest['prob_nina']
        
        return latest[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'fenomeno', 
                      'prob_nino', 'prob_nina', 'prob_neutral']]

# %%
def main():
    """Función principal para ejecutar el análisis completo."""
    print("🌊 Iniciando análisis avanzado de El Niño/La Niña...")
    
    # 1. Cargar datos
    print("📊 Cargando datos de NOAA...")
    detector = EnhancedDetector()
    predictor = ENSOPredictor(detector)
    
    # 2. Preparar datos
    print("🔧 Preparando datos con características adicionales...")
    df = predictor.prepare_data()
    print(f"   ✓ {len(df)} observaciones mensuales desde {df['ds'].min().year}")
    
    # 3. Entrenar modelo
    print("🤖 Entrenando modelo Prophet optimizado...")
    model = predictor.fit_model(df)
    print("   ✓ Modelo entrenado con éxito")
    
    # 4. Generar predicciones
    print("🔮 Generando predicciones para los próximos 3 años...")
    forecast = predictor.make_forecast(periods=36)
    
    # 5. Evaluar modelo
    print("📈 Evaluando rendimiento del modelo...")
    try:
        metrics = predictor.evaluate_model()
        print("   ✓ Métricas de evaluación:")
        for metric, value in metrics.items():
            print(f"     {metric}: {value:.4f}")
    except Exception as e:
        print(f"   ⚠️  Error en validación cruzada: {e}")
        print("   Continuando sin validación cruzada...")
    
    # 6. Crear visualizaciones
    print("📊 Creando visualizaciones...")
    fig = predictor.plot_forecast(figsize=(18, 12))
    
    # Guardar figura
    output_path = os.path.join(os.getcwd(), 'enhanced_enso_forecast.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Gráfico guardado: {output_path}")
    
    # 7. Mostrar predicciones recientes
    print("\n🎯 Predicciones para los próximos 12 meses:")
    latest_pred = predictor.get_latest_predictions(12)
    
    for _, row in latest_pred.iterrows():
        fecha = row['ds'].strftime('%Y-%m')
        valor = row['yhat']
        fenomeno = row['fenomeno']
        intervalo = f"[{row['yhat_lower']:.2f}, {row['yhat_upper']:.2f}]"
        
        print(f"   {fecha}: {valor:+.2f} ({fenomeno}) - IC 85%: {intervalo}")
    
    print("\n✅ Análisis completado exitosamente!")
    return predictor

# %%
if __name__ == "__main__":
    predictor = main()