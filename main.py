# %%
import pandas as pd
import numpy as np
import datetime
from prophet import Prophet
from typing import Union, Dict, Any
import os

# %%
class Detector:
    def __init__(self):
        """Carga los datos una sola vez al crear la instancia."""
        self.data = {}
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
        
        # Convertir a formato largo y crear diccionario
        df_long = df.melt(id_vars="YEAR", var_name="MONTH", value_name="ANOMALY")
        df_long["ANOMALY"] = pd.to_numeric(df_long["ANOMALY"], errors="coerce")
        df_long["DATE"] = pd.to_datetime(df_long["YEAR"].astype(str) + df_long["MONTH"], format="%Y%b")
        
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
        """
        Consulta el fenómeno para una fecha específica.
        
        Args:
            fecha: Fecha a consultar
            
        Returns:
            str: 'El Niño', 'La Niña', 'Neutral' o 'Sin dato'
        """
        # Convertir fecha al primer día del mes
        fecha_norm = pd.to_datetime(fecha).replace(day=1)
        
        # Buscar en el diccionario
        return self.data.get(fecha_norm, np.nan)


# %%
def dataframe_detector():
    """
    Crea un DataFrame con las fechas y los valores de El Niño/La Niña.
    
    Returns:
        pd.DataFrame: DataFrame con columnas 'DS' (fecha) y 'Y' (valor)
    """
    detector= Detector()

    date_min = datetime.datetime(1900, 1, 1)
    date_max = datetime.datetime(2025, 7, 31)
    df = pd.DataFrame([{'ds': date, 'y': detector.consult(date)} for date in pd.date_range(start=date_min, end=date_max, freq='ME')])
    df['ds'] = pd.to_datetime(df['ds'])
    #nesecito el promedio de y con un ventana de 12 meses
    df['y'] = df['y'].rolling(window=12, min_periods=1).mean()
    # convertir ds a datetime y y a float
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = df['y'].astype(float)
    # eliminar filas donde y es NaN
    df = df.dropna(subset=['y'])
    # establecer ds como índice
    # df = df.set_index('ds')

    return df

# %%
df = dataframe_detector()
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=12*3, freq='ME')
forecast = m.predict(future)


# %%
fig = m.plot(
    fcst=forecast, 
    xlabel='Fecha', ylabel='Predicción fenómeno El Niño / La Niña (1: Niño, -1: Niña)', 
    figsize=(10*3, 6*1.2), 
    include_legend=True
)

fig.savefig(os.path.join(os.getcwd(), 'forecast_nino_nina.png'), bbox_inches='tight')


