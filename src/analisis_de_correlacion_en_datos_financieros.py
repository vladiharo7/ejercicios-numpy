import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Crear datos financieros simulados
np.random.seed(42)
fechas = pd.date_range('20200101', periods=500)
df_financiero = pd.DataFrame({
    'fecha': fechas,
    'sp500': np.cumsum(np.random.normal(0.0005, 0.01, 500)),
    'nasdaq': np.cumsum(np.random.normal(0.0007, 0.012, 500)),
    'petroleo': np.cumsum(np.random.normal(0.0001, 0.015, 500)),
    'oro': np.cumsum(np.random.normal(0.0003, 0.008, 500)),
    'bonos': np.cumsum(np.random.normal(0.0002, 0.003, 500))
})

# Añadir correlaciones realistas
df_financiero['nasdaq'] = df_financiero['sp500'] * 1.2 + np.random.normal(0, 0.5, 500)
df_financiero['oro'] = -0.3 * df_financiero['sp500'] + np.random.normal(0, 1, 500)

# Calcular rendimientos diarios (más relevante que precios absolutos)
activos = ['sp500', 'nasdaq', 'petroleo', 'oro', 'bonos']
for activo in activos:
    df_financiero[f'rend_{activo}'] = df_financiero[activo].pct_change() * 100

# Eliminar primera fila (NaN por el cálculo de rendimientos)
df_financiero = df_financiero.dropna()

# Calcular matriz de correlación de rendimientos
columnas_rend = [f'rend_{activo}' for activo in activos]
corr_rendimientos = df_financiero[columnas_rend].corr()

# Visualizar matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(corr_rendimientos, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlación entre rendimientos de activos financieros')
plt.tight_layout()
plt.show()

# Analizar correlaciones cambiantes (ventana móvil)
ventana = 60  # 60 días
corr_movil = df_financiero[columnas_rend].rolling(window=ventana).corr()

# Extraer correlación móvil entre S&P 500 y oro
corr_sp_oro = corr_movil.xs(('rend_sp500', 'rend_oro'), level=(0, 1))

# Visualizar correlación cambiante en el tiempo
plt.figure(figsize=(12, 6))
corr_sp_oro.plot()
plt.axhline(y=0, color='r', linestyle='--')
plt.title(f'Correlación móvil ({ventana} días) entre S&P 500 y Oro')
plt.ylabel('Correlación')
plt.ylim(-1, 1)
plt.show()
