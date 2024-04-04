# Importando dados
!pip install netCDF4
!pip install metpy==1.0
!pip install cartopy
!pip install metpy

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from netCDF4 import Dataset
import pandas as pd
import xarray as xr
import matplotlib.ticker as mticker
import matplotlib.cm as cm
from numpy import linspace
from datetime import datetime
from metpy.units import units
import numpy as np
import scipy.ndimage as ndimage
from matplotlib.cm import get_cmap
import matplotlib.colors as colors
import ipywidgets as widgets
import cartopy.crs as ccrs
import metpy.calc as mpcalc
from metpy.plots import add_metpy_logo, Hodograph, SkewT
import matplotlib.gridspec as gridspec
import math
import metpy as mt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from metpy.cbook import get_test_data
from metpy.interpolate import cross_section
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob, os
from metpy.plots import SkewT, Hodograph


# Defina as variáveis de fatia
lat_slice = -23.76
lon_slice = -45.76
level_slice = slice(1000, 10, 100)

# Funções para calcular os índices de instabilidade
def calculate_indices(p, T, Td):
    temp = df3['t'][:]
    df3['tc'] = temp
    T1 = df3['tc'].values * units.kelvin
    ur = df3['r'][:]
    df3['Td'] = temp - ((100 - ur) / 5)
    Td1 = df3['Td'].values * units.kelvin
    p = df3['level'].values * units.hPa

    # CAPE e CIN
    cape, cin = mpcalc.surface_based_cape_cin(p, T1, Td1)

    # Lifted Index
    li = mpcalc.lifted_index(p, T1, Td1)

    # Showalter Index
    si = mpcalc.showalter_index(p, T1, Td1)

    # K Index
    ki = mpcalc.k_index(p, T1, Td1)

    # Total Totals Index
    tt = mpcalc.total_totals_index(p, T1, Td1)

    return cape, cin, li, si, ki, tt

# Carregue o arquivo de dados
file = xr.open_dataset('skew_era.nc')

# Obtenha todas as horas disponíveis no arquivo
horas_disponiveis = pd.to_datetime(file.time.values)

# Loop sobre as horas disponíveis
for hora in horas_disponiveis:
    hora_str = hora.strftime('%Y-%m-%dT%H:%M:%S')
    hora_pandas_str = str(hora)
    data = file.sel(latitude=lat_slice, longitude=lon_slice, time=hora_str, method='nearest')
    df = data.to_dataframe()

    df2 = df.reset_index(level='level')
    df3 = df2.iloc[::-1]

    # Temperaturas para o gráfico
    temp = df3['t'][:]
    df3['tc'] = temp - 273.15
    T = df3['tc'].values * units.degC
    ur = df3['r'][:]
    df3['Td'] = temp - ((100 - ur) / 5)
    Td = df3['Td'].values - 273
    p = df3['level'].values * units.hPa
    u = df3['u'][:]
    v = df3['v'][:]

    # Calcular os índices de instabilidade
    #cape, cin, li, si, ki, tt = calculate_indices(p, T1, Td1)

    # Criando as figuras
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(3, 3)

    # Plot 
    skew = SkewT(fig, rotation=45, subplot=gs[:, :2])
    skew.plot(p, T, 'darkred', linestyle='solid', linewidth=2)
    skew.plot(p, Td, 'darkgreen', linestyle='solid', linewidth=2)
    skew.plot_barbs(p, u, v)
    skew.ax.set_ylim(1000, 100)

    # TÍtulos
    hora_title = hora.strftime('%Y%m%d%HZ')
    plt.title(f'Barra do Una (São Sebastião) - {hora_title}', fontsize=21)
    plt.xlabel('Temperatura (°C)', fontsize=16)
    plt.ylabel('Pressão (hPa)', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Adicionando linhas
    skew.plot_dry_adiabats(colors='r', linestyle='solid', alpha=0.3)
    skew.plot_moist_adiabats(colors='lightsteelblue', linestyle='solid', alpha=1.0)
    skew.plot_mixing_lines(colors='b', linestyle='dashed')
    skew.ax.set_xlim(-30, 40)

    # NCL
    temp = df3['t'][:]
    df3['tc'] = temp
    T1 = df3['tc'].values * units.kelvin
    ur = df3['r'][:]
    df3['Td'] = temp - ((100 - ur) / 5)
    Td1 = df3['Td'].values * units.kelvin
    p = df3['level'].values * units.hPa

    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T1[0], Td1[0])
    parcel_prof = mpcalc.parcel_profile(p, T1[0], Td1[0]).to('degC')
    skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='k')

    # Plotando parcela
    skew.plot(p, parcel_prof, 'k', linestyle=':', linewidth=2)
    skew.shade_cin(p, T1, parcel_prof, color='maroon')
    skew.shade_cape(p, T1, parcel_prof, color='navy')

# Funções para calcular os índices de instabilidade
    def calculate_indices(p, T1, Td1):
        # CAPE e CIN
        cape, cin = mpcalc.surface_based_cape_cin(p, T1, Td1)

        # Lifted Index
        li = mpcalc.lifted_index(p, T1, Td1)
        li_mean = np.nanmean(li.magnitude)  # Média dos valores de Lifted Index

        # Showalter Index
        si = mpcalc.showalter_index(p, T1, Td1)
        si_mean = np.nanmean(si.magnitude)  # Média dos valores de Showalter Index

        # K Index
        ki = mpcalc.k_index(p, T1, Td1)
        ki_mean = np.nanmean(ki.magnitude)  # Média dos valores de K Index

        # Total Totals Index
        tt = mpcalc.total_totals_index(p, T1, Td1)
        tt_mean = np.nanmean(tt.magnitude)  # Média dos valores de Total Totals Index

        return cape.magnitude, cin.magnitude, li_mean, si_mean, ki_mean, tt_mean

    # Calcular os índices de instabilidade
    cape, cin, li, si, ki, tt = calculate_indices(p, T1, Td1)

    # Adicionar valores dos índices de instabilidade aos gráficos
    box_text = (f'CAPE: {cape:.2f} J/kg\n'
                f'CINE: {cin:.2f} J/kg\n'
                f'Lifted Index: {li:.2f} °C\n'
                f'Showalter: {si:.2f} °C\n'
                f'K: {ki:.2f}\n'
                f'Total Totals: {tt:.2f}')

    plt.text(0.66, 0.69, box_text, transform=fig.transFigure, fontsize=15,
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

    # Export result
    fig.savefig('local/'f'ERA5_indices_{hora_str}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Feche a figura atual para liberar memória
