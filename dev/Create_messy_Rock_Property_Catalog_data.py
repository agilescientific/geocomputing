import pandas as pd
import numpy as np

"""
Deliberately messing up a data file for pandas and machine learning
data preparation practice.
"""
rng = np.random.default_rng(42)

# Load the data.
df = pd.read_csv('https://geocomp.s3.amazonaws.com/data/RPC_4_lithologies.csv')

# Change some 'sandstone' to 'SANDSTONE'.
target = (df.Description.str.contains('NAVAJO')) & (df.Lithology=='sandstone')
df.loc[target, 'Lithology'] = 'SANDST.'

# Add redundant information.
df['Source'] = 'RPC'
df['Index'] = np.arange(len(df)) + 63

# Add apparently redundant information, but actually it's going to be useful!
df['VpVs ratio'] = df.Vp / df.Vs

# Add dates to regularize.
# Easy to undo: pd.to_datetime([d for d in dates])
def random_dates(start, end, n=1):
    """Make random dates."""
    start_unix = start.value // 10**9
    end_unix = end.value // 10**9
    return pd.to_datetime(rng.integers(start_unix, end_unix, n), unit='s')

s = random_dates(pd.to_datetime('2001-01-01'), pd.to_datetime('2015-12-31'), n=len(df))
dates = [x.strftime('%Y-%m-%d') if rng.random() < 0.8 else x.strftime('%a %d %b %Y') for x in s]
df['YYYY-MM-DD'] = dates

# Mess with units: all of Vp.
df['DTP [μs/ft]'] = 1e6 * 0.3048 / df.Vp

# Remove some values of DTP, which you'll have to compute "somehow" :P
# Hint: we used the Vp to make the Vp:Vs ratio.
df.loc[df.Description.str.contains('FERRON'), 'DTP'] = np.nan

# Remove some more random values.
rows = rng.integers(0, 800, 40)
df.iloc[rows, df.columns.get_loc('DTP [μs/ft]')] = -999.25

# Mess with units: all of Vs.
df['Vs [km/s]'] = df.Vs / 1000

# Mess with the units of some RHOB rows.
df['Rho [kg/m³]'] = df.Rho_n
df.loc[df.Lithology=='dolomite', 'Rho [kg/m³]'] /= 1000

# Add some duplicate rows.
no_rows = rng.integers(5, 10)
df = pd.concat([df, df.iloc[rng.integers(0, 800, size=no_rows)]]).sort_index()

# Save a subset of the columns we have now.
features = ['Index', 'RPC', 'Source', 'YYYY-MM-DD', 'Description', 'Lithology', 'DTP [μs/ft]', 'Vs [km/s]', 'VpVs ratio', 'Rho [kg/m³]']
df[features].to_csv('../data/RPC_4_lithologies_Messy.csv', index=False, float_format='%.4f')
