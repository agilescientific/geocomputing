import pandas as pd
import numpy as np

rng = np.random.default_rng(87)

# THIS PREPARES THE DATASET FROM SEG's GITHUB.
# Most of the data (Hall's training data).
dg = pd.read_csv('https://raw.githubusercontent.com/seg/2016-ml-contest/master/facies_vectors.csv')
dg = dg.rename(columns={'NM_M': 'Marine', 'RELPOS': 'RelPos'})
dg['ILD'] = 10**dg['ILD_log10']
dg['Depth'] = dg['Depth'] * 0.3048
cols = ['Well Name', 'Depth', 'Formation', 'RelPos', 'Marine', 'GR', 'ILD',
       'DeltaPHI', 'PHIND', 'PE', 'Facies']
dg = dg[cols]

# ALEXANDER D and KIMZEY A have no PE; keeping them.
# Drop Recruit F9 well, which is synthetic.
dg = dg.loc[dg['Well Name'] != 'Recruit F9']

# STUART and CRAWFORD log data (test data).
di = pd.read_csv('https://raw.githubusercontent.com/seg/2016-ml-contest/master/nofacies_data.csv')
di = di.rename(columns={'NM_M': 'Marine', 'RELPOS': 'RelPos'})
di['ILD'] = 10**di['ILD_log10']
di = di[cols[:-1]]

# Labels are in another file.
dh = pd.read_csv('https://raw.githubusercontent.com/seg/2016-ml-contest/master/blind_stuart_crawford_core_facies.csv')
dh['Facies'] = dh['LithCode']

# Merge them.
df = pd.merge(
    left=di,
    right=dh,
    left_on=['Well Name', 'Depth'],
    right_on=['WellName', 'Depth.ft'],
    how='left',
).drop(columns=['WellName', 'Depth.ft', 'LithLabel', 'LithCode'])

# Regularize the facies a bit, and adjust the marine indicator.
df.loc[df.Facies == 11, 'Facies'] = 1
df.Marine -= 1.0  # Cast to float.

# Concatenate and save.
de = pd.concat([dg, df]).reset_index().drop(columns=['index'])

# Now we shall begin the mangling.
# Add an unregularised Completion Date per well to the DataFrame.
def random_dates(start, end, n=1):
    """Make random dates."""
    start_unix = start.value // 10**9
    end_unix = end.value // 10**9
    return pd.to_datetime(rng.integers(start_unix, end_unix, n), unit='s')

s = random_dates(pd.to_datetime('2001-01-01'), pd.to_datetime('2015-12-31'), n=len(de['Well Name'].unique()))
dates = [x.strftime('%Y-%m-%d') if rng.random() < 0.6 else x.strftime('%a %d %b %Y') for x in s]
for date, well in zip(dates, de['Well Name'].unique()):
    de.loc[de['Well Name'] == well, 'Completion Date'] = date

# Mangle the well names.
# Add . to some Cross H Cattle Wells
cond = (de['Well Name'] == 'CROSS H CATTLE') & de['Formation'].str.startswith('A1')
de.loc[cond, 'Well Name'] = 'CROSS H. CATTLE'

# Mess with depths.
# Change some depths to feet.
wells_ft = ['SHANKLE', 'NEWBY']
de.loc[de['Well Name'].isin(wells_ft), 'Depth'] = de[de['Well Name'].isin(wells_ft)]['Depth'] / 0.3048

# Make some depths negative.
de.loc[de['Well Name'] == 'NEWBY', 'Depth'] = de[de['Well Name'] == 'NEWBY']['Depth'] * -1

# Make some porosities fractional.
de.loc[de['Well Name'] == 'STUART', 'PHIND'] = de[de['Well Name'] == 'STUART']['PHIND'] / 100

# Change SH and LM in some logs to Shale and Limestone.
wells_ft = ['STUART', 'CRAWFORD']
for well in wells_ft:
    de.loc[de['Well Name'] == well, 'Formation'] = de.loc[de['Well Name'] == well, 'Formation'].str.replace('SH', 'Shale', regex=False)
    de.loc[de['Well Name'] == well, 'Formation'] = de.loc[de['Well Name'] == well, 'Formation'].str.replace('LM', 'Limestone', regex=False)

# Convert the numbers for the Facies into text descriptions.
# Could be more evil and have a typo in a couple descriptions in some wells?
lithologies = {
    1: 'Non-marine sandstone',
    2: 'Nonmarine coarse siltstone',
    3: 'Nonmarine fine siltstone',
    4: 'Marine siltstone and shale',
    5: 'Mudstone',
    6: 'Wackestone',
    7: 'Dolomite',
    8: 'Packstone-grainstone',
    9: 'Phylloid-algal bafflestone',
}
#de['Facies'] = de['Facies'].replace(lithologies)

# Add a redundant Index column?
de['Index'] = np.arange(len(de)) + 63  # Then have index=False when saving.
de['Source'] = 'KGS'

# Save out to csv.
features = ['Index', 'Well Name', 'Depth', 'Formation', 'RelPos', 'Marine', 'GR', 'ILD', 'DeltaPHI', 'PHIND', 'PE', 'Facies', 'Completion Date', 'Source']
de[features].to_csv('../data/Panoma_Field_Permian_RAW.csv', index=False, float_format='%.4f')

meta_dict = {
    'Index': ['Index for sorting records', ''],
    'Well Name': ['Name of the well that the record is from.', ''],
    'Depth': ['Measured depth below KB', ''],
    'Formation': ['Which formation the record is from. See accompanying paper.', ''],
    'RelPos': ['Position of the record relative to a facies interval. 0 at base, normalised to 1 per facies interval.', ''],
    'Marine': ['Whether a record is of a marine rock (1) or not (0).', ''],
    'GR': ['Gamma Ray - natural radioactivity.', 'Unit: API units'],
    'DeltaPHI': ['Difference between the Neutron Porosity Logs and the Density Porosity Log. Measure of porosity.', 'Unit: %'],
    'ILD': ['Induction Log Deep - Deep formation resistivity.', 'Unit: Ohm.m'],
    'PHIND': ['Nuclear Density for porosity. Average of DPhi and NPhi.', 'Unit: API units'],
    'PE': ['Photoelectric factor. Unit: '],
    'Facies': ['Numerical code for the Facies. Derived from core. See accompanying paper.', ''],
    'Completion Date': ['Date of completion of drilling.', ''],
    'Source': ['Original source of the data.', ''],
}
metadata = pd.DataFrame.from_dict(meta_dict, orient='index', columns=['Description', 'Units'])
metadata = metadata.reset_index().rename(columns={'index': 'Column Name'})

with pd.ExcelWriter('../data/Panoma_Field_Permian-RAW.xlsx') as writer:  
    metadata.to_excel(writer, sheet_name='metadata', index=False)
    de.to_excel(writer, sheet_name='data', index=False)
