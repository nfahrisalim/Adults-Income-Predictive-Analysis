import pandas as pd

# Baca file adult.data (misal dipisahkan dengan koma)
df = pd.read_csv('adult.data', header=None)

# Tambahkan header sesuai dataset (contoh, sesuaikan dengan kolom yang ada)
df.columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

# Simpan sebagai CSV
df.to_csv('adult.csv', index=False)