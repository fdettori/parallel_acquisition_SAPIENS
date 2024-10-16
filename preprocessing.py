import pandas as pd

# Carica i due file CSV in DataFrame
hand_gestures = pd.read_csv('data/hand_gestures.csv')
signal_data = pd.read_csv('data/signal_data.csv')

# Convertiamo i timestamp in float per evitare problemi
hand_gestures['timestamp'] = hand_gestures['timestamp'].astype(float)
signal_data['timestamp'] = signal_data['timestamp'].astype(float)

# Merge usando la logica del "nearest neighbor" sulla colonna timestamp
merged = pd.merge_asof(
    signal_data.sort_values('timestamp'),
    hand_gestures.sort_values('timestamp'),
    on='timestamp',
    direction='backward',  # Prende il gesto più vicino ma non futuro
    tolerance=0.5  # Limite di tolleranza di 0.5 secondi
)

# Gestiamo i casi dove non è presente una gesture associata
merged['gesture'].fillna('Unknown', inplace=True)

# Salviamo il risultato nel file preprocessing.csv
merged.to_csv('preprocessing.csv', index=False)

## Bisogna capire in che modo vadano pre-processati i dati, da letteratura
