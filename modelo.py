import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# 1. Gerar Dataset (>100 tuplas conforme regra do PDF)
np.random.seed(42)
n = 600
data = {
    'ano': np.random.randint(2015, 2025, n),
    'km': np.random.randint(5000, 160000, n),
    'motor': np.random.choice([1.0, 1.4, 1.6, 2.0], n)
}
df = pd.DataFrame(data)

# Lógica simulada: Preço base - depreciação
df['preco'] = 90000 + (df['motor']*25000) - ((2025-df['ano'])*4000) - (df['km']*0.10)

# 2. Treinar
X = df[['ano', 'km', 'motor']]
y = df['preco']
clf = LinearRegression()
clf.fit(X, y)

# 3. Salvar com Pickle (Obrigatório pelo PDF)
pickle.dump(clf, open('model.pkl', 'wb'))
print("Modelo treinado e salvo: model.pkl")