import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier


#Carrega o dataset
file = pd.read_csv('./wine_dataset.csv')

file['style'] = file['style'].replace('red', 0)
file['style'] = file['style'].replace('white', 1)

y = file['style']
x = file.drop('style', axis=1)

#Realiza o treinamento do modelo
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.9)
x_treino.shape

model = ExtraTreesClassifier()
model.fit(x_treino, y_treino)

Accu = pd.DataFrame([model.score(x_teste, y_teste)])
result = model.score(x_teste, y_teste)
print(Accu)

prev = model.predict(x_teste[20:23])
print(prev)