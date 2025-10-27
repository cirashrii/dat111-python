import pandas as pd
from sklearn.linear_model import LinearRegression
solgtLeil = [
    [80,    3.4,    8.2], # m2, Standard, Pris
    [108,   3.4,    9.4],
    [77,    1.5,    5.3],
    [135,   2.5,    9.3],
    [45,    3.1,    5.2]]
df = pd.DataFrame(columns=["m2","Std", "Price"], data=solgtLeil)
#df = pd.read_csv('house_data.csv') # Lese huspriser fra fil/dokument

X = df[['m2','Std']]
#X = df[['m2']]
y = df['Price']
model = LinearRegression()  #
model.fit(X, y)             # Tren modellen basert på solgte leiligheter
estLeil = [[100,2],[120,4], [220,1], [200,4], [50,4]]

dfInput = pd.DataFrame(columns=["m2","Std"], data=estLeil)
#dfInput = dfInput.drop("Std", axis=1)
y_pred = model.predict(dfInput) # Estimerer verdien
print("Leiligheter i Bergen sentrum Standard(4=best) ")
for i in range(y_pred.size):
    print(f"m2 {estLeil[i][0]:>3} Standard {estLeil[i][1]} Estimert pris {y_pred[i]:>5.1f}")


