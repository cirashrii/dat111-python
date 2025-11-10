# importing modules and packages
import pandas as pd
import random

df = pd.read_csv('temperatur.csv')
df["Month"] = 12
month = list(range(1,12))

monthfactor1 = [200,150,120,120,110,110,80,70,80,110,100,100,0]
monthfactor2 = [120,130,150,200,200,120,100,80,100,100,100,100,0]

df2 = pd.DataFrame(columns=['X','Y','Temperatur','Month'])
rows = len(df)
for irow in range(0, rows):
        xpos = df.iloc[irow, 0]
        if float(xpos) == 0:
            continue
        ypos = df.iloc[irow, 1]
        for month in range(1, 13):
            factor =  monthfactor2[month] if xpos > 9 else monthfactor1[month]
            nedbor = int(df.iloc[irow, 2] * factor / random.randrange(1220, 2000))
            nedbor = min(499,nedbor)
            df2.loc[len(df2)] = [xpos,ypos,nedbor,month]

df2.to_csv('TemperaturX.csv',index=False)
print(df2)