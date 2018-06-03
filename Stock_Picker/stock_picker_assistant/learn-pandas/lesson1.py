import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib




names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]

BabyDataSet = list(zip(names,births))

df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])

MaxName = df['Names'][df['Births'] == df['Births'].max()].values

print(MaxName)
