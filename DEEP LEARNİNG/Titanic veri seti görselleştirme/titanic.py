import numpy as np
import seaborn as sns 
import pandas as pd
import timeit 

#Veri setimizi indirelim.
titanic = sns.load_dataset('titanic')

#Veri setimizin özelliklerine bakalım.
titanic.info()

#Örnek bir sorgu yapalım. Yapacağımız sorgu aynı cinsiyet grubuna ait 1. ve 3. sınıf ve yaşamıyor olan
#yolcuların bilet ücretlerini , yolculukta yanlız olup olmadıklarını, hangi şehirden olduklarını gözlemleyelim.

titanic[
    (titanic.sex == 'female')
    & (titanic['class'].isin(['first' , 'third']))
    & (titanic.age > 30)
    & (titanic.survived == 0)
]
