'''
Sistema de prediccion del indice de calidad del aire de la ciudad de bogotá D.C
La prediccion se hara dependiendo de cada particula registrada
'''
import pandas as pd
import numpy as np
import datetime as dt
import os

def tareas_iniciales():
    # Importamos los datos de nuestra carpeta dataset
    air_data = pd.read_csv("dataset/minambiente-bogota-airQuality.csv")
    # Reemplazamos los valores vacios con valores nulos
    air_data.replace(r'^\s*$', np.NaN, regex=True, inplace= True)

    # Renombramos las columnas
    columns_air = ["FECHA","PM25","PM10","O3","NO2","SO2","CO"]
    air_data.columns = columns_air

    return air_data

def informacion():
    # Accediendo a la data
    data = tareas_iniciales()
    # INFORMACION DE LOS DATOS | Descripcion general
    print("\nDESCRIPCION GENERAL DE LOS DATOS\n\n",data.describe(include="all"))
    print("\nTIPO DE DATOS\n\n", data.dtypes)
    # Tamaño de los datos
    print("\nTAMAÑO DE LOS DATOS\n\n",data.shape)
    # Verificar los valores nulos
    # Como se puede ver, la columna SO2 esta casi vacia
    print("\nVALORES NULOS POR COLUMNA\n\n", data.isnull().sum())

    return data

def limpiando_datos():
    # Accediendo a la data 
    data = informacion()
    # LIMPIEZA DE DATOS
    # La columma de SO2 tambien se debe eliminar, debido que hay pocos registros 
    data.drop(['SO2'], axis = 1, inplace = True)
    # Eliminamos los valores con registros en 0
    # En este caso gracias a la informacion shape, sabemos que hay de 2200 registros
    # La columna con menos registros es CO seguido de NO2, el cual son 650
    # Los cuales son muchos para generarlos usando metodos estadisticos
    # Por ello, directamente se eliminaran todos los datos nulos
    data.dropna(subset=["CO","O3","NO2","PM25","PM10"],axis=0, inplace = True)
    # Tipo de datos 
    # Como se pudo identificar al ejecutar la funcion dtype sobre el dataframe
    # Todas las columnas y valores son de tipo object, es decir son strings aunque
    # Realmente muchos de estos valores son de tipo numericos y/o fechas
    # Por ello es necesario cambio el tipo de dato de cada columna
    # data['FECHA'] = pd.to_datetime(data['FECHA'])
    tmp = data['FECHA'].str.split('/')
    data['FECHA'] = (tmp.str[0]+tmp.str[1]+tmp.str[2]).astype(int)

    for i in ['PM25','PM10','O3','NO2','CO']:
        data[i] = data[i].astype('int')

    print("\nTIPO DE DATOS\n\n", data.dtypes)
    return data

# Una vez que tenemos todos los datos limpios
# Guardamos el dataframe en otro CSV para realizar el entrenamiento

def exportando():
    data = limpiando_datos()
    # Comprobamos si la direccion para guardan el dataset limpio existe
    if os.path.exists('train'):
        
        data.to_csv('train/minambiente_train_data.csv')
            
    else:
        os.mkdir('train')
        data.to_csv('train/minambiente_train_data.csv')

def main():
    exportando()

if __name__ == "__main__":
    main()