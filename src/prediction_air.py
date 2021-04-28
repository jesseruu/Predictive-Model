import preprocesamiento as pre

def comprueba_data():
    if not pre.os.path.exists('train'):
        pre.main()

def data():
    ica_quality = pre.pd.read_csv('train/minambiente_train_data.csv')
    ica_quality.drop(['Unnamed: 0'], axis = 1, inplace = True)

if __name__ == '__main__':
    comprueba_data()
    data()