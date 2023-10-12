
import pandas as pd
import statsmodels.api as sm
import numpy as np
#from tensorflow import keras
import os
from sklearn.preprocessing import StandardScaler

from scipy import stats

import pickle



os.environ['KMP_DUPLICATE_LIB_OK']='True'

NN3 = pickle.load(open('D:\Ирина\Бауманка_ДатаСайнс\VKR\model\lr_model.pkl', 'rb'))
#NN3 = scikit-learn.saving.load_model('model\\lr_model')
#NN3 = keras.models.load_model('model\\NN3')


#from google.colab.output import eval_js
#print(eval_js("google.colab.kernel.proxyPort(5000)"))

from flask import Flask, render_template, request
app = Flask(__name__)
#app = Flask(__name__, template_folder=r'D:\Programming\Python_files\Flask_projects\flask 2304\flask 2304\KFH2304\KFH\templates')




# Функция денормализации по yeojohnson
#  inverse of the Yeo-Johnson transformation
def yeojohnson_inverse(X_trans, lambda_):
  '''
  if X >= 0 and lambda_ == 0:
    X = np.exp(X_trans) - 1
  elif X >= 0 and lambda_ != 0:
      X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
  elif X < 0 and lambda_ != 2:
      X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
  elif X < 0 and lambda_ == 2:
      X = 1 - np.exp(-X_trans)
  '''

  if lambda_ == 0:
    X = np.exp(X_trans) - 1
  elif lambda_ != 0:
    X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1  

  return X








def check_params(x1, x2):
    message = f"X1 = {x1}, X2 = {x2}"
    return message

def print_params_for_NN(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    if (x1 == "" or x2 == "" or x3 == "" or x4 == "" or x5 == "" or x6 == "" or x7 == "" or x8 == "" or x9 == "" or x10 == "" or x11 == ""):
        message = "ОШИБКА! Вы не ввели параметры."
    #elif(float(x1) >= 0 and float(x2) >=0 and float(x3) >=0 and float(x4) >=0 and float(x5) >=0 and float(x6) >=0 and float(x7) >=0 and float(x8) >=0 and float(x9) >=0 and float(x10) >=0 and float(x11)>=0):
    elif(float(x1) >= 0):
        message = f"Соотношение матрица-наполнитель: {x1}\nПлотность, кг/м3: {x2}\nмодуль упругости, ГПа: {x3}\
        Количество отвердителя, м.%: {x4}\nСодержание эпоксидных групп,%_2: {x5}\nТемпература вспышки, С_2: {x6}\
        Поверхностная плотность, г/м2: {x7}\nПотребление смолы, г/м2: {x8}\nУгол нашивки, град: {x9}\nШаг нашивки: {x10}\
        Плотность нашивки: {x11}"
    else:
        message = "ОШИБКА! Введенные значения должны быть больше или равны 0."
    return message

def calculate_NN(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    
    x8d = 0
    x9d = 0
    
    X_use = pd.DataFrame([[x1,x2,x3, x4, x5, x6, x7, x8d, x9d, x8, x9, x10, x11]],columns=['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа', 'Количество отвердителя, м.%', 'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2', 'Поверхностная плотность, г/м2', '1', '2', 'Потребление смолы, г/м2', 'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки'])
    #X_use = np.array([[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]])
    
    
    '''

   # dummy_column = np.zeros(shape = (X_use.shape[0], 1))
    cols_indexes = [0, 3, 5] # Индексы фиктивных столбцов (для Standard Scaler)
    X_use_6cols = np.insert(X_use, 0, dummy_column.T, axis = 1)
    X_use_6cols = np.insert(X_use_6cols, 3, dummy_column.T, axis = 1)
    X_use_6cols = np.insert(X_use_6cols, 5, dummy_column.T, axis = 1)
    
    # Нормализация по yeojohnson
    lamda_list = [0.013865117288407803,
                  -1.3939833064592597,
                  -0.0542790101104967,
                  0,
                  -0.030491839284671637,
                  0] # Список значений lambda для каждого столбца (у фиктивных = 0)
    
    X_use_6cols_normalized = X_use_6cols.copy()
    
    #power_tranformer = PowerTransformer(method = 'yeo-johnson', standardize = False)
    
    #power_tranformer
    X_use_6cols_normalized[:, 1] = stats.yeojohnson(X_use_6cols_normalized[:, 1], lmbda = lamda_list[1]) 
    X_use_6cols_normalized[:, 2] = stats.yeojohnson(X_use_6cols_normalized[:, 2], lmbda = lamda_list[2]) 
    X_use_6cols_normalized[:, 4] = stats.yeojohnson(X_use_6cols_normalized[:, 4], lmbda = lamda_list[4]) 
    
    '''
    # Загрузка standardscaler
    #with open('/content/drive/My Drive/Colab Notebooks/scaler.pkl','wb') as f:
        #pickle.dump(standadscaler, f)
    with open('D:\Ирина\Бауманка_ДатаСайнс\VKR\scaler\scaler_robust.sav','rb') as f:
        standardscaler = pickle.load(f)
    
    '''
    # Стандартизация
    X_use_6cols_standard = standadscaler.transform(X_use_6cols_normalized)
    
    # Удаление фиктивных столбцов
    X_use_standard = np.delete(X_use_6cols_standard, 5, axis = 1)
    X_use_standard = np.delete(X_use_standard, 3, axis = 1)
    X_use_standard = np.delete(X_use_standard, 0, axis = 1)
    '''
    
    X_norm = standardscaler.transform(X_use)
    X_norm = np.delete(X_norm, [7, 8], 1)
    
    
    model_loaded = NN3
    y_pred_norm = model_loaded.predict(X_norm)
    print('y_pred_use =', y_pred_norm)
    '''
    
    y_pred_use_standard = model_loaded.predict(X_use_standard) # Прогноз по нормализованным и стандартизированным
    print('y_pred_use_standard =', y_pred_use_standard)
    
    y_pred_use_standard_6cols = np.zeros(shape = (X_use.shape[0], 6))
    y_pred_use_standard_6cols[:, 0] = y_pred_use_standard[:, 0]
    
    # Дестандартизация Y_pred
    y_pred_use_standard_inv = standadscaler.inverse_transform(y_pred_use_standard_6cols)[:, 0]
    print('y_pred_use_standard_inv =', y_pred_use_standard_inv)
    
    
    '''
    
    
    '''
    # Денормализация Y_pred (scikit-learn)
    y_pred_use_normalized_inv_1 = y_pred_use_standard_inv.copy()
    y_pred_use_normalized_inv_1 = power_transformer_Y.inverse_transform(X = y_pred_use_normalized_inv_1.reshape(-1, 1))
    print('y_pred_use_normalized_inv_1 =', y_pred_use_normalized_inv_1)
    '''
   #y_pred_use = standardscaler.inverse_transform(y_pred_norm)
   
    
    '''
    # Денормализация Y_pred (2 способ, собственная функция)
    y_pred_use_normalized_inv_2 = y_pred_use_standard_inv.copy()
    y_pred_use_normalized_inv_2 = yeojohnson_inverse(X_trans = y_pred_use_normalized_inv_2, lambda_ = lamda_list[0])
    print('y_pred_use_normalized_inv =', y_pred_use_normalized_inv_2)
    '''
    
    
    message = ": ".join(["Прогноз:", str(y_pred_norm)])
    
    
    
    
    
    # Содержание
    
    '''
    
    standardscaler = StandardScaler()
    transform_X = standardscaler.fit_transform(X)
    
    #res = NN3.predict(transform_X)
    res = 1
    
    
    # create empty table with 3 fields
    Predict_dataset_like = np.zeros(shape=(len(X), 3) )
    # put the predicted values in the right field
    Predict_dataset_like[:,0] = res #res[:,0]
    # inverse transform and then select the right field
    res_inverse = standardscaler.inverse_transform(Predict_dataset_like)[:,0]
    message = ": ".join(["Доход, тыс. руб.", str(res_inverse[0])])
    '''
    
    
    return message

@app.route("/", methods=["post", "get"])
def index():
    message = ''
    message2 = ''
    x1=0
    x2=0
    x3=0
    x3=0
    x4=0
    x5=0
    x6=0
    x7=0
    x8=0
    x9=0
    x10=0
    x11=0

    if request.method == "POST":
        x1 = request.form.get("x1")
        x2 = request.form.get("x2")
        x3 = request.form.get("x3")
        x4 = request.form.get("x4")
        x5 = request.form.get("x5")
        x6 = request.form.get("x6")
        x7 = request.form.get("x7")
        x8 = request.form.get("x8")
        x9 = request.form.get("x9")
        x10 = request.form.get("x10")
        x11 = request.form.get("x11")

        
        message = print_params_for_NN(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11) 
        if "ОШИБКА" not in message:
            message2 = calculate_NN(float(x1), float(x2), float(x3), float(x4), float(x5), float(x6), float(x7), float(x8), float(x9), float(x10), float(x11))
        
    return render_template("index.html", message=message, message2=message2, x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6, x7=x7, x8=x8, x9=x9, x10=x10, x11=x11)

'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    '''
app.run()
