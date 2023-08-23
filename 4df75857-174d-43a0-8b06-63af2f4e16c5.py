#!/usr/bin/env python
# coding: utf-8

# # Interconnect: Empresa proveedora de telecomunicaciones

# Al operador de telecomunicaciones Interconnect le gustaría poder pronosticar su tasa de cancelación de clientes. Si se descubre que un usuario o usuaria planea irse, se le ofrecerán códigos promocionales y opciones de planes especiales. El equipo de marketing de Interconnect ha recopilado algunos de los datos personales de sus clientes, incluyendo información sobre sus planes y contratos.

# # Servicios de Interconnect

# 1. Comunicación por teléfono fijo. El teléfono se puede conectar a varias líneas de manera simultánea.
# 2. Internet. La red se puede configurar a través de una línea telefónica (DSL, *línea de abonado digital*) o a través de un cable de fibra óptica.
# 
# Algunos otros servicios que ofrece la empresa incluyen:
# 
# - Seguridad en Internet: software antivirus (*ProtecciónDeDispositivo*) y un bloqueador de sitios web maliciosos (*SeguridadEnLínea*).
# - Una línea de soporte técnico (*SoporteTécnico*).
# - Almacenamiento de archivos en la nube y backup de datos (*BackupOnline*).
# - Streaming de TV (*StreamingTV*) y directorio de películas (*StreamingPelículas*)
# 
# La clientela puede elegir entre un pago mensual o firmar un contrato de 1 o 2 años. Puede utilizar varios métodos de pago y recibir una factura electrónica después de una transacción.

# # Descripción de los datos

# Los datos consisten en archivos obtenidos de diferentes fuentes:
# 
# - `contract.csv` — información del contrato;
# - `personal.csv` — datos personales del cliente;
# - `internet.csv` — información sobre los servicios de Internet;
# - `phone.csv` — información sobre los servicios telefónicos.

# ## ANALISIS EXPLORATORIO
# 
# 1. Importacion de bibliotecas y bd
# 2. Analisis exploratorio EDA
# 

# ### Importacion de bibliotecas 

# In[1]:


#importamos las bibliotecas y las bases de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer


# ### Importacion de bases de datos 

# #### Analisis primario de Contract

# In[2]:


#importamos las bd de contract
contract=pd.read_csv('/datasets/final_provider/contract.csv')


# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Separa el analisis de cada dataset en subsecciones
# </div>

# In[3]:


#imprimimos las primera filas de la base de datos "contract"
contract.head()


# In[4]:


#verificamos los valores ausentes y duplicados en la bd de contratc
print('Valores ausentes en contract:', contract.isnull().sum().sum())
print('Valores duplicados en contract:', contract.duplicated().sum())


# In[5]:


#imprimimos la informacion general de la base de datos "contract"
contract.info()


# OBSERVACIONES:
# - BeginDate y EndDate deben ser de tipo datetime. CustomerID puede permanecer como está. Type, PaperlessBilling y PaymentMethod deben ser de tipo category. El total de cargos puede cambiarse a flotante ademas no tienen valores duplicados

# #### Analisis primario de Personal

# In[6]:


#importamos las bd de personal
personal=pd.read_csv('/datasets/final_provider/personal.csv')


# In[7]:


#imprimimos las primera filas de la base de datos "personal"
personal.head()


# In[8]:


#verificamos los valores ausentes y duplicados en la bd de personal
print('Valores ausentes en personal:', personal.isna().sum().sum())
print('Valores duplicados en personal:', personal.duplicated().sum())


# In[9]:


#imprimimos la informacion general de la base de datos "personal"
personal.info()


# OBSERVACIONES:
# - Aparte de customerID, todas las columnas deben cambiarse al tipo bolean ademas no existen datos duplicados

# #### Analisis primario de Internet

# In[10]:


#importamos las bd de internet
internet=pd.read_csv('/datasets/final_provider/internet.csv')


# In[11]:


#imprimimos las primera filas de la base de datos "internet"
internet.head()


# In[12]:


#verificamos los valores ausentes y duplicados en la bd de internet
print('Valores ausentes en internet:', internet.isna().sum().sum())
print('Valores duplicados en internet:', internet.duplicated().sum())


# In[13]:


#imprimimos la informacion general de la base de datos "internet"
internet.info()


# OBSERVACIONES:
# - Aparte de customerID, todas las demás columnas deben convertirse en categorías. También vemos que en ésta hay menos filas, lo que significa que no todos los clientes están abonados al servicio de Internet ademas, este db no tiene valores duplicados

# #### Analisis primario de Phone

# In[14]:


#importamos las bd de phone
phone=pd.read_csv('/datasets/final_provider/phone.csv')


# In[15]:


#imprimimos las primera filas de la base de datos "phone"
phone.head()


# In[16]:


#verificamos los valores ausentes y duplicados en la bd de phone
print('Valores ausentes en phone:', phone.isna().sum().sum())
print('Valores duplicados en phone:', phone.duplicated().sum())


# In[17]:


#imprimimos la informacion general de la base de datos "phone"
phone.info()


# OBSERVACIONES:
# - Menos filas que las primeras 2 tablas, la columna MultipleLines debe hacerse categórica

# Hemos visto las diferentes tablas,estas tablas deben de ser depuradas y trabajadas para ser fusionadas en algun momento. Esto último provocará inevitablemente que falten valores, ya que las tablas no tienen el mismo número de filas, por lo que estos datos deben de ser tratados luego de la union para poder identificarlos. Cabe recalcar que no existen valores duplicados en el db

# ## Procesamiento de la Data

# Empezaremos con la información del contrato y haremos que TotalCharges sea de tipo float. Cambiaremos las columnas necesarias a tipo categorical. Al tratar con EndDate, observamos algunas filas donde toma el valor 'No' (el cliente no se fue). Utilizaremos esto para crear una columna 'churn' que tomará el valor 0 cuando el cliente no se haya ido, y 1 cuando lo haya hecho. Esta nueva columna se convierte en nuestro objetivo

# In[18]:


#Contract
contract['Type']=contract['Type'].astype('category')
contract['PaperlessBilling']=contract['PaperlessBilling'].astype('category')
contract['PaymentMethod']=contract['PaymentMethod'].astype('category')
contract['TotalCharges']=pd.to_numeric(contract['TotalCharges'], errors='coerce')
contract['churn'] = (contract['EndDate'] != "No").astype("int")


# También podemos crear una columna "days" obteniendo la diferencia entre EndDate y BeginDate. En este punto siguen siendo de tipo objeto. Sustituiremos cada 'No' en EndDate por el día en que se tomaron los datos el 1 de febrero de 2020. Esta fecha se introducirá siguiendo el formato presente en la columna (2020-02-01 00:00:00) para facilitar el cambio de la columna a tipo datetime. Entonces podemos obtener la diferencia y almacenarla en la columna 'days'

# In[19]:


contract['EndDate']=contract['EndDate'].replace('No','2020-02-01 00:00:00')
contract['BeginDate']=pd.to_datetime(contract['BeginDate'], format='%Y-%m-%d')
contract['EndDate']=pd.to_datetime(contract['EndDate'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
contract.info()


# In[20]:


contract['days']=(contract['EndDate'] - contract['BeginDate']).dt.days
contract['days'].unique()


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Bien añadida la cantidad de dias de permanencia en la empresa
# </div>

# Ahora eliminamos las columnas "Date" porque ya no tienen ninguna utilidad.

# In[21]:


contract_final=contract.drop(['BeginDate', 'EndDate'], axis=1)


# También podemos eliminar las filas con valores que faltan al cambiar TotalCharges a tipo float

# In[22]:


#contract_final.dropna(inplace=True)


# Imprimimos la informacion del dataset para verificar si se realizaron los cambios anteriormente ejecutados

# In[23]:


contract_final.info()


# Ahora trabajaremos el dataset de Personal

# In[24]:


#Personal
personal['gender']=personal['gender'].astype('category')
personal['Partner'] = personal['Partner'].astype("category")
personal['Dependents'] = personal['Dependents'].astype("category")
personal['SeniorCitizen'] = personal['SeniorCitizen'].astype("category")


# In[25]:


personal.info()


# No vamos a hacer las columnas de las otras tablas categóricas todavía, ya que harán valores NaN que tendremos que reemplazar con un valor, por lo que sólo seguiremos adelante con la fusión. Fusionaremos todas las tablas en la columna customerID. En primer lugar, contract_final a las tablas de personal

# In[26]:


#Merging
merge_1=pd.merge(personal, contract_final, how="left", on="customerID")
display(merge_1)


# Verificamos la existencia de valores ausentes en el df

# In[27]:


merge_1.isnull().sum()


# In[28]:


merge_1[merge_1['TotalCharges'].isnull()]


# Vamos a rellenar los datos ausentes en TotalCharges con 0 ya que es el dia donde se toman los datos, por lo que esos clientes no se han ido aun

# In[29]:


merge_1["TotalCharges"]=merge_1["TotalCharges"].fillna(0)


# In[30]:


#revisamos que se hayan realizado los cambios 
merge_1.isnull().sum()


# Segunda fusión, la tabla merge_1 con las informacion de phone

# In[31]:


merge_2=pd.merge(merge_1, phone, how="left", on="customerID")
merge_2.head()


# Ahora realizaremos la ultima unificacion que es la tabla de merge_2 con la tabla de internet

# In[32]:


data_final=pd.merge(merge_2, internet, how="left", on="customerID")
data_final.head()


# In[33]:


#imrpimimos la informacion de la data final unificada
data_final.info()


# In[34]:


#cambiamos la columna churn a tipo entero
data_final['churn']=data_final['churn'].astype('int') 


# Ahora podemos sustituir los valores que faltan. Como los usuarios que tienen valores omitidos en sus filas no están suscritos al servicio en esas columnas, podemos sustituir los valores omitidos por "No"

# In[35]:


#sustituimos de los valores omitidos por "No"
str_cols = data_final.columns[data_final.dtypes=='object']
data_final[str_cols] = data_final[str_cols].fillna('No')


# In[36]:


#imrpimimos la informacion de la data final unificada
data_final.info()


# Ya podemos hacer nuestras conversiones de tipo categórico

# In[37]:


#converting to categorical columns
data_final['OnlineSecurity'] = (data_final['OnlineSecurity'] == "Yes").astype("category")
data_final['OnlineBackup'] = (data_final['OnlineBackup'] == "Yes").astype("category")
data_final['TechSupport'] = (data_final['TechSupport'] == "Yes").astype("category")
data_final['DeviceProtection'] = (data_final['DeviceProtection'] == "Yes").astype("category")
data_final['StreamingTV'] = (data_final['StreamingTV'] == "Yes").astype("category")
data_final['StreamingMovies'] = (data_final['StreamingMovies'] == "Yes").astype("category")
data_final['InternetService'] = data_final['InternetService'].astype('category')
data_final['MultipleLines'] = (data_final['MultipleLines'] == "Yes").astype("category")


# In[38]:


#imrpimimos la informacion de la data final unificada
data_final.info()


# Ya hemos procesado nuestros datos. Ahora podemos pasar al Análisis Exploratorio de Datos

# ## Análisis exploratorio de datos

# Exploraremos el desequilibrio de clases en nuestro conjunto de datos. Los que se fueron frente a los que se quedaron

# In[39]:


#personas que se quedaron vs las que se fueron
data_final.groupby('churn')[['customerID']].count()


# OBSERVACION: Más de 5.000 personas se quedaron y algo menos de 1.800se marcharon

# **Features vs Churn**

# Vamos a trazar gráficos para ver cómo se distribuye el churn entre las demás características

# In[40]:


#realizamos gráficos con características frente al churn. Boxplots para características discretas, gráficos de barras para valores categóricos.
fig, axs = plt.subplots(4, len(data_final.columns) // 4, figsize=(20,20))
axs = axs.flatten()
cols=list(set(data_final.columns) - set(['customerID']))
for col, ax in zip(cols, axs):
    if data_final[col].dtype=='float64':
        data_final.boxplot(column=col, by='churn', ax=ax)
        plt.suptitle('')
    else:
        df = data_final.groupby([col, 'churn'])['churn'].count().unstack()
        df.plot(kind='bar', stacked=False, label='#churn (neg, pos)', ax=ax)
        plt.legend(loc='upper left')
    
plt.tight_layout()
plt.show()


# OBSERVACIONES:
# 
# Lo que podemos deducir es que la mayoría de las personas que se fueron:
# 
# - Tenían facturación sin papel, especialmente a través de cheque electrónico
# - No eran personas mayores
# - No tenían uno o varios de los servicios del abono por internet
# - Pagaban mes a mes
# - Sus cuotas mensuales eran normalmente más altas (unos 80 USD de media), eran usuarios más recientes según el diagrama de caja de "días" (con algunos valores atípicos) y pagaban menos en total, ya que no se quedaban mucho tiempo (aunque observamos muchos valores atípicos).

# **Matriz de correlaciones**

# In[41]:


data_final.corr()


# OBSERVACIONES:
# - Ninguna de las características numéricas tiene una fuerte correlación con el churn. Existe una correlación notablemente fuerte entre el número de días y las cuotas totales pagadas.
# 
# - A continuación, analizaremos las cuotas mensuales de los abonados a los servicios de teléfono e Internet y, para cada servicio, compararemos su distribución entre los que se dieron de baja y los que no se dieron de baja.

# **Cargos mensuales de los abonados al servicio telefónico**

# In[42]:


phone_users=data_final[data_final['customerID'].isin(phone['customerID'])]
phone_users.info()


# Distribución de las tarifas mensuales entre los usuarios de teléfono

# In[43]:


phone_users['MonthlyCharges'].hist(bins=20)
plt.xlabel('Tarifas mensuales')
plt.ylabel('Frecuencia')
plt.title('Cargos mensuales entre clientes telefónicos')
plt.show()


# In[44]:


phone_users.boxplot(column='MonthlyCharges')
plt.title('Cargos mensuales entre clientes telefónicos')
plt.show()


# In[45]:


phone_users['MonthlyCharges'].describe()


# Observaciones:
# - Un usuario típico de teléfono suele cobrar entre unos 45 (percentil 25) y unos 91 (percentil 75). Sin embargo, según el histograma, hay un buen número de personas (unas 1.500) que pagan entre 20 y 30 mensuales. Comprobemos ahora la distribución entre los que se fueron y los que se quedaron

# In[46]:


phone_users[phone_users['churn']==1]['MonthlyCharges'].hist(bins=20, alpha=0.5, label='churn=1')
phone_users[phone_users['churn']==0]['MonthlyCharges'].hist(bins=20, alpha=0.5, label='churn=0')
plt.legend(loc='upper right')
plt.xlabel('Tarifas mensuales')
plt.title('Histograma de gastos mensuales por rotación (usuarios de teléfono)')
plt.show()


# Observaciones:
# - Las distribuciones son bastante similares, aunque la de los que se marcharon es mucho menor

# In[47]:


phone_users.boxplot(column='MonthlyCharges', by='churn')
plt.suptitle('')
plt.show()


# Observaciones:
# - Los que se marcharon pagaron cuotas mensuales más altas que los que se quedaron

# In[48]:


print('Distribución mensual de los gastos para los clientes que se fueron',
      phone_users[phone_users['churn']==1]['MonthlyCharges'].describe())


# In[49]:


print('Distribución mensual de los gastos para los clientes que se quedaron',
      phone_users[phone_users['churn']==0]['MonthlyCharges'].describe())


# Observaciones:
# - Los que se marcharon solían pagar entre 70 (percentil 25) y 95 (percentil 75). Los que se quedaron pagaban entre 24 (percentil 25) y 90 (percentil 75)

# - **Cargos mensuales de los abonados al servicio de Internet**

# Vamos a repetir los pasos anteriores para los abonados a Internet

# In[50]:


internet_users=data_final[data_final['customerID'].isin(internet['customerID'])]
internet_users.info()


# In[51]:


internet_users['MonthlyCharges'].hist(bins=20)
plt.xlabel('Tarifas mensuales')
plt.ylabel('Frecuencia')
plt.title('Cargos mensuales entre clientes de internet')
plt.show()


# Observaciones:
# - Se observan dos picos: uno en torno a 50 y otro en torno a 80. Está sesgado a la derecha

# In[52]:


internet_users.boxplot(column='MonthlyCharges')
plt.show()


# In[53]:


internet_users['MonthlyCharges'].describe()


# Observaciones:
# - Los usuarios de Internet suelen cobrar entre unos 60 (percentil 25) y unos 95 (percentil 75) mensuales. Desglosemos ahora la distribución por churn

# In[54]:


internet_users[internet_users['churn']==1]['MonthlyCharges'].hist(bins=20, alpha=0.5, label='churn=1')
internet_users[internet_users['churn']==0]['MonthlyCharges'].hist(bins=20, alpha=0.5, label='churn=0')
plt.legend(loc='upper right')
plt.xlabel('Monthly Charges')
plt.title('Histograma de gastos mensuales por rotación (usuarios de Internet)')
plt.show()


# Observaciones:
# - De nuevo, observamos distribuciones similares, aunque las frecuencias varían (la mayoría de las personas se quedaron)

# In[55]:


internet_users.boxplot(column='MonthlyCharges', by='churn')
plt.suptitle('')
plt.show()


# In[56]:


print('Distribución mensual de los gastos para los clientes que se fueron',
      internet_users[internet_users['churn']==1]['MonthlyCharges'].describe())


# In[57]:


print('Distribución mensual de los gastos para los clientes que se quedaron',
      internet_users[internet_users['churn']==0]['MonthlyCharges'].describe())


# Observaciones:
# - Los que se marcharon solían pagar entre 69 (percentil 25) y 95 (percentil 75). Los que se quedaron pagaban entre 59 (percentil 25) y 94 (percentil 75).
# 
# - En conclusión, tanto para los usuarios de Internet como para los de teléfono, los que se fueron pagaron más al mes.

# ## Entrenamiento del modelo 

# Terminemos de procesar nuestro conjunto de datos
# 
# Podemos eliminar la columna customerID, ya que no será de utilidad.

# In[58]:


data_final=data_final.drop('customerID', axis=1)


# ### Division de conjuntos en entrenamiento y prueba

# Ahora podemos dividir nuestros datos en conjuntos de entrenamiento y prueba. Utilizaremos 3 tipos de datos. Uno con Ordinal Encoding and Scaling, el segundo con One-Hot Encoding and scaling, el tercero escalado sin codificación. Cada uno de estos será upsampled debido al desequilibrio de clase que existe. Nuestra función de upsampling es la siguiente:

# - **Funcion de Upsampling**

# In[59]:


#creamos una función llamada upsample que toma como argumentos las características, el objetivo y el número de repetición
def upsample(features, target, repeat):
    
    #obtenemos las características de la clase negativa
    features_zeros = features[target == 0]
    
    #obtenemos las características de la clase positiva
    features_ones = features[target == 1]
    
    #obtenemos la clase negativa del objetivo
    target_zeros = target[target == 0]
    
    #obtenemos la clase positiva del objetivo
    target_ones = target[target == 1]
    
    #realizamos el upsmaple de  las características combinando las características de la clase negativa y las características de la clase positiva repetida
    features_ups = pd.concat([features_zeros] + [features_ones] * repeat)
    
    #realizamos el upsmaple del objetivo combinando el objetivo de clase negativo y el objetivo de clase positivo repetido
    target_ups = pd.concat([target_zeros] + [target_ones] * repeat)
    
    #mezclamos las características y los objetivos sobremuestreados resultantes
    features_ups, target_ups = shuffle(features_ups, target_ups, random_state=12345)
    
    #retornamos las características y el objetivo sobremuestreados resultantes
    return features_ups, target_ups 


# - **Ordinal Encoding and Scaling**

# In[60]:


data_mod=data_final.copy()
cat_feat = data_mod.columns[data_mod.dtypes=='category']

#creamos una instancia del codificador ordinal
encoder=OrdinalEncoder()

#codificamos las columnas
data_mod[cat_feat]=encoder.fit_transform(data_mod[cat_feat])

#revisamos la informacion de la data modificada
data_mod.info()


# In[61]:


#las características serán todas las columnas excepto la columna churn
features=data_mod.drop('churn', axis=1)

#El objetivo será la columna churn
target=data_mod['churn']

#realizamos la separacion de los conjuntos
features_train, features_test, target_train, target_test=train_test_split(features, target,                                                                                      test_size=0.25,                                                                                     random_state=12345)
print(features_train.shape)
print(target_train.shape)
print(features_test.shape)
print(target_test.shape)


# In[62]:


#verificamos el desequilibrio de clases
target_train.value_counts(normalize=True) 


# In[63]:


#realizamos el upsample a  el conjunto de características de entrenamiento y el objetivo introduciéndolos en la función upsample 
#con una repetición de 3
feat_ups, targ_ups = upsample(features_train, target_train, 3)

#imprimimos las dimensiones de los conjuntos sobremuestreados
print(feat_ups.shape, targ_ups.shape)


# In[64]:


#Escalado de columnas numéricas

#creamos una lista con los nombres numéricos de las columnas
numeric = ['MonthlyCharges', 'TotalCharges', 'days']

#llamamos a la función de escala
scaler = StandardScaler()

#entrenamos el escalador con los datos de las columnas numéricas
scaler.fit(feat_ups[numeric])

#transformamos los datos en valores escalados
feat_ups[numeric] = scaler.transform(feat_ups[numeric])

#remplazamos los datos de las columnas con los datos escalados
features_test[numeric] = scaler.transform(features_test[numeric])

#imprimimos las primeras 5 filas del df
feat_ups.head()


# - **One-Hot Encoding and scaling**

# Realizamos los pasos similares al anterios metodo, solo que aqui la data se maneja obteniendo los dummies del conjuto de datos

# In[65]:


#obtenemos los dummies de los datos
data_ohe=pd.get_dummies(data_final, drop_first=True)
data_ohe.head()


# In[66]:


#las características serán todas las columnas excepto la columna churn
feat=data_ohe.drop('churn', axis=1)

#El objetivo será la columna churn
target=data_ohe['churn']

#realizamos la separacion de los conjuntos
f_train, f_test, t_train, t_test=train_test_split(feat, target, test_size=0.25, random_state=12345)


print(f_train.shape)
print(t_train.shape)
print(f_test.shape)
print(t_test.shape)


# In[67]:


f_ohe, t_ohe = upsample(f_train, t_train, 3)
print(f_ohe.shape, t_ohe.shape)


# In[68]:


#Escalado de columnas numéricas

#creamos una lista con los nombres numéricos de las columnas
numeric = ['MonthlyCharges', 'TotalCharges', 'days']

#llamamos a la función de escala
scaler = StandardScaler()

#entrenamos el escalador con los datos de las columnas numéricas
scaler.fit(f_ohe[numeric])

#transformamos los datos en valores escalados
f_ohe[numeric] = scaler.transform(f_ohe[numeric])

#remplazamos los datos de las columnas con los datos escalados
f_test[numeric] = scaler.transform(f_test[numeric])

#imprimimos las primeras 5 filas del df
f_ohe.head()


# - **Escalado sin codificación (CatBoost y LGBM)**

# In[69]:


#creaamos una copia de nuestro db
no_enc=data_final.copy()

#las características serán todas las columnas excepto la columna churn
feat=no_enc.drop('churn', axis=1)

#El objetivo será la columna churn
target=no_enc['churn']

#realizamos la separacion de los conjuntos
ft_train, ft_test, tr_train, tr_test=train_test_split(feat, target,test_size=0.25,random_state=12345)
                                                                                     
                                                                                      
print(ft_train.shape)
print(tr_train.shape)
print(ft_test.shape)
print(tr_test.shape)


# In[70]:


f_ne, t_ne = upsample(ft_train, tr_train, 3)
print(f_ne.shape, t_ne.shape)


# In[71]:


#Escalado de columnas numéricas

#creamos una lista con los nombres numéricos de las columnas
numeric = ['MonthlyCharges', 'TotalCharges', 'days']

#llamamos a la función de escala
scaler = StandardScaler()

#entrenamos el escalador con los datos de las columnas numéricas
scaler.fit(f_ne[numeric])

#transformamos los datos en valores escalados
f_ne[numeric] = scaler.transform(f_ne[numeric])

#remplazamos los datos de las columnas con los datos escalados
ft_test[numeric] = scaler.transform(ft_test[numeric])

#imprimimos las primeras 5 filas del df
f_ne.head()


# ### Evaluacion de modelos 

# #### Regresion Lineal

# In[72]:


#Validación cruzada de regresión logística en datos con Ordinal Encoding and scaling
lr = LogisticRegression(solver='liblinear')
lr_score=cross_val_score(lr, feat_ups, targ_ups, scoring='roc_auc', cv=5)
print(lr_score.mean())


# In[73]:


#Entrenamiento de la regresión logística en datos con One-Hot Encoding and scaling
lr = LogisticRegression(solver='liblinear')
lr_score=cross_val_score(lr, f_ohe, t_ohe, scoring='roc_auc', cv=5)
print(lr_score.mean())


# #### Bosques aleatorios

# - **Validación cruzada de bosques aleatorios  utilizando datos con Ordinal Encoding and scaling**

# In[74]:


for depth in range(21, 26):
    rf=RandomForestClassifier(n_estimators=40, max_depth=depth, random_state=12345)
    rf_score=cross_val_score(rf, feat_ups, targ_ups, scoring='roc_auc', cv=5)
    print('Max_depth', depth, 'score:', rf_score.mean())


# - **Validación cruzada de bosques aleatorios  utilizando datos con One-Hot Encoding and scaling**

# In[75]:


for depth in range(21, 26):
    rf=RandomForestClassifier(n_estimators=40, max_depth=depth, random_state=12345)
    rf_score=cross_val_score(rf, f_ohe, t_ohe, scoring='roc_auc', cv=5)
    print('Max_depth', depth, 'score:', rf_score.mean())


# #### CatBoost

# - **Validación cruzada de CatBoost utilizando datos con Ordinal Encoding and scaling**

# In[76]:


cb=CatBoostClassifier(loss_function='Logloss',
                      learning_rate= 0.1,
                      random_seed=12345)
cb_score=cross_val_score(cb, feat_ups, targ_ups, scoring='roc_auc', cv=5)


# In[77]:


print(cb_score.mean())


# - **Validación cruzada de CatBoost utilizando datos escalados sin codificación**

# In[78]:


cat_feat=['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
'TechSupport', 'StreamingTV', 'StreamingMovies', 'gender', 'SeniorCitizen', 'Partner',
'Dependents', 'Type', 'PaperlessBilling', 'PaymentMethod']
cb=CatBoostClassifier(loss_function='Logloss',
                      learning_rate= 0.1,
                      random_seed=12345)
cb.fit(f_ne, t_ne, cat_features=cat_feat, verbose=False, plot=False)
cb_score=cross_val_score(cb, feat_ups, targ_ups, scoring='roc_auc', cv=5)


# In[79]:


print(cb_score.mean())


# #### LightGBM

# - **Validación cruzada de LGBM  utilizando datos escalados sin codificación**

# In[80]:


lgbm=LGBMClassifier(objective='binary',
                    learning_rate= 0.1,
                    random_state=12345)
lgbm_score=cross_val_score(lgbm, feat_ups, targ_ups, scoring='roc_auc', cv=5)
print(lgbm_score.mean())


# - **Validación cruzada de LGBM  utilizando datos con Ordinal Encoding and scaling**

# In[81]:


lgbm=LGBMClassifier(objective='binary',
                    learning_rate= 0.1,
                    random_state=12345)
lgbm_score=cross_val_score(lgbm, f_ne, t_ne, scoring='roc_auc', cv=5)
print(lgbm_score.mean())


# ## Prueba final

# El mejor de todos nuestros modelos según la validación cruzada fue el modelo CatBoost con una puntuación del 98,1%. Ahora vamos a probar el modelo utilizando el conjunto de prueba. Utilizaremos el modelo CatBoost que utilizó el codificador incorporado

# In[83]:


cat_feat=['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
'TechSupport', 'StreamingTV', 'StreamingMovies', 'gender', 'SeniorCitizen', 'Partner',
'Dependents', 'Type', 'PaperlessBilling', 'PaymentMethod']
cb=CatBoostClassifier(loss_function='Logloss',
                      learning_rate= 0.1,
                      random_seed=12345)
cb.fit(f_ne, t_ne, cat_features=cat_feat, verbose=False, plot=False)
pred=cb.predict(ft_test)
acc=accuracy_score(tr_test, pred)
probab=cb.predict_proba(ft_test)
auc=roc_auc_score(tr_test, probab[:,1])
print('AUC-ROC =', auc)
print('Accuracy =', acc)


# ## Conclusion

# - Analizamos los datos y descubrimos que los clientes que se marcharon solían pagar cuotas mensuales más altas
# - La mayoría de los clientes que se marcharon pagaban electrónicamente, por lo que es posible que queramos revisar ese sistema. - Entrenamos los modelos Logistic Regression, Random Forest, CatBoost y LightGBM
# - Tras la validación cruzada, la mejor puntuación (0,98) en el conjunto de entrenamiento correspondió al modelo CatBoost. Lo probamos en el conjunto de pruebas y obtuvimos una puntuación AUC-ROC de ~0,91 y una precisión de ~86%.

# # Plan de trabajo

# Pasos a seguir:
# 
# 1. Examinar y estudiar los datos de cada tabla:
# 
# - Realizar un análisis exploratorio inicial de los datos.
# - Resolver cualquier problema identificado en los datos.
# - Visualizar los datos para comprenderlos mejor.
# 
# 2. Preparar los datos para el entrenamiento del modelo:
# 
# - Combinar los datos y realizar ingeniería de datos si es necesario.
# - Etiquetar los datos considerando el objetivo de predicción.
# - Escalar los datos si es necesario para abordar diferencias en la escala de las características.
# 
# 3. Entrenar y probar el modelo:
# 
# - Dividir los datos en un conjunto de entrenamiento y un conjunto de validación en una proporción de 75:25.
# - Entrenar el modelo utilizando el conjunto de entrenamiento y realizar predicciones para el conjunto de validación.
# - Guardar las predicciones y las respuestas correctas para el conjunto de validación.
# - Mostrar las fechas de cancelación predichas, así como los valores de AUC-ROC y precisión del modelo.
# 
# 4. Analizar los resultados obtenidos para evaluar el rendimiento del modelo.
# 
# Objetivo del plan de proyecto:
# 
# - El objetivo principal que se busca alcanzar con este plan de trabajo es desarrollar un modelo de pronóstico de tasa de cancelación de clientes para el operador de telecomunicaciones Interconnect. El modelo tiene como finalidad identificar aquellos clientes que tienen la intención de darse de baja en el servicio, permitiendo al equipo de marketing intervenir a tiempo para retener a estos clientes mediante ofertas personalizadas, códigos promocionales u opciones de planes especiales.

# Hipotesis:
# - Las personas que se fueron, realmente se fueron por que pagaban mas que las que se quedaron 
# - El metodo de pago de los clientes no se ajustaba a lo que podia el cliente 
# Preguntas:
# - Los clientes se fueron sin recibir una promocion por parte de la empresa (retencion del cliente)?
# - Se les ofrece codigos erroneos a los clientes (mala venta)?

# In[ ]:




