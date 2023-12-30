# Pacotes a serem utilizados.
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Lendo banco de dados.
dados = pd.read_csv( 'datasets/customer-churn.csv' )

# Shape.
dados.shape

# Head.
dados.head()

# Info.
dados.info()

# Transformando strings de não e sim em números.
traducao_dic = { 'Sim': 1, 'Nao': 0 }

# Modificando manualmente.
dadosmodificados = dados[ [ 'Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn' ] ].replace( traducao_dic )

# Head.
dadosmodificados.head()

# Transformação por get_dummies.
dummie_dados = pd.get_dummies( 
                              
    dados.drop( [ 'Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn' ], 
               
               axis = 1 ), dtype = int )

# Junção dos dados transformados com os que já tinhamos.
dados_final = pd.concat( [ dadosmodificados, dummie_dados ], axis = 1 )

# Como ver as informações completas.
# pd.set_option( 'display.max_columns', 39 )

# Head.
dados_final.head()


    #                                 # 
    # Olhando balanceamento dos dados #
    #                                 #
    
# Olhando balanceamento.
dados_final[ 'Churn' ].value_counts( 1 )

# Separando variáveis explicativas e resposta.
X = dados_final.drop( 'Churn', axis = 1 )
y = dados_final[ 'Churn' ]

# Balanceamento (usando oversampling e o método smote para criar cópias parecidas).
smt = SMOTE( random_state = 123 ) # Instancia um objeto da classe SMOTE.
X, y = smt.fit_resample( X, y ) # Realiza a reamostragem do conjunto de dados.

# Juntando novamente. Concatena a variável target (y) com as features (X).
dados_final = pd.concat( [ X, y ], axis = 1 ) 

# Olhando balanceamento.
dados_final[ 'Churn' ].value_counts( 1 )


    #            # 
    # Usando KNN #
    #            #

# Separando variáveis explicativas e resposta.
X = dados_final.drop( 'Churn', axis = 1 )
y = dados_final[ 'Churn' ]
    
# Biblioteca para padronizar os dados (importante no KNN, que usa distâncias).
norm = StandardScaler()

# Padronizando.
X_normalizado = norm.fit_transform( X )

# Treino e Teste.
X_treino, X_teste, y_treino, y_teste = train_test_split( X_normalizado, 
                                                         y,
                                                         test_size = 0.3,
                                                         random_state = 123 )

# Instanciar o modelo (criamos o modelo) - por padrão são 5 vizinhos.    
knn = KNeighborsClassifier( metric = 'euclidean' ) # usando distância euclidiana.

# Treinando o modelo com os dados de treino.
knn.fit( X_treino, y_treino )

# Testando o modelo com os dados de teste.
predito_knn = knn.predict( X_teste )


    #             # 
    # Naive Bayes #
    #             #
    
# Criando o modelo.    
bnb = BernoulliNB()

# Treinando o modelo com os dados de treino.
bnb.fit( X_treino, y_treino )   
    
# Testando o modelo com os dados de teste.
predito_bnb = bnb.predict( X_teste )


    #                   # 
    # Árvore de Decisão #
    #                   #
    
# Criando o modelo.    
dtc = DecisionTreeClassifier( criterion = 'entropy', random_state = 42 )

# Treinando o modelo com os dados de treino.
dtc.fit( X_treino, y_treino )  

# Verificar a importância de cada atributo.
dtc.feature_importances_

# Testando o modelo com os dados de teste.
predito_dtc = dtc.predict( X_teste )


    #                   # 
    # Resultados Gerais #
    #                   #
    
#            Predito Sim  Predito Não  
# Real Sim        VP          FN 
# Real Não        FP          VN  
    
# Matriz de Confusão (KNN).
print( confusion_matrix( y_teste, predito_knn ) )    

# Matriz de Confusão (Naive Bayes).
print( confusion_matrix( y_teste, predito_bnb ) )    

# Matriz de Confusão (Árvore de Decisão).
print( confusion_matrix( y_teste, predito_dtc ) ) 

# Acurácia (KNN).
print( round( accuracy_score( y_teste, predito_knn ) * 100, 2 ) )    

# Acurácia (Naive Bayes).
print( round( accuracy_score( y_teste, predito_bnb ) * 100, 2 ) )    

# Acurácia (Árvore de Decisão).
print( round( accuracy_score( y_teste, predito_dtc ) * 100, 2 ) ) 

# Precisão (KNN) - Classificados corretamente entre os positivos.
print( round( precision_score( y_teste, predito_knn ) * 100, 2 ) )    

# Precisão (Naive Bayes) - Classificados corretamente entre os positivos.
print( round( precision_score( y_teste, predito_bnb ) * 100, 2 ) )    

# Precisão (Árvore de Decisão) - Classificados corretamente entre os positivos.
print( round( precision_score( y_teste, predito_dtc ) * 100, 2 ) ) 

# Recall (KNN) - VP / ( VP + FN ).
print( round( recall_score( y_teste, predito_knn ) * 100, 2 ) )    

# Recall (Naive Bayes) - VP / ( VP + FN ).
print( round( recall_score( y_teste, predito_bnb ) * 100, 2 ) )    

# Recall (Árvore de Decisão) - VP / ( VP + FN ).
print( round( recall_score( y_teste, predito_dtc ) * 100, 2 ) ) 

# F1 Score (KNN) - F1 Score usa tanto a Precisão como o Recall.
print( round( f1_score( y_teste, predito_knn ) * 100, 2 ) )    

# F1 Score (Naive Bayes) - F1 Score usa tanto a Precisão como o Recall.
print( round( f1_score( y_teste, predito_bnb ) * 100, 2 ) )    

# F1 Score (Árvore de Decisão) - F1 Score usa tanto a Precisão como o Recall.
print( round( f1_score( y_teste, predito_dtc ) * 100, 2 ) ) 