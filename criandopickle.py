# Pacotes a serem utilizados.
import pandas as pd
import plotly.express as px
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pickle


    #                      # 
    # Lendo banco de dados #
    #                      #
    
# Lendo banco de dados.
dados = pd.read_csv( 'datasets/marketing_investimento.csv' )

# Head.
dados.head()

# Info.
dados.info()


    #                               # 
    # Olhando variáveis categóricas #
    #                               #

# Aderência Investimento.
px.histogram( dados, x = 'aderencia_investimento', text_auto = True )

# Estado Civil por Aderência Investimento.
px.histogram( dados, x = 'estado_civil', text_auto = True, color = 'aderencia_investimento', barmode = 'group' )

# Escolaridade por Aderência Investimento.
px.histogram( dados, x = 'escolaridade', text_auto = True, color = 'aderencia_investimento', barmode = 'group' )

# Inadimplência por Aderência Investimento.
px.histogram( dados, x = 'inadimplencia', text_auto = True, color = 'aderencia_investimento', barmode = 'group' )

# Fez empréstimo por Aderência Investimento.
px.histogram( dados, x = 'fez_emprestimo', text_auto = True, color = 'aderencia_investimento', barmode = 'group' )


    #                             # 
    # Olhando variáveis numéricas #
    #                             #
    
# Idade.    
px.box( dados, x = 'idade', color = 'aderencia_investimento' )

# Saldo.    
px.box( dados, x = 'saldo', color = 'aderencia_investimento' )

# Tempo do Último Contato.    
px.box( dados, x = 'tempo_ult_contato', color = 'aderencia_investimento' )

# Número de Contatos.    
px.box( dados, x = 'numero_contatos', color = 'aderencia_investimento' )


    #                                 #
    # Separando e transformando dados #
    #                                 #

# Variáveis explicativas e resposta.
x = dados.drop( 'aderencia_investimento', axis = 1 )
y = dados[ 'aderencia_investimento' ]

# Colunas.
colunas = x.columns

# Criando OneHotEncoder caso não seja binário.
# Mandando remanescer as demais colunas e sem fazer sparse.
one_hot = make_column_transformer( (
    OneHotEncoder( drop = 'if_binary' ),
    [ 'estado_civil', 'escolaridade', 'inadimplencia', 'fez_emprestimo' ]
),
    remainder = 'passthrough',
    sparse_threshold = 0 )

# Fitando e transformando.
x = one_hot.fit_transform( x )

# Pegando nomes das colunas do OneHotEncoder.
one_hot.get_feature_names_out( colunas )

# DataFrame.
pd.DataFrame( x, columns = one_hot.get_feature_names_out( colunas ) )

        
    #                               #
    # Transformando a variável alvo #
    #                               #
    
# LabelEncoder.
label_ecoder = LabelEncoder()
y = label_ecoder.fit_transform( y ) 
y


    #                  #
    # Ajustando modelo #
    #                  #

# Treino e teste.
x_treino, x_teste, y_treino, y_teste = train_test_split( x, y, stratify = y,  random_state = 5 )

# Criando árvore de decisão.
arvore = DecisionTreeClassifier( random_state = 5 )
arvore.fit( x_treino, y_treino )
arvore.predict( x_teste )
arvore.score( x_teste, y_teste )

# Colocando os nomes que eu quero para as colunas.
nome_colunas = [ 'casado (a)',
                 'divorciado (a)',
                 'solteiro (a)',
                 'fundamental',
                 'medio',
                 'superior',
                 'inadimplencia',
                 'fez_emprestimo',
                 'idade',
                 'saldo',
                 'tempo_ult_contato',
                 'numero_contatos' ]

# Árvore graficamente.
plt.figure( figsize = ( 15, 6 ) )
plot_tree( arvore, filled = True, class_names = [ 'não', 'sim' ], fontsize = 1, feature_names = nome_colunas );
arvore.score( x_treino, y_treino ) # overfitting total.

# Tentando evitar overfitting.
arvore = DecisionTreeClassifier( max_depth = 3, random_state = 5 ) # criando árvore com profundidade de 3.
arvore.fit( x_treino, y_treino )
arvore.score( x_treino, y_treino ) # resultado para treino.
arvore.score( x_teste, y_teste ) # resultado para teste.

# Olhando graficamente.
plt.figure( figsize = ( 15, 6 ) )
plot_tree( arvore, filled = True, class_names = [ 'não', 'sim' ], fontsize = 7, feature_names = nome_colunas );

# Normalizando dados (para realizar o KNN).
normalizacao = MinMaxScaler()
x_treino_normalizado = normalizacao.fit_transform( x_treino )

# Exibindo em dataframe.
pd.DataFrame( x_treino_normalizado )


    #     #
    # KNN #
    #     #

# KNN (precisa normalizar).
knn = KNeighborsClassifier()
knn.fit( x_treino_normalizado, y_treino )
x_teste_normalizado = normalizacao.transform( x_teste )
knn.score( x_teste_normalizado, y_teste )


    #                                       #   
    # Escolhendo e salvando o melhor modelo #
    #                                       #

# Acurácias.
print( f'Acurácia Árvore: { arvore.score( x_teste, y_teste ) }' )
print( f'Acurácia KNN: { knn.score( x_teste_normalizado, y_teste ) }' )


    #                                        # 
    # Salvando pickles do OneHot e do Modelo #
    #                                        #

# Salvando OneHot.
with open( 'modelo_onehotenc.pkl', 'wb' ) as arquivo:
    pickle.dump( one_hot, arquivo )
    
# Salvando Árvore.
with open( 'modelo_arvore.pkl', 'wb' ) as arquivo:
    pickle.dump( arvore, arquivo )
    
        
    #                       #
    # Escorando nova pessoa #
    #                       # 
    
# Novo dado.    
novo_dado = {
    'idade': [ 45 ],
    'estado_civil':[ 'solteiro (a)' ],
    'escolaridade':[ 'superior' ],
    'inadimplencia': [ 'nao' ],
    'saldo': [ 23040 ],
    'fez_emprestimo': [ 'nao' ],
    'tempo_ult_contato': [ 800 ],
    'numero_contatos': [ 4 ]
}

# Jogando em um dataframe.
novo_dado = pd.DataFrame( novo_dado )
novo_dado

# Lendo pickles.
modelo_one_hot = pd.read_pickle( 'modelo_onehotenc.pkl' )
modelo_arvore = pd.read_pickle( 'modelo_arvore.pkl' )

# Escorando.
novo_dado = modelo_one_hot.transform( novo_dado )
modelo_arvore.predict( novo_dado )