#!/usr/bin/env python
# coding: utf-8

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio
# 
# - Base de Dados: https://drive.google.com/drive/folders/1ThpK_nfciHuTnUuIxDiLqhuPsWsqV-q7?usp=share_link

# ### Passo a Passo de um Projeto de Ciência de Dados
# 
# - Passo 1: Entendimento do Desafio
# - Passo 2: Entendimento da Área/Empresa
# - Passo 3: Extração/Obtenção de Dados
# - Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# - Passo 5: Análise Exploratória
# - Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
# - Passo 7: Interpretação de Resultados

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio
# - TV, Jornal e Rádio estão em milhares de reais
# - Vendas estão em milhões

# #### Importar a Base de dados

# In[1]:


# importação da biblioteca pandas e atribuindo um apelido à ela
import pandas as pd


# In[2]:


# importação da base de dados e atribuindo a base de dados a uma variável
tabela = pd.read_csv(
    r'C:\Users\Virtual Office\Python\Módulo 50 - Python aplicação no mercado de trabalho\Intensivão de Python\Aula 4\advertising.csv'
    )
display(tabela)


# #### Análise Exploratória
# - Vamos tentar visualizar como as informações de cada item estão distribuídas
# - Vamos ver a correlação entre cada um dos itens

# In[3]:


# Camandos de análise para compreensão dos dados que estão na base de dados, informações gerais, tipo, valores vazios e correlação.
print(tabela.info())
print(tabela.isna().sum())
display(tabela.describe())
display(tabela.corr())


# In[4]:


# importação das bibliotecas de gráficos 
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


# Analisando de forma visual a correlação que os dados tem entre si através do mapa de calor
# Quanto mais próximo de 1, maior a correlação; quanto mais próximo de -1 menor é a correlação
plt.figure(figsize = (15,5))

sns.heatmap(tabela.corr(), cmap = 'Reds', annot = True)

plt.show()


# #### Com isso, podemos partir para a preparação dos dados para treinarmos o Modelo de Machine Learning
# 
# - Separando em dados de treino e dados de teste

# In[6]:


# x -> Dados que serão usados como característica para prever Y
x = tabela[['TV', 'Radio', 'Jornal']]
# y -> Variável a ser prevista
y = tabela['Vendas']

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size= 0.3)



# #### Temos um problema de regressão - Vamos escolher os modelos que vamos usar:
# 
# - Regressão Linear
# - RandomForest (Árvore de Decisão)

# In[7]:


# Importação das duas AI que serão testadas 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# Variáveis recebendo os modelos de AI
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()
# Treinando a AI
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)


# #### Teste da AI e Avaliação do Melhor Modelo
# 
# - Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece

# In[8]:


from sklearn.metrics import r2_score
# Configurando a AI para previsão
previsao_arvoredecisao = modelo_regressaolinear.predict(x_teste)
previsao_regressaolinear = modelo_arvoredecisao.predict(x_teste)
# Resultado em percentual da precisão da previsão dos preços por cada modelo de AI
print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste,previsao_arvoredecisao))


# #### Visualização Gráfica das Previsões

# In[9]:


# Conferindo por gráfico de linha a previsão de cada modelo de AI
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['Previsao Regressao Linear'] = previsao_regressaolinear
tabela_auxiliar['Previsao Arvore Decisao'] = previsao_arvoredecisao

plt.figure(figsize=(15,5))
sns.lineplot(tabela_auxiliar)
plt.show()


# #### Como fazer uma nova previsão?

# In[10]:


# Importação de dados diferentes para AI prever o preço
tabela_nova = pd.read_csv(
    r'C:\Users\Virtual Office\Python\Módulo 50 - Python aplicação no mercado de trabalho\Intensivão de Python\Aula 4\novos.csv'
)
display(tabela_nova)


# In[11]:


# Previsão dos preços
previsao = modelo_arvoredecisao.predict(tabela_nova)
print(previsao)

