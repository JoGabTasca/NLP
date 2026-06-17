# NLP - Notebooks de Processamento de Linguagem Natural

Este repositório reúne notebooks desenvolvidos em Google Colab para estudar tarefas clássicas de Processamento de Linguagem Natural (PLN/NLP), com foco em pré-processamento textual, extração de características, análise de sentimentos e classificação de discurso ofensivo.

Os notebooks usam principalmente Python, pandas, NLTK, spaCy, scikit-learn, matplotlib e seaborn. A ideia geral é transformar textos em representações numéricas, treinar modelos de Machine Learning e avaliar os resultados com métricas como acurácia, F1-score e matriz de confusão.

## Arquivos do Projeto

| Arquivo | Tema | Abrir no Colab |
|---|---|---|
| `PLN_aula07.ipynb` | Aula prática de raspagem, limpeza textual e TF-IDF manual com letras de músicas | [Abrir](https://colab.research.google.com/github/JoGabTasca/NLP/blob/main/PLN_aula07.ipynb) |
| `Analise_sentimento_NLP.ipynb` | Análise de sentimentos em avaliações do dataset Olist | [Abrir](https://colab.research.google.com/github/JoGabTasca/NLP/blob/main/Analise_sentimento_NLP.ipynb) |
| `TAREFA_Analise_Sentimento_Olist.ipynb` | Tarefa de melhoria de modelo para análise de sentimentos no Olist | [Abrir](https://colab.research.google.com/github/JoGabTasca/NLP/blob/main/TAREFA_Analise_Sentimento_Olist.ipynb) |
| `classificacao_discurso_odio_olid_br.ipynb` | Classificação de linguagem ofensiva usando o dataset OLID-BR | [Abrir](https://colab.research.google.com/github/JoGabTasca/NLP/blob/main/classificacao_discurso_odio_olid_br.ipynb) |

## Visão Geral dos Notebooks

### `PLN_aula07.ipynb`

Notebook de aula voltado para conceitos fundamentais de PLN. Ele coleta letras de músicas da banda Linkin Park no site Vagalume, organiza os dados em um DataFrame e aplica etapas de limpeza e normalização textual.

Principais etapas:

- Raspagem de dados com `requests` e `BeautifulSoup`.
- Criação de uma base com nome da música, link, álbum e letra.
- Limpeza do texto com expressões regulares.
- Remoção de stopwords em inglês usando NLTK.
- Lematização com spaCy (`en_core_web_sm`).
- Construção de vocabulário a partir das letras limpas.
- Cálculo manual de TF, IDF e TF-IDF.

Este notebook é útil para entender, passo a passo, como a representação TF-IDF funciona antes de usar implementações prontas como `TfidfVectorizer` do scikit-learn.

### `Analise_sentimento_NLP.ipynb`

Notebook de análise de sentimentos usando avaliações do Brazilian E-Commerce Public Dataset by Olist. O objetivo é classificar comentários de usuários como positivos ou negativos a partir do texto das avaliações.

Principais etapas:

- Carregamento do arquivo `olist_order_reviews_dataset.csv` diretamente de uma URL.
- Remoção de colunas que não são necessárias para a tarefa.
- Tratamento de duplicatas e comentários sem texto.
- Junção de título e mensagem da avaliação em uma única coluna textual.
- Criação dos rótulos de sentimento a partir da nota da avaliação.
- Análise exploratória dos scores das reviews.
- Pré-processamento do texto com limpeza, stopwords e lematização em português.
- Vetorização com Bag of Words e TF-IDF.
- Treinamento de Regressão Logística.
- Comparação dos resultados entre Bag of Words e TF-IDF.
- Testes com novas frases simulando avaliações de produto.

O notebook conclui que a diferença entre Bag of Words e TF-IDF é pequena nesse problema, mas o TF-IDF apresenta desempenho ligeiramente melhor.

### `TAREFA_Analise_Sentimento_Olist.ipynb`

Este notebook parte da mesma proposta de análise de sentimentos no Olist, mas inclui uma etapa adicional de tarefa: tentar melhorar o desempenho da Regressão Logística usando outros algoritmos ou ajustes de hiperparâmetros.

Principais etapas:

- Repetição do pipeline base de análise de sentimentos no dataset Olist.
- Pré-processamento textual e criação de representações com Bag of Words e TF-IDF.
- Treinamento da Regressão Logística como modelo baseline.
- Avaliação do baseline com acurácia, F1-score e matriz de confusão.
- Testes com frases novas para observar o comportamento do classificador.
- Comparação com outros modelos, incluindo:
  - `LinearSVC`
  - `MultinomialNB`
  - `ComplementNB`
  - variações de `LogisticRegression`
- Busca por melhores valores do parâmetro `C` no `LinearSVC`.

Resultados observados no notebook:

- Baseline com Regressão Logística e TF-IDF: acurácia aproximada de `0.8948` e F1 ponderado de `0.8946`.
- Melhor modelo encontrado na comparação inicial: `LinearSVC` com `C=0.1`, com acurácia aproximada de `0.8956`.
- Melhor ajuste fino observado: `LinearSVC` com `C=0.12`, com acurácia e F1 ponderado próximos de `0.8960`.

Este é o notebook mais completo para a tarefa de análise de sentimentos, pois além de reproduzir o pipeline base, também testa alternativas para melhorar o modelo.

### `classificacao_discurso_odio_olid_br.ipynb`

Notebook dedicado à classificação de linguagem ofensiva em português brasileiro usando o dataset OLID-BR, disponível no Hugging Face. A tarefa consiste em classificar textos como ofensivos (`OFF`) ou não ofensivos (`NOT`).

Principais etapas:

- Instalação das dependências necessárias no Colab.
- Carregamento dos arquivos de treino e teste do dataset OLID-BR.
- Descrição dos campos do dataset, incluindo categorias como insulto, racismo, xenofobia, sexismo e outras marcações.
- Análise exploratória da distribuição das classes.
- Visualização do comprimento dos textos por classe.
- Exibição de exemplos de textos ofensivos e não ofensivos.
- Pré-processamento com:
  - conversão para minúsculas;
  - remoção de URLs e menções;
  - filtragem de caracteres;
  - remoção de stopwords, preservando negações importantes como `não` e `nem`;
  - lematização com spaCy em português.
- Conversão dos rótulos `OFF` e `NOT` para valores binários.
- Vetorização com Bag of Words e TF-IDF.
- Treinamento de Regressão Logística.
- Avaliação com acurácia, F1-score ponderado, F1-score macro, relatório de classificação e matriz de confusão.
- Análise das palavras mais relevantes para cada classe.
- Testes com novos comentários.
- Discussão das limitações de modelos clássicos para detecção de discurso ofensivo.

Resultados observados no notebook:

- Regressão Logística com Bag of Words: acurácia aproximada de `0.8481`, F1 ponderado de `0.8140` e F1 macro de `0.5674`.
- Regressão Logística com TF-IDF: acurácia aproximada de `0.8533`, F1 ponderado de `0.7863` e F1 macro de `0.4604`.

Apesar da acurácia razoável, o notebook mostra uma limitação importante: como o dataset é desbalanceado, o modelo tende a priorizar a classe majoritária. Por isso, o F1 macro é essencial para perceber que o desempenho na classe minoritária pode ser fraco.

## Principais Conceitos Trabalhados

- Raspagem de dados textuais.
- Limpeza e normalização de texto.
- Tokenização.
- Remoção de stopwords.
- Lematização.
- Bag of Words.
- TF-IDF.
- Treinamento e avaliação de modelos supervisionados.
- Regressão Logística.
- LinearSVC.
- Naive Bayes.
- Métricas de classificação.
- Matriz de confusão.
- Desbalanceamento de classes.
- Limitações de modelos clássicos em tarefas de linguagem natural.

## Ordem Sugerida de Estudo

1. `PLN_aula07.ipynb` - para entender limpeza textual, vocabulário e TF-IDF de forma mais manual.
2. `Analise_sentimento_NLP.ipynb` - para ver um pipeline completo de análise de sentimentos.
3. `TAREFA_Analise_Sentimento_Olist.ipynb` - para acompanhar a melhoria do modelo com novos algoritmos.
4. `classificacao_discurso_odio_olid_br.ipynb` - para aplicar conceitos semelhantes em uma tarefa mais difícil e sensível.

## Como Executar

Os notebooks foram pensados para execução no Google Colab. Para executar:

1. Abra o notebook desejado pelo link da tabela.
2. Execute as células na ordem.
3. Quando necessário, autorize a instalação de pacotes como `spacy` e o download dos modelos de linguagem.
4. Verifique se o ambiente do Colab está conectado à internet, pois alguns notebooks carregam datasets externos.

## Observações

- Alguns notebooks baixam modelos do spaCy durante a execução, como `pt_core_news_sm` e `en_core_web_sm`.
- Os resultados podem variar ligeiramente conforme versão das bibliotecas, ambiente de execução e estado dos datasets externos.
- As tarefas de análise de sentimentos e discurso ofensivo são simplificações didáticas. Em aplicações reais, é importante considerar vieses, desbalanceamento de classes, validação mais rigorosa e modelos contextualizados.
