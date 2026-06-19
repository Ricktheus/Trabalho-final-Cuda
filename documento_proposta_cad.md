# Proposta de Trabalho Prático - Computação de Alto Desempenho

**Integrantes do Grupo:**
- HENRIQUE M. M. MIRANDA - Matrícula: 202405479
- CINDY STEPHANIE GOMES RABELO - Matrícula: 202403898
- EDUARDO DIAS PEIXOTO - Matrícula: 202010395
- LUIANY GONCALVES CARVALHO - Matrícula: 202303351

## 1. Introdução e Descrição do Problema
Dentro do campo da mineração de dados em larga escala (*Big Data*), as técnicas de *Clustering* sofrem com um grande gargalo: a necessidade de se validar a qualidade dos agrupamentos obtidos. O **Índice de Dunn** (Dunn Index) é uma das métricas mais eficientes para esta validação: ele avalia se os agrupamentos são compactos (distância intra-cluster máxima mínima) e bem separados (distância inter-cluster mínima grande).
Contudo, calcular o Índice de Dunn requer computar a distância de cada ponto a todos os outros pontos e agrupamentos. A complexidade deste cálculo cresce quadraticamente, ou seja, $O(N^2)$, tornando a validação sequencial por meio de processadores CPU inviável quando se lida com milhões de registros. O problema que este projeto resolve é justamente essa severa limitação de escalabilidade no cálculo matemático de validação de *clusters*.

## 2. Objetivos da Aplicação
O objetivo deste trabalho prático é implementar uma versão paralelizada do cálculo do Índice de Dunn utilizando bibliotecas e arquiteturas de fluxo paralelo, mitigando a elevada carga quadrática de distâncias.
Os objetivos específicos incluem:
- Transferir a carga massiva da computação aritmética das Matrizes de Distâncias (intra e inter *clusters*) para os múltiplos núcleos de execução paralela.
- Explorar arquiteturas com elevado *Data Parallelism* utilizando CUDA.
- Analisar a performance e os gargalos residuais focados em transferências de memória entre os *hosts* e *devices*.

## 3. Resultados Esperados
O principal resultado esperado é a obtenção de um ganho notável de performance (*Speed-up*) quando comparada à mesma rotina de validação executada sequencialmente. Espera-se ilustrar de maneira concreta que a validação qualitativa do particionamento de Big Data em CPU é fundamentalmente um problema restritivo de tempo computacional (limitante temporal) e que se torna plenamente viável dentro de ecossistemas paralelos de hardware. O trabalho proporcionará excelente fluência do grupo em mapeamento matricial linearizado.

## 4. Ferramentas de Programação e Arquitetura 
Para a etapa de desenvolvimento do algoritmo paralelo, faremos o uso estrito e primário da linguagem **C/C++** instrumentada com as funções da plataforma **CUDA** da NVIDIA (ou associada com OpenMP para tarefas do Host).

## 5. Artigo Baseline
Este projeto fundamentar-se-á inteiramente na arquitetura e modelagem propostas na literatura de Alto Desempenho descrita a seguir:
**Artigo:** *Parallel and scalable Dunn Index for the validation of big data clusters* (Publicado originalmente no periódico *Parallel Computing*, Elsevier).
