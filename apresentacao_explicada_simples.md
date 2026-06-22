# A Apresentação Explicada em Linguagem Simples

> **O que é este documento:** uma explicação **slide a slide**, em palavras do dia a dia, do que a apresentação quer dizer. Sem fórmulas complicadas — só a ideia por trás de cada tela. Serve para qualquer pessoa entender o trabalho (e para você ganhar segurança na hora de falar).

---

## 🧩 A ideia do trabalho em 3 frases

1. Computadores conseguem **separar dados em grupos** parecidos (isso se chama *clustering*). Mas é preciso **dar uma nota** dizendo se os grupos ficaram bons.
2. Calcular essa nota é **muito lento** quando há muitos dados, porque exige comparar **cada ponto com todos os outros**.
3. Nós fizemos esse cálculo rodar numa **placa de vídeo (GPU)**, que faz milhares de continhas ao mesmo tempo, e ficou **até ~25× mais rápido** — sem errar nenhum resultado.

> **Analogia geral:** é como corrigir 100 mil provas. Um professor sozinho (a CPU) demora muito. Nós colocamos um **estádio cheio de ajudantes** (a GPU) corrigindo todas ao mesmo tempo.

---

## Slide 1 — Capa

**O que aparece:** o título do trabalho, os nomes da equipe e a disciplina.

**Em palavras simples:** é a "porta de entrada". Diz quem somos e qual é o assunto: acelerar, usando a placa de vídeo, o cálculo de três "notas de qualidade" de agrupamentos.

**Resumo:** "Olá, somos a equipe X e vamos mostrar como deixamos esse cálculo muito mais rápido."

---

## Slide 2 — O Problema

**O que aparece:** o que é *clustering*, por que precisamos validar, e por que é lento.

**Em palavras simples:** o computador junta coisas parecidas em grupos. Mas será que ele agrupou bem? Para saber, medimos duas coisas: se cada grupo está **bem unido por dentro** e se os grupos estão **bem afastados uns dos outros**. O problema é que, para medir isso, precisamos comparar **todo mundo com todo mundo**.

> **Analogia dos apertos de mão:** numa sala onde todos apertam a mão de todos, o número de apertos explode. Com 100 mil pessoas, são **bilhões** de apertos. Se você **dobra** o número de pessoas, o trabalho **quadruplica**. É isso que deixa tudo lento.

**Resumo:** medir a qualidade dos grupos é caríssimo porque cresce ao quadrado do número de dados.

---

## Slide 3 — As Três "Notas" (métricas)

**O que aparece:** as três fórmulas — Dunn, Silhueta e Davies-Bouldin.

**Em palavras simples:** são três jeitos diferentes de dar nota ao agrupamento:
- **Dunn:** olha o **pior caso** — o grupo mais "gordo" e os dois grupos mais "grudados". Nota alta = grupos compactos e separados.
- **Silhueta:** dá uma nota **para cada ponto**: "você se sente mais em casa no seu grupo ou no grupo vizinho?". Depois tira a média.
- **Davies-Bouldin:** compara o **centro** de cada grupo com os dos vizinhos. Nota baixa = grupos apertados e com centros distantes.

> **Por que três?** Cada uma enxerga de um ângulo. Juntas, dão um retrato mais confiável.

**Resumo:** três medidas que, juntas, dizem se o agrupamento ficou bom.

---

## Slide 4 — De onde partimos (o artigo-base)

**O que aparece:** o artigo científico que inspirou o trabalho e como nós fazemos diferente.

**Em palavras simples:** existe um artigo famoso que resolve o mesmo problema, mas de outro jeito: ele usa **vários computadores juntos** e faz uma **estimativa** (um "chute educado", olhando só uma parte dos dados). Nós fazemos o contrário: usamos **uma única placa de vídeo** e calculamos o valor **exato**, sem chutar.

> **Analogia da sopa:** para saber se a sopa está boa, o artigo **prova uma colherada** (rápido, mas aproximado). Nós **provamos a panela inteira**, só que com um fogão muito mais potente.

**Resumo:** mesmo problema, caminho diferente — eles aproximam de forma distribuída, nós calculamos exato numa GPU.

---

## Slide 5 — Objetivos

**O que aparece:** o que nos propusemos a fazer.

**Em palavras simples:** queremos (1) jogar o trabalho pesado para a placa de vídeo, (2) gastar **pouca memória** para conseguir lidar com muitos dados, (3) fazer uma comparação **honesta** com a CPU, (4) **provar** que o resultado está certo e (5) **medir** o quanto ficou mais rápido.

**Resumo:** ser mais rápido **e** continuar certo — e provar as duas coisas.

---

## Slide 6 — A grande sacada: *Matrix-free*

**O que aparece:** uma tabela mostrando que guardar todas as distâncias é impossível, e a solução.

**Em palavras simples:** a forma "ingênua" seria **guardar numa tabela gigante** a distância de cada ponto a cada outro. Mas essa tabela, para 100 mil pontos, ocuparia **80 GB** — não cabe na placa de vídeo nem na memória do computador. Nossa sacada (**matrix-free**) é **não guardar nada**: calculamos cada distância **na hora em que precisamos** e jogamos fora. Aí a memória cai de 80 GB para **uns 3 MB**.

> **Analogia da tabuada:** em vez de **imprimir e guardar** uma tabuada gigante de 100 mil × 100 mil, é mais fácil **fazer a conta na hora** ("7×8") sempre que precisar.

**Resumo:** não guardamos a tabela de distâncias; recalculamos na hora. **É isso que destravou os 100 mil pontos.**

---

## Slide 7 — Como funciona na prática (Dunn)

**O que aparece:** o funcionamento do "motorzinho" (kernel) que calcula o Dunn na GPU.

**Em palavras simples:** organizamos a placa de vídeo como um **exército**. Cada **ponto** ganha um **pelotão de 256 ajudantes**. O pelotão varre todos os outros pontos, calcula as distâncias e guarda só o que importa (o maior e o menor valor). Depois, os 256 resultados são combinados num só **em poucos passos**, tipo um **torneio mata-mata** (256 → 128 → ... → 1).

> **Detalhe esperto:** o ponto principal fica numa "memória de bancada" super-rápida, perto dos ajudantes, em vez de ir buscar no "almoxarifado" distante toda hora.

**Resumo:** muitos ajudantes calculando distâncias ao mesmo tempo, e combinando os resultados rapidinho.

---

## Slide 8 — Como funciona (Silhueta e Davies-Bouldin)

**O que aparece:** os motores da Silhueta e do Davies-Bouldin.

**Em palavras simples:** a **Silhueta** funciona parecido com o Dunn (um pelotão por ponto), mas vai somando distâncias **por grupo**. O **Davies-Bouldin** é o **mais fácil**: ele não precisa comparar todo mundo com todo mundo — usa só os **centros** dos grupos, então é **muito mais rápido** e quase não pesa no tempo.

> **Termo traduzido:** "*atomicAdd*" é só uma regra de **"um de cada vez"** para várias contas não se atrapalharem ao escrever no mesmo lugar (como uma vaquinha onde cada um põe dinheiro na vez).

**Resumo:** Silhueta é pesada como o Dunn; Davies-Bouldin é leve porque só olha os centros.

---

## Slide 9 — A comparação justa (CPU)

**O que aparece:** os três "competidores" — CPU com 1 núcleo, CPU com vários núcleos (OpenMP) e a GPU.

**Em palavras simples:** para o resultado ser **honesto**, não comparamos a GPU só com a CPU "devagar" (1 núcleo). Também comparamos com a CPU usando **todos os núcleos disponíveis** (OpenMP), tudo na **mesma máquina**.

> **Detalhe honesto:** o computador grátis que usamos (Colab) tem só **2 "meio-núcleos"** que dividem **um núcleo de verdade**. Por isso a "CPU turbinada" quase não ficou mais rápida que a CPU normal — e nós **assumimos isso abertamente** na análise.

**Resumo:** comparamos a GPU com o melhor que a CPU daquela máquina conseguia fazer.

---

## Slide 10 — Provando que está certo (3 camadas)

**O que aparece:** as três formas de checar que os resultados estão corretos.

**Em palavras simples:** antes de comemorar a velocidade, **provamos que não estamos errando**:
1. Montamos um exemplo de **gabarito conhecido** (a resposta tem que dar exatamente 2,0 — e deu).
2. Comparamos com uma **biblioteca famosa e confiável** (o scikit-learn) — bateu, com diferença minúscula.
3. Conferimos que a **GPU dá o mesmo número que a CPU**, sempre.

> **Regra rígida:** se qualquer uma dessas checagens falhar, o programa **para na hora**. Ou seja, só medimos velocidade quando o resultado já está comprovadamente certo.

**Resumo:** velocidade não vale nada se o resultado estiver errado — então provamos que está certo de três jeitos.

---

## Slide 11 — Resultado: o tempo (gráfico)

**O que aparece:** um gráfico do tempo gasto conforme os dados aumentam.

**Em palavras simples:** as linhas da CPU **disparam para cima** quando os dados crescem; a linha da GPU fica **lá embaixo, quase deitada**. Em 100 mil pontos, a CPU leva **~80 segundos** e a GPU leva **~3 segundos**.

> **Por que as duas linhas de CPU estão quase juntas?** Porque (como dito no slide 9) a versão "turbinada" quase não acelerou naquela máquina.

**Resumo:** quanto mais dados, maior a distância entre a CPU (lenta) e a GPU (rápida).

---

## Slide 12 — Resultado: o quanto acelerou (gráfico)

**O que aparece:** um gráfico mostrando "quantas vezes mais rápido" a GPU foi.

**Em palavras simples:** quanto **maior** o problema, **maior** a vantagem da GPU. Em 100 mil pontos, a GPU foi quase **25 vezes** mais rápida. Com poucos dados a vantagem é pequena, porque o "tempo de ligar a máquina" pesa mais que o cálculo em si.

> **Analogia:** ligar o estádio de ajudantes só compensa quando há **muito** trabalho. Para corrigir 3 provas, o professor sozinho é mais prático; para 100 mil, o estádio ganha disparado.

**Resumo:** a GPU compensa cada vez mais conforme os dados crescem — até ~25×.

---

## Slide 13 — Onde o tempo é gasto (gráfico)

**O que aparece:** uma "pizza/barra" mostrando em qual etapa a GPU gasta o tempo.

**Em palavras simples:** quase **todo** o tempo da GPU (~99,9%) é gasto no Dunn e na Silhueta — justamente as duas que comparam todo mundo com todo mundo. Copiar os dados para a placa e calcular o Davies-Bouldin é **quase instantâneo**.

> **O que isso ensina:** o gargalo é a **conta** em si, não o "transporte" dos dados. Então não adianta otimizar o transporte — é o cálculo que manda.

**Resumo:** o tempo está quase todo nas duas contas pesadas; o resto é desprezível.

---

## Slide 14 — Tabela de números

**O que aparece:** a tabela final com tempos e velocidades para vários tamanhos.

**Em palavras simples:** é o "placar" oficial. A linha mais importante é a de **100 mil pontos**: GPU em **3,24 s** contra **~80 s** da CPU = **~25× mais rápido**, e **100% de acerto** em todos os tamanhos.

**Resumo:** o resumo numérico que prova tudo o que dissemos: muito mais rápido, e sempre certo.

---

## Slide 15 — Conclusões da análise

**O que aparece:** os principais aprendizados.

**Em palavras simples:**
- **Quanto mais dados, melhor para a GPU** (a vantagem cresce).
- **Fomos honestos** sobre a CPU (a versão turbinada quase não ajudou naquela máquina), mas a GPU ganha mesmo assim.
- O **peso está nas contas**, não no transporte de dados.
- Dá para ser **ainda mais rápido** com uma placa melhor ou usando precisão "leve" (float).

**Resumo:** rápido, honesto e correto — e com espaço para ficar ainda melhor.

---

## 📖 Mini-glossário (termos traduzidos)

| Termo técnico | Em linguagem simples |
|---|---|
| **Clustering** | Juntar coisas parecidas em grupos, sozinho. |
| **Validar clusters** | Dar uma nota dizendo se os grupos ficaram bons. |
| **CPU** | O cérebro do computador: poucos "trabalhadores" muito fortes. |
| **GPU (placa de vídeo)** | Um estádio com milhares de "trabalhadores" simples, todos juntos. |
| **CUDA** | A "língua" para mandar tarefas para a placa de vídeo. |
| **O(N²)** | "Dobrou os dados? O trabalho quadruplicou." |
| **Matrix-free** | Não guardar a tabela gigante de distâncias; calcular na hora. |
| **Kernel** | O "motorzinho" (programa) que roda dentro da GPU. |
| **Memória compartilhada** | Uma bancada rápida perto dos trabalhadores. |
| **OpenMP** | Fazer a CPU usar vários núcleos ao mesmo tempo. |
| **Speed-up** | Quantas vezes a GPU foi mais rápida que a CPU. |
| **scikit-learn** | Uma biblioteca famosa e confiável, usada como "gabarito". |
| **float / double** | Régua de milímetro (rápida) vs. de micrômetro (mais exata). |

---

> **Frase final para fechar a apresentação:** "Pegamos um cálculo que era lento demais, fizemos rodar numa placa de vídeo de um jeito que **economiza memória**, e ficamos **até 25 vezes mais rápidos** — sem errar uma conta sequer."
