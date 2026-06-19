# Como compilar o artigo (`artigo.tex`)

O artigo segue o **template SBC** (exigido pela SSCAD 2026, máx. 12 páginas).
Ele precisa do `sbc-template.sty` (e, opcionalmente, do `sbc.bst`), que **não**
fazem parte deste repositório por serem distribuídos pela SBC.

## Opção A — Overleaf (recomendado, mais fácil)

1. Acesse [overleaf.com](https://www.overleaf.com) e crie um projeto em branco.
2. Faça upload de **4 arquivos** deste repositório:
   - `artigo.tex`
   - `bench_tempo.png`
   - `bench_speedup.png`
   - `bench_breakdown.png`
3. Adicione o template SBC ao projeto, de um destes jeitos:
   - **Mais simples:** no Overleaf, clique em *New Project → Templates* e procure
     por **"SBC Conferences"** (ou "Brazilian Computer Society"); copie o
     `sbc-template.sty` (e `sbc.bst`) de lá para o seu projeto; **ou**
   - Baixe o pacote oficial em **SBC → Documentos Institucionais → "Templates
     para Artigos e Capítulos de Livros"** e envie o `sbc-template.sty` para o
     projeto.
4. Em *Menu → Compiler*, selecione **pdfLaTeX** e compile.

## Opção B — Local (TeX Live / MiKTeX)

```bash
# coloque sbc-template.sty (e sbc.bst) na mesma pasta do artigo.tex
pdflatex artigo.tex
pdflatex artigo.tex   # 2x para resolver referências
```

## Observações

- Os **e-mails** no cabeçalho são um *placeholder* (`@discente.ufg.br`).
  Ajuste para os e-mails reais do grupo antes de submeter.
- Os **números e gráficos** já são os reais (T4 no Colab, até N=100.000).
- Se quiser regenerar os gráficos, rode `python gerar_graficos.py`
  (lê o `resultados_benchmark.csv`).
- Verifique o **limite de 12 páginas** após a compilação.
