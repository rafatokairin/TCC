# Artigo LNCS — versão em português (pt-BR)

Tradução do artigo de conferência (`../lncs/`) para o português brasileiro,
mantendo o formato **Springer LNCS**, as mesmas tabelas/números, figuras e
citações. Usa `\usepackage[brazil]{babel}`.

As classes do LNCS não são redistribuíveis; obtenha-as uma vez:

```bash
# a partir de paper/lncs-ptbr/
wget https://ftp.springer.de/pub/tex/latex/llncs/latex2e/llncs2e.zip
unzip llncs2e.zip 'llncs.cls' 'splncs04.bst'
```

## Compilar

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

As figuras já estão em `figures/` (as mesmas da versão em inglês, geradas pela
execução válida na RTX 4060). Os números das tabelas são idênticos aos da versão
em inglês — este é o mesmo estudo, apenas em português.
