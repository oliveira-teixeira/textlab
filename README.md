# TextLab - Ferramenta de Análise Textual Qualitativa

Ferramenta interativa para análise textual qualitativa com visualizações avançadas, construída em React.

## Funcionalidades

- **Gerenciamento de Corpus**: Importação e organização de múltiplos textos com filtragem por corpus
- **Estatísticas Textuais**: Frequência de palavras, hapax legomena, type-token ratio
- **TF-IDF**: Cálculo de relevância de termos por documento
- **Análise de Coocorrência**: Rede de palavras com layout de forças e curvas bezier suaves
- **Bigramas**: Rede de bigramas com frequência e força de associação
- **CHD/Reinert**: Classificação Hierárquica Descendente para segmentação temática
- **AFC**: Análise Fatorial de Correspondências com projeção bidimensional
- **Nuvem de Palavras**: Visualização com d3-cloud
- **Árvore de Palavras (Word Tree)**: Exploração de contextos com curvas orgânicas
- **KWIC**: Concordância com contexto de palavras-chave
- **Análise de Sentimento**: Classificação de polaridade por documento
- **Especificidades**: Termos característicos por corpus
- **Visualizações**: Heatmap, Treemap, Sunburst, Radar, Cluster

## Tecnologias

- React 18 + Vite
- Tailwind CSS
- D3.js (d3-cloud, d3-hierarchy)
- Lucide React (ícones)
- SVG com zoom/pan interativo

## Instalação

```bash
npm install
npm run dev
```

## Testes

```bash
# Testes React (Jest)
npm test

# Testes Python (pytest)
npm run test:python
```

## Build

```bash
npm run build
npm run preview
```

## Licença

MIT
