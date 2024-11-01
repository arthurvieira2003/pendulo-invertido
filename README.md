# Sistema de Controle de Pêndulo Invertido

Este projeto implementa diferentes abordagens de controle para o problema do pêndulo invertido sobre um carro, utilizando técnicas de Lógica Fuzzy, Algoritmos Genéticos e Sistemas Neuro-Fuzzy.

## Estrutura do Projeto

O projeto é composto por três arquivos principais:

### fuzzy.py

Implementa o controlador fuzzy básico e o simulador do pêndulo invertido. Este módulo contém:

- Definição das variáveis linguísticas (ângulo, velocidade angular, posição, etc.)
- Regras fuzzy para o controle
- Simulador físico do sistema
- Funções para visualização e análise de resultados

### genetico_fuzzy.py

Implementa um otimizador genético para ajustar os parâmetros do sistema fuzzy. Características:

- Otimização das funções de pertinência
- População de 300 indivíduos
- 150 gerações de evolução
- Função fitness considerando múltiplos objetivos (estabilização, energia, etc.)

### neuro_fuzzy.py

Implementa um sistema híbrido neuro-fuzzy que combina redes neurais com lógica fuzzy. Principais aspectos:

- Rede neural MLP para aprendizado
- Sistema fuzzy para conhecimento base
- Combinação adaptativa das saídas

## Requisitos

```
numpy>=1.21.0
scikit-fuzzy>=0.4.2
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
deap>=1.3.1
```

## Como Executar

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Execute um dos controladores:

Controlador Fuzzy básico:

```bash
python fuzzy.py
```

Controlador Genético-Fuzzy:

```bash
python genetico_fuzzy.py
```

Controlador Neuro-Fuzzy:

```bash
python neuro_fuzzy.py
```

## Resultados Esperados

Cada controlador irá gerar:

- Gráficos mostrando a evolução temporal do ângulo do pêndulo, posição do carro e força aplicada
- Visualização da posição final do sistema
- Métricas de desempenho (erro máximo, energia consumida, tempo de estabilização)

## Comparação entre as Abordagens

### Controlador Fuzzy Básico

Vantagens:

- Implementação mais simples e direta
- Regras intuitivas e facilmente interpretáveis
- Comportamento previsível e estável
- Menor custo computacional

Desvantagens:

- Desempenho limitado pela expertise do projetista
- Menor capacidade de adaptação
- Pode requerer muito ajuste manual

Cenários ideais:

- Sistemas bem conhecidos e modelados
- Quando a interpretabilidade é crucial
- Aplicações com recursos computacionais limitados

### Controlador Genético-Fuzzy

Vantagens:

- Otimização automática dos parâmetros
- Pode encontrar soluções não-óbvias
- Bom equilíbrio entre desempenho e interpretabilidade

Desvantagens:

- Processo de otimização computacionalmente intensivo
- Resultado pode variar entre execuções
- Tempo significativo de treinamento

Cenários ideais:

- Quando há tempo disponível para otimização
- Sistemas com múltiplos objetivos conflitantes
- Quando se busca melhor desempenho mantendo interpretabilidade

### Controlador Neuro-Fuzzy

Vantagens:

- Combina aprendizado com conhecimento especialista
- Maior capacidade de adaptação
- Potencial para melhor desempenho

Desvantagens:

- Maior complexidade de implementação
- Requer dados de treinamento
- Menor interpretabilidade que sistemas puramente fuzzy

Cenários ideais:

- Sistemas com dados disponíveis para treinamento
- Quando se busca máximo desempenho
- Aplicações que requerem adaptação em tempo real

## Conclusão

A escolha do controlador mais adequado depende das características específicas do problema:

- Para sistemas simples e bem definidos, o controlador fuzzy básico pode ser suficiente
- Quando se busca otimização automática sem perder interpretabilidade, o genético-fuzzy é uma boa escolha
- Para máximo desempenho e adaptabilidade, o neuro-fuzzy é mais indicado, desde que haja dados disponíveis
