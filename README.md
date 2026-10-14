# Bloco Suvaco de Cristo — Catalogação e análise do acervo

[Português](#português) | [English](#english) | [Español](#español)

## Português

Projeto de processamento e análise do acervo histórico do Bloco Suvaco de Cristo, do Rio de Janeiro. Compara métodos de ampliação de imagens, geração de legendas e classificação multirrótulo para apoiar a catalogação do acervo.

**Acervo privado:** as imagens, os metadados e as descrições do acervo não são públicos. O acesso depende de autorização dos responsáveis pelo acervo; este repositório não concede acesso nem permissão para redistribuir esses dados.

### Funcionalidades

- **Ampliação de imagens:** nearest, bilinear, bicubic, Lanczos, EDSR, ESRGAN e Real-ESRGAN.
- **Legendagem:** BLIP, GIT e ViT-GPT2, com tradução para português, além de modelos multimodais via API NVIDIA.
- **Classificação multirrótulo:** modelos supervisionados sobre embeddings DINOv2 e ConvNeXt V2, com otimização por Optuna, e classificação zero-shot com LLMs.
- **Avaliação:** métricas de classificação, similaridade entre legendas e descrições humanas, nitidez, contraste, artefatos e autoconsistência das imagens ampliadas.
- **Análise:** relatórios, tabelas e gráficos sobre o dataset e os experimentos.

### Estrutura

```text
.
├── config/
│   └── prompts.yaml              # Contexto, prompts e schemas das respostas
├── dataset/                      # Dados privados locais, fora do Git
│   ├── dataset_completo.csv
│   ├── palavras_chave_processed.csv
│   ├── catalogacao_with_local_paths.csv
│   ├── downloaded/               # Arquivos originais
│   ├── splits/                   # Divisão de treino/teste
│   └── upscaled/                 # Imagens ampliadas
├── results/                      # Predições, métricas, modelos e análises
├── src/
│   ├── analysis/                 # Análises do dataset e dos experimentos
│   ├── core/
│   │   ├── caption/              # Geração de legendas
│   │   ├── classification/       # Classificação supervisionada e por LLM
│   │   ├── data/                 # Carregamento do dataset e dos splits
│   │   ├── dataset/              # Preparação legada, ainda não versionada
│   │   ├── evaluation/           # Cálculo de métricas
│   │   ├── features/             # Extração de embeddings visuais
│   │   ├── llm_predictor/        # Integração com a API NVIDIA
│   │   ├── upscaler/             # Ampliação de imagens
│   │   └── utils/                # Utilitários de imagens
│   ├── run/                      # Pipelines Bash para múltiplos modelos
│   └── scripts/
│       ├── data/                 # Criação da divisão treino/teste
│       ├── execution/            # Processamento e treinamento
│       └── evaluation/           # Avaliação das saídas
├── .env.example
├── requirements.txt
└── README.md
```

### Instalação

Use Python 3.10 ou superior. Execute os comandos a partir da raiz do repositório. GPU com CUDA é recomendada para os modelos locais; os exemplos menores podem ser executados em CPU. O primeiro uso dos modelos pode baixar pesos da internet.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

# Dependências usadas pelas análises, ainda ausentes de requirements.txt
python -m pip install matplotlib seaborn

cp .env.example .env
```

No PowerShell, ative o ambiente com `venv\Scripts\Activate.ps1` e copie o exemplo com `Copy-Item .env.example .env`.

Preencha `NVIDIA_API_KEY` no `.env` para gerar legendas ou classificações via API. A implementação atual de `LLMFactory` oferece somente o provedor `nvidia`. O modelo é escolhido com `--model`; os pipelines incluem Llama 4 Maverick, Llama 4 Scout e Mistral Small, sujeitos à disponibilidade na API.

O `.env.example` também contém `GOOGLE_SERVICE_ACCOUNT_JSON` para credenciais de acesso ao acervo no Google Drive. Essas credenciais não são necessárias para processar imagens que já estejam disponíveis localmente.

### Dados necessários

O acervo é de acesso restrito e não acompanha o clone do repositório. Com autorização dos responsáveis, disponibilize os seguintes arquivos no ambiente local:

| Caminho | Uso |
| --- | --- |
| `dataset/dataset_completo.csv` | Metadados, incluindo `LocalPath` com caminhos válidos para as imagens e `Item` para associação dos rótulos. |
| `dataset/downloaded/` | Imagens originais referenciadas no CSV. |
| `dataset/palavras_chave_processed.csv` | Rótulos binários por item; as colunas diferentes de `Id` e `Item` são tratadas como categorias. |
| `dataset/catalogacao_with_local_paths.csv` | Referências humanas para a análise de legendas; por padrão, usa `Descrição (informal)`. |

**Preparação automática precisa de ajuste:** `src/scripts/execution/prepare_dataset.py` calcula a raiz incorretamente e procura `process.py` e `download.py` em `src/src/core/dataset/`. Os arquivos de preparação existem localmente em `src/core/dataset/`, mas ainda não estão versionados. Para executar os experimentos enquanto esse fluxo não for corrigido, utilize os CSVs já preparados e as imagens disponibilizados pelos responsáveis mediante autorização, e confira os caminhos em `LocalPath`.

Crie a divisão estratificada multirrótulo, com aproximadamente 70% para treino e 30% para teste:

```bash
python -m src.scripts.data.create_train_test_split \
  --dataset dataset/dataset_completo.csv \
  --categories dataset/palavras_chave_processed.csv \
  --output dataset/splits/split_latest.json \
  --test-size 0.3
```

O comando também gera `train.csv` e `test.csv` em `dataset/splits/`. Preserve o arquivo de split usado em cada experimento para comparar os modelos sobre o mesmo conjunto de teste.

### Executar experimentos individuais

Os exemplos de ampliação e legendagem usam apenas 10 imagens para uma execução inicial. Remova `--limit 10` para processar o conjunto completo.

**Ampliar imagens e avaliar a saída:**

```bash
python -m src.scripts.execution.upscale_images \
  --dataset dataset/dataset_completo.csv \
  --model bicubic --scale 2 --limit 10

python -m src.scripts.evaluation.evaluate_upscale_quality \
  --upscaled-dir dataset/upscaled/bicubic_2x \
  --original-dir dataset/downloaded \
  --output-dir results/upscale
```

**Gerar legendas com BLIP:**

```bash
python -m src.scripts.execution.caption_images_classical \
  --dataset dataset/dataset_completo.csv \
  --backend blip --batch-size 4 --limit 10 \
  --output-dir results/captions/classical
```

A tradução para português está habilitada por padrão; use `--no-translate` para desativá-la. O `--output-dir` explícito alinha a saída com os pipelines, pois o padrão deste script usa `results/caption/classical`, no singular.

**Gerar legendas com LLM:**

```bash
python -m src.scripts.execution.llm_generate_captions \
  --dataset dataset/dataset_completo.csv \
  --config config/prompts.yaml \
  --provider nvidia \
  --model mistralai/mistral-small-3.1-24b-instruct-2503 \
  --prompt-key caption --limit 10 \
  --output-dir results/captions/llm
```

**Treinar classificadores supervisionados:**

```bash
python -m src.scripts.execution.train_supervised_classifiers \
  --split dataset/splits/split_latest.json \
  --feature-model dinov2-small \
  --classifiers lda rf \
  --optimize --n-trials 15 --cv-folds 3 \
  --output-dir results/classification/supervised_classifiers
```

**Classificar imagens do conjunto de teste com LLM:**

```bash
python -m src.scripts.execution.llm_classify_images \
  --split dataset/splits/split_latest.json \
  --categories dataset/palavras_chave_processed.csv \
  --config config/prompts.yaml \
  --provider nvidia \
  --model mistralai/mistral-small-3.1-24b-instruct-2503 \
  --prompt-key classification/zero_shot_basic \
  --limit 10
```

Edite os textos e schemas em [`config/prompts.yaml`](config/prompts.yaml). As estratégias disponíveis são `classification/zero_shot_basic`, `classification/zero_shot_with_descriptions`, `classification/zero_shot_cot` e `classification/zero_shot_dual_role`. Modelo, temperatura, tokens e paralelismo são definidos pelos argumentos dos scripts. Consulte as opções com `python -m <módulo> --help`.

### Pipelines completos

Os scripts Bash em [`src/run/`](src/run/) executam múltiplos modelos. Crie o ambiente em `venv/`, como na instalação, pois os scripts de processamento ativam esse caminho. Tenha os dados preparados e o split antes da classificação.

```bash
# Ampliação e avaliação
bash src/run/upscale_all.sh
bash src/run/evaluate_upscale.sh

# Legendagem e avaliação
bash src/run/caption_classical_all.sh
bash src/run/llm_caption_all.sh
bash src/run/evaluate_captions.sh

# Classificação e avaliação
bash src/run/supervised_classify_all.sh
bash src/run/llm_classify_all.sh
bash src/run/evaluate_classification.sh
```

Esses pipelines processam conjuntos completos e, nos passos com LLM, fazem chamadas à API. Ajuste os modelos e parâmetros nos arquivos `.sh` conforme os recursos disponíveis.

### Análises e resultados

Após gerar as saídas correspondentes:

```bash
python src/analysis/dataset_analyzer.py
python src/analysis/upscale_analyzer.py
python src/analysis/classification_analyzer.py
python src/analysis/caption_analyzer.py
```

A análise de legendas usa, por padrão, `dataset/catalogacao_with_local_paths.csv`; altere o CSV e a coluna de referência com `--dataset` e `--reference-column`. Para uma análise sem BERTScore ou embeddings, use `--no-bertscore --no-embeddings`.

| Diretório | Conteúdo |
| --- | --- |
| `dataset/upscaled/<método>_<escala>x/` | Imagens ampliadas. |
| `results/upscale/` | Métricas por método e análises em `analysis/`. |
| `results/captions/classical/<backend>/` | Legendas dos modelos locais e métricas. |
| `results/captions/llm/<modelo>/<prompt>/` | Legendas estruturadas dos LLMs. |
| `results/captions/analysis/` | Comparações entre legendas e referências humanas. |
| `results/classification/supervised_classifiers/<embedding>/<classificador>/` | Modelos treinados, predições e métricas. |
| `results/classification/llm/<modelo>/<prompt>/` | Classificações dos LLMs e métricas. |
| `results/classification/analysis/` | Comparações entre classificadores e estratégias de prompt. |
| `results/dataset/analysis/` | Estatísticas e gráficos exploratórios do acervo. |

Os caminhos de classificação supervisionada acima são os usados pelos pipelines e pelo exemplo deste README. Sem `--output-dir`, o script individual usa `results/classification/supervised`.

### Versionamento e solução de problemas

O [`.gitignore`](.gitignore) exclui ambientes virtuais, caches, credenciais, dados do acervo, novos resultados de experimentos e pesos de modelos. O `.env.example` permanece versionado. A regra `/dataset/` se aplica somente ao acervo na raiz, permitindo versionar o código em `src/core/dataset/`, que era ocultado pela regra anterior. Arquivos de resultados já rastreados pelo Git continuam versionados e suas alterações seguem aparecendo no `git status`.

Resultados que reproduzam imagens, metadados ou descrições do acervo também devem respeitar as restrições de acesso e redistribuição. O `.gitignore` não remove arquivos já versionados nem seu histórico.

- **Dataset ou imagens ausentes:** confira os arquivos necessários e os caminhos em `LocalPath`; o preparador automático precisa ter os caminhos corrigidos e seus módulos versionados antes de ser utilizado em um clone novo.
- **Memória insuficiente:** reduza `--batch-size` na legendagem ou no treinamento; na ampliação, ajuste `--tile-size` ou use `--device cpu`.
- **Erros de API:** confira `NVIDIA_API_KEY`, disponibilidade do modelo e limites da conta; reduza `--max-workers` se necessário. Para Llama 4 Scout, os pipelines usam `--max-image-dim 512`.
- **Falha ao importar bibliotecas de gráficos:** instale `matplotlib` e `seaborn` no ambiente ativo.

## English

Research pipeline for cataloging the historical archive of Bloco Suvaco de Cristo. It compares image upscaling, caption generation and multilabel classification, with evaluation metrics and visual reports.

The archive images, metadata and descriptions are private. Access requires authorization from the archive custodians; this repository grants neither access nor redistribution rights. Outputs reproducing archive content are subject to the same restrictions.

Run commands from the repository root using the setup and examples in the Portuguese section. Authorized users must provide archive data locally; the automatic dataset preparation script currently resolves an incorrect path, and its legacy helper modules are not yet tracked by Git. LLM tasks use the NVIDIA API and require `NVIDIA_API_KEY`. Batch pipelines live in `src/run/`, individual commands in `src/scripts/`, and analysis tools in `src/analysis/`.

## Español

Pipeline de investigación para catalogar el acervo histórico del Bloco Suvaco de Cristo. Compara ampliación de imágenes, generación de descripciones y clasificación multietiqueta, con métricas de evaluación e informes visuales.

Las imágenes, los metadatos y las descripciones del acervo son privados. El acceso requiere autorización de los responsables del acervo; este repositorio no concede acceso ni derechos de redistribución. Los resultados que reproduzcan contenido del acervo están sujetos a las mismas restricciones.

Ejecute los comandos desde la raíz del repositorio siguiendo la instalación y los ejemplos de la sección en portugués. Los usuarios autorizados deben disponer de los datos localmente; el preparador automático actualmente calcula una ruta incorrecta y sus módulos auxiliares aún no están versionados en Git. Las tareas con LLM utilizan la API NVIDIA y requieren `NVIDIA_API_KEY`. Los pipelines están en `src/run/`, los comandos individuales en `src/scripts/` y las herramientas de análisis en `src/analysis/`.
