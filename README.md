# Bloco Suvaco de Cristo - Catalogação de Acervo

Projeto para catalogação, processamento e análise do acervo histórico do Bloco Suvaco de Cristo usando IA.

## 🎭 Sobre o Projeto

Este projeto automatiza a catalogação e análise de imagens, vídeos e documentos do acervo histórico do Bloco Suvaco de Cristo, um tradicional bloco de carnaval do Rio de Janeiro.

Utiliza modelos de linguagem multimodais (Vision Language Models) para:
- Gerar descrições detalhadas das imagens
- Identificar elementos culturais e históricos
- Extrair metadados automaticamente
- Organizar e indexar o acervo

## 📁 Estrutura do Projeto

```
.
├── config/
│   └── prompt.yaml                  # Configuração de prompts e modelo
├── dataset/
│   ├── Descrição do acervo.xlsx    # Arquivo Excel original
│   ├── catalogacao_with_hyperlinks.csv
│   ├── palavras_chave_processed.csv
│   ├── dataset_completo.csv        # Dataset final
│   └── downloaded/                  # Imagens baixadas
├── src/
│   ├── core/
│   │   ├── dataset/                # Módulo de preparação de dataset
│   │   │   ├── extract_hyperlinks.py
│   │   │   ├── download_and_process.py
│   │   │   ├── prepare_dataset.py
│   │   │   └── GOOGLE_DRIVE_SETUP.md
│   │   └── llm_predictor/          # Módulo de predição LLM
│   │       ├── llm_factory.py
│   │       ├── nvidia_predictor.py
│   │       ├── gemini_predictor.py
│   │       └── openai_predictor.py
│   └── scripts/                    # Scripts de processamento
│       ├── classify_images.py      # Classificação de imagens
│       └── README.md
├── prepare_dataset.py              # Script principal de preparação
├── classify_images.py              # Script principal de classificação
└── README.md                       # Este arquivo
```

## 🚀 Quick Start

### 1. Instalação

```bash
# Clonar repositório
git clone <url-do-repo>
cd bloco_suvaco_de_cristo

# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configurar Google Drive API

Siga as instruções em [`src/core/dataset/GOOGLE_DRIVE_SETUP.md`](src/core/dataset/GOOGLE_DRIVE_SETUP.md) para:
1. Criar service account no Google Cloud
2. Baixar `credentials.json`
3. Compartilhar pasta do Google Drive com a service account

### 3. Preparar Dataset

```bash
# Extrai hyperlinks do Excel, baixa arquivos e processa dados
python3 prepare_dataset.py
```

Este comando irá:
- ✅ Extrair hyperlinks do Excel
- ✅ Baixar todos os arquivos do Google Drive
- ✅ Processar palavras-chave (X → 1, vazio → 0)
- ✅ Mesclar tabelas
- ✅ Filtrar itens excluídos
- ✅ Gerar `dataset/dataset_completo.csv`

### 4. Classificar Imagens

```bash
# Classifica todas as imagens usando LLM multimodal
python3 classify_images.py
```

Este comando irá:
- ✅ Carregar dataset completo
- ✅ Processar cada imagem com IA
- ✅ Gerar descrições estruturadas
- ✅ Salvar resultados em JSON e CSV

## 📊 Outputs Gerados

### Dataset Completo
`dataset/dataset_completo.csv` - Dataset mesclado com:
- Metadados de catalogação
- Hyperlinks do Google Drive
- Caminhos locais dos arquivos
- Palavras-chave binárias
- Apenas itens válidos (Repositório ≠ 0)

### Classificações
`dataset/classifications_YYYYMMDD_HHMMSS.json` - Resultados completos em JSON
`dataset/classifications_YYYYMMDD_HHMMSS.csv` - Resultados tabulares

## ⚙️ Configuração

### config/prompt.yaml

Arquivo de configuração principal para classificação de imagens:

```yaml
# Modelo a usar
model:
  provider: "nvidia"  # nvidia, openai, gemini
  name: "microsoft/phi-4-multimodal-instruct"

# Parâmetros de geração
generation_config:
  temperature: 0.0
  max_output_tokens: 8000
  top_p: 0.95
```

Para customizar:
1. **Trocar modelo**: Altere `model.provider` e `model.name`
2. **Ajustar criatividade**: Modifique `temperature` (0.0-1.0)
3. **Modificar prompts**: Edite `system_prompt` e `user_prompt_template`

### .env (Opcional)

```bash
GOOGLE_SERVICE_ACCOUNT_JSON=credentials.json
```

## 🛠️ Módulos Principais

### src/core/dataset/
Módulo para preparação de dados:
- **extract_hyperlinks.py**: Extrai hyperlinks do Excel
- **download_and_process.py**: Baixa arquivos e processa dados
- **prepare_dataset.py**: Orquestra todo o processo

Veja [`src/core/dataset/README.md`](src/core/dataset/README.md)

### src/scripts/
Scripts de processamento:
- **classify_images.py**: Classifica imagens usando LLM multimodal

Veja [`src/scripts/README.md`](src/scripts/README.md)

### src/core/llm_predictor/
Módulo para integração com LLMs:
- Suporta NVIDIA, OpenAI, Gemini
- Factory pattern para fácil extensão

## 📈 Performance

### Paralelismo

Ambos os scripts suportam processamento paralelo:

**Download de arquivos:**
```python
max_workers=10  # Em download_and_process.py
```

**Classificação de imagens:**
```python
max_workers=4   # Em classify_images.py
```

### Estimativas de Tempo

**Preparação do dataset (1000+ arquivos):**
- Download: ~30-60 minutos (10 workers)
- Processamento: ~5 segundos

**Classificação (1000 imagens):**
- Phi-4 multimodal: ~5-8 horas (4 workers)
- GPT-4V: ~2-3 horas (4 workers)

## 🔍 Análise dos Resultados

### Com Pandas

```python
import pandas as pd

# Carregar classificações
df = pd.read_csv('dataset/classifications_20251008_090000.csv')

# Ver estatísticas
print(df['status'].value_counts())
print(df['epoca_estimada'].value_counts())

# Filtrar por ano
df_2023 = df[df['meta_ano'] == 2023]

# Buscar por palavra-chave
carnaval = df[df['palavras_chave'].str.contains('carnaval', case=False)]
```

### Com Excel

Abra os arquivos CSV diretamente no Excel para:
- Filtrar e ordenar
- Criar tabelas dinâmicas
- Gerar gráficos

## 🐛 Troubleshooting

### Erro: "Credenciais não encontradas"
- Verifique se `credentials.json` está na raiz
- Siga [`src/core/dataset/GOOGLE_DRIVE_SETUP.md`](src/core/dataset/GOOGLE_DRIVE_SETUP.md)

### Erro: "Permission denied" no download
- Compartilhe a pasta do Google Drive com o email da service account

### Classificação lenta
- Reduza `max_workers` se houver erros de memória
- Use GPU se disponível
- Considere modelo mais leve

### JSON inválido na resposta
- Simplifique o prompt em `config/prompt.yaml`
- Reduza `max_output_tokens`
- Teste com outro modelo

## 📝 Workflow Completo

```bash
# 1. Configurar credenciais
# Seguir GOOGLE_DRIVE_SETUP.md

# 2. Preparar dataset
python3 prepare_dataset.py

# 3. Classificar imagens
python3 classify_images.py

# 4. Analisar resultados
# Abrir CSVs com Pandas/Excel
```

## 🎯 Próximos Passos

- [ ] Interface web para navegação do acervo
- [ ] Busca semântica com embeddings
- [ ] Detecção automática de rostos
- [ ] Timeline interativa do acervo
- [ ] Exportação para formatos de arquivo
- [ ] API REST para consultas

## 📄 Licença

[Adicionar licença]

## 👥 Contribuindo

Contribuições são bem-vindas! Por favor:
1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📧 Contato

[Adicionar informações de contato]

---

Desenvolvido com ❤️ para preservar a história do Bloco Suvaco de Cristo 🎭
