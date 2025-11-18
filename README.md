# Bloco Suvaco de Cristo - Archive Cataloging

**[English](#english) | [Português](#português) | [Español](#español)**

---

## English

Project for cataloging, processing, and analysis of the historical archive of Bloco Suvaco de Cristo using AI.

### 🎭 About the Project

This project automates the cataloging and analysis of images, videos, and documents from the historical archive of Bloco Suvaco de Cristo, a traditional carnival block from Rio de Janeiro.

It uses multimodal language models (Vision Language Models) to:
- Generate detailed image descriptions
- Identify cultural and historical elements
- Automatically extract metadata
- Organize and index the archive

### 📁 Project Structure

```
.
├── config/
│   └── prompt.yaml                  # Prompts and model configuration
├── dataset/
│   ├── Descrição do acervo.xlsx    # Original Excel file
│   ├── catalogacao_with_hyperlinks.csv
│   ├── palavras_chave_processed.csv
│   ├── dataset_completo.csv        # Final dataset
│   └── downloaded/                  # Downloaded images
├── src/
│   ├── core/
│   │   ├── dataset/                # Dataset preparation module
│   │   │   ├── extract_hyperlinks.py
│   │   │   ├── download_and_process.py
│   │   │   ├── prepare_dataset.py
│   │   │   └── GOOGLE_DRIVE_SETUP.md
│   │   └── llm_predictor/          # LLM prediction module
│   │       ├── llm_factory.py
│   │       ├── nvidia_predictor.py
│   │       ├── gemini_predictor.py
│   │       └── openai_predictor.py
│   └── scripts/                    # Processing scripts
│       ├── classify_images.py      # Image classification
│       └── README.md
├── prepare_dataset.py              # Main preparation script
├── classify_images.py              # Main classification script
└── README.md                       # This file
```

### 🚀 Quick Start

#### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd bloco_suvaco_de_cristo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 2. Configure Google Drive API

Follow the instructions in [`src/core/dataset/GOOGLE_DRIVE_SETUP.md`](src/core/dataset/GOOGLE_DRIVE_SETUP.md) to:
1. Create a service account in Google Cloud
2. Download `credentials.json`
3. Share Google Drive folder with the service account

#### 3. Prepare Dataset

```bash
# Extract hyperlinks from Excel, download files and process data
python3 prepare_dataset.py
```

This command will:
- ✅ Extract hyperlinks from Excel
- ✅ Download all files from Google Drive
- ✅ Process keywords (X → 1, empty → 0)
- ✅ Merge tables
- ✅ Filter excluded items
- ✅ Generate `dataset/dataset_completo.csv`

#### 4. Classify Images

```bash
# Classify all images using multimodal LLM
python3 classify_images.py
```

This command will:
- ✅ Load complete dataset
- ✅ Process each image with AI
- ✅ Generate structured descriptions
- ✅ Save results in JSON and CSV

### 📊 Generated Outputs

#### Complete Dataset
`dataset/dataset_completo.csv` - Merged dataset with:
- Cataloging metadata
- Google Drive hyperlinks
- Local file paths
- Binary keywords
- Only valid items (Repository ≠ 0)

#### Classifications
`dataset/classifications_YYYYMMDD_HHMMSS.json` - Complete results in JSON
`dataset/classifications_YYYYMMDD_HHMMSS.csv` - Tabular results

### ⚙️ Configuration

#### config/prompt.yaml

Main configuration file for image classification:

```yaml
# Model to use
model:
  provider: "nvidia"  # nvidia, openai, gemini
  name: "microsoft/phi-4-multimodal-instruct"

# Generation parameters
generation_config:
  temperature: 0.0
  max_output_tokens: 8000
  top_p: 0.95
```

To customize:
1. **Change model**: Modify `model.provider` and `model.name`
2. **Adjust creativity**: Modify `temperature` (0.0-1.0)
3. **Modify prompts**: Edit `system_prompt` and `user_prompt_template`

#### .env (Optional)

```bash
GOOGLE_SERVICE_ACCOUNT_JSON=credentials.json
```

### 🛠️ Main Modules

#### src/core/dataset/
Module for data preparation:
- **extract_hyperlinks.py**: Extract hyperlinks from Excel
- **download_and_process.py**: Download files and process data
- **prepare_dataset.py**: Orchestrate entire process

See [`src/core/dataset/README.md`](src/core/dataset/README.md)

#### src/scripts/
Processing scripts:
- **classify_images.py**: Classify images using multimodal LLM

See [`src/scripts/README.md`](src/scripts/README.md)

#### src/core/llm_predictor/
Module for LLM integration:
- Supports NVIDIA, OpenAI, Gemini
- Factory pattern for easy extension

### 📈 Performance

#### Parallelism

Both scripts support parallel processing:

**File download:**
```python
max_workers=10  # In download_and_process.py
```

**Image classification:**
```python
max_workers=4   # In classify_images.py
```

#### Time Estimates

**Dataset preparation (1000+ files):**
- Download: ~30-60 minutes (10 workers)
- Processing: ~5 seconds

**Classification (1000 images):**
- Phi-4 multimodal: ~5-8 hours (4 workers)
- GPT-4V: ~2-3 hours (4 workers)

### 🔍 Analyzing Results

#### With Pandas

```python
import pandas as pd

# Load classifications
df = pd.read_csv('dataset/classifications_20251008_090000.csv')

# View statistics
print(df['status'].value_counts())
print(df['epoca_estimada'].value_counts())

# Filter by year
df_2023 = df[df['meta_ano'] == 2023]

# Search by keyword
carnaval = df[df['palavras_chave'].str.contains('carnaval', case=False)]
```

#### With Excel

Open CSV files directly in Excel to:
- Filter and sort
- Create pivot tables
- Generate charts

### 🐛 Troubleshooting

#### Error: "Credentials not found"
- Verify `credentials.json` is in the root
- Follow [`src/core/dataset/GOOGLE_DRIVE_SETUP.md`](src/core/dataset/GOOGLE_DRIVE_SETUP.md)

#### Error: "Permission denied" on download
- Share Google Drive folder with service account email

#### Slow classification
- Reduce `max_workers` if memory errors occur
- Use GPU if available
- Consider lighter model

#### Invalid JSON in response
- Simplify prompt in `config/prompt.yaml`
- Reduce `max_output_tokens`
- Test with another model

### 📝 Complete Workflow

```bash
# 1. Configure credentials
# Follow GOOGLE_DRIVE_SETUP.md

# 2. Prepare dataset
python3 prepare_dataset.py

# 3. Classify images
python3 classify_images.py

# 4. Analyze results
# Open CSVs with Pandas/Excel
```

### 🎯 Next Steps

- [ ] Web interface for archive navigation
- [ ] Semantic search with embeddings
- [ ] Automatic face detection
- [ ] Interactive archive timeline
- [ ] Export to archive formats
- [ ] REST API for queries

### 📄 License

[Add license]

### 👥 Contributing

Contributions are welcome! Please:
1. Fork the project
2. Create a branch for your feature
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

### 📧 Contact

[Add contact information]

---

Developed with ❤️ to preserve the history of Bloco Suvaco de Cristo 🎭

---

## Português

Projeto para catalogação, processamento e análise do acervo histórico do Bloco Suvaco de Cristo usando IA.

### 🎭 Sobre o Projeto

Este projeto automatiza a catalogação e análise de imagens, vídeos e documentos do acervo histórico do Bloco Suvaco de Cristo, um tradicional bloco de carnaval do Rio de Janeiro.

Utiliza modelos de linguagem multimodais (Vision Language Models) para:
- Gerar descrições detalhadas das imagens
- Identificar elementos culturais e históricos
- Extrair metadados automaticamente
- Organizar e indexar o acervo

### 📁 Estrutura do Projeto

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

### 🚀 Quick Start

#### 1. Instalação

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

#### 2. Configurar Google Drive API

Siga as instruções em [`src/core/dataset/GOOGLE_DRIVE_SETUP.md`](src/core/dataset/GOOGLE_DRIVE_SETUP.md) para:
1. Criar service account no Google Cloud
2. Baixar `credentials.json`
3. Compartilhar pasta do Google Drive com a service account

#### 3. Preparar Dataset

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

#### 4. Classificar Imagens

```bash
# Classifica todas as imagens usando LLM multimodal
python3 classify_images.py
```

Este comando irá:
- ✅ Carregar dataset completo
- ✅ Processar cada imagem com IA
- ✅ Gerar descrições estruturadas
- ✅ Salvar resultados em JSON e CSV

### 📊 Outputs Gerados

#### Dataset Completo
`dataset/dataset_completo.csv` - Dataset mesclado com:
- Metadados de catalogação
- Hyperlinks do Google Drive
- Caminhos locais dos arquivos
- Palavras-chave binárias
- Apenas itens válidos (Repositório ≠ 0)

#### Classificações
`dataset/classifications_YYYYMMDD_HHMMSS.json` - Resultados completos em JSON
`dataset/classifications_YYYYMMDD_HHMMSS.csv` - Resultados tabulares

### ⚙️ Configuração

#### config/prompt.yaml

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

#### .env (Opcional)

```bash
GOOGLE_SERVICE_ACCOUNT_JSON=credentials.json
```

### 🛠️ Módulos Principais

#### src/core/dataset/
Módulo para preparação de dados:
- **extract_hyperlinks.py**: Extrai hyperlinks do Excel
- **download_and_process.py**: Baixa arquivos e processa dados
- **prepare_dataset.py**: Orquestra todo o processo

Veja [`src/core/dataset/README.md`](src/core/dataset/README.md)

#### src/scripts/
Scripts de processamento:
- **classify_images.py**: Classifica imagens usando LLM multimodal

Veja [`src/scripts/README.md`](src/scripts/README.md)

#### src/core/llm_predictor/
Módulo para integração com LLMs:
- Suporta NVIDIA, OpenAI, Gemini
- Factory pattern para fácil extensão

### 📈 Performance

#### Paralelismo

Ambos os scripts suportam processamento paralelo:

**Download de arquivos:**
```python
max_workers=10  # Em download_and_process.py
```

**Classificação de imagens:**
```python
max_workers=4   # Em classify_images.py
```

#### Estimativas de Tempo

**Preparação do dataset (1000+ arquivos):**
- Download: ~30-60 minutos (10 workers)
- Processamento: ~5 segundos

**Classificação (1000 imagens):**
- Phi-4 multimodal: ~5-8 horas (4 workers)
- GPT-4V: ~2-3 horas (4 workers)

### 🔍 Análise dos Resultados

#### Com Pandas

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

#### Com Excel

Abra os arquivos CSV diretamente no Excel para:
- Filtrar e ordenar
- Criar tabelas dinâmicas
- Gerar gráficos

### 🐛 Troubleshooting

#### Erro: "Credenciais não encontradas"
- Verifique se `credentials.json` está na raiz
- Siga [`src/core/dataset/GOOGLE_DRIVE_SETUP.md`](src/core/dataset/GOOGLE_DRIVE_SETUP.md)

#### Erro: "Permission denied" no download
- Compartilhe a pasta do Google Drive com o email da service account

#### Classificação lenta
- Reduza `max_workers` se houver erros de memória
- Use GPU se disponível
- Considere modelo mais leve

#### JSON inválido na resposta
- Simplifique o prompt em `config/prompt.yaml`
- Reduza `max_output_tokens`
- Teste com outro modelo

### 📝 Workflow Completo

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

### 🎯 Próximos Passos

- [ ] Interface web para navegação do acervo
- [ ] Busca semântica com embeddings
- [ ] Detecção automática de rostos
- [ ] Timeline interativa do acervo
- [ ] Exportação para formatos de arquivo
- [ ] API REST para consultas

### 📄 Licença

[Adicionar licença]

### 👥 Contribuindo

Contribuições são bem-vindas! Por favor:
1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

### 📧 Contato

[Adicionar informações de contato]

---

Desenvolvido com ❤️ para preservar a história do Bloco Suvaco de Cristo 🎭

---

## Español

Proyecto para catalogación, procesamiento y análisis del acervo histórico del Bloco Suvaco de Cristo usando IA.

### 🎭 Sobre el Proyecto

Este proyecto automatiza la catalogación y análisis de imágenes, videos y documentos del acervo histórico del Bloco Suvaco de Cristo, un tradicional bloque de carnaval de Río de Janeiro.

Utiliza modelos de lenguaje multimodales (Vision Language Models) para:
- Generar descripciones detalladas de las imágenes
- Identificar elementos culturales e históricos
- Extraer metadatos automáticamente
- Organizar e indexar el acervo

### 📁 Estructura del Proyecto

```
.
├── config/
│   └── prompt.yaml                  # Configuración de prompts y modelo
├── dataset/
│   ├── Descrição do acervo.xlsx    # Archivo Excel original
│   ├── catalogacao_with_hyperlinks.csv
│   ├── palavras_chave_processed.csv
│   ├── dataset_completo.csv        # Dataset final
│   └── downloaded/                  # Imágenes descargadas
├── src/
│   ├── core/
│   │   ├── dataset/                # Módulo de preparación de dataset
│   │   │   ├── extract_hyperlinks.py
│   │   │   ├── download_and_process.py
│   │   │   ├── prepare_dataset.py
│   │   │   └── GOOGLE_DRIVE_SETUP.md
│   │   └── llm_predictor/          # Módulo de predicción LLM
│   │       ├── llm_factory.py
│   │       ├── nvidia_predictor.py
│   │       ├── gemini_predictor.py
│   │       └── openai_predictor.py
│   └── scripts/                    # Scripts de procesamiento
│       ├── classify_images.py      # Clasificación de imágenes
│       └── README.md
├── prepare_dataset.py              # Script principal de preparación
├── classify_images.py              # Script principal de clasificación
└── README.md                       # Este archivo
```

### 🚀 Inicio Rápido

#### 1. Instalación

```bash
# Clonar repositorio
git clone <url-del-repo>
cd bloco_suvaco_de_cristo

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

#### 2. Configurar Google Drive API

Siga las instrucciones en [`src/core/dataset/GOOGLE_DRIVE_SETUP.md`](src/core/dataset/GOOGLE_DRIVE_SETUP.md) para:
1. Crear service account en Google Cloud
2. Descargar `credentials.json`
3. Compartir carpeta de Google Drive con la service account

#### 3. Preparar Dataset

```bash
# Extrae hyperlinks del Excel, descarga archivos y procesa datos
python3 prepare_dataset.py
```

Este comando:
- ✅ Extraerá hyperlinks del Excel
- ✅ Descargará todos los archivos de Google Drive
- ✅ Procesará palabras clave (X → 1, vacío → 0)
- ✅ Fusionará tablas
- ✅ Filtrará elementos excluidos
- ✅ Generará `dataset/dataset_completo.csv`

#### 4. Clasificar Imágenes

```bash
# Clasifica todas las imágenes usando LLM multimodal
python3 classify_images.py
```

Este comando:
- ✅ Cargará dataset completo
- ✅ Procesará cada imagen con IA
- ✅ Generará descripciones estructuradas
- ✅ Guardará resultados en JSON y CSV

### 📊 Resultados Generados

#### Dataset Completo
`dataset/dataset_completo.csv` - Dataset fusionado con:
- Metadatos de catalogación
- Hyperlinks de Google Drive
- Rutas locales de los archivos
- Palabras clave binarias
- Solo elementos válidos (Repositorio ≠ 0)

#### Clasificaciones
`dataset/classifications_YYYYMMDD_HHMMSS.json` - Resultados completos en JSON
`dataset/classifications_YYYYMMDD_HHMMSS.csv` - Resultados tabulares

### ⚙️ Configuración

#### config/prompt.yaml

Archivo de configuración principal para clasificación de imágenes:

```yaml
# Modelo a usar
model:
  provider: "nvidia"  # nvidia, openai, gemini
  name: "microsoft/phi-4-multimodal-instruct"

# Parámetros de generación
generation_config:
  temperature: 0.0
  max_output_tokens: 8000
  top_p: 0.95
```

Para personalizar:
1. **Cambiar modelo**: Modifique `model.provider` y `model.name`
2. **Ajustar creatividad**: Modifique `temperature` (0.0-1.0)
3. **Modificar prompts**: Edite `system_prompt` y `user_prompt_template`

#### .env (Opcional)

```bash
GOOGLE_SERVICE_ACCOUNT_JSON=credentials.json
```

### 🛠️ Módulos Principales

#### src/core/dataset/
Módulo para preparación de datos:
- **extract_hyperlinks.py**: Extrae hyperlinks del Excel
- **download_and_process.py**: Descarga archivos y procesa datos
- **prepare_dataset.py**: Orquesta todo el proceso

Ver [`src/core/dataset/README.md`](src/core/dataset/README.md)

#### src/scripts/
Scripts de procesamiento:
- **classify_images.py**: Clasifica imágenes usando LLM multimodal

Ver [`src/scripts/README.md`](src/scripts/README.md)

#### src/core/llm_predictor/
Módulo para integración con LLMs:
- Soporta NVIDIA, OpenAI, Gemini
- Patrón Factory para fácil extensión

### 📈 Rendimiento

#### Paralelismo

Ambos scripts soportan procesamiento paralelo:

**Descarga de archivos:**
```python
max_workers=10  # En download_and_process.py
```

**Clasificación de imágenes:**
```python
max_workers=4   # En classify_images.py
```

#### Estimaciones de Tiempo

**Preparación del dataset (1000+ archivos):**
- Descarga: ~30-60 minutos (10 workers)
- Procesamiento: ~5 segundos

**Clasificación (1000 imágenes):**
- Phi-4 multimodal: ~5-8 horas (4 workers)
- GPT-4V: ~2-3 horas (4 workers)

### 🔍 Análisis de Resultados

#### Con Pandas

```python
import pandas as pd

# Cargar clasificaciones
df = pd.read_csv('dataset/classifications_20251008_090000.csv')

# Ver estadísticas
print(df['status'].value_counts())
print(df['epoca_estimada'].value_counts())

# Filtrar por año
df_2023 = df[df['meta_ano'] == 2023]

# Buscar por palabra clave
carnaval = df[df['palavras_chave'].str.contains('carnaval', case=False)]
```

#### Con Excel

Abra los archivos CSV directamente en Excel para:
- Filtrar y ordenar
- Crear tablas dinámicas
- Generar gráficos

### 🐛 Solución de Problemas

#### Error: "Credenciales no encontradas"
- Verifique que `credentials.json` esté en la raíz
- Siga [`src/core/dataset/GOOGLE_DRIVE_SETUP.md`](src/core/dataset/GOOGLE_DRIVE_SETUP.md)

#### Error: "Permission denied" en descarga
- Comparta la carpeta de Google Drive con el email de la service account

#### Clasificación lenta
- Reduzca `max_workers` si hay errores de memoria
- Use GPU si está disponible
- Considere un modelo más ligero

#### JSON inválido en respuesta
- Simplifique el prompt en `config/prompt.yaml`
- Reduzca `max_output_tokens`
- Pruebe con otro modelo

### 📝 Flujo de Trabajo Completo

```bash
# 1. Configurar credenciales
# Seguir GOOGLE_DRIVE_SETUP.md

# 2. Preparar dataset
python3 prepare_dataset.py

# 3. Clasificar imágenes
python3 classify_images.py

# 4. Analizar resultados
# Abrir CSVs con Pandas/Excel
```

### 🎯 Próximos Pasos

- [ ] Interfaz web para navegación del acervo
- [ ] Búsqueda semántica con embeddings
- [ ] Detección automática de rostros
- [ ] Línea de tiempo interactiva del acervo
- [ ] Exportación a formatos de archivo
- [ ] API REST para consultas

### 📄 Licencia

[Agregar licencia]

### 👥 Contribuyendo

¡Las contribuciones son bienvenidas! Por favor:
1. Haga fork del proyecto
2. Cree una rama para su feature
3. Haga commit de sus cambios
4. Haga push a la rama
5. Abra un Pull Request

### 📧 Contacto

[Agregar información de contacto]

---

Desarrollado con ❤️ para preservar la historia del Bloco Suvaco de Cristo 🎭
