# Configuração do Google Drive API

Este guia mostra como obter as credenciais necessárias para baixar arquivos do Google Drive usando uma **Service Account**.

## Passo 1: Criar Projeto no Google Cloud Console

1. Acesse [Google Cloud Console](https://console.cloud.google.com/)
2. Faça login com sua conta Google
3. Clique em **"Select a project"** (topo da página) → **"NEW PROJECT"**
4. Nome do projeto: `bloco-suvaco-cristo` (ou qualquer nome)
5. Clique em **"CREATE"**

## Passo 2: Ativar a Google Drive API

1. No menu lateral, vá em **"APIs & Services"** → **"Library"**
2. Busque por **"Google Drive API"**
3. Clique na API e depois em **"ENABLE"**

## Passo 3: Criar Service Account

1. No menu lateral, vá em **"APIs & Services"** → **"Credentials"**
2. Clique em **"CREATE CREDENTIALS"** → **"Service account"**
3. Preencha os campos:
   - **Service account name**: `bloco-suvaco-drive-reader`
   - **Service account ID**: (será preenchido automaticamente)
   - **Description**: `Service account para ler arquivos do Google Drive`
4. Clique em **"CREATE AND CONTINUE"**
5. Na seção "Grant this service account access to project" (opcional), clique em **"CONTINUE"**
6. Na seção "Grant users access to this service account" (opcional), clique em **"DONE"**

## Passo 4: Criar e Baixar a Chave JSON

1. Na página de **Credentials**, você verá sua service account listada em "Service Accounts"
2. Clique no **email da service account** (ex: `bloco-suvaco-drive-reader@...`)
3. Vá para a aba **"KEYS"**
4. Clique em **"ADD KEY"** → **"Create new key"**
5. Selecione **"JSON"** como tipo de chave
6. Clique em **"CREATE"**
7. Um arquivo JSON será baixado automaticamente para seu computador

**IMPORTANTE**: Esse arquivo JSON contém as credenciais da sua service account. Guarde-o em segurança!

## Passo 5: Compartilhar o Google Drive com a Service Account

Para que a service account possa acessar os arquivos do Google Drive, você precisa compartilhar a pasta/arquivos com ela:

1. Abra o **email da service account** que você copiou (algo como `bloco-suvaco-drive-reader@seu-projeto.iam.gserviceaccount.com`)
2. No **Google Drive**, vá até a pasta que contém os arquivos do acervo
3. Clique com o botão direito na pasta → **"Share"** (Compartilhar)
4. Cole o **email da service account** no campo de compartilhamento
5. Defina as permissões como **"Viewer"** (Leitor)
6. Clique em **"Share"** (Compartilhar)

Alternativamente, se os arquivos já forem públicos ou estiverem em um Drive compartilhado, você não precisa compartilhar explicitamente.

## Passo 6: Configurar o Projeto

1. **Copie o arquivo JSON baixado** para a raiz do projeto:
   ```bash
   mv ~/Downloads/seu-projeto-xxxxx.json /caminho/para/projeto/credentials.json
   ```

2. **(Opcional) Configure o arquivo `.env`** se quiser usar um nome diferente:
   ```bash
   cp .env.example .env
   ```

   Edite o `.env` e adicione:
   ```bash
   GOOGLE_SERVICE_ACCOUNT_JSON=credentials.json
   ```

## Passo 7: Testar a Autenticação

Execute o script:
```bash
python3 src/data_loader.py
```

O que vai acontecer:
1. ✅ O script vai autenticar automaticamente usando a service account
2. ✅ Não é necessário abrir o navegador ou fazer login
3. ✅ Os arquivos começarão a ser baixados imediatamente!

## Estrutura de Arquivos Após Configuração

```
.
├── .env                          # Configurações (opcional)
├── credentials.json              # Credenciais da service account (NÃO commitar!)
├── dataset/
│   ├── Descrição do acervo.xlsx
│   └── downloaded/              # Arquivos baixados do Drive
│       ├── arquivo1.jpg
│       ├── arquivo2.mp4
│       └── ...
```

## Troubleshooting

### Erro: "Google Drive service account credentials not found"
- Verifique se o arquivo `credentials.json` está na raiz do projeto
- Se você usou um nome diferente, configure a variável `GOOGLE_SERVICE_ACCOUNT_JSON` no `.env`

### Erro: "Permission denied" ou "File not found"
- Verifique se você compartilhou a pasta/arquivos do Google Drive com o email da service account
- Confirme que a permissão é pelo menos "Viewer" (Leitor)
- Verifique se os links no Excel são válidos e apontam para arquivos acessíveis

### Erro: "Invalid JSON credentials"
- Certifique-se de que baixou o arquivo JSON correto do Google Cloud Console
- Verifique se o arquivo não está corrompido (deve ser um JSON válido)

### Erro ao baixar arquivos
- Verifique se a Google Drive API está habilitada no seu projeto
- Confirme que os links no Excel são válidos
- Verifique se você tem cota suficiente na Google Cloud (normalmente não é problema para uso pessoal)

## Notas de Segurança

⚠️ **IMPORTANTE**:
- **NUNCA** commit o arquivo `credentials.json` no git
- **NUNCA** compartilhe o arquivo JSON da service account publicamente
- O arquivo `credentials.json` já está no `.gitignore`
- A service account tem acesso apenas de leitura ao Google Drive

## Vantagens da Service Account

✅ **Sem autenticação manual**: Não precisa abrir navegador ou fazer login
✅ **Automação fácil**: Perfeito para scripts e CI/CD
✅ **Mais seguro**: Credenciais isoladas da sua conta pessoal
✅ **Controle de acesso**: Pode revogar acesso a qualquer momento

## Próximos Passos

Após a configuração, você pode:
- Executar o script para baixar todos os arquivos automaticamente
- Usar os dados para treinar modelos
- Fazer análise exploratória do acervo
- Integrar com pipelines de CI/CD
