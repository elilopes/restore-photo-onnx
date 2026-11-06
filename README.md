# restore-photo-onnx

Sistema de restauração de fotos antigas com interface web React + Node.js + worker Python com modelos ONNX.

**O que faz:**
- Remove rasgos, riscos e manchas (modelo LaMa ou fallback OpenCV)
- Coloriza fotos em preto e branco (modelo DDColor)
- Recupera cores desbotadas (CLAHE + vibrance no espaço Lab)
- Amplia a resolução em 4× (modelo Real-ESRGAN)
- Aumenta a nitidez (unsharp mask no canal de luminância)

---

## Pré-requisitos

| Componente | Versão mínima |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| Redis | 7+ (ou Memurai no Windows — veja abaixo) |

**GPU (opcional):** instale `onnxruntime-gpu` no lugar de `onnxruntime` no `worker/requirements.txt` e defina `ORT_PROVIDERS=CUDAExecutionProvider` no `.env`.

---

## Instalação sem Docker (Windows / PowerShell)

### 1. Copiar o projeto

Extraia o ZIP em `C:\restore-photo-onnx` (ou qualquer pasta). Abra o PowerShell nessa pasta.

### 2. Redis (fila de jobs)

O sistema usa Redis para a fila BullMQ. No Windows, a opção mais simples é o **Memurai** (compatível com Redis):

1. Baixe o instalador em <https://www.memurai.com/get-memurai>
2. Instale e deixe o serviço rodando (porta padrão 6379)
3. Teste: `memurai-cli ping` → deve retornar `PONG`

Alternativa: WSL2 com `apt install redis-server` e `redis-server --daemonize yes`.

### 3. Worker Python

```powershell
cd worker
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. API Node.js

```powershell
cd ..\api
npm install
npm run build
```

### 5. Frontend

```powershell
cd ..\web
npm install
npm run build
```

### 6. Configuração

Copie `.env.example` para `.env` na raiz e ajuste se necessário:

```ini
WEB_PORT=8080
MAX_UPLOAD_MB=25
RETENTION_HOURS=24
ORT_PROVIDERS=CPUExecutionProvider
MAX_CONCURRENCY=1
QUEUE_CONCURRENCY=1
```

---

## Download dos modelos ONNX

Os modelos **não estão incluídos** no projeto (são arquivos grandes). Você pode baixá-los de duas formas:

### Opção A — Pela interface web (recomendado)

1. Inicie o sistema (veja "Rodando o sistema" abaixo)
2. Acesse `http://localhost:5173` no navegador
3. Role a página até a seção **Modelos de IA**
4. Clique em **Baixar modelos automaticamente**

O download acontece em background. A página mostra o progresso de cada modelo.

### Opção B — Pelo terminal

```powershell
cd C:\restore-photo-onnx\worker
.venv\Scripts\Activate.ps1
python ..\scripts\download_models.py
```

Parâmetros opcionais:
```powershell
# Baixar só um modelo
python ..\scripts\download_models.py --only inpaint
python ..\scripts\download_models.py --only colorize
python ..\scripts\download_models.py --only superres

# Forçar re-download mesmo que o arquivo já exista
python ..\scripts\download_models.py --force
```

### Opção C — Download manual

Baixe os arquivos abaixo e salve na pasta `C:\restore-photo-onnx\models\` com os nomes **exatos**:

| Arquivo | Função | Tamanho | Link |
|---|---|---|---|
| `lama_fp32.onnx` | Remove rasgos e riscos | ~200 MB | [Hugging Face](https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx) |
| `ddcolor-tiny-fp16.onnx` | Colorização P&B | ~60 MB | [Hugging Face](https://huggingface.co/edgetools/ddcolor/resolve/main/ddcolor-tiny-fp16.onnx) |
| `realesrgan_x4plus.onnx` | Ampliação 4× | ~65 MB | [Hugging Face](https://huggingface.co/Qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx) |

> **Atenção:** as URLs do Hugging Face são de terceiros e podem mudar. Se algum download falhar (erro 404), verifique o repositório correspondente no Hugging Face para a URL atualizada e edite `scripts/download_models.py`.

O sistema funciona **sem os modelos** usando o pipeline clássico (OpenCV) como fallback — os resultados são mais simples, mas a restauração básica continua funcionando.

---

## Rodando o sistema

Abra **três terminais** (PowerShell) na pasta `C:\restore-photo-onnx`:

### Terminal 1 — Worker Python

```powershell
cd worker
.venv\Scripts\Activate.ps1
uvicorn app.main:app --port 8001 --reload
```

### Terminal 2 — API + fila Node.js

```powershell
cd api
node dist/server.js &       # API Fastify na porta 3001
node dist/worker.js         # Worker BullMQ (consome a fila Redis)
```

Ou em terminais separados:
```powershell
# Terminal 2a
cd api && node dist/server.js

# Terminal 2b
cd api && node dist/worker.js
```

### Terminal 3 — Frontend (desenvolvimento)

```powershell
cd web
npm run dev
```

Acesse `http://localhost:5173` no navegador.

---

## Rodando com Docker (Windows 10/11 com WSL2)

Requisitos:
- Windows 10 22H2+ ou Windows 11 64-bit
- Docker Desktop instalado (<https://www.docker.com/products/docker-desktop/>)
- Virtualização habilitada na BIOS (Intel VT-x ou AMD-V)

```powershell
# Na raiz do projeto — baixar modelos primeiro:
docker compose --profile setup run --rm models

# Iniciar o sistema:
docker compose up
```

Acesse `http://localhost:8080`.

---

## Estrutura do projeto

```
restore-photo-onnx/
├── api/          Node.js (Fastify + BullMQ) — gateway HTTP e fila
├── worker/       Python (FastAPI + onnxruntime + OpenCV) — inferência
├── web/          React + Vite — interface do usuário
├── scripts/      download_models.py — utilitário de download dos modelos
├── storage/      uploads e resultados (criado automaticamente)
├── models/       modelos ONNX (criado automaticamente; não versionado)
├── docker-compose.yml
└── .env.example
```

---

## Testes

```powershell
# Worker Python (47 testes)
cd worker && .venv\Scripts\Activate.ps1
python -m pytest tests/ -v

# API Node.js (24 testes)
cd api && npm test

# Frontend React (19 testes)
cd web && npm test
```

---

## Configurações avançadas

| Variável de ambiente | Padrão | Descrição |
|---|---|---|
| `ORT_PROVIDERS` | `CPUExecutionProvider` | `CUDAExecutionProvider` para GPU NVIDIA |
| `MAX_CONCURRENCY` | `1` | Jobs simultâneos no worker Python |
| `MAX_UPLOAD_MB` | `25` | Tamanho máximo do upload |
| `RETENTION_HOURS` | `24` | Horas até os arquivos serem apagados |
| `WORKER_URL` | `http://localhost:8001` | URL do worker Python vista pela API |
| `REDIS_URL` | `redis://localhost:6379` | URL do Redis/Memurai |

---

## Licença

MIT
