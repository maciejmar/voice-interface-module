# Voice Gateway

Reużywalny mikroserwis głosowy dla agentów AI. Realizuje dwukierunkowy pipeline audio–tekst–audio bez modyfikowania istniejących agentów.

## Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                        Voice Gateway                        │
│                                                             │
│  audio in  ──► STT ──► query text ──► Agent HTTP POST      │
│  (WAV/MP3)    (Whisper              (FastAPI, LangGraph…)   │
│               large-v3)                    │                │
│                                     answer text             │
│                                            │                │
│  audio out ◄── TTS ◄────────────────────────               │
│  (WAV)        (Piper)                                       │
└─────────────────────────────────────────────────────────────┘
```

## Decyzje architektoniczne

| Pytanie | Decyzja | Uzasadnienie |
|---------|---------|--------------|
| Dlaczego osobny mikroserwis? | HTTP POST do agentów | Zero ingerencji w kod agentów; obsługuje binarne strumienie audio |
| Dlaczego nie MCP? | MCP udostępnia narzędzia **dla** agentów, nie jest frontend-em | Odwrócona rola — głos to interfejs użytkownika, nie narzędzie agenta |
| Dlaczego nie node LangGraph? | Wiąże z konkretną wersją grafu; nie obsługuje plików binarnych | Reużywalność: jeden serwis dla N agentów |
| STT: Whisper large-v3 | Najlepsza jakość języka polskiego | Działa na GPU; offline; CTranslate2 = szybko |
| TTS: Piper | VITS, offline, CPU | Nie zajmuje VRAM; native Polish voices |

## Quick-start

### 1. Pobierz modele (maszyna z internetem)

```bash
bash download_models.sh
# modele lądują w ./models/stt/ i ./models/tts/
```

### 2. Uruchom

```bash
# utwórz sieć jeśli nie istnieje
docker network create ai-network

docker compose up --build
```

Serwis startuje na porcie `8100`. Pierwsze uruchomienie trwa ~60 s (ładowanie Whisper).

### 3. Sprawdź

```bash
curl http://localhost:8100/health
curl http://localhost:8100/agents
```

---

## Air-gapped (środowisko bez internetu)

### Zmień bazowy obraz Docker

W `Dockerfile` zamień pierwszą linię:

```dockerfile
FROM repo.bank.com.pl/docker/nvidia/cuda:12.4.1-runtime-ubuntu22.04
```

Usuń też blok `RUN wget ...` pobierający modele TTS w Dockerfile (modele będą z volumenu).

### Eksport/import obrazu Docker

```bash
# na maszynie z internetem
docker build -t voice-gateway:latest .
docker save voice-gateway:latest | gzip > voice-gateway.tar.gz
scp voice-gateway.tar.gz user@server:/data/apps/voice-gateway/

# na serwerze air-gapped
docker load < voice-gateway.tar.gz
```

### Offline pip cache

```bash
# na maszynie z internetem
pip download -r app/requirements.txt -d ./pip-cache
scp -r pip-cache user@server:/data/apps/voice-gateway/

# w Dockerfile (air-gapped)
COPY pip-cache /pip-cache
RUN pip3 install --no-index --find-links=/pip-cache -r requirements.txt
```

### Skopiuj modele

```bash
scp -r ./models/ user@server:/data/apps/voice-gateway/
```

---

## API Reference

### `GET /health`

```bash
curl http://localhost:8100/health
```

```json
{
  "status": "ok",
  "stt_model_loaded": true,
  "stt_model_size": "large-v3",
  "stt_device": "cuda",
  "piper_binary_exists": true,
  "piper_model_exists": true,
  "agents": ["gacek", "sufler"]
}
```

---

### `GET /agents`

```bash
curl http://localhost:8100/agents
```

---

### `POST /stt` — tylko rozpoznawanie mowy

```bash
curl -X POST http://localhost:8100/stt \
  -F "audio=@nagranie.wav"
```

```json
{"transcription": "Jakie masz godziny otwarcia?"}
```

---

### `POST /tts` — tylko synteza mowy

```bash
curl -X POST http://localhost:8100/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Dzień dobry, w czym mogę pomóc?"}' \
  --output odpowiedz.wav
```

---

### `POST /voice/{agent_name}` — pełny pipeline głosowy

```bash
curl -X POST http://localhost:8100/voice/gacek \
  -F "audio=@pytanie.wav" \
  --output odpowiedz.wav \
  -D -   # pokaż nagłówki z X-Transcription i X-Agent-Response
```

Nagłówki odpowiedzi:
- `X-Transcription` — rozpoznany tekst
- `X-Agent-Response` — tekstowa odpowiedź agenta (max 500 znaków)
- `X-Agent-Name` — nazwa użytego agenta

---

### `POST /voice/{agent_name}/json` — pipeline z JSON

```bash
curl -X POST http://localhost:8100/voice/gacek/json \
  -F "audio=@pytanie.wav"
```

```json
{
  "transcription": "Co to jest RAG?",
  "agent_name": "gacek",
  "agent_response": {"answer": "RAG to..."},
  "answer_text": "RAG to...",
  "audio_base64": "UklGRi...",
  "audio_content_type": "audio/wav"
}
```

---

### `POST /text/{agent_name}` — tekst → agent → audio

```bash
curl -X POST http://localhost:8100/text/sufler \
  -H "Content-Type: application/json" \
  -d '{"text": "Podsumuj ostatni raport"}' \
  --output odpowiedz.wav
```

---

### `WebSocket /ws/voice/{agent_name}` — streaming real-time

```python
import asyncio, websockets

async def main():
    uri = "ws://localhost:8100/ws/voice/gacek"
    async with websockets.connect(uri) as ws:
        with open("pytanie.wav", "rb") as f:
            await ws.send(f.read())   # binary frame(s)
        await ws.send("END")          # sygnał końca

        async for msg in ws:
            if isinstance(msg, str):
                print(msg)  # {"type": "transcription", "text": "..."}
            else:
                with open("odpowiedz.wav", "wb") as f:
                    f.write(msg)      # WAV audio

asyncio.run(main())
```

Sekwencja wiadomości od serwera:
1. `{"type": "transcription", "text": "..."}`
2. `{"type": "agent_response", "text": "...", "full_response": {...}}`
3. `<binary WAV>`

---

## Wymagania dla agenta AI

Aby agent mógł korzystać z Voice Gateway, musi spełniać poniższe wymagania.

### Transport
- Dostępny w sieci Docker `ai-network`
- Nasłuchuje na dowolnym porcie TCP

### Endpoint
- `POST /api/invoke` (lub inny URL zarejestrowany w `AGENTS_CONFIG`)
- Akceptuje `Content-Type: application/json`

### Format zapytania
```json
{"query": "tekst pytania użytkownika"}
```

### Format odpowiedzi
- Status HTTP `200`
- `Content-Type: application/json`
- Odpowiedź pod jednym z kluczy (Voice Gateway sprawdza w tej kolejności):

```json
{"answer": "..."}
{"response": "..."}
{"output": "..."}
{"result": "..."}
{"content": "..."}
{"message": "..."}
{"text": "..."}
```

Wartość może być stringiem lub obiektem z pod-kluczem `text`, `content`, `answer` lub `message`.

### Timeout
- Agent musi odpowiedzieć w ciągu **120 sekund** (konfigurowalny przez `AGENT_TIMEOUT`)

---

## Podłączenie nowego agenta

Edytuj zmienną `AGENTS_CONFIG` w `docker-compose.yml`:

```yaml
environment:
  AGENTS_CONFIG: |
    {
      "moj-agent": {
        "url": "http://moj-agent:8000/api/invoke",
        "description": "Opis mojego agenta"
      }
    }
```

**Zero zmian w kodzie.** Restart serwisu:

```bash
docker compose up -d
```

---

## Zasoby GPU

| Komponent | Urządzenie | Zużycie VRAM |
|-----------|-----------|--------------|
| Whisper large-v3 (STT) | GPU (CUDA) | ~3 GB |
| Piper (TTS) | CPU | 0 MB |
| **Łącznie** | | **~3 GB** |

Serwer H100 NVL (94 GB VRAM) ma duży zapas na równoległe agenty LLM.
