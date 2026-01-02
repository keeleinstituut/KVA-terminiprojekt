# KVA-terminiprojekt

## Seadistamine

Täida fail `.env` (vt `.env.example`).

Määra failis `docker-compose.yml`:
```yaml
volumes:
  - {pg_andmed}:/var/lib/postgresql/data
  - {qdrant_andmed}:/qdrant/storage
  - {credentials.json}:/app/config/credentials.json
```

Käivitamine:
```
docker compose up
```

Autentimiseks loo `credentials.json`. Süsteem toetab kahte rolli:
- `kylaline`: Piiratud ligipääs (ainult otsing)
- `admin` (või muu kasutaja): Täisligipääs (otsing, failide haldus, debug)

Näide:
```json
{"kylaline": "", "admin": "parool123"}
```

## Pipeline

```
┌───────────────────────────────────────────────────────────────────────────┐
│  SISEND: termin (nt "valvur"), limit=5                                    │
└─────────────────────┬─────────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  1. PÄRINGU LAIENDAMINE + OTSING (paralleelselt iga kategooria jaoks)     │
│                                                                           │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐  │
│  │   DEFINITSIOONID    │ │   SEOTUD TERMINID   │ │  KASUTUSKONTEKSTID  │  │
│  ├─────────────────────┤ ├─────────────────────┤ ├─────────────────────┤  │
│  │ LLM laiendab:       │ │ LLM laiendab:       │ │ LLM laiendab:       │  │
│  │ → "valvur defini-   │ │ → "valvur, vanem-   │ │ → "valvuri töö,     │  │
│  │    tsioon, tähendus"│ │    valvur, saatja"  │ │    ülesanded"       │  │
│  │         ↓           │ │         ↓           │ │         ↓           │  │
│  │ Otsing "valvur"     │ │ Otsing "valvur"     │ │ Otsing "valvur"     │  │
│  │ → 10 tulemust       │ │ → 10 tulemust       │ │ → 10 tulemust       │  │
│  │ + otsing iga        │ │ + otsing iga        │ │ + otsing iga        │  │
│  │   laiendatud        │ │   laiendatud        │ │   laiendatud        │  │
│  │   terminiga         │ │   terminiga         │ │   terminiga         │  │
│  │         ↓           │ │         ↓           │ │         ↓           │  │
│  │ Kombineeri (~20-30) │ │ Kombineeri (~20-30) │ │ Kombineeri (~20-30) │  │
│  │ + [Reranking/sort]  │ │ + [Reranking/sort]  │ │ + [Reranking/sort]  │  │
│  │         ↓           │ │         ↓           │ │         ↓           │  │
│  │ → 5 parimat lõiku   │ │ → 5 parimat lõiku   │ │ → 5 parimat lõiku   │  │
│  └──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘  │
│             │                       │                       │             │
│      definitsioonidele        terminitele             kasutusele          │
│       sobivad lõigud         sobivad lõigud          sobivad lõigud       │
└─────────────┼───────────────────────┼───────────────────────┼─────────────┘
              │                       │                       │
              ▼                       ▼                       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  2. PARALLEELNE LLM-ANALÜÜS (iga kategooria oma lõikudega)                │
│                                                                           │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐  │
│  │ "Leia definitsioo-  │ │ "Leia seotud        │ │ "Leia kasutus-      │  │
│  │  nid lõikudest"     │ │  terminid lõikudest"│ │  näited lõikudest"  │  │
│  └──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘  │
└─────────────┼───────────────────────┼───────────────────────┼─────────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  VÄLJUND: struktureeritud terminoloogiakirje                              │
│  - definitsioonid                                                         │
│  - seotud terminid                                                        │
│  - kasutuskontekstid                                                      │
│  - vaata ka                                                               │
└───────────────────────────────────────────────────────────────────────────┘
```

**Otsingu loogika (iga kategooria jaoks):**
1. Otsing **algse päringuga** ("valvur") → 2×*limit* tulemust
2. Otsing **iga laiendatud terminiga** → *limit* tulemust per termin
3. Kombineeri + eemalda duplikaadid
4. Reranking/sort kasutades **algset päringut**
5. → ***limit* parimat lõiku** LLM-ile

## Konfiguratsioon

Failis `config/config.json`:

```json
{
  "llm": {"model": "gpt-4o"},
  "hybrid_search": {"enabled": true},
  "reranking": {"enabled": true, "model": "BAAI/bge-reranker-base"}
}
```

### Reranking

Reranking parandab otsingutulemuste kvaliteeti, tõstes asjakohased lõigud esikohale cross-encoder mudeli abil.

**Mudeli valikud (testitud CPU-l, 20 tulemust):**

| Mudel | Aeg | Kvaliteet | Märkused |
|-------|-----|-----------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~2.2s | Hea | Kiireim, inglise-keskne |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~2.7s | Parem | Veidi aeglasem |
| `BAAI/bge-reranker-base` | ~4.2s | Hea+ | **Soovitatav** - parem eesti keele tugi |
| `BAAI/bge-reranker-v2-m3` | ~14s | Hea+ | Sama kvaliteet kui base, 3x aeglasem |

**Sisse lülitamine:**
1. Muuda `config/config.json`:
   ```json
   "reranking": {"enabled": true, "model": "BAAI/bge-reranker-base"}
   ```
2. Taaskäivita backend:
   ```bash
   docker compose restart backend
   ```
3. Oota, kuni mudel laadib (10-80s sõltuvalt mudelist, esimesel korral laaditakse alla)

**Kontrollimine:**
```bash
curl http://localhost:8000/health | jq '.reranking_enabled, .reranker_model'
# peaks näitama: true ja mudeli nime
```

**Märkus:** Reranking lisab igale päringule viivise (sõltuvalt mudelist), aga parandab tulemuste kvaliteeti märkimisväärselt.
