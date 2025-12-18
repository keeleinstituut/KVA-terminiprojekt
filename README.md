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
┌─────────────────────────────────────────────────────────────────────────┐
│  SISEND: termin (nt "eririietus")                                       │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PÄRINGU LAIENDAMINE (LLM)                                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐            │
│  │ definitsioonid  │ │ seotud terminid │ │ kasutuskontekst │            │
│  │ → "eririietus   │ │ → "spetsiaalne  │ │ → "eririietus   │            │
│  │    on ..."      │ │    riietus"     │ │    kasutamine"  │            │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘            │
└───────────┼───────────────────┼───────────────────┼─────────────────────┘
            │                   │                   │
            ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  HÜBRIIDOTSING (Qdrant)                                                 │
│  dense (e5-large) + sparse (BM25) → lõigud dokumendist                  │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PARALLEELNE LLM PÄRING                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐            │
│  │ definitsioonid  │ │ seotud terminid │ │ kasutuskontekst │            │
│  │ ← lõigud A      │ │ ← lõigud B      │ │ ← lõigud C      │            │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘            │
└───────────┼───────────────────┼───────────────────┼─────────────────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  VÄLJUND: struktureeritud terminoloogiakirje                            │
│  - definitsioonid (puudusid antud näites)                               │
│  - seotud terminid (kombinesoon, jope, tunked - Politseiametniku... )   │
│  - kasutuskontekstid ("Mootorratturi eririietus..." - Politseiametni...)│
│  - vaata ka (vormiriietus, eraldusmärgid)                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Konfiguratsioon

Mudeli muutmine failis `config/config.json`:
```json
{
  "llm": {"model": "gpt-4o"},
  "hybrid_search": {"enabled": true},
  "reranking": {"enabled": false}
}
```
