# ⚡ JobVeezie
**Aggregatore di offerte di lavoro con AI locale — powered by Llama 3**

> Scraping multi-piattaforma · Matching AI su GPU NVIDIA · Zero cloud · 100% privacy

---

## 🚀 Installazione rapida

### 1. Prerequisiti di sistema

- Python **3.10+**
- [Ollama](https://ollama.ai) installato e in esecuzione
- GPU NVIDIA con **≥ 8GB VRAM** (Blackwell o superiore consigliato)

### 2. Scarica il modello AI

```bash
ollama pull llama3.1:8b
```

> Il modello in quantizzazione Q4_K_M occupa ~4.5GB VRAM, lasciando ampio margine
> per il contesto di inferenza su GPU con 8GB VRAM.

### 3. Installa le dipendenze Python

```bash
pip install -r requirements.txt
```

### 4. Avvia l'applicazione

```bash
streamlit run app.py
```

L'app si aprirà automaticamente su `http://localhost:8501`

---

## 📁 Struttura del progetto

```
JobVeezie/
├── app.py            # Interfaccia Streamlit (UI, navigazione, render card)
├── engine.py         # Core engine: scraping JobSpy + AI Ollama + threading
├── utils.py          # Parsing PDF, formattazione, utility UI
├── requirements.txt  # Dipendenze Python
└── README.md         # Questa guida
```

---

## 🎯 Come funziona

### Flusso principale

```
CV (PDF) ──► utils.py ──► testo estratto
                               │
Parametri ricerca              ▼
(Ruolo + Città) ──► engine.py ──► JobSpy scraping
                               │       (LinkedIn, Indeed, Glassdoor)
                               ▼
                         Ollama / Llama 3
                         (analisi AI locale)
                               │
                               ▼
                    app.py ──► Card con score 1-10
```

### Paginazione intelligente (Lazy Loading)

1. L'utente avvia la ricerca → scarica e analizza i **primi 10 annunci** (Pagina 1)
2. Mentre legge la Pagina 1, un **thread daemon in background** scarica e analizza la Pagina 2
3. Quando l'utente clicca "Prossima Pagina", i dati sono già pronti in cache → **risposta istantanea**
4. Il ciclo continua: ogni pagina visualizzata triggera il prefetch della successiva

### AI Matching

Il prompt inviato a Llama 3 include:
- Testo completo del CV (max 2500 caratteri)
- Descrizione dell'annuncio (max 1500 caratteri)
- Istruzioni per rispondere in **JSON strutturato**

Output AI per ogni annuncio:
- `punteggio` (1-10)
- `motivazione` (spiegazione in una frase)
- `competenze_match` (tag verdi: skill già presenti nel CV)
- `competenze_mancanti` (tag rossi: skill da acquisire)

---

## ⚙️ Configurazione avanzata

Modifica le costanti in `engine.py`:

| Costante | Default | Descrizione |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | Modello Ollama da usare |
| `ANNUNCI_PER_PAGINA` | `10` | Dimensione di ogni pagina |
| `SORGENTI_DEFAULT` | `["linkedin", "indeed", "glassdoor"]` | Piattaforme di scraping |
| `MAX_RISULTATI_TOTALI` | `100` | Limite massimo annunci per sessione |

### Ottimizzazioni GPU (in `engine.py` → `analizza_compatibilita_ai`)

```python
options={
    "temperature": 0.1,   # Bassa = output deterministico
    "num_ctx": 4096,      # Context window (4K è sufficiente per il task)
    "num_gpu": 99,        # Usa tutti i layer disponibili in GPU
    "num_thread": 8,      # Thread CPU per pre/post processing
}
```

---

## 🔧 Troubleshooting

**Ollama non risponde**
```bash
# Verifica che il servizio sia attivo
ollama serve
# In un altro terminale:
ollama list
```

**Errore di scraping (LinkedIn/Indeed)**
- Aumenta il delay tra richieste (JobSpy lo gestisce automaticamente)
- Prova con `hours_old=168` (7 giorni) per più risultati

**CV non leggibile**
- Assicurati che il PDF non sia scansionato (deve contenere testo selezionabile)
- Per PDF scansionati, considera l'aggiunta di OCR con `pytesseract`

---

## 📊 Requisiti hardware consigliati

| Componente | Minimo | Consigliato |
|---|---|---|
| GPU VRAM | 6GB | 8GB+ (es. RTX 5070) |
| RAM | 8GB | 16GB+ |
| CPU | 4 core | 8+ core |
| Storage | 10GB liberi | 20GB+ |
