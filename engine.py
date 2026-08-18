# ============================================================
# engine.py - Core engine: scraping JobSpy + AI Ollama
# JobVeezie - Aggregatore di lavoro con AI locale
# ============================================================

import logging
import re
import threading
import time
from typing import Callable, Optional

import ollama
import pandas as pd
from jobspy import scrape_jobs

from utils import pulisci_testo

# Configurazione logging
logger = logging.getLogger("jobveezie.engine")

# ─────────────────────────────────────────────
# COSTANTI DI CONFIGURAZIONE
# ─────────────────────────────────────────────

# Modello Ollama ottimizzato per GPU NVIDIA Blackwell 8GB VRAM.
# Llama 3.1 8B in quantizzazione Q4_K_M occupa ~4.5GB VRAM, lasciando
# margine sufficiente per il contesto durante l'inferenza.
OLLAMA_MODEL = "llama3.1:8b"

# Numero di annunci per "pagina" di risultati
ANNUNCI_PER_PAGINA = 10

# Sorgenti di scraping abilitate
SORGENTI_DEFAULT = ["linkedin", "indeed", "glassdoor"]

# Timeout massimo per una singola chiamata all'AI (secondi)
TIMEOUT_AI = 60

# Numero massimo totale di annunci da scaricare in una sessione
MAX_RISULTATI_TOTALI = 100


# ─────────────────────────────────────────────
# SCRAPING CON JOBSPY
# ─────────────────────────────────────────────

def cerca_lavori(
    ruolo: str,
    localita: str,
    numero_risultati: int = ANNUNCI_PER_PAGINA,
    sorgenti: list = None,
    offset: int = 0,
) -> list[dict]:
    """
    Esegue lo scraping delle offerte di lavoro tramite python-jobspy.

    Args:
        ruolo:            Titolo del ruolo cercato (es. "Python Developer").
        localita:         Città o area geografica (es. "Milano, Italia").
        numero_risultati: Quanti annunci recuperare in questa chiamata.
        sorgenti:         Lista di piattaforme da interrogare.
        offset:           Offset per la paginazione lato sorgente (non supportato
                          uniformemente da tutti i provider, usato dove possibile).

    Returns:
        Lista di dizionari, ciascuno rappresentante un annuncio di lavoro.
        Restituisce lista vuota in caso di errore.
    """
    if sorgenti is None:
        sorgenti = SORGENTI_DEFAULT

    logger.info(
        f"[Scraping] Avvio ricerca: ruolo='{ruolo or '(tutti)'}', località='{localita}', "
        f"risultati={numero_risultati}, sorgenti={sorgenti}"
    )

    # Se il ruolo è vuoto, usa un termine generico per la ricerca per sola località
    termine_ricerca = ruolo.strip() if ruolo and ruolo.strip() else "lavoro"

    try:
        df: pd.DataFrame = scrape_jobs(
            site_name=sorgenti,
            search_term=termine_ricerca,
            location=localita,
            results_wanted=numero_risultati,
            # country_indeed supporta le ricerche Indeed internazionali
            country_indeed="italy",
            # Delay tra richieste per evitare ban (in secondi)
            hours_old=72,
        )

        if df is None or df.empty:
            logger.warning("[Scraping] Nessun risultato trovato.")
            return []

        # Converte il DataFrame in lista di dizionari, sostituendo NaN con None
        annunci = df.where(pd.notnull(df), None).to_dict(orient="records")
        logger.info(f"[Scraping] Recuperati {len(annunci)} annunci.")
        return annunci

    except Exception as e:
        logger.error(f"[Scraping] Errore durante lo scraping: {e}", exc_info=True)
        return []


# ─────────────────────────────────────────────
# INTERAZIONE CON OLLAMA (AI LOCALE)
# ─────────────────────────────────────────────

def costruisci_prompt_matching(testo_cv: str, descrizione_lavoro: str, titolo: str) -> str:
    """
    Costruisce il prompt ottimizzato per l'analisi di compatibilità CV-annuncio.
    Progettato per Llama 3 con output JSON strutturato e conciso,
    per massimizzare le prestazioni su GPU con 8GB VRAM.

    Args:
        testo_cv:          Testo estratto dal CV dell'utente.
        descrizione_lavoro: Descrizione dell'annuncio di lavoro.
        titolo:            Titolo del ruolo nell'annuncio.

    Returns:
        Stringa del prompt formattato.
    """
    # Pulizia e troncamento per rispettare il context window del modello
    cv_pulito = pulisci_testo(testo_cv, max_chars=2500)
    desc_pulita = pulisci_testo(descrizione_lavoro, max_chars=1500)

    prompt = f"""Sei un recruiter esperto. Analizza la compatibilità tra il CV e l'annuncio.

CV:
{cv_pulito}

ANNUNCIO - {titolo}:
{desc_pulita}

Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. NESSUN testo prima o dopo. NESSUN markdown. NESSUNA spiegazione. Solo JSON puro:
{{"punteggio": <numero 1-10>, "motivazione": "<frase breve>", "competenze_match": ["<skill1>"], "competenze_mancanti": ["<gap1>"]}}"""

    return prompt


def analizza_compatibilita_ai(
    testo_cv: str,
    annuncio: dict,
    modello: str = None,   # None = auto-rileva al primo uso
) -> dict:
    """
    Invoca Ollama per analizzare la compatibilità tra CV e annuncio.
    Se modello è None, rileva automaticamente il miglior modello disponibile.
    """
    # Auto-rilevamento modello (fatto una sola volta, poi riutilizzato)
    if modello is None:
        modello = rileva_modello_ottimale()
    titolo = annuncio.get("title", "Posizione non specificata")
    descrizione = annuncio.get("description", "")

    if not descrizione or len(descrizione.strip()) < 30:
        # Senza descrizione assegniamo un punteggio neutro invece di escludere l'annuncio
        logger.debug(f"[AI] '{titolo}': descrizione assente/troppo corta, assegnato score neutro.")
        return {
            "punteggio": 5,
            "motivazione": "Descrizione non disponibile — valutazione automatica neutra.",
            "competenze_match": [],
            "competenze_mancanti": [],
        }

    prompt = costruisci_prompt_matching(testo_cv, descrizione, titolo)

    try:
        logger.debug(f"[AI] Analisi in corso per: '{titolo}'")
        inizio = time.time()

        # Chiamata all'API Ollama locale.
        # Il parametro 'options' permette di ottimizzare l'uso della VRAM:
        # - num_gpu: numero di layer da caricare in GPU (0 = auto-detect)
        # - num_ctx: context window (2048 è sufficiente per il nostro task)
        # - temperature: bassa per risposte più deterministiche
        risposta = ollama.chat(
            model=modello,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,       # Bassa temperatura = output più preciso
                "num_ctx": 4096,          # Context window ottimizzato per 8GB VRAM
                "num_gpu": 99,            # Usa tutti i layer disponibili in GPU
                "num_thread": 8,          # Thread CPU per pre/post processing
            },
        )

        durata = time.time() - inizio
        logger.debug(f"[AI] '{titolo}' analizzato in {durata:.1f}s")

        testo_risposta = risposta["message"]["content"].strip()
        return _parse_risposta_ai(testo_risposta, titolo)

    except Exception as e:
        logger.error(f"[AI] Errore durante l'analisi di '{titolo}': {e}")
        return {
            "punteggio": 0,
            "motivazione": f"Errore AI: {str(e)[:100]}",
            "competenze_match": [],
            "competenze_mancanti": [],
        }


def _parse_risposta_ai(testo: str, titolo: str) -> dict:
    """
    Effettua il parsing della risposta JSON di Ollama con fallback robusto.
    Gestisce casi in cui il modello inserisce testo extra prima/dopo il JSON.

    Args:
        testo:  Risposta grezza del modello.
        titolo: Titolo dell'annuncio (per logging).

    Returns:
        Dizionario con i campi del risultato AI.
    """
    import json

    # Valore di fallback in caso di parsing fallito
    fallback = {
        "punteggio": 0,
        "motivazione": "Impossibile analizzare la risposta AI.",
        "competenze_match": [],
        "competenze_mancanti": [],
    }

    try:
        # Tentativo 1: parsing diretto
        risultato = json.loads(testo)
        return _valida_risultato_ai(risultato)

    except json.JSONDecodeError:
        pass

    try:
        # Tentativo 2: estrae il blocco JSON tramite regex (gestisce testo extra)
        match = re.search(r'\{[^{}]*\}', testo, re.DOTALL)
        if match:
            risultato = json.loads(match.group(0))
            return _valida_risultato_ai(risultato)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Tentativo 3: parsing risposta in stile markdown (**Punteggio:** 2)
    try:
        match_score = re.search(r'\*{0,2}[Pp]unteggio\*{0,2}\s*:?\*{0,2}\s*(\d+)', testo)
        match_motiv = re.search(r'\*{0,2}[Mm]otivazione\*{0,2}\s*:?\*{0,2}\s*(.+?)(?:\n|$)', testo)
        if match_score:
            punteggio = min(10, max(1, int(match_score.group(1))))
            motivazione = match_motiv.group(1).strip().lstrip('*').strip() if match_motiv else "Analisi parziale."
            logger.debug(f"[AI] Parsing markdown fallback per '{titolo}': score={punteggio}")
            return {**fallback, "punteggio": punteggio, "motivazione": motivazione}
    except Exception:
        pass

    logger.error(f"[AI] Parsing completamente fallito per '{titolo}'. Risposta: {testo[:200]}")
    return fallback


def _valida_risultato_ai(dati: dict) -> dict:
    """
    Valida e normalizza i campi del risultato AI.

    Args:
        dati: Dizionario grezzo dalla risposta del modello.

    Returns:
        Dizionario validato con tutti i campi richiesti.
    """
    punteggio = dati.get("punteggio", 0)

    # Assicura che il punteggio sia nel range 1-10
    if isinstance(punteggio, str):
        punteggio = int(re.sub(r'\D', '', punteggio) or 0)
    punteggio = min(10, max(0, int(punteggio)))

    return {
        "punteggio":            punteggio,
        "motivazione":          str(dati.get("motivazione", "N/A"))[:300],
        "competenze_match":     list(dati.get("competenze_match", [])),
        "competenze_mancanti":  list(dati.get("competenze_mancanti", [])),
    }


# ─────────────────────────────────────────────
# PIPELINE COMPLETA: SCRAPING + AI ANALYSIS
# ─────────────────────────────────────────────

def analizza_pagina(
    annunci: list[dict],
    testo_cv: str,
    callback_progresso: Optional[Callable] = None,
) -> list[dict]:
    """
    Esegue l'analisi AI su una lista di annunci già scaricati.
    Arricchisce ogni annuncio con il campo 'ai_score' e i dettagli AI.

    Args:
        annunci:             Lista di annunci da analizzare.
        testo_cv:            Testo estratto dal CV dell'utente.
        callback_progresso:  Funzione opzionale chiamata dopo ogni analisi
                             con argomento (indice, totale, annuncio_analizzato).

    Returns:
        Lista di annunci arricchiti con i campi AI.
    """
    risultati = []

    for i, annuncio in enumerate(annunci):
        logger.debug(f"[Pipeline] Analisi {i+1}/{len(annunci)}: {annuncio.get('title', 'N/A')}")

        # Esegue l'analisi AI
        analisi = analizza_compatibilita_ai(testo_cv, annuncio)

        # Arricchisce il dizionario dell'annuncio con i risultati AI
        annuncio_arricchito = {
            **annuncio,
            "ai_score":               analisi["punteggio"],
            "ai_motivazione":         analisi["motivazione"],
            "ai_competenze_match":    analisi["competenze_match"],
            "ai_competenze_mancanti": analisi["competenze_mancanti"],
        }
        risultati.append(annuncio_arricchito)

        # Notifica il progresso al chiamante (es. per aggiornare la UI)
        if callback_progresso:
            try:
                callback_progresso(i + 1, len(annunci), annuncio_arricchito)
            except Exception as e:
                logger.warning(f"[Pipeline] Errore nel callback progresso: {e}")

    return risultati


class MotoreRicercaAsync:
    """
    Motore di ricerca asincrono con supporto al lazy loading.

    Gestisce il download e l'analisi AI degli annunci in thread separati,
    implementando la logica di paginazione intelligente:
    - Pagina 1: scarica e analizza i primi ANNUNCI_PER_PAGINA annunci
    - Mentre l'utente legge la pagina 1, un thread in background
      prepara già la pagina 2 (e successive)
    """

    def __init__(self, ruolo: str, localita: str, testo_cv: str):
        self.ruolo = ruolo
        self.localita = localita
        self.testo_cv = testo_cv

        # Cache delle pagine già analizzate: {numero_pagina: [annunci]}
        self._cache_pagine: dict[int, list] = {}

        # Thread attivi per il prefetch
        self._thread_prefetch: dict[int, threading.Thread] = {}

        # Lock per accesso thread-safe alla cache
        self._lock = threading.Lock()

        # Flag per segnalare che non ci sono più risultati da scaricare
        self._fine_risultati: bool = False

        logger.info(f"[MotoreAsync] Inizializzato per ruolo='{ruolo}', località='{localita}'")

    def recupera_pagina(self, numero_pagina: int) -> list[dict]:
        """
        Recupera una pagina di annunci analizzati.
        Se la pagina è già in cache, la restituisce immediatamente.
        Altrimenti esegue scraping + analisi AI sincrona, poi avvia
        il prefetch della pagina successiva in background.

        Args:
            numero_pagina: Numero di pagina (1-indexed).

        Returns:
            Lista di annunci analizzati e filtrati (score >= 1).
        """
        # Controlla se la pagina è già in cache
        with self._lock:
            if numero_pagina in self._cache_pagine:
                logger.info(f"[MotoreAsync] Pagina {numero_pagina} servita dalla cache.")
                self._avvia_prefetch(numero_pagina + 1)
                return self._cache_pagine[numero_pagina]

        # La pagina non è in cache: esegui scraping + analisi
        logger.info(f"[MotoreAsync] Avvio scraping pagina {numero_pagina}...")
        offset = (numero_pagina - 1) * ANNUNCI_PER_PAGINA

        annunci_grezzi = cerca_lavori(
            ruolo=self.ruolo,
            localita=self.localita,
            numero_risultati=ANNUNCI_PER_PAGINA,
            offset=offset,
        )

        if not annunci_grezzi:
            logger.warning(f"[MotoreAsync] Nessun annuncio trovato per pagina {numero_pagina}.")
            with self._lock:
                self._fine_risultati = True
                self._cache_pagine[numero_pagina] = []
            return []

        # Analisi AI sincrona per la pagina richiesta
        annunci_analizzati = analizza_pagina(annunci_grezzi, self.testo_cv)

        with self._lock:
            self._cache_pagine[numero_pagina] = annunci_analizzati

        # Avvia il prefetch della prossima pagina in background
        self._avvia_prefetch(numero_pagina + 1)

        return annunci_analizzati

    def _avvia_prefetch(self, numero_pagina: int) -> None:
        """
        Avvia un thread in background per pre-caricare la pagina indicata.
        Non fa nulla se la pagina è già in cache o in prefetch,
        o se i risultati sono esauriti.

        Args:
            numero_pagina: Numero della pagina da pre-caricare.
        """
        with self._lock:
            # Skip se già in cache, già in prefetch, o risultati finiti
            if (
                numero_pagina in self._cache_pagine
                or numero_pagina in self._thread_prefetch
                or self._fine_risultati
            ):
                return

        logger.info(f"[MotoreAsync] Avvio prefetch in background per pagina {numero_pagina}.")

        def _lavoro_prefetch():
            offset = (numero_pagina - 1) * ANNUNCI_PER_PAGINA
            annunci_grezzi = cerca_lavori(
                ruolo=self.ruolo,
                localita=self.localita,
                numero_risultati=ANNUNCI_PER_PAGINA,
                offset=offset,
            )

            if not annunci_grezzi:
                with self._lock:
                    self._fine_risultati = True
                    self._cache_pagine[numero_pagina] = []
                logger.info(f"[MotoreAsync] Prefetch pagina {numero_pagina}: nessun risultato, fine dati.")
                return

            annunci_analizzati = analizza_pagina(annunci_grezzi, self.testo_cv)

            with self._lock:
                self._cache_pagine[numero_pagina] = annunci_analizzati
                # Rimuove il thread dalla lista dei prefetch attivi
                self._thread_prefetch.pop(numero_pagina, None)

            logger.info(f"[MotoreAsync] Prefetch pagina {numero_pagina} completato: {len(annunci_analizzati)} annunci.")

        thread = threading.Thread(
            target=_lavoro_prefetch,
            name=f"prefetch-pagina-{numero_pagina}",
            daemon=True,  # Thread daemon: viene terminato se il processo principale esce
        )

        with self._lock:
            self._thread_prefetch[numero_pagina] = thread

        thread.start()

    def pagina_disponibile(self, numero_pagina: int) -> bool:
        """
        Controlla se una pagina è già pronta in cache (senza bloccare).

        Args:
            numero_pagina: Numero della pagina da controllare.

        Returns:
            True se la pagina è in cache, False se è ancora in elaborazione.
        """
        with self._lock:
            return numero_pagina in self._cache_pagine

    def ha_piu_pagine(self, numero_pagina_corrente: int) -> bool:
        """
        Determina se esistono altre pagine di risultati dopo quella corrente.

        Args:
            numero_pagina_corrente: Numero della pagina attualmente visualizzata.

        Returns:
            True se ci potrebbero essere altre pagine disponibili.
        """
        with self._lock:
            return not self._fine_risultati

    def stato_cache(self) -> dict:
        """Restituisce lo stato attuale della cache per debugging."""
        with self._lock:
            return {
                "pagine_in_cache":   list(self._cache_pagine.keys()),
                "prefetch_attivi":   list(self._thread_prefetch.keys()),
                "fine_risultati":    self._fine_risultati,
                "totale_analizzati": sum(len(v) for v in self._cache_pagine.values()),
            }


# ─────────────────────────────────────────────
# UTILITY DI SISTEMA
# ─────────────────────────────────────────────

def get_modelli_disponibili() -> list[str]:
    """
    Restituisce la lista dei nomi dei modelli installati in Ollama.
    Restituisce lista vuota se Ollama non è raggiungibile.
    """
    try:
        risposta = ollama.list()
        # La struttura può variare tra versioni di ollama-python: gestiamo entrambe
        modelli = risposta.get("models", [])
        nomi = []
        for m in modelli:
            # Versioni recenti restituiscono oggetti con attributo .model o .name
            nome = None
            if isinstance(m, dict):
                nome = m.get("model") or m.get("name")
            else:
                nome = getattr(m, "model", None) or getattr(m, "name", None)
            if nome:
                nomi.append(nome)
        return nomi
    except Exception as e:
        logger.warning(f"[Ollama] Impossibile recuperare lista modelli: {e}")
        return []


# Cache del modello rilevato automaticamente (evita chiamate ripetute a ollama.list)
_modello_cache: str = None


def rileva_modello_ottimale() -> str:
    """
    Seleziona automaticamente il miglior modello disponibile.
    Il risultato viene messo in cache per evitare chiamate ripetute a Ollama.
    """
    global _modello_cache
    if _modello_cache:
        return _modello_cache
    nomi = get_modelli_disponibili()
    if not nomi:
        _modello_cache = OLLAMA_MODEL
        return _modello_cache

    preferenze = [
        "llama3.1:8b-instruct",   # → llama3.1:8b-instruct-q4_K_M  ✅ ideale
        "llama3.1",               # qualsiasi altra variante llama3.1
        "llama3.2",               # llama3.2:1b come fallback leggero
        "qwen2.5:7b-instruct",    # qwen instruct come alternativa valida
        "qwen2.5",                # qualsiasi qwen2.5
        "llama3",
        "llama2",
        "mistral",
        "deepseek",
        "phi3",
        "gemma",
    ]
    for pref in preferenze:
        for nome in nomi:
            if pref in nome.lower():
                logger.info(f"[Ollama] Modello auto-rilevato: '{nome}'")
                _modello_cache = nome
                return _modello_cache

    # Fallback: prende il primo disponibile
    logger.info(f"[Ollama] Nessun modello preferito trovato, uso: '{nomi[0]}'")
    _modello_cache = nomi[0]
    return _modello_cache


def verifica_ollama() -> tuple[bool, str]:
    """
    Verifica che Ollama sia in esecuzione e che il modello richiesto sia disponibile.

    Returns:
        Tupla (successo: bool, messaggio: str).
    """
    try:
        nomi_modelli = get_modelli_disponibili()

        if not nomi_modelli:
            return False, "Nessun modello trovato. Esegui: ollama pull llama3.1:8b"

        modello_base = OLLAMA_MODEL.split(":")[0]
        modelli_compatibili = [m for m in nomi_modelli if modello_base in m]

        if modelli_compatibili:
            return True, f"Ollama OK ✅ — Modello '{modelli_compatibili[0]}' disponibile."
        else:
            modello_auto = rileva_modello_ottimale()
            return (
                True,
                f"Modello '{OLLAMA_MODEL}' non trovato. "
                f"Verrà usato automaticamente: '{modello_auto}'. "
                f"Disponibili: {', '.join(nomi_modelli)}"
            )

    except Exception as e:
        return False, f"Ollama non raggiungibile: {e}. Assicurati che il servizio sia avviato."