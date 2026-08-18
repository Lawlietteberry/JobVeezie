# ============================================================
# utils.py - Parsing del CV e funzioni di utilità
# JobVeezie - Aggregatore di lavoro con AI locale
# ============================================================

import io
import logging
import re
from typing import Optional

import PyPDF2
import streamlit as st

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("jobveezie.utils")


# ─────────────────────────────────────────────
# PARSING DEL CV
# ─────────────────────────────────────────────

def estrai_testo_da_pdf(file_bytes: bytes) -> str:
    """
    Estrae il testo grezzo da un file PDF caricato dall'utente.

    Args:
        file_bytes: Contenuto binario del file PDF.

    Returns:
        Stringa con il testo estratto, oppure stringa vuota in caso di errore.
    """
    testo_estratto = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        logger.info(f"CV caricato: {len(reader.pages)} pagine rilevate.")

        for i, pagina in enumerate(reader.pages):
            testo_pagina = pagina.extract_text()
            if testo_pagina:
                testo_estratto.append(testo_pagina)
                logger.debug(f"  → Pagina {i+1}: {len(testo_pagina)} caratteri estratti.")
            else:
                logger.warning(f"  → Pagina {i+1}: nessun testo leggibile (potrebbe essere scansionata).")

        testo_completo = "\n".join(testo_estratto)
        logger.info(f"Testo totale estratto dal CV: {len(testo_completo)} caratteri.")
        return testo_completo

    except Exception as e:
        logger.error(f"Errore durante il parsing del PDF: {e}")
        return ""


def pulisci_testo(testo: str, max_chars: int = 3000) -> str:
    """
    Pulisce e normalizza il testo estratto dal CV o dagli annunci.
    Rimuove spazi multipli, righe vuote eccessive e tronca se troppo lungo.

    Args:
        testo:     Testo grezzo da pulire.
        max_chars: Numero massimo di caratteri (per ottimizzare il prompt all'AI).

    Returns:
        Testo pulito e troncato.
    """
    if not testo:
        return ""

    # Rimuove caratteri non stampabili (esclusi newline e tab)
    testo = re.sub(r'[^\x20-\x7E\n\t\u00C0-\u024F]', ' ', testo)

    # Sostituisce sequenze di spazi/tab con un singolo spazio
    testo = re.sub(r'[ \t]+', ' ', testo)

    # Riduce righe vuote multiple a una sola riga vuota
    testo = re.sub(r'\n{3,}', '\n\n', testo)

    # Trim globale
    testo = testo.strip()

    # Troncamento per rispettare il limite di token del modello Llama locale
    if len(testo) > max_chars:
        testo = testo[:max_chars] + "\n[... troncato per lunghezza ...]"
        logger.debug(f"Testo troncato a {max_chars} caratteri.")

    return testo


# ─────────────────────────────────────────────
# FORMATTAZIONE E UI HELPERS
# ─────────────────────────────────────────────

def formatta_stipendio(lavoro: dict) -> str:
    """
    Formatta le informazioni sullo stipendio di un annuncio in modo leggibile.

    Args:
        lavoro: Dizionario con i dati dell'annuncio di lavoro.

    Returns:
        Stringa formattata con stipendio o "Non specificato".
    """
    min_sal = lavoro.get("min_amount")
    max_sal = lavoro.get("max_amount")
    valuta = lavoro.get("currency", "€")
    intervallo = lavoro.get("interval", "")

    if min_sal and max_sal:
        return f"{valuta} {int(min_sal):,} – {int(max_sal):,} / {intervallo}"
    elif min_sal:
        return f"Da {valuta} {int(min_sal):,} / {intervallo}"
    elif max_sal:
        return f"Fino a {valuta} {int(max_sal):,} / {intervallo}"
    else:
        return "Non specificato"


def colore_punteggio(punteggio: int) -> str:
    """
    Restituisce un colore esadecimale in base al punteggio di matching AI.

    Args:
        punteggio: Intero da 1 a 10.

    Returns:
        Codice colore esadecimale.
    """
    if punteggio >= 9:
        return "#00E676"   # Verde brillante - Match eccellente
    elif punteggio >= 7:
        return "#69F0AE"   # Verde chiaro - Buon match
    elif punteggio >= 5:
        return "#FFD740"   # Giallo - Match moderato
    else:
        return "#FF5252"   # Rosso - Match basso


def emoji_punteggio(punteggio: int) -> str:
    """Restituisce un'emoji rappresentativa del punteggio."""
    if punteggio >= 9:
        return "🔥"
    elif punteggio >= 7:
        return "✅"
    elif punteggio >= 5:
        return "🟡"
    else:
        return "❌"


def sorgente_emoji(sorgente: str) -> str:
    """
    Restituisce un'emoji per identificare visivamente la piattaforma sorgente.

    Args:
        sorgente: Nome della piattaforma (es. 'linkedin', 'indeed').

    Returns:
        Stringa con emoji identificativa.
    """
    mapping = {
        "linkedin":  "💼 LinkedIn",
        "indeed":    "🔍 Indeed",
        "glassdoor": "🏢 Glassdoor",
        "zip_recruiter": "📋 ZipRecruiter",
    }
    return mapping.get(str(sorgente).lower(), f"🌐 {sorgente}")


def calcola_statistiche(annunci_analizzati: list) -> dict:
    """
    Calcola statistiche aggregate sugli annunci analizzati.

    Args:
        annunci_analizzati: Lista di dizionari con dati + punteggio AI.

    Returns:
        Dizionario con statistiche riassuntive.
    """
    if not annunci_analizzati:
        return {}

    punteggi = [a.get("ai_score", 0) for a in annunci_analizzati if a.get("ai_score")]
    qualificati = [a for a in annunci_analizzati if a.get("ai_score", 0) >= 7]

    return {
        "totale_analizzati":  len(annunci_analizzati),
        "totale_qualificati": len(qualificati),
        "punteggio_medio":    round(sum(punteggi) / len(punteggi), 1) if punteggi else 0,
        "punteggio_max":      max(punteggi) if punteggi else 0,
    }


def tronca_descrizione(testo: str, max_chars: int = 300) -> str:
    """
    Tronca una descrizione lunga per la visualizzazione nelle card UI.

    Args:
        testo:     Testo completo della descrizione.
        max_chars: Numero massimo di caratteri da mostrare.

    Returns:
        Testo troncato con ellissi, oppure testo originale se sufficientemente corto.
    """
    if not testo or len(testo) <= max_chars:
        return testo or "Descrizione non disponibile."
    return testo[:max_chars].rsplit(' ', 1)[0] + "…"
