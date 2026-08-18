# ============================================================
# app.py - Interfaccia Streamlit principale
# JobVeezie - Aggregatore di lavoro con AI locale
# Avvia con: streamlit run app.py
# ============================================================

import logging
import threading
import time

import streamlit as st

from engine import (
    ANNUNCI_PER_PAGINA,
    OLLAMA_MODEL,
    MotoreRicercaAsync,
    verifica_ollama,
)
from utils import (
    calcola_statistiche,
    colore_punteggio,
    emoji_punteggio,
    estrai_testo_da_pdf,
    formatta_stipendio,
    sorgente_emoji,
    tronca_descrizione,
)

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jobveezie.app")


# ─────────────────────────────────────────────
# CONFIGURAZIONE PAGINA STREAMLIT
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="JobVeezie — AI Job Aggregator",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS PERSONALIZZATO
# Stile dark moderno ispirato a Veezie, con accent giallo/verde
# ─────────────────────────────────────────────

st.markdown("""
<style>
  /* Font Google */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

  /* Root variables */
  :root {
    --bg-dark:      #0D0F14;
    --bg-card:      #161A22;
    --bg-card-hover:#1E2330;
    --accent:       #C8F135;
    --accent-dim:   #8FAF20;
    --text-main:    #E8EAF0;
    --text-muted:   #7A8099;
    --border:       #252A38;
    --green-bright: #00E676;
    --yellow:       #FFD740;
    --red:          #FF5252;
  }

  /* Sfondo globale */
  .stApp { background-color: var(--bg-dark); }
  .main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #111318 !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif; }

  /* Titolo principale */
  .jobveezie-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
  }
  .jobveezie-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
  }

  /* Card annuncio */
  .job-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
  }
  .job-card:hover {
    border-color: var(--accent-dim);
    background: var(--bg-card-hover);
  }

  /* Badge punteggio */
  .score-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 30px;
    margin-bottom: 0.6rem;
  }

  /* Titolo job nella card */
  .job-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-main);
    margin: 0.3rem 0 0.1rem;
  }
  .job-company {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: var(--accent);
    font-weight: 500;
  }
  .job-meta {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
  }
  .job-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: #B0B8CC;
    line-height: 1.6;
    margin-top: 0.7rem;
    padding-top: 0.7rem;
    border-top: 1px solid var(--border);
  }
  .job-ai-note {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #9BA8C0;
    font-style: italic;
    margin-top: 0.5rem;
  }

  /* Tag competenze */
  .tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 4px;
    margin: 2px 3px 2px 0;
  }
  .tag-match   { background: #1A3020; color: #00E676; border: 1px solid #2A5030; }
  .tag-missing { background: #301A1A; color: #FF6B6B; border: 1px solid #502A2A; }

  /* Stats bar */
  .stats-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1.2rem;
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    align-items: center;
  }
  .stat-item {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-muted);
  }
  .stat-value {
    font-size: 1.3rem;
    color: var(--accent);
    font-weight: 700;
    display: block;
  }

  /* Paginazione */
  .pagination-area {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    color: var(--text-muted);
  }

  /* Status Ollama */
  .ollama-status {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    padding: 8px 14px;
    border-radius: 8px;
    margin-top: 0.5rem;
  }
  .ollama-ok      { background: #1A3020; color: #00E676; border: 1px solid #2A5030; }
  .ollama-error   { background: #301A1A; color: #FF6B6B; border: 1px solid #502A2A; }

  /* Bottoni Streamlit override */
  .stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    color: var(--text-main) !important;
    transition: all 0.15s !important;
  }
  .stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--bg-card-hover) !important;
  }

  /* Barra progresso */
  .stProgress > div > div > div { background: var(--accent) !important; }

  /* Input text */
  .stTextInput > div > div > input,
  .stSelectbox > div > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
  }

  /* Divisore */
  hr { border-color: var(--border) !important; }

  /* Nascondi footer Streamlit */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INIZIALIZZAZIONE SESSION STATE
# ─────────────────────────────────────────────

def inizializza_session_state():
    """
    Inizializza tutte le variabili di stato della sessione Streamlit.
    Chiamata una sola volta all'avvio dell'app.
    """
    defaults = {
        # Dati CV
        "testo_cv":          "",
        "nome_cv":           "",

        # Parametri di ricerca
        "ruolo_ricerca":     "",
        "localita_ricerca":  "",

        # Motore di ricerca asincrono (istanza di MotoreRicercaAsync)
        "motore":            None,

        # Paginazione
        "pagina_corrente":   1,

        # Ricerca attiva
        "ricerca_avviata":   False,

        # Stato UI
        "mostra_filtrati":   False,   # Mostra tutti gli annunci di default

        # Flag elaborazione
        "elaborazione_in_corso": False,
    }

    for chiave, valore in defaults.items():
        if chiave not in st.session_state:
            st.session_state[chiave] = valore


inizializza_session_state()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 💼 JobVeezie")
    st.markdown("*Il tuo aggregatore di lavoro con AI locale*")
    st.markdown("---")

    # --- Sezione CV ---
    st.markdown("### 📄 Il tuo CV")
    file_cv = st.file_uploader(
        "Carica il tuo CV (PDF)",
        type=["pdf"],
        help="Il CV verrà usato dall'AI per calcolare il punteggio di compatibilità.",
    )

    if file_cv is not None:
        if file_cv.name != st.session_state.get("nome_cv"):
            with st.spinner("Estrazione testo dal CV..."):
                testo = estrai_testo_da_pdf(file_cv.read())

            if testo:
                st.session_state.testo_cv = testo
                st.session_state.nome_cv  = file_cv.name
                st.success(f"✅ CV caricato: {file_cv.name}")
            else:
                st.error("❌ Impossibile estrarre testo dal PDF. Prova con un PDF non scansionato.")
        else:
            st.success(f"✅ CV attivo: {st.session_state.nome_cv}")

    st.markdown("---")

    # --- Sezione Parametri Ricerca ---
    st.markdown("### 🔍 Parametri Ricerca")

    ruolo_input = st.text_input(
        "Ruolo cercato *(opzionale)*",
        value=st.session_state.ruolo_ricerca,
        placeholder="es. Python Developer, Data Scientist... (opzionale)",
    )

    localita_input = st.text_input(
        "Città / Area geografica",
        value=st.session_state.localita_ricerca,
        placeholder="es. Milano, Roma, Remote...",
    )

    st.markdown("---")

    # --- Sezione Filtri ---
    st.markdown("### ⚙️ Opzioni")

    mostra_solo_top = st.checkbox(
        "Filtra: mostra solo annunci con score ≥ 7",
        value=st.session_state.mostra_filtrati,
        help="Di default vengono mostrati tutti gli annunci. Attiva per vedere solo i più compatibili col tuo CV.",
    )
    st.session_state.mostra_filtrati = mostra_solo_top

    st.markdown("---")

    # --- Status Ollama ---
    st.markdown("### 🤖 Stato AI Engine")
    if st.button("🔄 Verifica Ollama"):
        with st.spinner("Connessione ad Ollama..."):
            ok, messaggio = verifica_ollama()
        classe = "ollama-ok" if ok else "ollama-error"
        st.markdown(f'<div class="ollama-status {classe}">{messaggio}</div>', unsafe_allow_html=True)

    from engine import rileva_modello_ottimale
    modello_attivo = rileva_modello_ottimale()
    st.markdown(f'<div class="ollama-status ollama-ok">Modello: {modello_attivo}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Pulsante Avvia Ricerca ---
    avvia_ricerca = st.button(
        "🚀 Avvia Ricerca",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.elaborazione_in_corso,
    )


# ─────────────────────────────────────────────
# LOGICA AVVIO RICERCA
# ─────────────────────────────────────────────

if avvia_ricerca:
    # Validazione input
    if not st.session_state.testo_cv:
        st.error("⚠️ Prima carica il tuo CV dalla sidebar.")
        st.stop()
    if not localita_input.strip():
        st.error("⚠️ Inserisci la città o area geografica.")
        st.stop()

    # Aggiorna i parametri di ricerca nello stato
    st.session_state.ruolo_ricerca    = ruolo_input.strip()
    st.session_state.localita_ricerca = localita_input.strip()
    st.session_state.pagina_corrente  = 1
    st.session_state.ricerca_avviata  = True
    st.session_state.elaborazione_in_corso = False

    # Crea una nuova istanza del motore asincrono
    st.session_state.motore = MotoreRicercaAsync(
        ruolo=st.session_state.ruolo_ricerca,
        localita=st.session_state.localita_ricerca,
        testo_cv=st.session_state.testo_cv,
    )

    st.rerun()


# ─────────────────────────────────────────────
# HEADER PRINCIPALE
# ─────────────────────────────────────────────

st.markdown('<div class="jobveezie-header">⚡ JobVeezie</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="jobveezie-subtitle">Aggregatore intelligente di offerte di lavoro '
    '— powered by Llama 3 · AI locale · Zero cloud</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# AREA PRINCIPALE: RISULTATI
# ─────────────────────────────────────────────

if not st.session_state.ricerca_avviata:
    # Stato iniziale: mostra messaggio di benvenuto
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="stats-box" style="flex-direction:column; gap:0.5rem;">
          <span class="stat-value">01</span>
          <span class="stat-item">📄 Carica il tuo CV (PDF) dalla sidebar</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stats-box" style="flex-direction:column; gap:0.5rem;">
          <span class="stat-value">02</span>
          <span class="stat-item">🔍 Imposta ruolo e città di ricerca</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stats-box" style="flex-direction:column; gap:0.5rem;">
          <span class="stat-value">03</span>
          <span class="stat-item">🚀 Premi "Avvia Ricerca" e lascia lavorare l'AI</span>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "💡 **Tip:** JobVeezie usa Llama 3 in locale per analizzare ogni annuncio e "
        "assegnargli un punteggio di compatibilità con il tuo CV. "
        "Solo gli annunci con score ≥ 7 vengono mostrati (configurabile)."
    )
    st.stop()


# ─────────────────────────────────────────────
# CARICAMENTO E VISUALIZZAZIONE PAGINA
# ─────────────────────────────────────────────

motore: MotoreRicercaAsync = st.session_state.motore
pagina = st.session_state.pagina_corrente

# Header ricerca attiva
ruolo_display = st.session_state.ruolo_ricerca or "Tutti i ruoli"
st.markdown(f"""
<div style="font-family:'DM Sans',sans-serif; margin-bottom:1rem;">
  <span style="color:#7A8099;">Ricerca:</span>
  <span style="color:#E8EAF0; font-weight:700;"> {ruolo_display}</span>
  <span style="color:#7A8099;"> in </span>
  <span style="color:#E8EAF0; font-weight:700;">{st.session_state.localita_ricerca}</span>
  <span style="color:#C8F135; font-family:'Space Mono',monospace; font-size:0.85rem; margin-left:1rem;">
    Pagina {pagina}
  </span>
</div>
""", unsafe_allow_html=True)

# Verifica se la pagina è già in cache o va caricata
pagina_pronta = motore.pagina_disponibile(pagina)

if not pagina_pronta:
    # La pagina non è in cache: avvia il caricamento in background e mostra progresso reale
    st.markdown(f"#### ⏳ Caricamento pagina {pagina}...")
    status_placeholder = st.empty()
    progress_bar = st.progress(0, text="Connessione alle piattaforme di lavoro...")

    # Avvia il caricamento in un thread separato se non già partito
    chiave_thread = f"_thread_caricamento_p{pagina}"
    if chiave_thread not in st.session_state:
        st.session_state[chiave_thread] = True

        def _carica_pagina_background():
            motore.recupera_pagina(pagina)

        t = threading.Thread(target=_carica_pagina_background, daemon=True)
        t.start()

    # Polling: attende al massimo 2s per mostrare aggiornamento visivo, poi rerun
    for step in range(10):
        time.sleep(0.4)
        progress_bar.progress((step + 1) * 10, text=f"Analisi AI in corso {'.' * ((step % 3) + 1)}")
        if motore.pagina_disponibile(pagina):
            break

    if motore.pagina_disponibile(pagina):
        # Pulizia flag thread dalla session_state
        st.session_state.pop(chiave_thread, None)
        progress_bar.empty()
        status_placeholder.empty()
        st.rerun()
    else:
        # Non ancora pronta: forza rerun per continuare il polling
        st.rerun()

else:
    # Pagina già in cache: recupero istantaneo
    annunci_pagina = motore.recupera_pagina(pagina)


# ─────────────────────────────────────────────
# FILTRO E STATISTICHE
# ─────────────────────────────────────────────

if not annunci_pagina:
    st.warning("⚠️ Nessun annuncio trovato per questa pagina. Prova con parametri diversi.")
    if pagina > 1:
        if st.button("← Torna alla pagina precedente"):
            st.session_state.pagina_corrente -= 1
            st.rerun()
    st.stop()

# Applica filtro score
if st.session_state.mostra_filtrati:
    annunci_visibili = [a for a in annunci_pagina if a.get("ai_score", 0) >= 7]
else:
    annunci_visibili = annunci_pagina

# Statistiche della pagina corrente
stats = calcola_statistiche(annunci_pagina)

st.markdown(f"""
<div class="stats-box">
  <div class="stat-item"><span class="stat-value">{stats.get('totale_analizzati', 0)}</span>Analizzati</div>
  <div class="stat-item"><span class="stat-value" style="color:#00E676;">{stats.get('totale_qualificati', 0)}</span>Score ≥ 7</div>
  <div class="stat-item"><span class="stat-value" style="color:#FFD740;">{stats.get('punteggio_medio', 0)}</span>Score medio</div>
  <div class="stat-item"><span class="stat-value" style="color:#C8F135;">{stats.get('punteggio_max', 0)}/10</span>Score max</div>
  <div class="stat-item" style="margin-left:auto;"><span style="color:#7A8099;font-size:0.8rem;">
    {"🔎 Filtro attivo: solo score ≥ 7" if st.session_state.mostra_filtrati else "📋 Tutti gli annunci — scegli tu a cosa candidarti"}
  </span></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RENDER CARD ANNUNCI
# ─────────────────────────────────────────────

if not annunci_visibili:
    st.info(
        "🤔 Nessun annuncio con score ≥ 7 in questa pagina. "
        "Prova la pagina successiva o disattiva il filtro dalla sidebar."
    )
else:
    for annuncio in annunci_visibili:
        score       = annuncio.get("ai_score", 0)
        titolo      = annuncio.get("title", "Titolo non disponibile")
        azienda     = annuncio.get("company", "Azienda non specificata")
        localita    = annuncio.get("location", "N/A")
        tipo_lavoro = annuncio.get("job_type", "")
        sorgente    = annuncio.get("site", "N/A")
        url         = annuncio.get("job_url", "#")
        descrizione = annuncio.get("description", "")
        stipendio   = formatta_stipendio(annuncio)
        motivazione = annuncio.get("ai_motivazione", "")
        match_tags  = annuncio.get("ai_competenze_match", [])
        gap_tags    = annuncio.get("ai_competenze_mancanti", [])

        # Costruisce l'HTML dei tag competenze
        tags_match_html = "".join([
            f'<span class="tag tag-match">✓ {t}</span>'
            for t in match_tags[:5]
        ])
        tags_gap_html = "".join([
            f'<span class="tag tag-missing">✗ {t}</span>'
            for t in gap_tags[:3]
        ])

        # Colore del badge in base al punteggio
        colore = colore_punteggio(score)
        emoji  = emoji_punteggio(score)

        st.markdown(f"""
        <div class="job-card">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem;">
            <div>
              <div class="score-badge" style="background: {colore}22; color: {colore}; border: 1px solid {colore}44;">
                {emoji} {score}/10
              </div>
              <div class="job-title">{titolo}</div>
              <div class="job-company">{azienda}</div>
              <div class="job-meta">
                📍 {localita}
                {'&nbsp;&nbsp;|&nbsp;&nbsp;💰 ' + stipendio if stipendio != 'Non specificato' else ''}
                {'&nbsp;&nbsp;|&nbsp;&nbsp;🕐 ' + tipo_lavoro if tipo_lavoro else ''}
                &nbsp;&nbsp;|&nbsp;&nbsp;{sorgente_emoji(sorgente)}
              </div>
            </div>
            <div>
              <a href="{url}" target="_blank" style="
                font-family: 'Space Mono', monospace;
                font-size: 0.8rem;
                color: #C8F135;
                text-decoration: none;
                border: 1px solid #C8F13544;
                padding: 6px 14px;
                border-radius: 6px;
                background: #C8F13510;
                white-space: nowrap;
              ">Apri annuncio →</a>
            </div>
          </div>

          <div class="job-desc">{tronca_descrizione(descrizione)}</div>

          {f'<div class="job-ai-note">💡 {motivazione}</div>' if motivazione else ''}

          {f'<div style="margin-top:0.6rem;">{tags_match_html}{tags_gap_html}</div>' if (match_tags or gap_tags) else ''}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────
# PAGINAZIONE
# ─────────────────────────────────────────────

col_prev, col_info, col_next = st.columns([1, 2, 1])

with col_prev:
    pagina_prec_disabilitata = (pagina <= 1)
    if st.button(
        "← Pagina Precedente",
        disabled=pagina_prec_disabilitata,
        use_container_width=True,
    ):
        st.session_state.pagina_corrente -= 1
        st.rerun()

with col_info:
    # Indicatore pagina con stato prefetch
    prossima_pronta = motore.pagina_disponibile(pagina + 1)
    indicatore_prefetch = "⚡ pronta" if prossima_pronta else "⏳ in caricamento..."

    st.markdown(f"""
    <div class="pagination-area">
      <span>Pagina <strong style="color:#C8F135;">{pagina}</strong></span>
      <span style="font-size:0.75rem;">· Prossima: {indicatore_prefetch}</span>
    </div>
    """, unsafe_allow_html=True)

with col_next:
    prossima_pagina_disponibile = motore.ha_piu_pagine(pagina)
    if st.button(
        "Prossima Pagina →",
        disabled=not prossima_pagina_disponibile,
        use_container_width=True,
    ):
        st.session_state.pagina_corrente += 1
        st.rerun()


# ─────────────────────────────────────────────
# DEBUG INFO (espandibile, solo per sviluppo)
# ─────────────────────────────────────────────

with st.expander("🛠️ Debug Info", expanded=False):
    st.json(motore.stato_cache())
    st.write(f"**Pagina corrente:** {pagina}")
    st.write(f"**Annunci in questa pagina:** {len(annunci_pagina)}")
    st.write(f"**Annunci visibili (filtro):** {len(annunci_visibili)}")
    st.write(f"**CV caricato:** {'Sì' if st.session_state.testo_cv else 'No'} "
             f"({len(st.session_state.testo_cv)} caratteri)")