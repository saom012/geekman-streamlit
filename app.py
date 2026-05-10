"""
GeekMAN — Streamlit web interface.

Companion site for the paper

    Masud, M. R., Treves, B. & Faloutsos, M.
    "Disambiguating usernames across platforms: the GeekMAN approach."
    Social Network Analysis and Mining 14, 177 (2024).
    https://doi.org/10.1007/s13278-024-01321-x
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from geekman import (
    compute_batch,
    compute_similarity_detailed,
    load_word_list,
)

# --------------------------------------------------------------------------- #
# Page setup                                                                  #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="GeekMAN — Username Matching Across Online Networks",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://github.com/mrayhanulmasud/geekman",
        "Report a bug": "https://github.com/mrayhanulmasud/geekman/issues",
        "About": (
            "GeekMAN — match usernames across online platforms. "
            "Companion site for *Disambiguating usernames across platforms: "
            "the GeekMAN approach* (SNAM 2024)."
        ),
    },
)

# --------------------------------------------------------------------------- #
# Styling                                                                     #
# --------------------------------------------------------------------------- #
#
# Design language: warm cream background, white surfaces, deep navy accent,
# generous whitespace, tabular type for numbers. The CSS below is intentionally
# scoped through `gm-*` classes and a few `.stButton`, `.stTabs`, `.stTextInput`
# overrides so it does not fight Streamlit upgrades.
#

st.markdown(
    """
    <style>
      /* ─────────────────  GLOBAL  ───────────────── */

      :root {
        --gm-bg:           #FAF7F2;
        --gm-surface:      #FFFFFF;
        --gm-border:       #E8E1D5;
        --gm-border-2:     #D8CFBF;
        --gm-text:         #111827;
        --gm-muted:        #6B7280;
        --gm-muted-2:      #9CA3AF;
        --gm-primary:      #1E40AF;
        --gm-primary-700:  #1E3A8A;
        --gm-primary-50:   #EEF2FF;
        --gm-strong:       #166534;
        --gm-medium:       #A16207;
        --gm-weak:         #991B1B;
      }

      html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     "Inter", Roboto, Helvetica, Arial, sans-serif;
        color: var(--gm-text);
      }

      /* tighten layout */
      .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 3rem !important;
        max-width: 920px;
      }

      /* hide Streamlit chrome we don't want */
      header[data-testid="stHeader"] { background: transparent; }
      footer { visibility: hidden; height: 0; }
      .viewerBadge_container__1QSob,
      [data-testid="stDecoration"] { display: none !important; }

      /* ─────────────────  TOP BAR  ───────────────── */

      .gm-topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: .35rem 0 1.1rem 0;
        border-bottom: 1px solid var(--gm-border);
        margin-bottom: 1.6rem;
      }
      .gm-brand {
        display: flex; align-items: center; gap: .55rem;
        font-weight: 700; font-size: 1.05rem; letter-spacing: -.01em;
        color: var(--gm-text);
      }
      .gm-brand .gm-logo-dot {
        width: 9px; height: 9px; border-radius: 50%;
        background: var(--gm-primary); display: inline-block;
      }
      .gm-nav { display: flex; gap: 1.4rem; }
      .gm-nav a {
        color: var(--gm-muted); text-decoration: none;
        font-size: .9rem; font-weight: 500;
        transition: color .15s ease;
      }
      .gm-nav a:hover { color: var(--gm-primary); }

      /* ─────────────────  HERO  ───────────────── */

      .gm-hero { margin: .4rem 0 1.8rem 0; }
      .gm-hero h1 {
        font-size: 2.3rem; font-weight: 700;
        letter-spacing: -.02em; line-height: 1.1;
        margin: 0 0 .55rem 0; color: var(--gm-text);
      }
      .gm-hero .gm-tagline {
        font-size: 1.05rem; color: var(--gm-muted);
        margin: 0 0 .9rem 0; line-height: 1.55;
        max-width: 640px;
      }
      .gm-hero .gm-byline {
        font-size: .82rem; color: var(--gm-muted-2);
        letter-spacing: .02em;
      }
      .gm-hero .gm-byline a {
        color: var(--gm-muted); text-decoration: none;
        border-bottom: 1px dotted var(--gm-border-2);
      }
      .gm-hero .gm-byline a:hover { color: var(--gm-primary); }

      /* ─────────────────  TABS  ───────────────── */

      .stTabs [data-baseweb="tab-list"] {
        gap: 1.6rem;
        border-bottom: 1px solid var(--gm-border);
      }
      .stTabs [data-baseweb="tab"] {
        padding: .55rem 0 .8rem 0 !important;
        background: transparent !important;
        font-weight: 500; color: var(--gm-muted);
      }
      .stTabs [aria-selected="true"] {
        color: var(--gm-text) !important;
        font-weight: 600;
      }
      .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--gm-primary) !important;
        height: 2px !important;
      }

      /* ─────────────────  STEP STRIP  ───────────────── */
      /* small "1 · 2 · 3" guidance row used at the top of each tab */

      .gm-howto {
        background: var(--gm-surface);
        border: 1px solid var(--gm-border);
        border-radius: 10px;
        padding: .85rem 1rem;
        margin: 1rem 0 1.2rem 0;
        display: flex; gap: 1.2rem; flex-wrap: wrap;
        font-size: .9rem;
      }
      .gm-howto-step {
        display: flex; align-items: center; gap: .55rem;
        color: var(--gm-text);
      }
      .gm-howto-step .gm-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 22px; height: 22px; border-radius: 50%;
        background: var(--gm-primary-50); color: var(--gm-primary);
        font-size: .78rem; font-weight: 600;
      }

      /* ─────────────────  INPUTS  ───────────────── */

      div[data-baseweb="input"] > div {
        border: 1px solid var(--gm-border) !important;
        border-radius: 8px !important;
        background: var(--gm-surface) !important;
        transition: border-color .15s, box-shadow .15s;
      }
      div[data-baseweb="input"] > div:focus-within {
        border-color: var(--gm-primary) !important;
        box-shadow: 0 0 0 3px rgba(30, 64, 175, .12) !important;
      }
      .stTextInput label,
      .stFileUploader label,
      .stSelectbox label {
        color: var(--gm-text) !important;
        font-weight: 500 !important;
        font-size: .92rem !important;
      }

      /* ─────────────────  BUTTONS  ───────────────── */

      .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all .15s ease;
      }
      /* primary CTA */
      .stButton > button[kind="primary"] {
        background: var(--gm-primary);
        border-color: var(--gm-primary);
        padding: .65rem 1rem;
      }
      .stButton > button[kind="primary"]:hover {
        background: var(--gm-primary-700);
        border-color: var(--gm-primary-700);
      }
      .stButton > button[kind="primary"]:disabled {
        background: var(--gm-border-2); border-color: var(--gm-border-2);
        color: var(--gm-muted);
      }
      /* secondary (example chips) */
      .stButton > button[kind="secondary"] {
        background: var(--gm-surface);
        border: 1px solid var(--gm-border);
        color: var(--gm-text);
        font-size: .85rem;
        font-weight: 500;
        padding: .45rem .75rem;
      }
      .stButton > button[kind="secondary"]:hover {
        border-color: var(--gm-primary);
        color: var(--gm-primary);
      }

      /* ─────────────────  RESULT CARD  ───────────────── */

      .gm-result {
        margin-top: 1.4rem; padding: 1.4rem 1.5rem;
        background: var(--gm-surface);
        border: 1px solid var(--gm-border);
        border-radius: 12px;
      }
      .gm-result .gm-result-label {
        font-size: .75rem; color: var(--gm-muted);
        letter-spacing: .08em; text-transform: uppercase;
        margin-bottom: .35rem;
      }
      .gm-result .gm-result-value {
        font-feature-settings: "tnum" 1;
        font-size: 2.6rem; font-weight: 700; line-height: 1;
        margin-bottom: .35rem;
      }
      .gm-result .gm-result-hint {
        color: var(--gm-muted); font-size: .92rem;
      }
      .gm-result .gm-result-pair {
        margin-top: .9rem; padding-top: .9rem;
        border-top: 1px dashed var(--gm-border);
        font-size: .85rem; color: var(--gm-muted);
      }
      .gm-result .gm-result-pair code {
        background: transparent; color: var(--gm-text);
        font-size: .9rem;
      }

      /* progress bar tint */
      .stProgress > div > div > div { background: var(--gm-primary) !important; }

      /* ─────────────────  EXPANDER & TABLES  ───────────────── */

      details summary { font-weight: 500; }
      .stDataFrame, .stDataEditor { border-radius: 10px; }

      /* ─────────────────  ALERTS  ───────────────── */

      div[data-testid="stAlert"] { border-radius: 10px; }

      /* ─────────────────  ABOUT / CITE COPY  ───────────────── */

      .gm-prose h3 {
        font-size: 1.1rem; font-weight: 600; margin-top: 1.6rem;
        color: var(--gm-text);
      }
      .gm-prose p, .gm-prose li {
        color: var(--gm-text); line-height: 1.65;
        font-size: .96rem;
      }
      .gm-prose .gm-abstract {
        background: var(--gm-surface);
        border: 1px solid var(--gm-border);
        border-left: 3px solid var(--gm-primary);
        border-radius: 8px;
        padding: 1rem 1.2rem; margin: .8rem 0;
        font-size: .95rem; color: var(--gm-text); line-height: 1.7;
      }
      .gm-prose a { color: var(--gm-primary); text-decoration: none; }
      .gm-prose a:hover { text-decoration: underline; }

      /* ─────────────────  FOOTER  ───────────────── */

      .gm-footer {
        margin-top: 3rem; padding-top: 1.1rem;
        border-top: 1px solid var(--gm-border);
        color: var(--gm-muted-2); font-size: .8rem;
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: .6rem;
      }
      .gm-footer a {
        color: var(--gm-muted); text-decoration: none;
      }
      .gm-footer a:hover { color: var(--gm-primary); }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources                                                            #
# --------------------------------------------------------------------------- #

WORD_LIST_PATH = Path(__file__).parent / "resources" / "curated_word_list.txt"


@st.cache_resource(show_spinner="Loading dictionary (≈46k words, one time only)…")
def get_word_list() -> set[str]:
    if not WORD_LIST_PATH.exists():
        st.error(
            f"Dictionary file not found at `{WORD_LIST_PATH}`. "
            "Please ensure `resources/curated_word_list.txt` is present."
        )
        st.stop()
    return load_word_list(str(WORD_LIST_PATH))


WORD_LIST = get_word_list()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

EXAMPLE_PAIRS: list[tuple[str, str]] = [
    ("Anon-Exploiter", "An0n3xpl0it3r"),
    ("BlackHat42",     "blackhat_42"),
    ("DragonSlayer",   "xX_Dragon_Xx"),
]


def _score_color(score: float) -> str:
    if score >= 0.80:
        return "var(--gm-strong)"
    if score >= 0.50:
        return "var(--gm-medium)"
    return "var(--gm-weak)"


def _score_hint(score: float) -> str:
    if score >= 0.80:
        return "Strong match — these usernames likely belong to the same person."
    if score >= 0.50:
        return "Moderate match — worth a closer look."
    if score > 0.0:
        return "Weak match — these usernames probably belong to different people."
    return "No meaningful overlap detected."


def _set_example(u1_val: str, u2_val: str) -> None:
    """Callback used by the example chips.

    This runs *between* script re-runs (i.e. before any widget renders),
    which is the only safe time to mutate session state for keys bound
    to widgets.
    """
    st.session_state["u1"] = u1_val
    st.session_state["u2"] = u2_val


# --------------------------------------------------------------------------- #
# Top bar                                                                     #
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="gm-topbar">
      <div class="gm-brand">
        <span class="gm-logo-dot"></span> GeekMAN
      </div>
      <nav class="gm-nav">
        <a href="https://doi.org/10.1007/s13278-024-01321-x"
           target="_blank" rel="noopener">Paper</a>
        <a href="https://github.com/mrayhanulmasud/geekman"
           target="_blank" rel="noopener">GitHub</a>
      </nav>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Hero                                                                        #
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="gm-hero">
      <h1>GeekMAN</h1>
      <p class="gm-tagline">
        Username matching across online networks. GeekMAN scores how likely
        two usernames belong to the same person — accounting for
        capitalisation, leet-speak, dictionary tokens, and more.
      </p>
      <p class="gm-byline">
        From the paper
        <a href="https://doi.org/10.1007/s13278-024-01321-x" target="_blank"
           rel="noopener"><em>Disambiguating usernames across platforms:
        the GeekMAN approach</em></a>
        · Masud, Treves, Faloutsos · Social Network Analysis and Mining (2024)
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Tabs                                                                        #
# --------------------------------------------------------------------------- #

tab_compare, tab_batch, tab_about, tab_cite = st.tabs(
    ["Compare", "Batch upload", "About", "Citation"]
)


# --------------------------------------------------------------------------- #
# Tab — Compare                                                               #
# --------------------------------------------------------------------------- #

with tab_compare:
    st.markdown(
        """
        <div class="gm-howto">
          <div class="gm-howto-step">
            <span class="gm-num">1</span> Enter two usernames
          </div>
          <div class="gm-howto-step">
            <span class="gm-num">2</span> Click <em>Compute similarity</em>
          </div>
          <div class="gm-howto-step">
            <span class="gm-num">3</span> Read the score (0–1)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # seed state on first run *before* the widgets render
    if "u1" not in st.session_state:
        st.session_state["u1"] = "Anon-Exploiter"
    if "u2" not in st.session_state:
        st.session_state["u2"] = "An0n3xpl0it3r"

    col_a, col_b = st.columns(2)
    with col_a:
        u1 = st.text_input(
            "Username 1",
            key="u1",
            placeholder="e.g. Anon-Exploiter",
            max_chars=80,
        )
    with col_b:
        u2 = st.text_input(
            "Username 2",
            key="u2",
            placeholder="e.g. An0n3xpl0it3r",
            max_chars=80,
        )

    # examples — using on_click callback so we never mutate widget-bound
    # session state from inside the script body (which is what was causing
    # the StreamlitAPIException previously).
    st.caption("Or try one of these examples:")
    ex_cols = st.columns(len(EXAMPLE_PAIRS))
    for col, (a, b) in zip(ex_cols, EXAMPLE_PAIRS):
        col.button(
            f"{a}  ↔  {b}",
            width="stretch",
            key=f"ex_{a}_{b}",
            on_click=_set_example,
            args=(a, b),
        )

    st.write("")
    compute = st.button(
        "Compute similarity",
        type="primary",
        width="stretch",
        disabled=not (u1.strip() and u2.strip()),
    )

    if compute:
        with st.spinner("Computing similarity…"):
            breakdown = compute_similarity_detailed(
                u1.strip(), u2.strip(), WORD_LIST
            )

        colour = _score_color(breakdown.score)
        st.markdown(
            f"""
            <div class="gm-result">
              <div class="gm-result-label">Similarity score</div>
              <div class="gm-result-value" style="color: {colour};">
                {breakdown.score:.3f}
              </div>
              <div class="gm-result-hint">{_score_hint(breakdown.score)}</div>
              <div class="gm-result-pair">
                <code>{u1.strip()}</code> &nbsp;↔&nbsp; <code>{u2.strip()}</code>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(breakdown.score, 0.0), 1.0))

        with st.expander("How was this score computed?"):
            st.markdown(
                "GeekMAN combines six per-feature similarity signals and "
                "returns the **maximum** as the final score. "
                "The dominant signal for this pair is highlighted by being "
                "closest to the final value."
            )
            comp_df = pd.DataFrame(
                [{"Feature": k, "Score": v}
                 for k, v in breakdown.component_scores.items()]
            )
            st.dataframe(
                comp_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                },
            )

            st.markdown("**How each username was decomposed:**")
            chunk_df = pd.DataFrame(
                {
                    "View": list(breakdown.chunkifications_1.keys()),
                    f"Username 1 — {u1}": [
                        v if not isinstance(v, list) else ", ".join(v) or "—"
                        for v in breakdown.chunkifications_1.values()
                    ],
                    f"Username 2 — {u2}": [
                        v if not isinstance(v, list) else ", ".join(v) or "—"
                        for v in breakdown.chunkifications_2.values()
                    ],
                }
            )
            st.dataframe(chunk_df, hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# Tab — Batch upload                                                          #
# --------------------------------------------------------------------------- #

with tab_batch:
    st.markdown(
        """
        <div class="gm-howto">
          <div class="gm-howto-step">
            <span class="gm-num">1</span> Prepare a CSV with two columns
          </div>
          <div class="gm-howto-step">
            <span class="gm-num">2</span> Upload below
          </div>
          <div class="gm-howto-step">
            <span class="gm-num">3</span> Download the scored results
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample = pd.DataFrame(EXAMPLE_PAIRS, columns=["username_1", "username_2"])

    with st.expander("CSV format and template", expanded=False):
        st.markdown(
            "Your file should be **comma-separated** with two columns of "
            "username pairs. A header row (`username_1,username_2`) is "
            "supported but not required — files without a header are also "
            "handled. Each row becomes one scored pair."
        )
        st.dataframe(sample, hide_index=True, width="stretch")
        st.download_button(
            "Download a sample CSV",
            data=sample.to_csv(index=False).encode("utf-8"),
            file_name="geekman_sample.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        try:
            raw = uploaded.getvalue().decode("utf-8", errors="replace")
            df_try = pd.read_csv(io.StringIO(raw))
            if {"username_1", "username_2"}.issubset(df_try.columns):
                df = df_try[["username_1", "username_2"]].copy()
            elif df_try.shape[1] >= 2:
                df = df_try.iloc[:, :2].copy()
                df.columns = ["username_1", "username_2"]
            else:
                df = pd.read_csv(
                    io.StringIO(raw),
                    header=None,
                    names=["username_1", "username_2"],
                )
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            st.stop()

        df = df.dropna().astype(str)
        if df.empty:
            st.warning("The uploaded file appears to be empty.")
            st.stop()

        n = len(df)
        st.success(f"Loaded **{n}** username pair{'s' if n != 1 else ''}.")
        st.dataframe(df.head(10), hide_index=True, width="stretch")

        if st.button(
            "Compute similarity scores",
            type="primary",
            width="stretch",
            key="batch_run",
        ):
            progress = st.progress(0.0, text="Scoring…")
            result = compute_batch(
                df,
                WORD_LIST,
                progress_callback=lambda p: progress.progress(
                    p, text=f"Scoring… {int(p * 100)}%"
                ),
            )
            progress.empty()
            st.success(f"Done — scored {n} pair{'s' if n != 1 else ''}.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean score", f"{result['sim_score'].mean():.3f}")
            c2.metric(
                "Strong matches (≥ 0.80)",
                int((result["sim_score"] >= 0.80).sum()),
            )
            c3.metric(
                "Weak matches (< 0.50)",
                int((result["sim_score"] < 0.50).sum()),
            )

            st.dataframe(
                result,
                hide_index=True,
                width="stretch",
                column_config={
                    "sim_score": st.column_config.ProgressColumn(
                        "sim_score",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                    ),
                },
            )

            st.download_button(
                "Download results as CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="geekman_sim_score.csv",
                mime="text/csv",
                type="primary",
            )


# --------------------------------------------------------------------------- #
# Tab — About                                                                 #
# --------------------------------------------------------------------------- #

with tab_about:
    st.markdown(
        """
        <div class="gm-prose">

        <h3>Abstract</h3>

        <div class="gm-abstract">
        How can we identify malicious hackers participating in different online
        platforms using their usernames only? Establishing the identity of a
        user across online platforms — security forums, GitHub, YouTube — is an
        essential capability for tracing malicious hackers. Although a hacker
        could pick arbitrary names, they often use the same or similar
        usernames as this helps them establish an online “brand”. We propose
        <strong>GeekMAN</strong>, a systematic human-inspired approach to
        identify similar usernames across online platforms focusing on
        technogeek platforms. GeekMAN decomposes each username into multiple
        views — capitalisation, symbols, digits, slang/leet-speak and
        dictionary words — and combines per-view similarities into a single
        score in [0, 1].
        </div>

        <h3>How it works</h3>
        <p>
        For each input username, GeekMAN extracts six different
        <em>chunkifications</em>: the lowercase form, a letters-only form,
        a CamelCase split, a symbol-delimited split, a digit-delimited split,
        and a dictionary-token split with leet-speak awareness
        (<code>0→o</code>, <code>3→e</code>, <code>4→a</code>, …). It then
        compares each pair of views with an appropriate string-similarity
        measure (Levenshtein, Jaccard, Monge–Elkan) and returns the
        <strong>maximum</strong> across the six per-view scores as the final
        similarity.
        </p>

        <h3>Authors</h3>
        <p>
          Md Rayhanul Masud · Ben Treves · Michalis Faloutsos<br>
          University of California, Riverside
        </p>

        <h3>Publications</h3>
        <ul>
          <li>
            <strong>Journal article (recommended).</strong>
            Masud, M. R., Treves, B. &amp; Faloutsos, M.
            <em>Disambiguating usernames across platforms: the GeekMAN
            approach.</em>
            Social Network Analysis and Mining 14, 177 (2024).
            <a href="https://doi.org/10.1007/s13278-024-01321-x"
               target="_blank" rel="noopener">doi.org/10.1007/s13278-024-01321-x</a>
          </li>
          <li>
            <strong>Conference paper.</strong>
            Masud, M. R., Treves, B. &amp; Faloutsos, M.
            <em>GeekMAN: Geek-oriented username Matching Across online
            Networks.</em>
            ASONAM 2023, ACM.
            <a href="https://doi.org/10.1145/3625007.3627498"
               target="_blank" rel="noopener">doi.org/10.1145/3625007.3627498</a>
          </li>
        </ul>

        <h3>Source code</h3>
        <p>
          The reference implementation and the data used in the paper are
          available at
          <a href="https://github.com/mrayhanulmasud/geekman"
             target="_blank" rel="noopener">github.com/mrayhanulmasud/geekman</a>.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Tab — Citation                                                              #
# --------------------------------------------------------------------------- #

with tab_cite:
    st.markdown(
        '<div class="gm-prose"><p>If GeekMAN is useful in your research, '
        'please cite the journal article. The conference version is also '
        'listed below.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("##### Journal article — recommended citation")
    st.markdown(
        "Masud, M. R., Treves, B. & Faloutsos, M. "
        "**Disambiguating usernames across platforms: the GeekMAN approach.** "
        "*Social Network Analysis and Mining* **14**, 177 (2024). "
        "[doi.org/10.1007/s13278-024-01321-x]"
        "(https://doi.org/10.1007/s13278-024-01321-x)"
    )

    bibtex_journal = """@article{masud2024disambiguating,
  title   = {Disambiguating usernames across platforms: the {GeekMAN} approach},
  author  = {Masud, Md Rayhanul and Treves, Ben and Faloutsos, Michalis},
  journal = {Social Network Analysis and Mining},
  volume  = {14},
  number  = {1},
  pages   = {177},
  year    = {2024},
  doi     = {10.1007/s13278-024-01321-x},
  url     = {https://doi.org/10.1007/s13278-024-01321-x}
}"""
    bib_tab, apa_tab, ieee_tab = st.tabs(["BibTeX", "APA", "IEEE"])
    with bib_tab:
        st.code(bibtex_journal, language="bibtex")
    with apa_tab:
        st.code(
            "Masud, M. R., Treves, B., & Faloutsos, M. (2024). "
            "Disambiguating usernames across platforms: the GeekMAN approach. "
            "Social Network Analysis and Mining, 14, 177. "
            "https://doi.org/10.1007/s13278-024-01321-x",
            language="text",
        )
    with ieee_tab:
        st.code(
            "M. R. Masud, B. Treves, and M. Faloutsos, "
            "\"Disambiguating usernames across platforms: the GeekMAN "
            "approach,\" Social Network Analysis and Mining, vol. 14, "
            "p. 177, 2024.",
            language="text",
        )

    st.markdown("---")
    st.markdown("##### Conference paper")
    st.markdown(
        "Masud, M. R., Treves, B. & Faloutsos, M. "
        "**GeekMAN: Geek-oriented username Matching Across online Networks.** "
        "In *Proc. ASONAM 2023*, ACM, 2023. "
        "[doi.org/10.1145/3625007.3627498]"
        "(https://doi.org/10.1145/3625007.3627498)"
    )
    bibtex_conf = """@inproceedings{masud2023geekman,
  title     = {{GeekMAN}: Geek-oriented username Matching Across online Networks},
  author    = {Masud, Md Rayhanul and Treves, Ben and Faloutsos, Michalis},
  booktitle = {Proceedings of the 2023 IEEE/ACM International Conference on
               Advances in Social Networks Analysis and Mining (ASONAM)},
  year      = {2023},
  publisher = {ACM},
  doi       = {10.1145/3625007.3627498},
  url       = {https://doi.org/10.1145/3625007.3627498}
}"""
    st.code(bibtex_conf, language="bibtex")


# --------------------------------------------------------------------------- #
# Footer                                                                      #
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="gm-footer">
      <span>© GeekMAN · University of California, Riverside</span>
      <span>
        <a href="https://doi.org/10.1007/s13278-024-01321-x"
           target="_blank" rel="noopener">Paper</a>
        &nbsp;·&nbsp;
        <a href="https://github.com/mrayhanulmasud/geekman"
           target="_blank" rel="noopener">GitHub</a>
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)
