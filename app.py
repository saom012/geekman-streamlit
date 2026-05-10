"""
GeekMAN — Streamlit web interface.

Companion site for the paper

    Masud, M. R., Treves, B. & Faloutsos, M.
    "Disambiguating usernames across platforms: the GeekMAN approach."
    Social Network Analysis and Mining 14, 177 (2024).
    https://doi.org/10.1007/s13278-024-01321-x

Three tabs:

    1. Regular Search — compare two usernames interactively
    2. File Upload   — batch-score a CSV of username pairs
    3. Cite This Work — citation in BibTeX / APA / IEEE
"""

from __future__ import annotations

import io
import os
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
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/mrayhanulmasud/geekman",
        "Report a bug": "https://github.com/mrayhanulmasud/geekman/issues",
        "About": (
            "GeekMAN matches geek-oriented usernames across online platforms. "
            "Companion site for the paper *Disambiguating usernames across "
            "platforms: the GeekMAN approach* (SNAM 2024)."
        ),
    },
)

# --------------------------------------------------------------------------- #
# Styling                                                                     #
# --------------------------------------------------------------------------- #

# A small amount of bespoke CSS to lift the default Streamlit look to
# something a bit more "academic-paper companion site"-feeling without
# becoming a maintenance burden. Keep this short on purpose.
st.markdown(
    """
    <style>
      /* tighten the top padding */
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 880px;}

      /* the hero block */
      .gm-hero {
          padding: 1.6rem 1.8rem;
          border-radius: 12px;
          background: linear-gradient(135deg, #fff5f5 0%, #ffffff 60%);
          border: 1px solid #f1d5d8;
          margin-bottom: 1.4rem;
      }
      .gm-hero h1 {
          font-size: 1.9rem; font-weight: 700; margin: 0 0 .35rem 0;
          color: #1f2937; letter-spacing: -.01em;
      }
      .gm-hero p { margin: 0; color: #4b5563; font-size: .98rem; line-height: 1.55; }
      .gm-hero .gm-pill {
          display: inline-block; font-size: .72rem; letter-spacing: .08em;
          text-transform: uppercase; color: #C8102E; font-weight: 700;
          margin-bottom: .35rem;
      }

      /* score card */
      .gm-score-card {
          padding: 1.2rem 1.4rem; border-radius: 10px; border: 1px solid #e5e7eb;
          background: #fafafa; margin-top: 1rem;
      }
      .gm-score-value { font-size: 2.4rem; font-weight: 700; line-height: 1; }
      .gm-score-label {
          color: #6b7280; font-size: .82rem; letter-spacing: .06em;
          text-transform: uppercase; margin-bottom: .35rem;
      }
      .gm-score-hint  { color: #6b7280; font-size: .9rem; margin-top: .5rem; }

      /* footer */
      .gm-footer {
          margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid #e5e7eb;
          color: #6b7280; font-size: .85rem; text-align: center;
      }
      .gm-footer a { color: #C8102E; text-decoration: none; }
      .gm-footer a:hover { text-decoration: underline; }

      /* example pill buttons */
      .stButton>button[kind="secondary"] { font-size: .85rem; }
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
    """Load the curated dictionary once per app session."""
    if not WORD_LIST_PATH.exists():
        st.error(
            f"Dictionary file not found at `{WORD_LIST_PATH}`. "
            "Please ensure `resources/curated_word_list.txt` is present."
        )
        st.stop()
    return load_word_list(str(WORD_LIST_PATH))


# Trigger early load so the first interaction feels snappy
WORD_LIST = get_word_list()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _score_color(score: float) -> str:
    """Map a similarity score to a hex colour for the score card."""
    if score >= 0.80:
        return "#15803d"   # green
    if score >= 0.50:
        return "#b45309"   # amber
    return "#b91c1c"       # red


def _score_hint(score: float) -> str:
    if score >= 0.80:
        return "Strong match — likely the same person."
    if score >= 0.50:
        return "Moderate match — worth a closer look."
    if score > 0.0:
        return "Weak match — probably different people."
    return "No meaningful overlap."


EXAMPLE_PAIRS: list[tuple[str, str]] = [
    ("Anon-Exploiter", "An0n3xpl0it3r"),
    ("BlackHat42",     "blackhat_42"),
    ("DragonSlayer",   "xX_Dragon_Xx"),
    ("hacker_joe",     "JoeHacker"),
]


# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "**GeekMAN** is a systematic, human-inspired approach to "
        "matching usernames of *technogeek* users across online platforms — "
        "security forums, GitHub, YouTube and beyond.\n\n"
        "Given two usernames it returns a similarity score in **[0, 1]** "
        "(1 = perfect match)."
    )

    st.markdown("### How it works")
    st.markdown(
        "The algorithm decomposes each username into multiple views — "
        "lowercase form, capitalisation, symbols, digits, slang/leet-speak, "
        "and dictionary words — and combines per-view similarities "
        "(Levenshtein, Jaccard, Monge–Elkan) into a single score."
    )

    st.markdown("### Links")
    st.markdown(
        "- 📄 [Journal paper (SNAM 2024)]"
        "(https://doi.org/10.1007/s13278-024-01321-x)\n"
        "- 📑 [Conference paper (ASONAM 2023)]"
        "(https://doi.org/10.1145/3625007.3627498)\n"
        "- 💻 [Source code on GitHub]"
        "(https://github.com/mrayhanulmasud/geekman)"
    )

    st.markdown("---")
    st.caption(
        "Built with [Streamlit](https://streamlit.io) and "
        "[py_stringmatching](https://github.com/anhaidgroup/py_stringmatching)."
    )


# --------------------------------------------------------------------------- #
# Hero                                                                        #
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="gm-hero">
      <span class="gm-pill">Research demo · SNAM 2024</span>
      <h1>Disambiguating usernames across platforms: the GeekMAN approach</h1>
      <p>Compare two usernames and get a similarity score in [0, 1].
      Use the <em>File Upload</em> tab to score thousands of pairs at once.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Tabs                                                                        #
# --------------------------------------------------------------------------- #

tab_search, tab_upload, tab_cite = st.tabs(
    ["🔎  Regular Search", "📁  File Upload", "📚  Cite This Work"]
)


# --------------------------------------------------------------------------- #
# Tab 1 — Regular Search                                                      #
# --------------------------------------------------------------------------- #

with tab_search:
    # initialise session state for prefilled examples
    if "u1" not in st.session_state:
        st.session_state.u1 = "Anon-Exploiter"
    if "u2" not in st.session_state:
        st.session_state.u2 = "An0n3xpl0it3r"

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

    # Try-an-example shortcuts
    with st.container():
        st.caption("Try an example:")
        ex_cols = st.columns(len(EXAMPLE_PAIRS))
        for col, (a, b) in zip(ex_cols, EXAMPLE_PAIRS):
            if col.button(f"{a}  ↔  {b}", use_container_width=True, key=f"ex_{a}_{b}"):
                st.session_state.u1 = a
                st.session_state.u2 = b
                st.rerun()

    st.write("")  # spacer
    compute = st.button(
        "Find Similarity Score",
        type="primary",
        use_container_width=True,
        disabled=not (u1.strip() and u2.strip()),
    )

    if compute:
        with st.spinner("Computing similarity…"):
            breakdown = compute_similarity_detailed(u1.strip(), u2.strip(), WORD_LIST)

        colour = _score_color(breakdown.score)
        st.markdown(
            f"""
            <div class="gm-score-card">
                <div class="gm-score-label">Similarity score</div>
                <div class="gm-score-value" style="color: {colour};">
                    {breakdown.score:.3f}
                </div>
                <div class="gm-score-hint">{_score_hint(breakdown.score)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # progress bar visual cue
        st.progress(min(max(breakdown.score, 0.0), 1.0))

        # detailed breakdown
        with st.expander("See how this score was computed"):
            st.markdown("**Per-feature similarity scores**")
            st.markdown(
                "The final score is the **maximum** across these six signals."
            )
            comp_df = pd.DataFrame(
                [
                    {"Feature": k, "Score": v}
                    for k, v in breakdown.component_scores.items()
                ]
            )
            st.dataframe(
                comp_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                },
            )

            st.markdown("**Decompositions of each username**")
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
            st.dataframe(chunk_df, hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 2 — File Upload                                                         #
# --------------------------------------------------------------------------- #

with tab_upload:
    st.markdown(
        "Upload a **CSV** file with two columns of username pairs. "
        "GeekMAN will score every row and return a downloadable CSV with a "
        "`sim_score` column appended."
    )

    with st.expander("CSV format and example", expanded=False):
        st.markdown(
            "- The file must be comma-separated.\n"
            "- It can either have a header row (`username_1,username_2`) "
            "or no header at all — both are handled.\n"
            "- Each row should contain exactly two values.\n"
        )
        sample = pd.DataFrame(
            EXAMPLE_PAIRS, columns=["username_1", "username_2"]
        )
        st.dataframe(sample, hide_index=True, use_container_width=True)
        st.download_button(
            "⬇ Download a sample CSV",
            data=sample.to_csv(index=False).encode("utf-8"),
            file_name="geekman_sample.csv",
            mime="text/csv",
            use_container_width=False,
        )

    uploaded = st.file_uploader(
        "Drag and drop a CSV file here",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        # Try to read with header first; if both columns aren't usernames, fall back.
        try:
            raw = uploaded.getvalue().decode("utf-8", errors="replace")
            df_try = pd.read_csv(io.StringIO(raw))
            if {"username_1", "username_2"}.issubset(df_try.columns):
                df = df_try[["username_1", "username_2"]].copy()
            elif df_try.shape[1] >= 2:
                df = df_try.iloc[:, :2].copy()
                df.columns = ["username_1", "username_2"]
            else:
                # single column → re-read without header
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
        st.dataframe(df.head(10), hide_index=True, use_container_width=True)

        if st.button(
            "Compute similarity scores",
            type="primary",
            use_container_width=True,
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

            # quick stats
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean score", f"{result['sim_score'].mean():.3f}")
            c2.metric("Strong matches (≥0.80)", int((result["sim_score"] >= 0.80).sum()))
            c3.metric("Weak matches (<0.50)", int((result["sim_score"] < 0.50).sum()))

            st.dataframe(
                result,
                hide_index=True,
                use_container_width=True,
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
                "⬇ Download results as CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="geekman_sim_score.csv",
                mime="text/csv",
                type="primary",
            )


# --------------------------------------------------------------------------- #
# Tab 3 — Cite This Work                                                      #
# --------------------------------------------------------------------------- #

with tab_cite:
    st.markdown(
        "If you use GeekMAN in your research, please cite the journal paper. "
        "Both the conference and journal versions are listed below."
    )

    st.markdown("#### Journal article (recommended citation)")
    st.markdown(
        "> Masud, M. R., Treves, B. & Faloutsos, M. **Disambiguating usernames "
        "across platforms: the GeekMAN approach.** *Social Network Analysis "
        "and Mining* **14**, 177 (2024). "
        "[https://doi.org/10.1007/s13278-024-01321-x]"
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
    st.markdown("**BibTeX**")
    st.code(bibtex_journal, language="bibtex")

    st.markdown("**APA**")
    st.code(
        "Masud, M. R., Treves, B., & Faloutsos, M. (2024). Disambiguating "
        "usernames across platforms: the GeekMAN approach. Social Network "
        "Analysis and Mining, 14, 177. "
        "https://doi.org/10.1007/s13278-024-01321-x",
        language="text",
    )

    st.markdown("**IEEE**")
    st.code(
        "M. R. Masud, B. Treves, and M. Faloutsos, \"Disambiguating usernames "
        "across platforms: the GeekMAN approach,\" Social Network Analysis "
        "and Mining, vol. 14, p. 177, 2024.",
        language="text",
    )

    st.markdown("---")
    st.markdown("#### Original conference paper")
    st.markdown(
        "> Masud, M. R., Treves, B. & Faloutsos, M. **GeekMAN: Geek-oriented "
        "username Matching Across online Networks.** In *Proc. ASONAM 2023*, "
        "ACM, 2023. "
        "[https://doi.org/10.1145/3625007.3627498]"
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
    st.markdown("**BibTeX**")
    st.code(bibtex_conf, language="bibtex")


# --------------------------------------------------------------------------- #
# Footer                                                                      #
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="gm-footer">
        Made with ❤ for academic research ·
        <a href="https://github.com/mrayhanulmasud/geekman">Source on GitHub</a> ·
        <a href="https://doi.org/10.1007/s13278-024-01321-x">Read the paper</a>
    </div>
    """,
    unsafe_allow_html=True,
)
