# GeekMAN

> **Disambiguating usernames across platforms: the GeekMAN approach**
> Companion website for the journal paper in *Social Network Analysis and
> Mining* (2024). [`https://doi.org/10.1007/s13278-024-01321-x`](https://doi.org/10.1007/s13278-024-01321-x)

GeekMAN is a systematic, human-inspired approach to matching usernames of
*technogeek* users across online platforms — security forums, GitHub,
YouTube, etc. Given two usernames it returns a similarity score in
**[0, 1]**, where 1 is a perfect match.

This repository contains both the original research notebook and a
[Streamlit](https://streamlit.io) web interface that exposes the algorithm
as an interactive demo.

---

## 🚀 Live demo

Once deployed, the app is available at
`https://<your-subdomain>.streamlit.app` (the original was
`geekman.streamlit.app`).

The interface offers three tabs:

1. **Regular Search** — compare two usernames interactively and see a
   per-feature breakdown of the score.
2. **File Upload** — upload a CSV of username pairs and download the
   results with a `sim_score` column appended.
3. **Cite This Work** — citations in BibTeX, APA and IEEE.

---

## 🖥 Run locally

```bash
git clone https://github.com/<your-username>/geekman.git
cd geekman
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app will open at <http://localhost:8501>.

---

## 📁 Project structure

```
geekman/
├── app.py                          # Streamlit web interface
├── geekman.py                      # Algorithm module (importable, tested)
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version (for Streamlit Cloud)
├── .streamlit/
│   └── config.toml                 # Theme + server settings
├── resources/
│   └── curated_word_list.txt       # ~46k-word dictionary
├── data/
│   └── sample.csv                  # Tiny sample to test File Upload
├── .gitignore
└── README.md
```

---

## 🧠 Algorithm overview

For each username GeekMAN computes six different *views*:

| View | Used by |
|---|---|
| Lowercase form | Levenshtein |
| Letters-only (no digits/symbols) | Levenshtein |
| Capitalisation chunks (CamelCase split) | Jaccard |
| Symbol-delimited chunks | Monge–Elkan |
| Digit-delimited letter chunks | Monge–Elkan |
| Dictionary-token chunks (slang-aware) | Monge–Elkan (weighted) |

The final similarity is the **maximum** across the six per-view scores.
Slang/leet-speak transformations (`0→o`, `3→e`, `4→a`, `@→a`, `!→i`, …)
are applied before dictionary chunking so that pairs like
`Anon-Exploiter` ↔ `An0n3xpl0it3r` are correctly matched.

---

## 📚 Citation

If you use GeekMAN in your research, please cite the journal paper:

```bibtex
@article{masud2024disambiguating,
  title   = {Disambiguating usernames across platforms: the {GeekMAN} approach},
  author  = {Masud, Md Rayhanul and Treves, Ben and Faloutsos, Michalis},
  journal = {Social Network Analysis and Mining},
  volume  = {14},
  number  = {1},
  pages   = {177},
  year    = {2024},
  doi     = {10.1007/s13278-024-01321-x}
}
```

The original conference version (ASONAM 2023):

```bibtex
@inproceedings{masud2023geekman,
  title     = {{GeekMAN}: Geek-oriented username Matching Across online Networks},
  author    = {Masud, Md Rayhanul and Treves, Ben and Faloutsos, Michalis},
  booktitle = {Proc. ASONAM},
  year      = {2023},
  publisher = {ACM},
  doi       = {10.1145/3625007.3627498}
}
```

---

## 🛠 Built with

- [Streamlit](https://streamlit.io)
- [py_stringmatching](https://github.com/anhaidgroup/py_stringmatching)
- [pandas](https://pandas.pydata.org)
