import csv
import html
import io
import math
import os
from pathlib import Path
from datetime import datetime
from typing import Any

import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import tiktoken
except Exception:
    tiktoken = None


_ENCODER_CACHE: dict[str, Any] = {}
TOKEN_PALETTE = [
    "#FFF3BF",
    "#D3F9D8",
    "#D0EBFF",
    "#FFE8CC",
    "#E5DBFF",
    "#FFD8E8",
]

EXPLICATIONS_MARKDOWN = """
## Quelques explications

L'objet de l'application est de montrer le caractère aléatoire de la génération des tokens par un modèle de LLM.

Elle propose, à partir d'une question simple, de :
- Générer 5 réponses différentes.
- Décomposer le process de génération token par token pour chacune des réponses.
- Proposer le process de génération token par token en ne retenant systématiquement que le token le plus probable (équivalent à une température égale à 0).

L'appli s'appuie sur ChatGPT 4o mini pour la génération des 5 réponses.

Par défaut, l'appli propose :
- Une question simple.
- Une température de 0.7.
- Une longueur max de la réponse à 30 mots.
- Le nombre de tokens probables sélectionnables (`top n logprobs`) à 5.

Ces paramètres peuvent être modifiés avant le lancement de l'application.

Les boutons numérotés de 1 à 5 permettent d'afficher le process de génération de la réponse, token par token.

Ce schéma illustre, de manière simplifiée, la génération de texte par les modèles de langage token par token (un token étant une unité de texte : morceau de mot, mot, ponctuation, etc.).

### 1) Ce que le modèle reçoit : contexte et prompt
À gauche, le modèle dispose d'un contexte (instructions, historique de la conversation, informations déjà fournies) et d'un prompt (la demande de l'utilisateur). Selon l'application, ce contexte peut être enrichi par des sources externes (Web search, RAG - Retrieval-Augmented Generation).

### 2) Découpage en tokens et traitement d'entrée
Le modèle convertit l'entrée en tokens. Cette étape standardise le contenu (mots, sous-mots, chiffres, ponctuation) afin qu'il puisse être traité mathématiquement.

### 3) Génération des tokens probables : une liste pondérée
Une fois l'entrée traitée, le modèle calcule la suite la plus plausible, non pas en produisant directement une phrase, mais en estimant une distribution de probabilités sur le prochain token.

### 4) Sélection du token : l'endroit où intervient l'aléatoire
L'étape suivante consiste à sélectionner un token parmi les candidats. C'est ici que l'application peut mettre en évidence le caractère aléatoire.

### 5) Production du texte : répétition du cycle
Une fois le token sélectionné, il devient un token de sortie, est ajouté au texte, puis le modèle recommence le cycle pour générer le token suivant (jusqu'à obtenir une phrase, un paragraphe, etc.).
"""

EXPLICATIONS_IMAGE_ANCHOR = (
    "Ce schéma illustre, de manière simplifiée, la génération de texte par les modèles de langage "
    "token par token (un token étant une unité de texte : morceau de mot, mot, ponctuation, etc.)."
)


def field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def format_prob(prob: float | None) -> str:
    if prob is None:
        return "n/a"
    return f"{prob:.4f}%"


def logprob_to_prob(logprob: Any) -> float | None:
    try:
        return math.exp(float(logprob)) * 100.0
    except Exception:
        return None


def visible_token(token: str) -> str:
    if token == "":
        return "<empty>"
    return token.replace(" ", "<sp>").replace("\n", "\\n").replace("\t", "\\t")


def word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def enforce_exact_word_count(text: str, target_words: int, pad_token: str = "PAD") -> tuple[str, int, int]:
    words = [w for w in text.split() if w.strip()]
    raw_count = len(words)
    if raw_count >= target_words:
        strict_words = words[:target_words]
    else:
        strict_words = words + [pad_token] * (target_words - raw_count)
    strict_text = " ".join(strict_words)
    return strict_text, raw_count, len(strict_words)


def rows_to_csv_bytes(rows: list[dict[str, str]]) -> bytes:
    if not rows:
        return b""
    output = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def get_encoder(model_name: str) -> Any:
    if tiktoken is None:
        return None
    key = model_name.strip() or "cl100k_base"
    if key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]
    try:
        encoder = tiktoken.encoding_for_model(key)
    except Exception:
        encoder = tiktoken.get_encoding("cl100k_base")
    _ENCODER_CACHE[key] = encoder
    return encoder


def tokenize_text(text: str, model_name: str) -> list[str] | None:
    encoder = get_encoder(model_name)
    if encoder is None:
        return None
    try:
        token_ids = encoder.encode(text, disallowed_special=())
        return [encoder.decode([token_id]) for token_id in token_ids]
    except Exception:
        return None


def token_span(token_text: str, index: int) -> str:
    color = TOKEN_PALETTE[index % len(TOKEN_PALETTE)]
    return (
        "<span style=\"display:inline; padding:0 0.06rem; border-radius:0.25rem; "
        f"background:{color};\">{html.escape(token_text)}</span>"
    )


def highlight_token_sequence(text: str, token_sequence: list[str]) -> str:
    if not text:
        return ""
    cursor = 0
    parts: list[str] = []
    for token_index, token_text in enumerate(token_sequence):
        if not token_text:
            continue
        token_for_search = token_text
        pos = text.find(token_for_search, cursor)
        if pos < 0 and "<sp>" in token_text:
            token_for_search = token_text.replace("<sp>", " ")
            pos = text.find(token_for_search, cursor)
        if pos < 0:
            continue
        parts.append(html.escape(text[cursor:pos]))
        parts.append(token_span(token_for_search, token_index))
        cursor = pos + len(token_for_search)
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def highlight_context_history(context_text: str, token_history: list[str]) -> str:
    marker = " | A: "
    pos = context_text.find(marker)
    if pos < 0:
        return highlight_token_sequence(context_text, token_history)

    left = context_text[: pos + len(marker)]
    right = context_text[pos + len(marker) :]
    return html.escape(left) + highlight_token_sequence(right, token_history)


def highlighted_tokens_html(text: str, model_name: str) -> str:
    tokens = tokenize_text(text, model_name)
    if not tokens:
        return html.escape(text)

    html_parts: list[str] = []
    for token_index, token in enumerate(tokens):
        html_parts.append(token_span(token, token_index))

    return "".join(html_parts)


def highlighted_candidates_html(text: str) -> str:
    if not text or text == "(not provided)":
        return html.escape(text)

    parts = text.split(" | ")
    html_parts: list[str] = []
    for i, part in enumerate(parts):
        if i > 0:
            html_parts.append(" | ")
        if " (" in part and part.endswith(")"):
            token_part, prob_part = part.rsplit(" (", 1)
            html_parts.append(token_span(token_part, i))
            html_parts.append(f" ({html.escape(prob_part)}")
        else:
            html_parts.append(token_span(part, i))
    return "".join(html_parts)


def render_tokenized_text(text: str, model_name: str, highlight_tokens_enabled: bool) -> None:
    if not highlight_tokens_enabled:
        st.write(text)
        return
    content = highlighted_tokens_html(text, model_name)
    st.markdown(
        f"<div style=\"white-space:pre-wrap; line-height:1.8;\">{content}</div>",
        unsafe_allow_html=True,
    )


def render_wrapped_table(
    rows: list[dict[str, str]],
    height_px: int = 420,
    highlight_tokens_enabled: bool = False,
    highlight_columns: set[str] | None = None,
    model_name: str = "",
    process_links_enabled: bool = False,
    fixed_row_height_px: int | None = None,
    fixed_height_columns: set[str] | None = None,
) -> None:
    if not rows:
        st.info("Aucune donnée à afficher.")
        return

    headers = list(rows[0].keys())
    highlight_columns = highlight_columns or set()
    fixed_height_columns = fixed_height_columns or set()
    width_bounds_by_column: dict[str, tuple[int, int]] = {
        "run": (3, 8),
        "token": (5, 10),
        "step": (3, 8),
        "answer_raw": (20, 36),
        "answer_strict": (20, 36),
        "words_raw": (5, 12),
        "words_strict": (5, 12),
        "chosen_token": (8, 24),
        "chosen_prob": (8, 14),
    }
    compact_columns = set(width_bounds_by_column.keys())
    compact_width_ch: dict[str, int] = {}
    for header in headers:
        if header not in compact_columns:
            continue
        max_len = len(str(header))
        for row in rows:
            cell_len = len(str(row.get(header, "")))
            if cell_len > max_len:
                max_len = cell_len
        min_width, max_width = width_bounds_by_column[header]
        compact_width_ch[header] = max(min_width, min(max_width, max_len + 2))

    html_parts = []
    html_parts.append(
        f"""
<div style="max-height:{height_px}px; overflow:auto; border:1px solid #E6E9EF; border-radius:8px;">
<table style="width:100%; border-collapse:collapse; table-layout:fixed; font-size:0.9rem;">
<colgroup>
"""
    )
    for header in headers:
        if header in compact_width_ch:
            html_parts.append(f"<col style=\"width:{compact_width_ch[header]}ch;\">")
        else:
            html_parts.append("<col>")
    html_parts.append(
        """
</colgroup>
<thead style="position:sticky; top:0; background:#F7F9FC; z-index:1;">
<tr>
"""
    )
    for header in headers:
        align = "center" if header in compact_columns else "left"
        nowrap = "white-space:nowrap;" if header in compact_columns else ""
        html_parts.append(
            f"<th style=\"padding:8px; border-bottom:1px solid #E6E9EF; text-align:{align}; {nowrap}\">{html.escape(str(header))}</th>"
        )
    html_parts.append("</tr></thead><tbody>")

    for row_index, row in enumerate(rows):
        html_parts.append("<tr>")
        for header in headers:
            cell = row.get(header, "")
            cell_value = "" if cell is None else str(cell)
            if process_links_enabled and header == "chosen_token":
                cell_html = token_span(cell_value, row_index)
            elif process_links_enabled and header == "context_before_token" and row_index > 0:
                token_history = [
                    str(previous_row.get("chosen_token", ""))
                    for previous_row in rows[:row_index]
                ]
                cell_html = highlight_context_history(cell_value, token_history)
            elif highlight_tokens_enabled and header in highlight_columns:
                if header in {"answer_raw", "answer_strict", "context_before_token"}:
                    cell_html = highlighted_tokens_html(cell_value, model_name)
                elif header == "chosen_token":
                    cell_html = token_span(cell_value, 0)
                elif header == "top_candidates":
                    cell_html = highlighted_candidates_html(cell_value)
                else:
                    cell_html = highlighted_tokens_html(cell_value, model_name)
            else:
                cell_html = html.escape(cell_value)
            align = "center" if header in compact_columns else "left"
            nowrap = "white-space:nowrap;" if header in compact_columns else ""
            fixed_height_style = ""
            if fixed_row_height_px is not None and (
                not fixed_height_columns or header in fixed_height_columns
            ):
                fixed_height_style = (
                    f"height:{fixed_row_height_px}px; max-height:{fixed_row_height_px}px; "
                    "overflow:hidden; box-sizing:border-box;"
                )
            html_parts.append(
                "<td style=\"padding:8px; border-bottom:1px solid #F0F2F6; vertical-align:top; "
                f"text-align:{align}; {nowrap} {fixed_height_style} white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere;\">"
                f"{cell_html}</td>"
            )
        html_parts.append("</tr>")

    html_parts.append("</tbody></table></div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_run_selector_column(rows: list[dict[str, str]], selected_run: str | None) -> None:
    st.markdown(
        """
<div class="run-token-header">
token
</div>
""",
        unsafe_allow_html=True,
    )
    for row in rows:
        run_id = str(row.get("run", ""))
        if not run_id:
            continue
        is_selected = selected_run is not None and run_id == str(selected_run)
        if st.button(
            run_id,
            key=f"select_raw_run_{run_id}",
            use_container_width=False,
            type="primary" if is_selected else "secondary",
        ):
            st.session_state.selected_raw_run = run_id
            st.session_state.show_selected_raw_table = True


def apply_run_button_compact_css() -> None:
    st.markdown(
        """
<style>
.run-token-header {
  width: max-content !important;
  min-width: 2.2rem !important;
  margin: 0 auto 2px auto !important;
  padding: 6px 8px !important;
  border: 1px solid #E6E9EF !important;
  border-radius: 8px !important;
  background: #F7F9FC !important;
  text-align: center !important;
  font-weight: 600 !important;
}

div[class*="st-key-select_raw_run_"] [data-testid="stButton"] > button,
div[class*="st-key-select_raw_run_"] button {
  min-height: 1.32rem !important;
  height: 1.32rem !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  padding-left: 0.30rem !important;
  padding-right: 0.30rem !important;
  line-height: 1 !important;
  width: auto !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}

div[class*="st-key-select_raw_run_"] [data-testid="stButton"] > button p,
div[class*="st-key-select_raw_run_"] button p {
  font-size: 0.82rem !important;
  margin: 0 !important;
  text-align: center !important;
}

div[class*="st-key-select_raw_run_"] {
  display: block !important;
  width: max-content !important;
  min-width: 2.2rem !important;
  border: 1px solid #E6E9EF !important;
  border-radius: 6px !important;
  background: #FFFFFF !important;
  text-align: center !important;
  margin-top: 0 !important;
  margin-left: auto !important;
  margin-right: auto !important;
  margin-bottom: 2px !important;
  padding: 0 !important;
}

div[class*="st-key-select_raw_run_"] [data-testid="stButton"],
div[class*="st-key-select_raw_run_"] [data-testid="stButton"] > div {
  display: flex !important;
  justify-content: center !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}

div[class*="st-key-select_raw_run_"] [data-testid="stVerticalBlock"] {
  gap: 0 !important;
}

div[class*="st-key-select_raw_run_"] [data-testid="stVerticalBlockBorderWrapper"] {
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  min-height: 0 !important;
}

div[class*="st-key-select_raw_run_"] [data-testid="stButton"] > button,
div[class*="st-key-select_raw_run_"] button {
  margin-left: auto !important;
  margin-right: auto !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def compact_context(text: str, max_chars: int = 140) -> str:
    normalized = text.replace("\n", "\\n").replace("\t", "\\t").strip()
    if not normalized:
        return "<debut>"
    if len(normalized) <= max_chars:
        return normalized
    return "..." + normalized[-max_chars:]


def build_messages(question: str, target_words: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a concise assistant. "
                f"Answer in exactly {target_words} words. "
                "Use a simple educational style."
            ),
        },
        {"role": "user", "content": question.strip()},
    ]


def chat_completion(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = top_logprobs
    return client.chat.completions.create(**kwargs)


def extract_text(response: Any) -> str:
    choices = field(response, "choices", [])
    if not choices:
        return ""
    message = field(choices[0], "message")
    return field(message, "content", "") or ""


def extract_logit_rows(
    response: Any, top_n: int, question_context: str
) -> tuple[list[dict[str, str]], str]:
    rows: list[dict[str, str]] = []
    greedy_tokens: list[str] = []
    chosen_prefix_tokens: list[str] = []
    chosen_prefix_visible_tokens: list[str] = []

    choices = field(response, "choices", [])
    if not choices:
        return rows, ""

    logprobs_obj = field(choices[0], "logprobs")
    token_items = field(logprobs_obj, "content", []) if logprobs_obj else []

    for idx, item in enumerate(token_items, start=1):
        chosen_prefix_visible = "".join(chosen_prefix_visible_tokens)
        answer_context = chosen_prefix_visible.replace("<sp>", " ") if chosen_prefix_visible else "<debut>"
        context_before_token = (
            f"Q: {compact_context(question_context, 90)} | "
            f"A: {answer_context}"
        )

        chosen_token = field(item, "token", "")
        chosen_prob = logprob_to_prob(field(item, "logprob"))

        alt_items = (field(item, "top_logprobs", []) or [])[:top_n]
        candidates: list[tuple[str, float | None]] = []
        for alt in alt_items:
            token = field(alt, "token", "")
            prob = logprob_to_prob(field(alt, "logprob"))
            candidates.append((token, prob))

        if candidates:
            best_token = max(candidates, key=lambda x: x[1] if x[1] is not None else -1.0)[0]
        else:
            best_token = chosen_token

        greedy_tokens.append(best_token)

        top_candidates = " | ".join(
            f"{visible_token(token)} ({format_prob(prob)})" for token, prob in candidates
        )

        rows.append(
            {
                "step": str(idx),
                "context_before_token": context_before_token,
                "chosen_token": visible_token(chosen_token),
                "chosen_prob": format_prob(chosen_prob),
                "top_candidates": top_candidates or "(not provided)",
            }
        )
        chosen_prefix_tokens.append(chosen_token)
        chosen_prefix_visible_tokens.append(visible_token(chosen_token))

    return rows, "".join(greedy_tokens)


def run_demo(
    client: Any,
    question: str,
    model: str,
    target_words: int,
    temperature: float,
    top_n: int,
) -> tuple[
    list[dict[str, str]],
    dict[str, list[dict[str, str]]],
    list[dict[str, str]],
    dict[str, str],
    dict[str, str],
]:
    base_messages = build_messages(question, target_words)
    independent_answers: list[dict[str, str]] = []
    independent_logit_rows_by_run: dict[str, list[dict[str, str]]] = {}

    for i in range(1, 6):
        response = chat_completion(
            client=client,
            model=model,
            messages=base_messages,
            temperature=temperature,
            max_tokens=120,
            logprobs=True,
            top_logprobs=top_n,
        )
        text = extract_text(response)
        text_strict, raw_count, strict_count = enforce_exact_word_count(text, target_words)
        run_id = str(i)
        independent_answers.append(
            {
                "run": run_id,
                "answer_raw": text,
                "words_raw": str(raw_count),
                "answer_strict": text_strict,
                "words_strict": str(strict_count),
            }
        )
        run_logit_rows, _ = extract_logit_rows(response, top_n=top_n, question_context=question.strip())
        independent_logit_rows_by_run[run_id] = run_logit_rows

    logits_response = chat_completion(
        client=client,
        model=model,
        messages=base_messages,
        temperature=0.0,
        max_tokens=120,
        logprobs=True,
        top_logprobs=top_n,
    )
    deterministic_text = extract_text(logits_response)
    logit_rows, greedy_text = extract_logit_rows(
        logits_response, top_n=top_n, question_context=question.strip()
    )
    deterministic_strict, det_raw_count, det_strict_count = enforce_exact_word_count(
        deterministic_text, target_words
    )
    greedy_strict, greedy_raw_count, greedy_strict_count = enforce_exact_word_count(
        greedy_text, target_words
    )

    deterministic_view = {
        "raw": deterministic_text,
        "raw_words": str(det_raw_count),
        "strict": deterministic_strict,
        "strict_words": str(det_strict_count),
    }
    greedy_view = {
        "raw": greedy_text,
        "raw_words": str(greedy_raw_count),
        "strict": greedy_strict,
        "strict_words": str(greedy_strict_count),
    }

    return independent_answers, independent_logit_rows_by_run, logit_rows, deterministic_view, greedy_view


def main() -> None:
    st.set_page_config(page_title="Token Generation Lab", layout="wide")
    page = str(st.query_params.get("page", "")).strip().lower()
    if page == "explications":
        st.title("Page d'explications")
        base_dir = Path(__file__).resolve().parent
        image_candidates = [
            base_dir / "Process_LLM.jpeg",
            base_dir / "Process_LLM.jpg",
        ]
        image_path = next((p for p in image_candidates if p.exists()), None)
        def render_explanations_image(path) -> None:
            if path is None:
                st.warning(
                    "Image introuvable. Place le fichier 'Process_LLM.jpeg' dans le dossier "
                    "'token_pedagogique_streamlit'."
                )
                return
            image_col, _ = st.columns([3, 6])
            with image_col:
                st.image(str(path), caption="Processus de génération token par token", use_container_width=True)
        if EXPLICATIONS_IMAGE_ANCHOR in EXPLICATIONS_MARKDOWN:
            before, after = EXPLICATIONS_MARKDOWN.split(EXPLICATIONS_IMAGE_ANCHOR, 1)
            st.markdown(before + EXPLICATIONS_IMAGE_ANCHOR)
            render_explanations_image(image_path)
            st.markdown(after)
        else:
            st.markdown(EXPLICATIONS_MARKDOWN)
            render_explanations_image(image_path)
        st.link_button("Retour à l'application", "?", use_container_width=False)
        return

    st.title("Comment une IA générative choisit ses mots")
    st.link_button("Quelques explications", "?page=explications", use_container_width=False)

    if OpenAI is None:
        st.error("Le package 'openai' est absent. Installe les dépendances d'abord.")
        st.code("pip install -r requirements.txt")
        st.stop()

    if "highlight_tokens_enabled" not in st.session_state:
        st.session_state.highlight_tokens_enabled = False
    if "highlight_generation_flow_enabled" not in st.session_state:
        st.session_state.highlight_generation_flow_enabled = False
    if "last_run_data" not in st.session_state:
        st.session_state.last_run_data = None
    if "selected_raw_run" not in st.session_state:
        st.session_state.selected_raw_run = None
    if "show_selected_raw_table" not in st.session_state:
        st.session_state.show_selected_raw_table = False

    with st.sidebar:
        st.header("Configuration")
        api_key_default = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OPENAI_API_KEY", value=api_key_default, type="password")
        model = st.text_input("Modèle", value="gpt-4o-mini")
        target_words = st.number_input("Nombre de mots cible", min_value=5, max_value=80, value=30)
        temperature = st.slider("Température (5 réponses)", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        top_n = st.slider("Top n logprobs", min_value=1, max_value=20, value=5, step=1)
        st.markdown("**Surlignage des tokens**")
        toggle_label = (
            "Désactiver surlignage"
            if st.session_state.highlight_tokens_enabled
            else "Activer surlignage"
        )
        if st.button(toggle_label, use_container_width=True):
            st.session_state.highlight_tokens_enabled = not st.session_state.highlight_tokens_enabled
        state_label = "ON" if st.session_state.highlight_tokens_enabled else "OFF"
        st.caption(f"Surlignage: {state_label}")
        if st.session_state.highlight_tokens_enabled and tiktoken is None:
            st.warning("Package manquant: installe 'tiktoken' pour un surlignage token réel.")
        st.markdown("**Mise en lumière du processus**")
        flow_label = (
            "Désactiver liaison token -> contexte"
            if st.session_state.highlight_generation_flow_enabled
            else "Activer liaison token -> contexte"
        )
        if st.button(flow_label, use_container_width=True):
            st.session_state.highlight_generation_flow_enabled = (
                not st.session_state.highlight_generation_flow_enabled
            )
        flow_state = "ON" if st.session_state.highlight_generation_flow_enabled else "OFF"
        st.caption(f"Liaison génération: {flow_state}")

    question = st.text_area(
        "Question courte",
        value="Quels sont les piliers de l'intelligence économique",
        height=110,
    )

    run_btn = st.button("Lancer la démonstration", type="primary")

    st.info(
        "Note : l'API expose seulement les top_n tokens les plus probables par position, "
        "pas tous les logits du vocabulaire complet. Les réponses affichées sont forcées "
        "au nombre de mots cible via post-traitement."
    )

    if run_btn:
        if not question.strip():
            st.warning("Saisis une question.")
        elif not api_key.strip():
            st.warning("Ajoute une clé API OpenAI dans la barre latérale.")
        else:
            client = OpenAI(api_key=api_key.strip())
            try:
                with st.spinner("Génération en cours..."):
                    (
                        answers,
                        independent_logit_rows_by_run,
                        logit_rows,
                        deterministic_view,
                        greedy_view,
                    ) = run_demo(
                        client=client,
                        question=question,
                        model=model.strip(),
                        target_words=int(target_words),
                        temperature=float(temperature),
                        top_n=int(top_n),
                    )
                st.session_state.last_run_data = {
                    "question": question.strip(),
                    "model": model.strip(),
                    "target_words": int(target_words),
                    "temperature": float(temperature),
                    "top_n": int(top_n),
                    "answers": answers,
                    "independent_logit_rows_by_run": independent_logit_rows_by_run,
                    "logit_rows": logit_rows,
                    "deterministic_view": deterministic_view,
                    "greedy_view": greedy_view,
                }
                st.session_state.selected_raw_run = None
                st.session_state.show_selected_raw_table = False
            except Exception as exc:
                st.error(f"Erreur API: {exc}")

    if st.session_state.last_run_data is None:
        st.info("Lance d'abord la démonstration pour afficher des résultats.")
        return

    run_data = st.session_state.last_run_data
    run_question = str(run_data["question"])
    run_model = str(run_data["model"])
    run_target_words = int(run_data["target_words"])
    run_temperature = float(run_data["temperature"])
    run_top_n = int(run_data["top_n"])
    answers = list(run_data["answers"])
    independent_logit_rows_by_run = dict(run_data.get("independent_logit_rows_by_run", {}))
    logit_rows = list(run_data["logit_rows"])
    deterministic_view = dict(run_data["deterministic_view"])
    greedy_view = dict(run_data["greedy_view"])

    st.subheader("1) Cinq réponses indépendantes (sans mémoire du run précédent)")
    apply_run_button_compact_css()
    selected_run = (
        str(st.session_state.selected_raw_run)
        if st.session_state.show_selected_raw_table and st.session_state.selected_raw_run is not None
        else None
    )
    select_col, answers_col = st.columns([0.45, 9.55], gap="small")
    with select_col:
        render_run_selector_column(answers, selected_run=selected_run)
    with answers_col:
        answers_for_table = []
        for row in answers:
            answers_for_table.append(
                {
                    "run": str(row.get("run", "")),
                    "answer_raw": row.get("answer_raw", ""),
                    "words_raw": row.get("words_raw", ""),
                    "answer_strict": row.get("answer_strict", ""),
                    "words_strict": row.get("words_strict", ""),
                }
            )
        render_wrapped_table(
            answers_for_table,
            height_px=320,
            highlight_tokens_enabled=st.session_state.highlight_tokens_enabled,
            highlight_columns={"answer_raw", "answer_strict"},
            model_name=run_model,
            fixed_row_height_px=32,
            fixed_height_columns={"run", "answer_raw", "words_raw", "answer_strict", "words_strict"},
        )

    if st.session_state.show_selected_raw_table:
        selected_run_id = st.session_state.selected_raw_run
        selected_rows = independent_logit_rows_by_run.get(str(selected_run_id), [])
        if selected_rows:
            st.markdown(
                f"**1.bis) Top logits par token pour la réponse brute sélectionnée (token {selected_run_id})**"
            )
            render_wrapped_table(
                selected_rows,
                height_px=360,
                highlight_tokens_enabled=st.session_state.highlight_tokens_enabled,
                highlight_columns={"context_before_token", "chosen_token", "top_candidates"},
                model_name=run_model,
                process_links_enabled=st.session_state.highlight_generation_flow_enabled,
            )
            if st.button("Masquer ce tableau", key="hide_selected_raw_table"):
                st.session_state.show_selected_raw_table = False
                st.session_state.selected_raw_run = None
        else:
            st.info("Aucun détail de logits disponible pour cette réponse brute.")
            if st.button("Masquer ce tableau", key="hide_selected_raw_table_empty"):
                st.session_state.show_selected_raw_table = False
                st.session_state.selected_raw_run = None

    st.subheader("2) Top logits par token (run déterministe, température=0)")
    if logit_rows:
        render_wrapped_table(
            logit_rows,
            height_px=460,
            highlight_tokens_enabled=st.session_state.highlight_tokens_enabled,
            highlight_columns={"context_before_token", "chosen_token", "top_candidates"},
            model_name=run_model,
            process_links_enabled=st.session_state.highlight_generation_flow_enabled,
        )
        st.caption(
            "La colonne context_before_token montre le contexte utilisé à chaque étape : "
            "question + tokens déjà générés avant le token courant."
        )
        if st.session_state.highlight_generation_flow_enabled:
            st.caption(
                "Mode liaison actif: le token choisi en ligne i et sa reprise dans le contexte "
                "de la ligne i+1 partagent la même couleur."
            )
    else:
        st.warning("Aucune information logprobs retournée par le modèle.")

    st.subheader("3) Réponse composée")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Sortie déterministe du modèle (température=0)**")
        render_tokenized_text(
            deterministic_view["strict"],
            model_name=run_model,
            highlight_tokens_enabled=st.session_state.highlight_tokens_enabled,
        )
        st.caption(
            f"Mots bruts: {deterministic_view['raw_words']} | "
            f"Mots affichés: {deterministic_view['strict_words']}"
        )
        with st.expander("Voir le texte brut"):
            render_tokenized_text(
                deterministic_view["raw"],
                model_name=run_model,
                highlight_tokens_enabled=st.session_state.highlight_tokens_enabled,
            )

    with col_b:
        st.markdown("**Composition gloutonne depuis top logits (top-1 par étape)**")
        render_tokenized_text(
            greedy_view["strict"],
            model_name=run_model,
            highlight_tokens_enabled=st.session_state.highlight_tokens_enabled,
        )
        st.caption(
            f"Mots bruts: {greedy_view['raw_words']} | "
            f"Mots affichés: {greedy_view['strict_words']}"
        )
        with st.expander("Voir le texte brut"):
            render_tokenized_text(
                greedy_view["raw"],
                model_name=run_model,
                highlight_tokens_enabled=st.session_state.highlight_tokens_enabled,
            )

    st.subheader("4) Export CSV")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows = [
        {
            "question": run_question,
            "model": run_model,
            "target_words": str(run_target_words),
            "temperature": str(run_temperature),
            "top_n": str(run_top_n),
            "deterministic_raw": deterministic_view["raw"],
            "deterministic_strict": deterministic_view["strict"],
            "greedy_raw": greedy_view["raw"],
            "greedy_strict": greedy_view["strict"],
        }
    ]
    answers_csv = rows_to_csv_bytes(answers)
    logits_csv = rows_to_csv_bytes(logit_rows)
    summary_csv = rows_to_csv_bytes(summary_rows)

    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        st.download_button(
            label="Télécharger réponses (CSV)",
            data=answers_csv,
            file_name=f"reponses_independantes_{timestamp}.csv",
            mime="text/csv",
        )
    with col_2:
        st.download_button(
            label="Télécharger top_logits (CSV)",
            data=logits_csv,
            file_name=f"top_logits_{timestamp}.csv",
            mime="text/csv",
            disabled=not bool(logit_rows),
        )
    with col_3:
        st.download_button(
            label="Télécharger synthèse (CSV)",
            data=summary_csv,
            file_name=f"synthese_demo_{timestamp}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
