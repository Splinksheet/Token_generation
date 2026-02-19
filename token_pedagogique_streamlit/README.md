# Token Generation Lab (Streamlit)

Projet autonome pour expliquer la génération de token avec OpenAI.

## Démarrage
(.venv) PS C:\Users\Caius Octavius\Code_python> cd token_pedagogique_streamlit
(.venv) PS C:\Users\Caius Octavius\Code_python\token_pedagogique_streamlit> streamlit run app.py

Cle API : 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

## Ce que fait l'app

- Prend une question courte.
- Génère 5 réponses indépendantes (mémoire remise à zéro à chaque appel).
- Lance un run deterministic (`temperature=0`) avec `logprobs`.
- Affiche les `top_n` tokens probables a chaque position.
- Reconstruit une reponse gloutonne (token le plus probable a chaque etape).
- Force les réponses affichées au nombre de mots cible (ex: 20 mots) via post-traitement.
- Permet l'export CSV des réponses, des top logits et d'une synthèse.

## Installation

```bash
cd token_pedagogique_streamlit
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Clé API

Sous PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

## Lancement

```bash
streamlit run app.py
```

## Remarque pedagogique

L'API expose des `top_logprobs` par token, pas les logits complets de tout le vocabulaire.
