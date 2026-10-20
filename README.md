# Medical PDF to Anki

Requires Python 3.10 or later. Install and launch from this directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).
Upload one or more PDFs, enter your OpenAI API key in the password field,
choose a deck name, and click **Generate flashcards**. Preview the cards and
download the `.apkg` file, then open it in Anki to import the deck. The JSON
download includes each card's supporting excerpt and physical PDF page number.

You can also set `OPENAI_API_KEY` locally and leave the password field blank.
The entered key stays in the current Streamlit session; the app does not save
it to disk or change environment variables. PDFs are processed in isolated
temporary files, removed after generation. Download bytes remain in session
memory until cleared or the session ends. Generation sends extracted PDF text
to OpenAI and incurs API charges. Downloads do not trigger another generation.

Export lecture slides to PDF first. Image-only pages require OCR beforehand;
this text-only pipeline does not interpret diagrams. Structured output enforces
card shape and source quotes are checked, but medical correctness and atomicity
still require human review.

Generation targets UK Year 4 clinical medical study, using British terminology.
It makes separate passes for focused overviews/OSCE pattern recognition (roughly
40% of the requested capacity) and specific recall (roughly 60%). Actual proportions
depend on source support and filtering. A short overview can test 2-5 related
features within one explicit scope, such as one age group or organ system; vague
whole-condition essays and unrelated lists are rejected. Detailed cards complement
these with mechanisms, discriminators, complications and clinical decisions.

A final structured model review checks level, focus and source support. Generation
and review use batches of up to six cards to limit output size. Incomplete or
invalid JSON, missing assessments, and duplicated or unexpected review indices
are discarded and retried in smaller batches, with at most four
recovery attempts per generation style or review pass and one retry for an
individual card. This may require several API requests per text chunk. If recovery
fails, no partial deck is exported and a concise error is displayed. Saved session
results are versioned and checked before rendering; incompatible results are cleared.
All answer components need supporting
excerpts; additional excerpts can cite other pages in that chunk. These appear
in the JSON and on the back of the Anki card. Anki tags distinguish `overview`,
`pattern` and `detail`; the interface shows their counts.

The pipeline works within text chunks: it cannot guarantee an exhaustive overview
of a topic spanning a whole document. Source limitations may produce fewer cards;
it must not invent clinical context or imply that a partial list is exhaustive.
This is an educational target, not certification against a university curriculum.
Regenerate existing decks to apply the balanced approach.

The original command-line interface remains available:

```powershell
$env:OPENAI_API_KEY = 'your-key'
python app.py lecture.pdf --output medicine.apkg
```

Run local tests with `python -m unittest discover -v`. Tests mock OpenAI requests
and do not spend API credits.
