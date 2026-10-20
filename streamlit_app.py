"""Run with: python -m streamlit run streamlit_app.py"""

import json
import os
from pathlib import Path
import tempfile
from typing import Callable

from openai import AuthenticationError, OpenAI, OpenAIError, RateLimitError
import pymupdf
import streamlit as st

from app import ModelOutputError, extract_pages, export_deck, generate_cards, make_chunks, normalize, validate_grounding

RESULT_SCHEMA_VERSION = 4


def current_deck_result(result: object) -> bool:
    """Validate stored data as well as its version; imports may survive hot reloads."""
    required = {'rows', 'apkg', 'review', 'count', 'rejected', 'duplicates',
                'skipped_pages', 'deck_name', 'files'}
    return (
        isinstance(result, dict)
        and required <= result.keys()
        and result.get('result_schema_version') == RESULT_SCHEMA_VERSION
        and result.get('educational_standard') == 'UK Year 4 balanced v2'
        and isinstance(result['rows'], list)
        and bool(result['rows'])
        and all(isinstance(row, dict) and row.get('card_type') in {'overview', 'pattern', 'detail'}
                for row in result['rows'])
    )


def build_downloads(
    uploads: list[tuple[str, bytes]],
    api_key: str,
    deck_name: str,
    model: str,
    cards_per_chunk: int,
    progress: Callable[[float, str], None] | None = None,
    minimum_cards_per_chunk: int = 6,
) -> dict:
    """Use isolated temporary files; return download bytes after cleanup.

    The API key is passed directly to this request's client, never written to disk
    or assigned to the process environment. No global cache contains user data.
    """
    if not uploads:
        raise ValueError('Upload at least one PDF.')
    if not api_key.strip() or not deck_name.strip() or not model.strip():
        raise ValueError('An API key, deck name, and model are required.')
    if not 1 <= cards_per_chunk <= 40:
        raise ValueError('Cards per chunk must be between 1 and 40.')
    if not 0 <= minimum_cards_per_chunk <= cards_per_chunk:
        raise ValueError('Minimum cards must be between 0 and the maximum cards per chunk.')
    with tempfile.TemporaryDirectory(prefix='medical_anki_') as directory:
        root = Path(directory)
        jobs = []
        skipped_pages = 0
        # Parse every file before making paid requests. Never use upload paths.
        for index, (filename, data) in enumerate(uploads):
            label = filename.replace('\\', '/').rsplit('/', 1)[-1]
            if not label.lower().endswith('.pdf') or not data:
                raise ValueError('Each upload must be a nonempty PDF.')
            path = root / f'input_{index}.pdf'
            path.write_bytes(data)
            try:
                pages = extract_pages(path)
                with pymupdf.open(path) as document:
                    skipped_pages += document.page_count - len(pages)
            except Exception as error:
                raise ValueError(
                    f'Upload {index + 1} could not be read. Use an unlocked PDF with selectable text; '
                    'scanned PDFs need OCR first.'
                ) from error
            jobs.extend((label, chunk) for chunk in make_chunks(pages, 12000))

        accepted = []
        chunk_reports = []
        seen = set()
        rejected = duplicates = 0
        with OpenAI(api_key=api_key.strip(), timeout=120.0, max_retries=3) as client:
            for index, (label, chunk) in enumerate(jobs):
                if progress:
                    progress(index / len(jobs), f'Generating chunk {index + 1} of {len(jobs)}')
                before = len(accepted)
                for card in generate_cards(client, chunk, model.strip(), cards_per_chunk,
                                           minimum=minimum_cards_per_chunk, excluded_questions=seen):
                    try:
                        validate_grounding(card, chunk)
                    except ValueError:
                        rejected += 1
                        continue
                    key = normalize(card.question).casefold()
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                    accepted.append((label, card))
                chunk_reports.append({'file': label, 'chunk': index + 1,
                                      'generated': len(accepted) - before,
                                      'minimum': minimum_cards_per_chunk,
                                      'maximum': cards_per_chunk})
        if not accepted:
            raise ValueError('No cards passed the source and UK Year 4 level checks. '
                             'Try material with more clinical detail or clearer text.')
        output = root / 'medical_flashcards.apkg'
        export_deck(accepted, output, deck_name.strip())
        rows = [{'source_file': label, **card.model_dump()} for label, card in accepted]
        result = {
            'result_schema_version': RESULT_SCHEMA_VERSION,
            'educational_standard': 'UK Year 4 balanced v2',
            'apkg': output.read_bytes(),
            'review': json.dumps(rows, ensure_ascii=False, indent=2).encode('utf-8'),
            'rows': rows,
            'chunk_reports': chunk_reports,
            'count': len(rows),
            'rejected': rejected,
            'duplicates': duplicates,
            'skipped_pages': skipped_pages,
            'deck_name': deck_name.strip(),
            'files': [name.replace('\\', '/').rsplit('/', 1)[-1] for name, _ in uploads],
        }
    if progress:
        progress(1.0, 'Deck ready')
    return result


def main() -> None:
    st.set_page_config(page_title='Medical PDF to Anki', page_icon='📚')
    st.title('Medical PDF to Anki')
    st.write('Upload your lecture PDFs to create active-recall flashcards with source references.')
    st.info('UK Year 4+: a mix of focused topic overviews, OSCE pattern recognition, '
            'and detailed recall. Each card has one defined clinical scope and is '
            'reviewed for level, focus, and source support.')
    st.caption('Export lecture slides to PDF first. Scanned pages need OCR; diagrams are not interpreted.')

    with st.form('generation'):
        uploads = st.file_uploader('Lecture PDFs', type=['pdf'], accept_multiple_files=True)
        key = st.text_input('OpenAI API key', type='password',
                            help='Leave blank to use OPENAI_API_KEY from your local environment.')
        deck_name = st.text_input('Deck name', value='Medicine::High Yield')
        with st.expander('Generation settings'):
            model = st.text_input('OpenAI model', value=os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'))
            cards_per_chunk = st.number_input('Maximum cards per text chunk',
                                              min_value=1, max_value=40, value=20)
            minimum_cards_per_chunk = st.number_input('Minimum cards per text chunk',
                min_value=0, max_value=40, value=6,
                help='Target after review and duplicate filtering. Up to two extra coverage passes '
                     'try to reach it. Unsupported material is never added just to meet the count. '
                     'Set 0 to disable the minimum target.')
        st.caption('Generation and educational review send extracted PDF text to OpenAI and incur '
                   'API usage charges. Small generation/review batches and automatic '
                   'retries may require several requests per text chunk. '
                   'Your entered key is kept in this session and is not saved to a file.')
        generate = st.form_submit_button('Generate flashcards', type='primary')

    if generate:
        st.session_state.pop('deck_result', None)
        api_key = key.strip() or os.getenv('OPENAI_API_KEY', '').strip()
        if not uploads:
            st.error('Upload at least one PDF.')
        elif not api_key:
            st.error('Enter your OpenAI API key or set OPENAI_API_KEY locally.')
        elif not deck_name.strip() or not model.strip():
            st.error('Enter a deck name and model.')
        elif minimum_cards_per_chunk > cards_per_chunk:
            st.error('Minimum cards per chunk cannot exceed the maximum.')
        else:
            bar = st.progress(0.0, text='Reading PDFs…')
            try:
                with st.spinner('Creating flashcards. Large lectures can take several minutes.'):
                    result = build_downloads(
                        [(upload.name, upload.getvalue()) for upload in uploads],
                        api_key, deck_name, model, int(cards_per_chunk),
                        progress=lambda value, message: bar.progress(value, text=message),
                        minimum_cards_per_chunk=int(minimum_cards_per_chunk),
                    )
                st.session_state['deck_result'] = result
            except AuthenticationError:
                st.error('OpenAI rejected the API key. Check it and try again.')
            except RateLimitError:
                st.error('OpenAI rate limit or quota reached. Check API billing or try again later.')
            except OpenAIError:
                st.error('OpenAI could not complete generation. Check connectivity, model access, '
                         'and API billing, then try again.')
            except ModelOutputError as error:
                st.error(str(error))
            except ValueError as error:
                st.error(str(error))
            except Exception:
                # Avoid echoing raw SDK exceptions or request details into the page.
                st.error('Generation failed; no new download was created. Check the PDFs and try again.')
            finally:
                bar.empty()

    result = st.session_state.get('deck_result')
    if result and not current_deck_result(result):
        st.session_state.pop('deck_result', None)
        st.info('Generate a new deck to include the balanced overview and detailed recall approach.')
        result = None
    if result:
        st.success(f"{result['count']} flashcards ready — {result['deck_name']}")
        st.text('Generated from: ' + ', '.join(result['files']))
        chunk_reports = result.get('chunk_reports', [])
        shortfalls = [row for row in chunk_reports if row['generated'] < row['minimum']]
        if shortfalls:
            st.warning(f"{len(shortfalls)} text chunk(s) did not reach the requested minimum after "
                       'coverage attempts and quality checks. Review the counts below; fewer cards '
                       'may reflect limited source material or filtering. Card count alone does not '
                       'guarantee complete topic coverage.')
        with st.expander('Cards per text chunk'):
            st.dataframe(chunk_reports, hide_index=True)
        if result['skipped_pages']:
            st.warning(f"{result['skipped_pages']} pages had no extractable text and were skipped.")
        st.caption(f"Filtered: {result['rejected']} unsupported excerpts, "
                   f"{result['duplicates']} duplicate questions.")
        counts = {kind: sum(row.get('card_type') == kind for row in result['rows'])
                  for kind in ('overview', 'pattern', 'detail')}
        st.caption(f"Card mix: {counts['overview']} overview · {counts['pattern']} pattern recognition · "
                   f"{counts['detail']} detailed recall")
        st.info('Review accuracy, coverage, and focus before studying. Broader cards cover a '
                'bounded clinical scope; they do not claim to list everything about a condition.')
        st.dataframe(result['rows'], hide_index=True)
        st.download_button('Download Anki deck (.apkg)', data=result['apkg'],
                           file_name='medical_flashcards.apkg', mime='application/octet-stream')
        st.download_button('Download review file (.json)', data=result['review'],
                           file_name='medical_flashcards.json', mime='application/json')
        if st.button('Clear generated deck'):
            del st.session_state['deck_result']
            st.rerun()


if __name__ == '__main__':
    main()
