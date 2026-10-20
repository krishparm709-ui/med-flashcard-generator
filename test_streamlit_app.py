from io import BytesIO
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import pymupdf


class InterfaceTests(unittest.TestCase):
    def pdf_bytes(self):
        with pymupdf.open() as doc:
            doc.new_page().insert_text((72, 72), 'The sinoatrial node is the primary pacemaker of the heart.')
            return doc.tobytes()

    def test_uploads_generate_downloadable_package_and_review(self):
        from app import Flashcard
        from streamlit_app import build_downloads
        card = Flashcard(question='What is the primary pacemaker of the heart?',
                         answer='Sinoatrial node', topic='Cardiology', source_page=1,
                         evidence='The sinoatrial node is the primary pacemaker of the heart.')
        invalid = card.model_copy(update={'evidence': 'An invented source excerpt.'})
        with patch('streamlit_app.OpenAI') as client, patch('streamlit_app.generate_cards', return_value=[card, card, invalid]):
            result = build_downloads([('../../lecture.pdf', self.pdf_bytes())],
                                     'test-key', 'Medicine', 'test-model', 20)
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['rejected'], 1)
        self.assertEqual(result['duplicates'], 1)
        self.assertEqual(json.loads(result['review'])[0]['source_file'], 'lecture.pdf')
        self.assertNotIn('test-key', result['review'].decode())
        self.assertEqual(result['chunk_reports'][0]['generated'], 1)
        with zipfile.ZipFile(BytesIO(result['apkg'])) as archive:
            self.assertIn('collection.anki2', archive.namelist())
        client.assert_called_once_with(api_key='test-key', timeout=120.0, max_retries=3)

    def test_bad_pdf_fails_before_any_api_call(self):
        from streamlit_app import build_downloads
        with patch('streamlit_app.OpenAI') as client:
            with self.assertRaises(ValueError):
                build_downloads([('bad.pdf', b'not a PDF')], 'test-key', 'Medicine', 'test-model', 20)
            client.assert_not_called()

    def test_interface_renders_and_validates_missing_upload(self):
        from streamlit.testing.v1 import AppTest
        ui = AppTest.from_file('streamlit_app.py').run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.number_input[1].value, 6)
        ui.button[0].click().run()
        self.assertFalse(ui.exception)
        self.assertIn('Upload at least one PDF', ui.error[0].value)

    def test_minimum_above_maximum_is_rejected_before_generation(self):
        from streamlit.testing.v1 import AppTest
        from streamlit_app import build_downloads
        with patch('streamlit_app.OpenAI') as client:
            with self.assertRaises(ValueError):
                build_downloads([('lecture.pdf', self.pdf_bytes())], 'test-key', 'Medicine',
                                'test-model', 4, minimum_cards_per_chunk=6)
            client.assert_not_called()

    def test_chunk_shortfalls_report_actual_exported_count(self):
        from app import Flashcard
        from streamlit_app import build_downloads
        card = Flashcard(question='What is the primary pacemaker of the heart?',
                         answer='Sinoatrial node', topic='Cardiology', source_page=1,
                         evidence='The sinoatrial node is the primary pacemaker of the heart.')
        with patch('streamlit_app.OpenAI'), patch('streamlit_app.generate_cards', return_value=[card, card]):
            result = build_downloads([('lecture.pdf', self.pdf_bytes())], 'test-key', 'Medicine',
                                     'test-model', 20, minimum_cards_per_chunk=6)
        self.assertEqual(result['chunk_reports'][0]['generated'], 1)
        self.assertEqual(result['chunk_reports'][0]['minimum'], 6)

    def test_stale_rows_without_card_type_are_invalidated(self):
        from streamlit.testing.v1 import AppTest
        ui = AppTest.from_file('streamlit_app.py')
        ui.session_state['deck_result'] = {
            'educational_standard': 'UK Year 4 balanced v2', 'result_schema_version': 3,
            'rows': [{'question': 'An older question without a card type'}],
            'apkg': b'old-package', 'review': b'[]', 'count': 1, 'rejected': 0,
            'duplicates': 0, 'skipped_pages': 0, 'deck_name': 'Old deck', 'files': ['old.pdf'],
        }
        ui.run()
        self.assertFalse(ui.exception)
        self.assertTrue(any('Generate a new deck' in message.value for message in ui.info))
        self.assertEqual(len(ui.get('download_button')), 0)

    def test_ui_downloads_persist_without_regeneration_and_clear_on_failure(self):
        from app import Flashcard
        from streamlit.testing.v1 import AppTest
        card = Flashcard(question='What is the primary pacemaker of the heart?',
                         answer='Sinoatrial node', topic='Cardiology', source_page=1,
                         evidence='The sinoatrial node is the primary pacemaker of the heart.')
        upload = SimpleNamespace(name='lecture.pdf', getvalue=self.pdf_bytes)
        with patch('streamlit.file_uploader', return_value=[upload]), \
             patch('openai.OpenAI'), \
             patch('app.generate_cards', return_value=[card]) as generate, \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            ui = AppTest.from_file('streamlit_app.py').run()
            ui.button[0].click().run(timeout=20)
            self.assertFalse(ui.exception)
            self.assertTrue(ui.success)
            self.assertEqual(len(ui.get('download_button')), 2)
            ui.run()
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(len(ui.get('download_button')), 2)
            generate.side_effect = RuntimeError('Failure details must not be displayed')
            ui.button[0].click().run(timeout=20)
            self.assertFalse(ui.exception)
            self.assertTrue(ui.error)
            self.assertEqual(len(ui.get('download_button')), 0)
            self.assertNotIn('Failure details', ui.error[0].value)


if __name__ == '__main__':
    unittest.main()
