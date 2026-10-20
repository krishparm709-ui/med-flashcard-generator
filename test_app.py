import importlib.util
from contextlib import closing
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class PipelineTests(unittest.TestCase):
    def test_app_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('app'))

    def test_pdf_to_anki(self):
        import app
        import pymupdf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / 'lecture.pdf'
            with pymupdf.open() as doc:
                doc.new_page().insert_text((72, 72), 'The sinoatrial node is the primary pacemaker of the heart.')
                doc.save(pdf)
            pages = app.extract_pages(pdf)
            chunks = list(app.make_chunks(pages, 200))
            self.assertEqual(len(chunks), 1)
            card = app.Flashcard(question='What is the primary pacemaker of the heart?',
                                 answer='Sinoatrial node', topic='Cardiology',
                                 source_page=1, evidence='The sinoatrial node is the primary pacemaker of the heart.')
            response = SimpleNamespace(status='completed', output=[], output_parsed=app.CardBatch(cards=[card]))
            review = SimpleNamespace(status='completed', output=[], output_parsed=app.CardReviewBatch(
                reviews=[app.CardReview(index=0, meets_level=True, atomic=True, source_supported=True,
                                        reason='Clinically relevant foundational knowledge.')]))
            empty = SimpleNamespace(status='completed', output=[], output_parsed=app.CardBatch(cards=[]))
            client = SimpleNamespace(responses=SimpleNamespace(parse=Mock(side_effect=[empty, response, review])))
            cards = app.generate_cards(client, chunks[0], 'test-model', 10)
            self.assertEqual(cards, [card])
            app.validate_grounding(card, chunks[0])
            with self.assertRaises(ValueError):
                app.validate_grounding(card.model_copy(update={'source_page': 2}), chunks[0])
            with self.assertRaises(ValueError):
                app.validate_grounding(card.model_copy(update={'evidence': 'Invented quote'}), chunks[0])
            output = root / 'deck.apkg'
            app.export_deck([(pdf.name, card)], output, 'Medicine')
            with zipfile.ZipFile(output) as archive:
                archive.extract('collection.anki2', root)
            with closing(sqlite3.connect(root / 'collection.anki2')) as db:
                self.assertEqual(db.execute('select count(*) from notes').fetchone()[0], 1)
                self.assertIn('Sinoatrial node', db.execute('select flds from notes').fetchone()[0])

    def test_level_review_filters_and_requires_complete_assessment(self):
        import app
        card = app.Flashcard(question='What is the primary pacemaker of the heart?',
                             answer='Sinoatrial node', topic='Cardiology', source_page=1,
                             evidence='The sinoatrial node is the primary pacemaker of the heart.')
        review = app.CardReviewBatch(reviews=[app.CardReview(
            index=0, meets_level=False, atomic=True, source_supported=True,
            reason='Insufficient clinical application for the requested level.')])
        response = SimpleNamespace(status='completed', output=[], output_parsed=review)
        client = SimpleNamespace(responses=SimpleNamespace(parse=Mock(return_value=response)))
        self.assertEqual(app.review_cards(client, [(1, card.evidence)], [card], 'test-model'), [])
        review.reviews[0].meets_level = True
        self.assertEqual(app.review_cards(client, [(1, card.evidence)], [card], 'test-model'), [card])
        for criterion in ('atomic', 'source_supported'):
            setattr(review.reviews[0], criterion, False)
            self.assertEqual(app.review_cards(client, [(1, card.evidence)], [card], 'test-model'), [])
            setattr(review.reviews[0], criterion, True)
        response.output_parsed = app.CardReviewBatch(reviews=[])
        with self.assertRaises(app.ModelOutputError):
            app.review_cards(client, [(1, card.evidence)], [card], 'test-model')

    def test_missing_duplicate_and_wrong_review_indices_recover(self):
        import app
        card = app.Flashcard(question='Which condition presents with sign A?', answer='Condition X',
            topic='Condition X', source_page=1, evidence='Condition X presents with sign A.')
        other = card.model_copy(update={'question': 'Which condition presents with sign B?'})
        def response(indices):
            return SimpleNamespace(status='completed', output=[], output_parsed=app.CardReviewBatch(
                reviews=[app.CardReview(index=index, meets_level=True, atomic=True,
                    source_supported=True, reason='Supported.') for index in indices]))
        for invalid_indices in ([0], [0, 0], [1, 2], [0, 1, 2]):
            with self.subTest(indices=invalid_indices):
                parse = Mock(side_effect=[response(invalid_indices), response([0]), response([0])])
                client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
                result = app.review_cards(client, [(1, card.evidence)], [card, other], 'test-model')
                self.assertEqual(result, [card, other])
                self.assertEqual(parse.call_count, 3)
                for call in parse.call_args_list[1:]:
                    import json
                    payload = json.loads(call.kwargs['input'])
                    self.assertEqual([c['index'] for c in payload['candidates']], [0])

    def test_persistently_missing_review_fails_with_bounded_retries(self):
        import app
        card = app.Flashcard(question='Which condition presents with sign A?', answer='Condition X',
            topic='Condition X', source_page=1, evidence='Condition X presents with sign A.')
        response = SimpleNamespace(status='completed', output=[], output_parsed=app.CardReviewBatch(reviews=[]))
        parse = Mock(return_value=response)
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        with self.assertRaises(app.ModelOutputError):
            app.review_cards(client, [(1, card.evidence)], [card], 'test-model')
        self.assertEqual(parse.call_count, 2)

    def test_chunking_does_not_lose_text(self):
        import app
        text = 'alpha beta gamma delta ' * 100
        chunks = list(app.make_chunks({1: text}, 100))
        self.assertEqual(' '.join(p[1] for chunk in chunks for p in chunk), ' '.join(text.split()))
        self.assertTrue(all(sum(len(t) for _, t in c) <= 100 for c in chunks))

    def test_broad_and_detailed_passes_and_multi_page_support(self):
        import app
        source = [(1, 'Condition X presents with sign A.'), (2, 'Condition X presents with sign B.')]
        overview = app.Flashcard(card_type='overview', question='Which two presenting signs suggest condition X?',
            answer='Sign A; sign B.', topic='Condition X', source_page=1, evidence=source[0][1],
            supporting_evidence=[app.SourceExcerpt(source_page=2, evidence=source[1][1])])
        detail = app.Flashcard(question='Which condition presents with sign A?', answer='Condition X',
            topic='Condition X', source_page=1, evidence=source[0][1])
        def completed(parsed):
            return SimpleNamespace(status='completed', output=[], output_parsed=parsed)
        reviews = app.CardReviewBatch(reviews=[app.CardReview(index=i, meets_level=True, atomic=True,
            source_supported=True, reason='Focused and supported.') for i in range(2)])
        parse = Mock(side_effect=[completed(app.CardBatch(cards=[overview])),
                                  completed(app.CardBatch(cards=[detail])), completed(reviews)])
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        self.assertEqual(app.generate_cards(client, source, 'test-model', 10), [overview, detail])
        self.assertIn('overview', parse.call_args_list[0].kwargs['instructions'])
        self.assertIn('detail', parse.call_args_list[1].kwargs['instructions'])
        app.validate_grounding(overview, source)
        with self.assertRaises(ValueError):
            app.validate_grounding(overview, source[:1])

    def test_refusal_and_incomplete_fail(self):
        import app
        for response in [SimpleNamespace(status='incomplete', output=[], output_parsed=None),
                         SimpleNamespace(status='completed', output=[SimpleNamespace(content=[SimpleNamespace(type='refusal')])], output_parsed=None)]:
            client = SimpleNamespace(responses=SimpleNamespace(parse=lambda **kwargs: response))
            with self.assertRaises(RuntimeError):
                app.generate_cards(client, [(1, 'Source')], 'test-model', 10)

    def test_minimum_requests_additional_approved_unique_cards(self):
        import app
        source = [(1, 'Condition X presents with sign A. Condition X presents with sign B.')]
        first = app.Flashcard(question='Which condition presents with sign A?', answer='Condition X',
            topic='Condition X', source_page=1, evidence='Condition X presents with sign A.')
        second = first.model_copy(update={'question': 'Which condition presents with sign B?',
                                         'evidence': 'Condition X presents with sign B.'})
        with patch('app.generate_card_style', side_effect=[[first], [], [first, second]]) as generate, \
             patch('app.review_cards', side_effect=lambda client, chunk, cards, model: cards):
            result = app.generate_cards(Mock(), source, 'test-model', 4, minimum=2)
        self.assertEqual(result, [first, second])
        self.assertEqual(generate.call_count, 3)
        self.assertIn(first.question, generate.call_args.kwargs['excluded_questions'])

    def test_minimum_cannot_exceed_maximum_and_topups_are_bounded(self):
        import app
        with self.assertRaises(ValueError):
            app.generate_cards(Mock(), [(1, 'Source')], 'test-model', 4, minimum=6)
        with patch('app.generate_card_style', return_value=[]) as generate:
            self.assertEqual(app.generate_cards(Mock(), [(1, 'Source')], 'test-model', 10, minimum=6), [])
        self.assertEqual(generate.call_count, 4)  # Two initial styles and two coverage attempts.

    def test_truncated_json_retries_in_smaller_batches(self):
        import app
        from pydantic import ValidationError
        try:
            app.CardBatch.model_validate_json('{"cards":[')
        except ValidationError as error:
            truncated = error
        card = app.Flashcard(question='Which condition presents with sign A?', answer='Condition X',
            topic='Condition X', source_page=1, evidence='Condition X presents with sign A.')
        response = SimpleNamespace(status='completed', output=[], output_parsed=app.CardBatch(cards=[card]))
        other = card.model_copy(update={'question': 'Which condition presents with sign B?'})
        second = SimpleNamespace(status='completed', output=[], output_parsed=app.CardBatch(cards=[other]))
        parse = Mock(side_effect=[truncated, response, second])
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        result = app.generate_card_style(client, [(1, card.evidence)], 'test-model', 2, 'detail')
        self.assertEqual(result, [card, other])
        self.assertIn('at most 1 cards', parse.call_args_list[1].kwargs['instructions'])

    def test_repeated_invalid_json_has_bounded_safe_error(self):
        import app
        from pydantic import ValidationError
        try:
            app.CardBatch.model_validate_json('{"cards":[PRIVATE_SOURCE')
        except ValidationError as error:
            truncated = error
        parse = Mock(side_effect=truncated)
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        with self.assertRaises(RuntimeError) as raised:
            app.generate_card_style(client, [(1, 'Source text')], 'test-model', 1, 'detail')
        self.assertNotIn('PRIVATE_SOURCE', str(raised.exception))
        self.assertLessEqual(parse.call_count, 2)

    def test_incomplete_review_retries_without_accepting_partial_cards(self):
        import app
        card = app.Flashcard(question='Which condition presents with sign A?', answer='Condition X',
            topic='Condition X', source_page=1, evidence='Condition X presents with sign A.')
        incomplete = SimpleNamespace(status='incomplete', output=[], output_parsed=None)
        completed = SimpleNamespace(status='completed', output=[], output_parsed=app.CardReviewBatch(
            reviews=[app.CardReview(index=0, meets_level=True, atomic=True, source_supported=True, reason='Supported.')]))
        parse = Mock(side_effect=[incomplete, completed])
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        self.assertEqual(app.review_cards(client, [(1, card.evidence)], [card], 'test-model'), [card])


if __name__ == '__main__':
    unittest.main()
