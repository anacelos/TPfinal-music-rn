import copy
import json
import unittest

from evaluation.subjective import finalize_context_descriptions as final


class TestContextDescriptionFinal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(l) for l in (final.BASE / "scenarios.jsonl").read_bytes().splitlines()]
        cls.audits = json.loads((final.BASE / "description_audit.json").read_bytes())["records"]
        cls.review = json.loads(final.REVIEW.read_bytes())

    def test_finalization_preserves_every_frozen_field_and_unreviewed_text(self):
        before = copy.deepcopy(self.rows)
        rows, audits = final.finalize_rows(self.rows, self.audits, self.review)
        self.assertEqual(self.rows, before)
        for a, b in zip(before, rows):
            for key in a:
                if key not in final.CHANGED_FIELDS:
                    self.assertEqual(a[key], b[key], key)
            if a["selection_index"] not in final.REVIEWED:
                self.assertEqual(a["description_ptbr"], b["description_ptbr"])
            self.assertEqual(b["public_context_heading"], "Contexto do jogo")
        self.assertEqual([r["selection_index"] for r in rows if r["manual_review_recommended"]], [181, 188])
        self.assertEqual(len(rows), 240)

    def test_queue_is_stable_separate_and_places_188_last(self):
        rows, _ = final.finalize_rows(self.rows, self.audits, self.review)
        queued = sorted(rows, key=lambda r: r["collection_order_v2"])
        self.assertEqual([r["collection_order_v2"] for r in queued], list(range(240)))
        self.assertEqual(queued[-1]["selection_index"], 188)
        for group in final.GROUPS:
            orders = [r["serve_order"] for r in queued if r["collection_priority_group"] == group]
            self.assertEqual(orders, sorted(orders))
        self.assertTrue(all(r["evidence_strength"] == "strong" for r in queued[:10]))
        self.assertTrue(all(r["evidence_strength"] == "moderate" for r in queued[10:14]))

    def test_queue_ignores_cost_duration_and_permutation(self):
        rows, _ = final.finalize_rows(self.rows, self.audits, self.review)
        altered = copy.deepcopy(rows)
        for r in altered:
            r.update(markov_matching_cost=-99999, neural_matching_cost=99999,
                     duration=0, presentation={"arbitrary": "irrelevant"})
        self.assertEqual(final.collection_orders(rows), final.collection_orders(altered))

    def test_special_cases_and_text_guards(self):
        rows, _ = final.finalize_rows(self.rows, self.audits, self.review)
        self.assertEqual(rows[31]["description_ptbr"], rows[198]["description_ptbr"])
        self.assertEqual(rows[188]["description_ptbr"], self.rows[188]["description_ptbr"])
        self.assertIn("demonstração", rows[41]["description_ptbr"])
        self.assertIn("protótipo", rows[61]["description_ptbr"])
        self.assertIn("animação", rows[239]["description_ptbr"])
        for r in rows:
            self.assertFalse(final.source.check_description(r["description_ptbr"])["issues"])

    def test_wrong_identity_missing_review_or_pewco_divergence_fails(self):
        bad = copy.deepcopy(self.review)
        bad["entries"][0]["external_id"] = "wrong"
        with self.assertRaises(ValueError):
            final.finalize_rows(self.rows, self.audits, bad)
        bad = copy.deepcopy(self.review)
        bad["entries"].pop()
        with self.assertRaises(ValueError):
            final.finalize_rows(self.rows, self.audits, bad)
        bad = copy.deepcopy(self.review)
        next(e for e in bad["entries"] if e["selection_index"] == 198)["after"] = "Um jogo de plataformas."
        with self.assertRaises(ValueError):
            final.finalize_rows(self.rows, self.audits, bad)

    def test_question_and_future_attention_are_exact_documentation(self):
        self.assertEqual(final.QUESTION, "Considerando a imagem e a descrição do jogo, quanto esta música combina com esse contexto?")
        self.assertEqual(final.SCALE, ["Não combina nada", "Combina pouco", "Mais ou menos", "Combina bem", "Combina muito bem"])
        self.assertEqual(final.SPOKEN_TEXT, "Este áudio é uma checagem de atenção. Para confirmar que você está prestando atenção à avaliação, selecione a opção ‘Não combina nada’ para esta faixa.")


if __name__ == "__main__":
    unittest.main()
