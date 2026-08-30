from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from action_catalog import (  # noqa: E402
    ACTION_CATALOG,
    ACTION_CATALOG_VERSION,
    ACTION_CODES,
    ActionContractError,
    compile_action_spec,
    compile_panel_action,
    derived_action_components,
    migrate_legacy_action_exact,
)


class ActionCatalogTests(unittest.TestCase):
    def spec(self, code="PRESS_CONTROL"):
        return {
            "actor_id": "char_hero",
            "action_code": code,
            "target": "alarm button",
            "start_state": "the alarm button is untouched",
            "end_state": "alarm button remains depressed and its indicator is lit",
        }

    def test_every_catalog_code_compiles_deterministically_and_is_hashed(self):
        self.assertEqual(tuple(ACTION_CATALOG), ACTION_CODES)
        self.assertGreaterEqual(len(ACTION_CODES), 30)
        for code in ACTION_CODES:
            first = compile_action_spec(
                self.spec(code), visible_character_ids=["char_hero"],
            )
            second = compile_action_spec(
                self.spec(code), visible_character_ids=["char_hero"],
            )
            self.assertEqual(first, second, code)
            self.assertEqual(first["catalog_version"], ACTION_CATALOG_VERSION)
            self.assertEqual(len(first["spec_sha256"]), 64)
            self.assertIn(ACTION_CATALOG[code]["h3_verb"], first["h3_action_en"])
            self.assertNotIn("  ", first["h3_action_en"])

    def test_compiler_removes_literal_target_overlap(self):
        compiled = compile_action_spec(
            self.spec(), visible_character_ids=["char_hero"],
        )
        self.assertNotIn("alarm button alarm button", compiled["h3_action_en"].casefold())
        self.assertNotIn("alarm button button", compiled["h3_action_en"].casefold())
        self.assertIn("ending with it remains depressed", compiled["h3_action_en"])

    def test_actor_must_be_visible_and_cast_must_be_unique(self):
        with self.assertRaisesRegex(ActionContractError, "actor_id must belong"):
            compile_action_spec(self.spec(), visible_character_ids=["char_other"])
        with self.assertRaisesRegex(ActionContractError, "must not contain duplicates"):
            compile_action_spec(self.spec(), visible_character_ids=["char_hero", "char_hero"])
        with self.assertRaisesRegex(ActionContractError, "non-empty"):
            compile_action_spec(self.spec(), visible_character_ids=[])

    def test_unknown_code_and_empty_contract_fields_fail_closed(self):
        broken = self.spec("FEEL_BRAVE")
        with self.assertRaisesRegex(ActionContractError, "approved catalog enum"):
            compile_action_spec(broken, visible_character_ids=["char_hero"])
        for key in ("actor_id", "target", "start_state", "end_state"):
            broken = self.spec()
            broken[key] = ""
            with self.subTest(key=key), self.assertRaises(ActionContractError):
                compile_action_spec(broken, visible_character_ids=["char_hero"])

    def test_legacy_alias_is_exact_and_unknown_word_is_not_guessed(self):
        self.assertEqual(migrate_legacy_action_exact("按下"), "PRESS_CONTROL")
        self.assertEqual(migrate_legacy_action_exact(" PRESS "), "PRESS_CONTROL")
        self.assertIsNone(migrate_legacy_action_exact("pressing hopefully"))
        compiled = compile_action_spec(
            {"sub": "char_hero", "verb": "按下", "obj": "报警按钮"},
            visible_character_ids=["char_hero"],
            start_state="按钮未触碰", end_state="按钮保持按下", allow_legacy=True,
        )
        self.assertEqual(compiled["action_code"], "PRESS_CONTROL")
        with self.assertRaisesRegex(ActionContractError, "exact registered alias"):
            compile_action_spec(
                {"sub": "char_hero", "verb": "勇敢按动", "obj": "报警按钮"},
                visible_character_ids=["char_hero"],
                start_state="按钮未触碰", end_state="按钮保持按下", allow_legacy=True,
            )

    def test_panel_duplicate_fields_must_agree_with_canonical_spec(self):
        compiled = compile_action_spec(self.spec(), visible_character_ids=["char_hero"])
        panel = {
            "character_ids": ["char_hero"],
            "action_spec": compiled,
            "action_code": compiled["action_code"],
            "action_components": derived_action_components(compiled),
            "visible_action": "tampered display prose",
        }
        self.assertEqual(compile_panel_action(panel), compiled)
        panel["action_code"] = "OPEN_OBJECT"
        with self.assertRaisesRegex(ActionContractError, "panel.action_code disagrees"):
            compile_panel_action(panel)

    def test_catalog_version_and_spec_hash_bind_canonical_fields(self):
        first = compile_action_spec(self.spec(), visible_character_ids=["char_hero"])
        changed = self.spec()
        changed["end_state"] = "alarm button springs back up"
        second = compile_action_spec(changed, visible_character_ids=["char_hero"])
        self.assertNotEqual(first["spec_sha256"], second["spec_sha256"])
        invalid = self.spec()
        invalid["catalog_version"] = "future/v99"
        with self.assertRaisesRegex(ActionContractError, "unsupported catalog_version"):
            compile_action_spec(invalid, visible_character_ids=["char_hero"])

    def test_stale_embedded_hash_compiled_text_and_panel_state_fail_closed(self):
        compiled = compile_action_spec(self.spec(), visible_character_ids=["char_hero"])
        invalid = dict(compiled)
        invalid["spec_sha256"] = "0" * 64
        with self.assertRaisesRegex(ActionContractError, "spec_sha256 does not match"):
            compile_action_spec(invalid, visible_character_ids=["char_hero"])
        invalid = dict(compiled)
        invalid["h3_action_en"] = "tampered compiled action"
        with self.assertRaisesRegex(ActionContractError, "h3_action_en does not match"):
            compile_action_spec(invalid, visible_character_ids=["char_hero"])
        panel = {
            "character_ids": ["char_hero"], "action_spec": compiled,
            "action_code": compiled["action_code"],
            "first_state": "different opening state", "final_state": compiled["end_state"],
        }
        with self.assertRaisesRegex(ActionContractError, "panel.first_state disagrees"):
            compile_panel_action(panel)


if __name__ == "__main__":
    unittest.main()
