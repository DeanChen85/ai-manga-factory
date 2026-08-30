from __future__ import annotations

import json
import hashlib
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))

import orchestrator
import render_service
import story_splitter
import task_store
from video_quality import select_edit_window


def target_llm_contract() -> dict:
    """Eight-shot platform story returned by the mocked MiniMax boundary."""
    contract = {
        "title": "写给十年后的信",
        "subtitle": "雨夜车站的回声",
        "story_bible": {
            "title": "写给十年后的信",
            "logline": "年轻快递员在废弃车站发现一封写给十年后自己的信。",
            "synopsis": "林舟在雨夜送件误入废弃车站，发现一封署名为十年后自己的信，并决定改变明天。",
            "genre": "温暖悬疑反转",
            "target_audience": "年轻成人 18-35",
            "themes": ["选择", "成长"],
            "continuity_rules": ["林舟始终穿深蓝防雨外套并携带黄色快递包"],
        },
        "character_bible": [{
            "character_id": "char_linzhou",
            "name": "林舟",
            "identity_prompt": "24岁中国男性，短黑发，棕色眼睛，清瘦脸型，左眉尾浅疤",
            "wardrobe_lock": "深蓝防雨外套，灰色连帽衫，黑色长裤，黄色快递包",
            "model_identity_tags_en": [
                "1boy", "adult male", "Chinese", "short black hair", "brown eyes",
                "slim face", "subtle scar at the outer end of the left eyebrow", "masculine face",
            ],
            "model_wardrobe_tags_en": [
                "navy blue rain jacket", "gray hoodie", "black long pants",
                "yellow courier shoulder bag", "practical black shoes",
            ],
            "voice_profile": {
                "language": "Chinese",
                "age": "young adult",
                "timbre": "warm restrained baritone",
                "pace": "medium slow",
                "accent": "standard Mandarin",
            },
        }],
        "visual_bible": {
            "style_name": "cinematic Chinese comic",
            "style_prompt": "cinematic Chinese comic animation, cold blue rainy-night ambience, warm gold interior contrast, consistent linework and materials",
            "global_negative_prompt": "identity drift, wardrobe change, random text, watermark",
            "aspect_ratio": "9:16",
        },
        "scene_bible": [
            {
                "scene_id": "scene_station_platform",
                "description": "废弃车站站台，斑驳雨棚，冷蓝路灯，积水倒影，远处封闭铁门",
                "positive_prompt": "空旷废弃车站站台，雨夜，冷蓝灯，积水倒影，无人物，无文字",
                "model_prompt_en": "single empty abandoned train station platform at rainy night, weathered canopy, cold blue lamps, wet reflective puddles, closed iron gate in the distance, one coherent environment view, background only",
                "panel_ids": ["panel_01_arrival", "panel_02_letter"],
            },
            {
                "scene_id": "scene_waiting_room",
                "description": "老旧候车室，木椅与停摆时钟，窗外雨幕，桌灯投下暖金光",
                "positive_prompt": "空旷老旧候车室，暖金桌灯，窗外雨夜，无人物，无文字",
                "model_prompt_en": "single empty old train station waiting room, wooden benches, stopped wall clock, rain-streaked windows, warm golden table lamp, cold blue rainy night outside, one coherent interior view, background only",
                "panel_ids": ["panel_03_choice"],
            },
        ],
        "panels": [
            {
                "panel_id": "panel_01_arrival",
                "name": "panel_01_arrival",
                "scene_id": "scene_station_platform",
                "character_ids": ["char_linzhou"],
                "continuity_group": "main",
                "previous_panel_id": None,
                "continuity_state_in": {"letter": "not found", "bag": "on shoulder"},
                "continuity_state_out": {"letter": "noticed", "bag": "on shoulder"},
                "first_frame": "林舟从站台入口步入画面，雨水沿深蓝外套滴落",
                "last_frame": "林舟停在长椅旁，视线落向一只干燥的白色信封",
                "motion": "林舟穿过雨幕，缓慢靠近长椅上的信封",
                "cuts": [{
                    "time_range": "0-10s",
                    "name": "arrival",
                    "intensity": "SMOOTH",
                    "shot_description": "竖屏中远景从积水倒影缓慢抬升并跟随林舟穿过雨幕，冷蓝路灯勾勒深蓝外套和黄色快递包。",
                }],
                "spoken_dialogue": [{"time_range": "2-4s", "speaker_id": "char_linzhou", "text": "这里早就停用了。"}],
                "subtitle_timeline": [],
                "on_screen_text": [{"time_range": "6-7s", "text": "十年后"}],
                "audio_cues": [{"time_range": "0-10s", "tag": "RAIN"}],
            },
            {
                "panel_id": "panel_02_letter",
                "name": "panel_02_letter",
                "scene_id": "scene_station_platform",
                "character_ids": ["char_linzhou"],
                "continuity_group": "main",
                "previous_panel_id": "panel_01_arrival",
                "continuity_state_in": {"letter": "noticed", "bag": "on shoulder"},
                "continuity_state_out": {"letter": "opened", "bag": "on shoulder"},
                "first_frame": "承接上一镜位置，林舟弯腰拿起长椅上的白色信封",
                "last_frame": "林舟展开信纸，惊讶地看见熟悉笔迹，衣着和快递包保持不变",
                "motion": "手部特写转向面部反应，信封在雨棚下保持干燥",
                "cuts": [{
                    "time_range": "0-10s",
                    "name": "letter",
                    "intensity": "TENSE",
                    "shot_description": "镜头从林舟戴雨水的手指捏住信封开始，平稳推近到他辨认熟悉笔迹时克制而震惊的面部反应。",
                }],
                "spoken_dialogue": [{"time_range": "4-6s", "speaker_id": "char_linzhou", "text": "这是我的字。"}],
                "subtitle_timeline": [],
                "on_screen_text": [],
                "audio_cues": [{"time_range": "1-2s", "tag": "PAPER"}],
            },
            {
                "panel_id": "panel_03_choice",
                "name": "panel_03_choice",
                "scene_id": "scene_waiting_room",
                "character_ids": ["char_linzhou"],
                "continuity_group": "main",
                "previous_panel_id": "panel_02_letter",
                "continuity_state_in": {"letter": "opened", "bag": "on shoulder"},
                "continuity_state_out": {"letter": "folded in pocket", "bag": "on shoulder"},
                "first_frame": "承接上一镜动作，林舟拿着展开的信纸进入候车室暖光区域",
                "last_frame": "林舟把信折好放进胸前口袋，转身走向雨夜中的出口",
                "motion": "林舟读完信后从迟疑转为坚定，收好信并走向出口",
                "cuts": [{
                    "time_range": "0-10s",
                    "name": "choice",
                    "intensity": "POWERFUL",
                    "shot_description": "暖金桌灯照亮林舟与信纸，镜头绕至正面捕捉他从迟疑转为坚定，随后拉远看他走向冷蓝雨夜出口。",
                }],
                "spoken_dialogue": [{"time_range": "5-7s", "speaker_id": "char_linzhou", "text": "明天还来得及。"}],
                "subtitle_timeline": [],
                "on_screen_text": [],
                "audio_cues": [{"time_range": "7-10s", "tag": "MUSIC_LIFT"}],
            },
        ],
    }
    templates = contract["panels"]
    roles = ["hook", "setup", "escalation", "escalation", "escalation", "reversal", "reversal", "close"]
    panels = []
    previous_id = None
    previous_state = {}
    for index, role in enumerate(roles, 1):
        panel = json.loads(json.dumps(templates[min((index - 1) // 3, 2)], ensure_ascii=False))
        panel_id = f"panel_{index:02d}_{role}"
        final_state = {"letter": f"visible state {index}", "bag": "on shoulder"}
        panel.update({
            "panel_id": panel_id, "name": panel_id,
            "previous_panel_id": previous_id,
            "continuity_state_in": previous_state,
            "continuity_state_out": final_state,
            "source_generation_duration_seconds": 10.125,
            "edit_duration_seconds": 3.75,
            "shot_role": role, "story_beat_id": f"beat_{role}",
            "visible_action": f"Lin slides the envelope across the desk until it stops under the lamp in shot {index}",
            "first_state": f"warning hidden at position {index}",
            "final_state": f"warning revealed at position {index}",
            "cause": "The previous clue forces Lin to inspect the next physical detail",
            "next_hook": "A newly revealed mark changes his next action",
            "camera_plan": {
                "shot_size": f"shot-size-{index}", "angle": f"angle-{index}",
                "movement": "controlled push", "composition": f"composition-{index}",
            },
            "transition": {"type": "hard_cut", "motivation": "causal beat advance"},
            "edit_hint": {
                "preferred_moment": "warning clears envelope", "edit_in_hint": "hand reaches",
                "edit_out_hint": "mark revealed",
            },
            "priority": "must_have", "group_shot_reason": "",
            "spoken_dialogue": ([{
                "time_range": "0.2-1.2s", "start_s": 0.2, "end_s": 1.2,
                "speaker_id": "char_linzhou", "text": "等等", "delivery_style": "quiet",
                "max_chars": 4,
            }] if index == 1 else []),
            "subtitle_timeline": [], "on_screen_text": [], "audio_cues": [],
            "first_frame": f"Lin reaches toward the warning in composition {index}",
            "last_frame": f"Lin reveals the warning in composition {index}",
        })
        panel["cuts"] = [{
            "time_range": "0-10.125s", "name": role, "intensity": "SMOOTH",
            "shot_description": f"A controlled vertical shot {index} follows Lin opening the envelope and lifting the warning while preserving his approved wardrobe and station geography.",
        }]
        panels.append(panel)
        previous_id = panel_id
        previous_state = final_state
    contract["story_beats"] = [{
        "beat_id": f"beat_{role}", "role": role,
        "dramatic_question": f"What changes during {role}?",
        "visible_proof": f"Lin performs a visible object action during {role}",
        "payoff_or_hook": f"The {role} action causes the next beat",
    } for role in dict.fromkeys(roles)]
    contract["panels"] = panels
    for scene in contract["scene_bible"]:
        scene["panel_ids"] = [
            panel["panel_id"] for panel in panels if panel["scene_id"] == scene["scene_id"]
        ]
    return contract


class TargetJourneyE2ETests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.old_env = dict(os.environ)
        os.environ["AI_FACTORY_ROOT"] = str(self.root)
        os.environ["AI_MANGA_PROJECTS_DIR"] = str(self.root / "projects")
        os.environ["AI_MANGA_JOB_DB"] = str(self.root / "state" / "jobs.sqlite3")
        task_store._default_store = None
        orchestrator.PROJECTS_DIR = self.root / "projects"
        self.ep_id = "target_e2e_20260810"

    def tearDown(self):
        task_store._default_store = None
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tempdir.cleanup()

    def _asset_generator(self, item, *_args, **_kwargs):
        source_id = item.get("character_id") or item.get("scene_id")
        path = self.root / "mock_comfy" / f"{source_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"mock-reference:{source_id}".encode("utf-8"))
        return {"prompt_id": f"mock-{source_id}", "reference_images": [str(path)]}

    def _tail_frame(self, _video: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"mock-continuity-tail")
        return destination

    def _submitter(self, panel, _output_path, **kwargs):
        job_id = kwargs["job_id"]
        kwargs["store"].update_job(job_id, status="submitted", prompt_id=f"prompt-{job_id}")
        return {"job_id": job_id, "prompt_id": f"prompt-{job_id}"}

    def _waiter(self, job_id, **kwargs):
        store = kwargs["store"]
        job = store.get_job(job_id)
        output = Path(job["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"mock-video:{job_id}".encode("utf-8"))
        artifact_sha = hashlib.sha256(output.read_bytes()).hexdigest()
        visual_sha = hashlib.sha256(f"visual:{job_id}".encode("utf-8")).hexdigest()
        store.update_job(
            job_id,
            status="succeeded",
            progress=1.0,
            output_path=str(output),
            preview_path=str(output),
            probe={
                "duration_seconds": 10.0,
                "video": {"width": 608, "height": 1056, "fps": 24.0},
            },
            metadata={
                **job["metadata"], "artifact_sha256": artifact_sha,
                "content_qa": {"passed": True, "analysis": {
                    "decoded_visual_sha256": visual_sha,
                    "perceptual_hashes": [visual_sha], "static": False,
                    "metrics": {"sample_count": 1},
                }, "reasons": []},
                "editorial_review": {"status": "pending"},
                "release": {"status": "pending"},
            },
        )
        return output

    def _prepared_v3_asset_review(self) -> Path:
        """Prepare a fully valid offline V3 contract with generated mock assets."""
        with mock.patch.object(
            story_splitter, "_call_m3",
            return_value=json.dumps(target_llm_contract(), ensure_ascii=False),
        ):
            episode = story_splitter.split_story(
                "A courier enters an abandoned station and finds a future letter.",
                topic="future letter", synopsis="A courier must act on a warning from the future.",
                target_audience="young adults", total_duration_seconds=30, shot_count=8,
                platform="douyin", prompt_mode="cinematic",
                visual_style="cinematic Chinese comic",
                style_enforcement="cinematic Chinese comic with stable identity and wardrobe",
                aspect_ratio="9:16", language="cn", api_key="offline-test-only",
            )
        draft = render_service.prepare_contract(self.ep_id, episode)
        render_service.approve_contract(
            self.ep_id, expected_hash=draft["pipeline"]["contract_hash"],
        )
        assets = orchestrator.prepare_all_assets(
            self.ep_id,
            character_generator=self._asset_generator,
            scene_generator=self._asset_generator,
        )
        self.assertEqual(assets["pipeline"]["assets_status"], "ready_for_approval")
        return Path(assets["project_dir"]) / "episode.json"

    def test_v3_asset_approval_and_production_gate_reject_panel_without_character_reference(self):
        episode_path = self._prepared_v3_asset_review()
        persisted = json.loads(episode_path.read_text(encoding="utf-8"))
        persisted["panels"][0]["character_ids"] = []
        episode_path.write_text(json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "must reference at least one visible character"):
            render_service.approve_assets(self.ep_id)
        gate = task_store.production_gate(self.ep_id)
        self.assertFalse(gate["ready"])
        self.assertIn("contract_invalid", gate["reasons"])
        self.assertTrue(any("visible character" in error for error in gate["contract_errors"]))

    def test_v3_asset_approval_and_production_gate_reject_invisible_dialogue_speaker(self):
        episode_path = self._prepared_v3_asset_review()
        persisted = json.loads(episode_path.read_text(encoding="utf-8"))
        persisted["panels"][0]["spoken_dialogue"][0]["speaker_id"] = "char_not_visible"
        episode_path.write_text(json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "speaker_id must reference a visible character"):
            render_service.approve_assets(self.ep_id)
        gate = task_store.production_gate(self.ep_id)
        self.assertFalse(gate["ready"])
        self.assertIn("contract_invalid", gate["reasons"])
        self.assertTrue(any("speaker_id must reference a visible character" in error for error in gate["contract_errors"]))

    def test_theme_to_complete_platform_package_with_resume(self):
        calls = []

        def fake_m3(system_prompt, user_prompt, **_kwargs):
            calls.append((system_prompt, user_prompt))
            return json.dumps(target_llm_contract(), ensure_ascii=False)

        with mock.patch.object(story_splitter, "_call_m3", side_effect=fake_m3):
            episode = story_splitter.split_story(
                "林舟在雨夜误入废弃车站，发现一封写给十年后自己的信。",
                topic="写给十年后的自己",
                synopsis="年轻快递员在废弃车站发现一封未来的来信，并做出改变。",
                target_audience="年轻成人 18-35",
                total_duration_seconds=30,
                shot_count=8,
                platform="douyin",
                prompt_mode="cinematic",
                visual_style="cinematic Chinese comic",
                style_enforcement="cinematic Chinese comic animation, cold blue rain against warm gold interiors, strictly consistent identity and wardrobe",
                aspect_ratio="9:16",
                language="cn",
                api_key="test-boundary-only",
            )

        self.assertIn("elite series head writer", calls[0][0])
        self.assertIn("写给十年后的自己", calls[0][1])
        self.assertEqual(episode["schema_version"], "ai-manga.prompt-package/v3")
        self.assertEqual(len(episode["panels"]), 8)
        self.assertTrue(all(panel["subtitle_source"] == "spoken_dialogue_derived" for panel in episode["panels"]))
        self.assertTrue(all(panel["on_screen_text"] == [] for panel in episode["panels"]))
        self.assertTrue(all(panel["prompt_package"]["h3_visible_text_policy"] == "forbidden" for panel in episode["panels"]))

        # This E2E fixture exercises the legacy expert path that goes straight
        # to releasable clips. The normal Web product defaults to proof,
        # promotion, then production and is covered by the H3 profile tests.
        episode.setdefault("render_settings", {})["production_strategy"] = "direct_production"
        for panel in episode["panels"]:
            panel.setdefault("prompt_package", {}).setdefault("render_settings", {})[
                "production_strategy"
            ] = "direct_production"

        draft = render_service.prepare_contract(self.ep_id, episode)
        self.assertEqual(draft["pipeline"]["contract_status"], "draft")
        self.assertEqual(len(draft["jobs"]), 8)
        with self.assertRaisesRegex(RuntimeError, "production gate blocked"):
            orchestrator.run_episode_jobs(self.ep_id)

        render_service.approve_contract(
            self.ep_id, expected_hash=draft["pipeline"]["contract_hash"]
        )
        assets = orchestrator.prepare_all_assets(
            self.ep_id,
            character_generator=self._asset_generator,
            scene_generator=self._asset_generator,
        )
        self.assertEqual(len(assets["assets"]["items"]), 3)
        approved = render_service.approve_assets(self.ep_id)
        self.assertEqual(approved["pipeline"]["assets_status"], "approved")
        for job in approved["jobs"]:
            roles = {entry["role"] for entry in job["metadata"]["inputs"]["reference_inputs"]}
            self.assertEqual(roles, {"character_reference", "scene_reference"})

        failed_once = {"done": False}

        def interrupted_wait(job_id, **kwargs):
            job = kwargs["store"].get_job(job_id)
            if job["panel_index"] == 2 and not failed_once["done"]:
                failed_once["done"] = True
                raise RuntimeError("simulated Comfy interruption")
            return self._waiter(job_id, **kwargs)

        common_patches = (
            mock.patch.object(orchestrator, "submit_render_job", side_effect=self._submitter),
            mock.patch.object(orchestrator, "wait_render_job", side_effect=interrupted_wait),
            mock.patch.object(orchestrator, "_extract_tail_frame", side_effect=self._tail_frame),
            mock.patch.object(orchestrator, "update_status"),
            mock.patch.object(
                orchestrator, "release_comfy_resources",
                return_value={"released": True, "reason": "test isolation"},
            ),
        )
        with common_patches[0], common_patches[1], common_patches[2], common_patches[3], common_patches[4]:
            first_run = orchestrator.run_episode_jobs(self.ep_id, poll_interval=0.01)
        self.assertEqual(
            [job["status"] for job in first_run["snapshot"]["jobs"]],
            ["succeeded", "failed"] + ["queued"] * 6,
        )
        first_clip = Path(first_run["snapshot"]["jobs"][0]["output_path"])
        first_clip_bytes = first_clip.read_bytes()

        render_service.resume(self.ep_id)
        with mock.patch.object(orchestrator, "submit_render_job", side_effect=self._submitter), \
             mock.patch.object(orchestrator, "wait_render_job", side_effect=self._waiter), \
             mock.patch.object(orchestrator, "_extract_tail_frame", side_effect=self._tail_frame), \
             mock.patch.object(orchestrator, "update_status"), \
             mock.patch.object(
                 orchestrator, "release_comfy_resources",
                 return_value={"released": True, "reason": "test isolation"},
             ):
            resumed = orchestrator.run_episode_jobs(self.ep_id, poll_interval=0.01)
        self.assertEqual([job["status"] for job in resumed["snapshot"]["jobs"]], ["succeeded"] * 8)
        self.assertEqual(first_clip.read_bytes(), first_clip_bytes)

        store = task_store.default_store()
        middle = store.list_jobs(self.ep_id)[1]
        store.update_job(middle["job_id"], status="failed", error="delivery gate test")
        with self.assertRaisesRegex(RuntimeError, "incomplete panel jobs"):
            render_service.export(self.ep_id, "douyin")
        store.update_job(middle["job_id"], status="succeeded", error=None)

        expected_hashes = {}
        expected_selections = {}
        for job in store.list_jobs(self.ep_id):
            artifact_sha = hashlib.sha256(Path(job["output_path"]).read_bytes()).hexdigest()
            expected_hashes[job["job_id"]] = artifact_sha
            selection = select_edit_window(
                {
                    "decoded_visual_sha256": (job["metadata"]["content_qa"]["analysis"])["decoded_visual_sha256"],
                    "algorithm": {"sample_fps": 2.0},
                    "metrics": {"adjacent_luma_changes": [12.0] * 24},
                },
                source_duration_seconds=10.125,
                requested_duration_seconds=float(job["metadata"]["inputs"]["shot_plan"]["edit_duration_seconds"]),
                source_artifact_sha256=artifact_sha,
            )
            expected_selections[job["job_id"]] = selection["selection_sha256"]
            store.update_job(job["job_id"], metadata={**job["metadata"], "edit_selection": selection})
            render_service.approve_job_review(
                self.ep_id, job["job_id"], expected_artifact_sha256=artifact_sha,
                expected_edit_selection_sha256=selection["selection_sha256"],
                reviewed_by="E2E editor", reason="mock shot tells its unique approved beat",
            )
        render_service.approve_episode_release(
            self.ep_id, expected_artifact_hashes=expected_hashes,
            expected_edit_selection_hashes=expected_selections,
            approved_by="E2E publisher", reason="all eight mock shots approved",
        )

        commands = []

        def fake_runner(command, **_kwargs):
            commands.append(command)
            Path(command[-1]).write_bytes(b"mock-final-video")
            return mock.Mock(returncode=0)

        def fake_probe(path, **_kwargs):
            return {
                "path": str(path),
                "size_bytes": 16,
                "duration_seconds": 30.0,
                "video": {
                    "codec": "h264", "width": 720, "height": 1280,
                    "fps": 30.0, "pixel_format": "yuv420p",
                },
                "audio": {"codec": "aac", "sample_rate": 48000, "channels": 2},
            }

        manifest = render_service.export(
            self.ep_id,
            "douyin",
            burn_subtitles=True,
            runner=fake_runner,
            probe_func=fake_probe,
            ffmpeg="ffmpeg",
            ffprobe="ffprobe",
            quality_analyzer=lambda path, **kwargs: {
                "decoded_visual_sha256": hashlib.sha256(
                    f"visual:{next(job['job_id'] for job in store.list_jobs(self.ep_id) if Path(job['output_path']).resolve() == Path(path).resolve())}".encode("utf-8")
                ).hexdigest(),
                "perceptual_hashes": [hashlib.sha256(Path(path).name.encode("utf-8")).hexdigest()],
                "static": False, "metrics": {"sample_count": 1},
            },
        )
        self.assertEqual(manifest["probe"]["video"]["width"], 720)
        self.assertEqual(manifest["probe"]["video"]["height"], 1280)
        self.assertEqual(manifest["preset"]["delivery_standard"], "720p-v1")
        self.assertTrue(manifest["subtitles"]["burned_in"])
        self.assertEqual(manifest["release_status"], "approved")
        filter_flag = "-filter_complex" if "-filter_complex" in commands[0] else "-vf"
        self.assertIn("subtitles=", commands[0][commands[0].index(filter_flag) + 1])
        with zipfile.ZipFile(manifest["package_path"]) as package:
            names = set(package.namelist())
            self.assertIn("final.mp4", names)
            self.assertIn("episode.json", names)
            self.assertTrue(any(name.endswith(".srt") for name in names))


if __name__ == "__main__":
    unittest.main()
