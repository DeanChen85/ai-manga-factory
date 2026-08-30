from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, call
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from ui_helpers import (
    approve_job_review_via_facade,
    approve_release_via_facade,
    asset_gate_ready,
    attach_reference_images,
    attach_scene_reference_images,
    creative_gate_ready,
    continuity_anchor_candidates,
    classify_shot_worklist,
    content_review_summary,
    derive_project_stage,
    earliest_qa_rejected_failed_job,
    existing_media_paths,
    job_counts,
    job_media_for_review,
    job_review_evidence,
    generation_wait_notice,
    generation_input_signature,
    merge_episode_asset_review_state,
    normalize_jobs,
    persisted_delivery_manifests,
    prioritize_preview_media,
    reviewed_asset_hashes,
    runtime_prompt_audit,
    rejection_requires_group_anchor,
    resume_stage2_via_facade,
    stage2_resume_eligibility,
    shot_readiness_rows,
    start_continuity_safe_via_facade,
    updated_incomplete_stage2_checkpoint,
    with_asset_approval,
    with_asset_review_status,
    with_creative_approval,
)


class UiHelperTests(unittest.TestCase):
    def test_group_anchor_requirement_distinguishes_timing_from_visual_rejects(self):
        self.assertFalse(rejection_requires_group_anchor({}))
        self.assertFalse(rejection_requires_group_anchor({
            "qa_rejection_audit": [{"category": "action_timing_or_edit_window"}],
        }))
        for category in (
            "identity_or_character", "composition_or_scene", "continuity_or_state", "other",
        ):
            with self.subTest(category=category):
                self.assertTrue(rejection_requires_group_anchor({
                    "qa_rejection_audit": [{"category": category}],
                }))
        self.assertTrue(rejection_requires_group_anchor({
            "qa_rejection_audit": [{"reason": "legacy unclassified reject"}],
        }))
        self.assertFalse(rejection_requires_group_anchor({
            "qa_rejection_audit": [{"reason": "late action", "at": "t1"}],
            "qa_rejection_classification": {
                "category": "action_timing_or_edit_window",
                "rejection_reason": "late action", "rejection_at": "t1",
            },
        }))
        self.assertTrue(rejection_requires_group_anchor({
            "qa_rejection_audit": [{"reason": "different reject", "at": "t2"}],
            "qa_rejection_classification": {
                "category": "action_timing_or_edit_window",
                "rejection_reason": "late action", "rejection_at": "t1",
            },
        }))

    def test_runtime_prompt_audit_reads_only_project_snapshot_and_verifies_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt = "subject_definitions: <Picture 1> approved."
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            graph = root / "panel.graph.json"
            graph.write_text(json.dumps({
                "prompt": prompt,
                "prompt_audit": {
                    "prompt_sha256": digest,
                    "reference_bundle_sha256": "refs",
                    "skill_version": "director/v1",
                    "official_prompt_shape": "official/v1",
                    "runtime_prompt_contract": "runtime/v1",
                },
                "settings": {"reference_bindings": [
                    {"model_label": "<Picture 1>", "role": "first_frame"},
                ]},
                "reference_images": [{
                    "role": "first_frame", "source_path": str(root / "frame.png"),
                    "sha256": "frame-sha",
                }],
            }, ensure_ascii=False), encoding="utf-8")
            audit = runtime_prompt_audit({
                "graph_path": str(graph),
                "metadata": {"prompt_sha256": digest},
            }, [root])
            self.assertTrue(audit["available"])
            self.assertTrue(audit["prompt_hash_matches"])
            self.assertEqual(audit["prompt"], prompt)
            self.assertEqual(audit["references"][0]["model_label"], "<Picture 1>")
            blocked = runtime_prompt_audit({"graph_path": str(graph)}, [root / "other"])
            self.assertFalse(blocked["available"])
            self.assertIn("拒绝读取", blocked["error"])

    def test_generation_wait_notice_is_truthful_about_timeout_and_stop(self):
        with patch("ui_helpers.time.strftime", return_value="2026-08-13 21:00:00"):
            notice = generation_wait_notice(1.0, 90)
        self.assertEqual(notice["timeout_seconds"], 90.0)
        self.assertIn("最长等待 90 秒", notice["headline"])
        self.assertIn("Stop 不是远端取消确认", notice["stop_help"])
        self.assertIn("不会保存半成品合同", notice["failure_help"])
        self.assertIn("不会自动重试付费请求", notice["failure_help"])

        two_stage = generation_wait_notice(1.0, 90, planned_calls=2)
        self.assertEqual(two_stage["planned_calls"], 2)
        self.assertIn("计划 2 次调用", two_stage["headline"])
        self.assertIn("每次最长等待 90 秒", two_stage["headline"])

    def test_stage2_checkpoint_resume_is_input_bound_and_calls_exact_facade(self):
        settings = {"topic": "雨夜", "shot_count": 7, "api_key": "must-not-bind"}
        signature = generation_input_signature("ep_resume", "两位核心人物", settings)
        self.assertEqual(
            signature,
            generation_input_signature(
                "ep_resume", "两位核心人物",
                {"topic": "雨夜", "shot_count": 7, "api_key": "different-secret"},
            ),
        )
        self.assertNotEqual(
            signature,
            generation_input_signature("ep_resume", "梗概已修改", settings),
        )
        checkpoint = {
            "checkpoint_sha256": "a" * 64,
            "stage1_status": "validated", "stage2_status": "failed",
            "protocol": "anthropic", "model": "MiniMax-M2.7",
        }
        ready = stage2_resume_eligibility(
            checkpoint,
            saved_ep_id="ep_resume", current_ep_id="ep_resume",
            saved_input_signature=signature, current_input_signature=signature,
            protocol="anthropic", model="MiniMax-M2.7",
        )
        self.assertTrue(ready["ready"])
        changed = stage2_resume_eligibility(
            checkpoint,
            saved_ep_id="ep_resume", current_ep_id="ep_resume",
            saved_input_signature=signature, current_input_signature="b" * 64,
            protocol="anthropic", model="MiniMax-M2.7",
        )
        self.assertFalse(changed["ready"])
        self.assertIn("输入或生成设置已变化", changed["reason"])

        handler = Mock(return_value={"panels": [1]})
        progress = Mock()
        result = resume_stage2_via_facade(
            handler, "两位核心人物", ep_id="ep_resume",
            checkpoint_hash="a" * 64,
            settings={"topic": "雨夜", "shot_count": 7},
            api_key="offline-only", progress_cb=progress,
        )
        self.assertEqual(result, {"panels": [1]})
        handler.assert_called_once_with(
            "两位核心人物",
            ep_id="ep_resume", checkpoint_hash="a" * 64,
            api_key="offline-only", progress_cb=progress,
            topic="雨夜", shot_count=7,
        )

    def test_stage2_failure_discovers_only_new_or_updated_incomplete_checkpoint(self):
        old = {
            "checkpoint_sha256": "a" * 64,
            "stage1_status": "validated", "stage2_status": "pending",
            "stage2_attempt_count": 0, "updated_at": "2026-08-14T00:00:00Z",
        }
        failed = {
            **old, "stage2_status": "failed", "stage2_attempt_count": 1,
            "updated_at": "2026-08-14T00:01:00Z",
        }
        self.assertEqual(
            updated_incomplete_stage2_checkpoint([old], [failed]), failed,
        )
        self.assertEqual(updated_incomplete_stage2_checkpoint([old], [old]), {})
        self.assertEqual(
            updated_incomplete_stage2_checkpoint([], [{**old, "stage2_status": "completed"}]),
            {},
        )

    def test_web_journey_uses_public_facade_and_exposes_both_gates(self):
        source = (Path(__file__).resolve().parents[1] / "pipeline" / "web_app.py").read_text(encoding="utf-8")
        helper_source = (Path(__file__).resolve().parents[1] / "pipeline" / "ui_helpers.py").read_text(encoding="utf-8")
        self.assertIn("门禁一", source)
        self.assertIn("门禁二", source)
        for public_call in (
            "prepare_contract", "approve_contract", "prepare_assets", "approve_assets",
            "start_production", "reject_asset", "retry_asset",
            "select_asset_references",
        ):
            self.assertIn(public_call, source)
        for forbidden in ("sqlite3", "render_video_h3", "subprocess", "ffmpeg"):
            self.assertNotIn(forbidden, source.lower())
        for series_journey in (
            "整季 V4", "总集数", "每集秒数", "season_outline",
            "split_series", "generate_series_episode", "update_series_outline_episode",
            "全季共享人物", "逐集连续大纲与 V3 审核卡",
        ):
            self.assertIn(series_journey, source)
        for series_facade in (
            "prepare_series", "approve_series", "register_episodes",
            "prepare_shared_assets", "approve_shared_assets", "reject_shared_asset",
            "retry_shared_asset", "start_episode", "start_series", "resume_episode",
            "resume_series", "retry_episode", "retry_series", "cancel_episode",
            "cancel_series", "status_series", "export_episode", "export_season",
        ):
            self.assertIn(series_facade, source)
        self.assertIn('@st.fragment(run_every="2s")', source)
        self.assertIn("检测并修复群像镜头人物覆盖", source)
        self.assertIn("群像镜头没有完整绑定同场人物", source)
        self.assertIn('st.session_state["flash_success"] = "创作合同已保存', source)
        self.assertIn('@st.fragment(run_every="2s")\ndef _asset_gate', source)
        self.assertIn('参考图片已全部生成并回传', source)
        self.assertIn('_render_media(refs, max_items=3)', source)
        self.assertIn('contract_valid=not bool(contract_errors)', source)
        self.assertIn('if st.session_state.get("local_dirty"):', source)
        self.assertIn('episode = local_episode', source)
        self.assertIn('整批拒绝并重生（画风 / 人物 / 场景不一致）', source)
        self.assertIn('f"{ep_id}:{kind}:{item_id}"', source)
        self.assertIn('并只启动一个后台 worker', source)
        self.assertIn('snapshot_episode(refreshed, rejected), rejected', source)
        self.assertIn('review_status in {"rejected", "failed"}', source)
        self.assertIn('"重试生成" if review_status == "failed"', source)
        self.assertIn('使用这些图片替换当前场景资产', source)
        self.assertIn('or bool(contract_errors)\n            or not backend_gate_ready', source)
        self.assertIn('or "character_ids unknown" in warning', source)
        self.assertIn('MODERN_URBAN_STYLE_PROMPT', source)
        self.assertIn('"youtube_shorts"', source)
        self.assertIn('"youtube"', source)
        self.assertIn('"竖屏720p｜抖音（720×1280）"', source)
        self.assertIn('"横屏720p｜YouTube（1280×720）"', source)
        self.assertIn("统一720p交付", source)
        self.assertIn('@st.fragment(run_every="2s")\ndef _delivery_panel(ep_id: str)', source)
        self.assertIn('_delivery_panel(ep_id)', source)
        self.assertIn('worker_result = _call_service(start_handler, ep_id, current_episode)', source)
        self.assertIn('本镜已重新排队并请求 worker 启动', source)
        self.assertIn('拒收本镜并重置后续连续镜头', source)
        self.assertIn('_service("reject_job")', source)
        self.assertIn('严格连续链下游已清除旧尾帧并等待重跑', source)
        self.assertIn('dirty=False,\n        )\n        st.rerun()', source)
        self.assertIn('approve_handler(ep_id, expected_hashes=expected_hashes)', source)
        self.assertIn('MiniMax H3 提示词大师重生本镜', source)
        self.assertIn('api_key=api_key or None', source)
        self.assertIn('os.environ.get("MiniMax_API_KEY"', source)
        self.assertIn('if _asset_sync_signature(episode) != before_sync:', source)
        self.assertIn('job_media_for_review(job, [PROJECTS_DIR, COMFYUI_OUTPUT])', source)
        self.assertIn('旧片仅归档供审计，不参与合片', source)
        self.assertIn('拒收审计片（历史版本，不参与合片）', source)
        self.assertIn('"烧录批准对白字幕"', source)
        self.assertIn('value=True,\n        key=f"delivery_burn_subtitles_{ep_id}"', source)
        self.assertIn('burn_subtitles=burn_subtitles', source)
        self.assertIn('subtitle_strict=True', source)
        self.assertIn('approved spoken_dialogue → 确定性后期字幕', source)
        self.assertIn('persisted_delivery_manifests(snapshot, [PROJECTS_DIR / ep_id])', source)
        self.assertIn('已恢复批准发布的平台交付', source)
        self.assertIn('历史技术导出，不可发布', source)
        self.assertIn('metrics[1].metric("burned_in"', source)
        self.assertIn('metrics[2].metric("subtitle_strict"', source)
        self.assertIn('"下载最终 MP4"', source)
        self.assertIn('"下载 delivery.zip"', source)
        self.assertIn('交付 Manifest 关键值', source)
        for manifest_value in ("output_path", "package_path", "manifest_path", "subtitles"):
            self.assertIn(f'"{manifest_value}"', source)
        self.assertIn('safe_root = earliest_qa_rejected_failed_job(jobs)', source)
        self.assertIn('continuity_anchor_candidates(', source)
        self.assertIn('_service("approve_continuity_anchor")', source)
        self.assertIn('_service("start_continuity_safe")', source)
        self.assertIn('start_continuity_safe_via_facade(', source)
        self.assertIn('str(safe_root["job_id"])', source)
        self.assertIn('preferred_voice="Microsoft Huihui Desktop"', helper_source)
        self.assertIn('motion="slow_push"', helper_source)
        self.assertIn('burn_subtitles=False', helper_source)
        self.assertNotIn('_call_service(safe_handler, ep_id, current_episode)', source)
        self.assertIn('value=False,\n                key=f"continuity_safe_confirm_{ep_id}"', source)
        self.assertIn('连续性安全模式（低动态，严格人物与字幕）', source)
        self.assertIn('这不是 H3 重抽或高动态生成，也不会自动启用', source)
        self.assertIn('metadata.get("render_mode") == "continuity_safe"', source)
        self.assertIn('continuity_safe：静态锚保底样片', source)
        self.assertIn('正式素材生成完成（待内容验收）', source)
        self.assertIn('预演生成完成（待审核晋级，禁止交付）', source)
        self.assertIn('自动 QA 通过', source)
        self.assertIn('批准当前正式镜头版本', source)
        self.assertIn('批准预演并晋级正式生产', source)
        self.assertIn('人物身份、脸型、发型、服装、人数和场景连续', source)
        self.assertIn('生成画面无字幕、标题、标签、Logo、招牌、乱码字形', source)
        self.assertIn('or not identity_confirmed', source)
        self.assertIn('or not clean_frame_confirmed', source)
        self.assertIn('最终发送给 MiniMax H3 的提示词与参照审计', source)
        self.assertIn('or not prompt_audit["prompt_hash_matches"]', source)
        self.assertIn('主体动作、首尾状态变化和本镜叙事信息', source)
        self.assertIn('静态锚保底样片，非剧情镜头', source)
        self.assertIn('approve_job_review_via_facade(', source)
        self.assertIn('approve_release_via_facade(', source)
        self.assertIn('expected_edit_selection_sha256', helper_source)
        self.assertIn('expected_edit_selection_hashes', helper_source)
        self.assertIn('edit_selection_sha256,', source)
        self.assertIn('review["edit_selection_hashes"]', source)
        self.assertIn('每镜 H3 固定生成 10.125 秒源素材', source)
        self.assertIn('预计 GPU 镜数', source)
        self.assertIn('镜数由短剧剪辑密度自动规划', source)
        self.assertIn('request_started_at = time.time()', source)
        self.assertIn('planned_calls=planned_calls', source)
        self.assertIn('将调用 MiniMax 2 次，两次可能分别计费', source)
        self.assertIn('split_story(\n                    synopsis, api_key=api_key or None, demo_mode=False,\n                    progress_cb=show_generation_stage, ep_id=ep_id, **settings,', source)
        self.assertIn('阶段 1 已保存，阶段 2 未完成', source)
        self.assertIn('只重试阶段 2（将只调用 MiniMax 1 次，可能计费）', source)
        self.assertIn('resume_stage2_via_facade(', source)
        self.assertIn('checkpoint_hash=str(resume_checkpoint["checkpoint_sha256"])', source)
        self.assertIn('split_story_checkpoint_inputs(synopsis, **settings)', source)
        self.assertIn('resume_checkpoint = match_stage1_checkpoint(', source)
        self.assertIn('creative_brief=checkpoint_inputs["creative_brief"]', source)
        self.assertIn('settings=checkpoint_inputs["settings"]', source)
        self.assertIn('progress_cb=show_generation_stage', source)
        self.assertIn('开始时间 {wait_notice[\'started_at\']}', source)
        self.assertIn('Stop 不是远端取消确认', helper_source)
        self.assertIn('系统不会自动重试，请检查错误后再手动点击生成', source)
        self.assertIn('主题、故事梗概、时长、镜数、语言和风格输入均保留', source)
        self.assertIn('minimax_configuration_status()', source)
        self.assertIn('MiniMax 配置已弃用', source)
        self.assertIn('导出硬门未通过', source)
        self.assertNotIn("载入单集生产台", source)

    def test_normalize_and_count_jobs(self):
        jobs = normalize_jobs({"jobs": [
            {"job_id": "a", "panel_name": "p1", "status": "succeeded"},
            {"job_id": "b", "panel_name": "p2", "status": "running"},
            {"job_id": "c", "panel_name": "p3", "status": "failed"},
        ]})
        self.assertEqual(jobs[0]["panel_id"], "p1")
        self.assertEqual(job_counts(jobs), {"total": 3, "success": 1, "active": 1, "failed": 1, "other": 0})

    def test_shot_readiness_rows_requires_contract_assets_job_and_provider_proof(self):
        panel = {
            "panel_id": "panel_01", "scene_id": "scene_room", "character_ids": ["char_a"],
            "visible_action": "林川把药盒推到顾远面前停住",
            "action_components": {
                "sub": "林川", "verb": "推", "obj": "药盒", "res": "药盒停在顾远面前",
            },
            "camera_plan": {
                "shot_size": "medium", "angle": "eye level",
                "movement": "slow push", "composition": "characters on thirds",
            },
            "first_state": "药盒在林川手中", "final_state": "药盒停在顾远面前",
            "first_frame": "林川握住药盒", "last_frame": "药盒停在桌面",
            "spoken_dialogue": [{
                "speaker_id": "char_a", "text": "拿好。", "start_s": 0.2, "end_s": 1.0,
            }],
            "subtitle_timeline": [{
                "speaker_id": "char_a", "text": "拿好。", "start_s": 0.2, "end_s": 1.0,
            }],
        }
        episode = {
            "character_bible": [{"character_id": "char_a", "reference_images": ["char.png"]}],
            "scene_bible": [{"scene_id": "scene_room", "reference_images": ["room.png"]}],
            "panels": [panel],
        }
        snapshot = {
            "pipeline": {"assets_status": "approved"},
            "assets": {"items": [
                {
                    "asset_type": "character", "source_id": "char_a", "status": "succeeded",
                    "approved": True, "content_hash": "char-hash", "reference_images": ["char.png"],
                },
                {
                    "asset_type": "scene", "source_id": "scene_room", "status": "succeeded",
                    "approved": True, "content_hash": "scene-hash", "reference_images": ["room.png"],
                },
            ]},
            "jobs": [{
                "job_id": "job_01", "panel_name": "panel_01", "status": "succeeded",
                "output_path": "panel_01.mp4",
                "metadata": {"render_mode": "h3", "artifact_sha256": "artifact-01"},
            }],
        }
        rows = {row["key"]: row for row in shot_readiness_rows(panel, episode, snapshot)}
        self.assertEqual(set(rows), {
            "identity", "action", "camera", "first_final", "references",
            "dialogue_subtitles", "asset_refs", "job_provider",
        })
        self.assertTrue(all(row["ready"] for row in rows.values()))

        broken_panel = dict(panel)
        broken_panel["action_components"] = {"sub": "林川", "verb": "", "obj": "药盒", "res": "停住"}
        broken_panel["camera_plan"] = {"shot_size": "medium"}
        broken_panel["subtitle_timeline"] = [{
            "speaker_id": "char_a", "text": "另一套台词", "start_s": 0.2, "end_s": 1.0,
        }]
        broken_snapshot = {
            **snapshot,
            "pipeline": {"assets_status": "ready_for_approval"},
            "jobs": [{
                **snapshot["jobs"][0],
                "metadata": {"artifact_sha256": "artifact-01"},
            }],
        }
        broken = {
            row["key"]: row
            for row in shot_readiness_rows(broken_panel, episode, broken_snapshot)
        }
        self.assertTrue(broken["action"]["blocking"])
        self.assertTrue(broken["camera"]["blocking"])
        self.assertTrue(broken["dialogue_subtitles"]["blocking"])
        self.assertTrue(broken["asset_refs"]["blocking"])
        self.assertTrue(broken["job_provider"]["blocking"])
        self.assertIn("未上报", broken["job_provider"]["detail"])

        unregistered = {
            row["key"]: row
            for row in shot_readiness_rows(panel, episode, {**snapshot, "jobs": []})
        }
        self.assertFalse(unregistered["job_provider"]["ready"])
        self.assertIn("尚未注册", unregistered["job_provider"]["detail"])

    def test_classify_shot_worklist_is_fail_closed_for_qa_hash_and_release(self):
        def successful_job(job_id, *, review=False, release=False):
            artifact = f"artifact-{job_id}"
            selection = f"selection-{job_id}"
            metadata = {
                "artifact_sha256": artifact,
                "edit_selection": {
                    "selection_sha256": selection,
                    "source_artifact_sha256": artifact,
                    "in_seconds": 0.5, "out_seconds": 2.5, "duration_seconds": 2.0,
                },
                "content_qa": {
                    "status": "passed", "passed": True,
                    "analysis": {
                        "decoded_visual_sha256": f"visual-{job_id}",
                        "source_path": f"{job_id}.mp4",
                    },
                },
            }
            if review:
                metadata["editorial_review"] = {
                    "status": "approved", "artifact_sha256": artifact,
                    "edit_selection_sha256": selection,
                }
            if release:
                metadata["release"] = {
                    "status": "approved", "artifact_sha256": artifact,
                    "edit_selection_sha256": selection,
                    "decoded_visual_sha256": f"visual-{job_id}",
                }
            return {
                "job_id": job_id, "panel_name": job_id, "status": "succeeded",
                "output_path": f"{job_id}.mp4", "metadata": metadata,
            }

        no_qa = successful_job("no_qa")
        no_qa["metadata"].pop("content_qa")
        jobs = [
            {"job_id": "failed", "panel_name": "failed", "status": "failed"},
            {"job_id": "running", "panel_name": "running", "status": "running"},
            no_qa,
            successful_job("human_pending"),
            successful_job("release_pending", review=True),
            successful_job("passed", review=True, release=True),
        ]
        result = classify_shot_worklist(
            jobs, {"jobs": jobs, "pipeline": {"release_status": "approved"}},
        )
        self.assertEqual(result["counts"], {
            "needs_attention": 2, "active": 1, "awaiting_review": 2,
            "passed": 1, "all": 6,
        })
        self.assertEqual(
            {item["job_id"] for item in result["needs_attention"]}, {"failed", "no_qa"},
        )
        self.assertEqual([item["job_id"] for item in result["active"]], ["running"])
        self.assertEqual(
            {item["job_id"] for item in result["awaiting_review"]},
            {"human_pending", "release_pending"},
        )
        self.assertEqual([item["job_id"] for item in result["passed"]], ["passed"])
        self.assertTrue(all(item.get("worklist_reason") for item in result["all"]))

        stale_release = successful_job("stale", review=True, release=True)
        stale_release["metadata"]["release"]["artifact_sha256"] = "old-artifact"
        stale = classify_shot_worklist(
            [stale_release], {
                "jobs": [stale_release], "pipeline": {"release_status": "approved"},
            },
        )
        self.assertEqual(stale["counts"]["passed"], 0)
        self.assertEqual(stale["counts"]["awaiting_review"], 1)
        self.assertIn("release", stale["awaiting_review"][0]["worklist_reason"])

        revoked = successful_job("revoked", review=True, release=True)
        revoked_result = classify_shot_worklist(
            [revoked], {"jobs": [revoked], "pipeline": {"release_status": "revoked"}},
        )
        self.assertEqual(revoked_result["counts"]["passed"], 0)
        self.assertEqual(revoked_result["counts"]["awaiting_review"], 1)

    def test_web_renders_readiness_and_defaults_to_attention_worklist(self):
        source = (Path(__file__).resolve().parents[1] / "pipeline" / "web_app.py").read_text(encoding="utf-8")
        self.assertIn("shot_readiness_rows(panel, episode, snapshot)", source)
        self.assertIn("逐镜生产就绪清单", source)
        self.assertIn("高级：原始 JSON", source)
        self.assertIn("classify_shot_worklist(jobs, snapshot)", source)
        self.assertIn('"镜头工作清单"', source)
        self.assertIn('index=0,', source)
        self.assertIn('for job in visible_jobs:', source)
        self.assertIn("总计数与生产控制始终基于全部任务", source)

    def test_content_review_gate_fails_closed_for_same_anchor_safe_chain(self):
        jobs = []
        for index in range(1, 7):
            jobs.append({
                "job_id": f"job_{index}", "panel_id": f"panel_{index}",
                "status": "succeeded",
                "metadata": {
                    "artifact_sha256": f"container-hash-{index}",
                    "render_mode": "continuity_safe",
                    "continuity_safe": {"source_anchor_sha256": "same-visual-anchor"},
                },
            })
        snapshot = {
            "jobs": jobs,
            "deliveries": ["historical.manifest.json"],
            "pipeline": {"release_status": "approved"},
        }
        summary = content_review_summary(snapshot)
        self.assertEqual(summary["technical_complete"], 6)
        self.assertEqual(summary["automated_qa_passed"], 0)
        self.assertEqual(summary["human_approved"], 0)
        self.assertEqual(summary["same_anchor_safe_count"], 6)
        self.assertFalse(summary["release_approved"])
        self.assertFalse(summary["ready_for_export"])
        self.assertEqual(
            derive_project_stage(
                {
                    "character_bible": [{"character_id": "char", "reference_images": ["char.png"]}],
                    "scene_bible": [{"scene_id": "scene", "reference_images": ["scene.png"]}],
                    "approval_state": {
                        "creative": {"story": True, "characters": True, "storyboard": True},
                        "assets": {"character_ids": ["char"], "scene_ids": ["scene"]},
                    },
                },
                snapshot,
                contract_persisted=True,
            ),
            "release_revoked",
        )

    def test_content_review_gate_requires_artifact_bound_qa_review_and_release(self):
        jobs = []
        hashes = {}
        selections = {}
        for index in range(1, 3):
            job_id = f"job_{index}"
            artifact = f"artifact-{index}"
            hashes[job_id] = artifact
            selection_hash = f"selection-{index}"
            selections[job_id] = selection_hash
            jobs.append({
                "job_id": job_id, "panel_id": f"panel_{index}", "status": "succeeded",
                "metadata": {
                    "artifact_sha256": artifact,
                    "edit_selection": {
                        "selection_sha256": selection_hash,
                        "source_artifact_sha256": artifact,
                        "in_seconds": 1.0, "out_seconds": 3.0,
                        "duration_seconds": 2.0,
                    },
                    "content_qa": {
                        "passed": True,
                        "analysis": {
                            "decoded_visual_sha256": f"visual-{index}",
                            "source_path": f"clip-{index}.mp4",
                        },
                    },
                    "editorial_review": {
                        "status": "approved", "artifact_sha256": artifact,
                        "edit_selection_sha256": selection_hash,
                    },
                    "release": {
                        "status": "approved", "artifact_sha256": artifact,
                        "decoded_visual_sha256": f"visual-{index}",
                        "edit_selection_sha256": selection_hash,
                    },
                },
                "output_path": f"clip-{index}.mp4",
            })
        snapshot = {
            "jobs": jobs,
            "pipeline": {"release_status": "approved"},
            "release": {
                "status": "approved", "approved_artifact_hashes": hashes,
                "approved_edit_selection_hashes": selections,
            },
        }
        summary = content_review_summary(snapshot)
        self.assertEqual(summary["automated_qa_passed"], 2)
        self.assertEqual(summary["human_approved"], 2)
        self.assertTrue(summary["release_approved"])
        self.assertTrue(summary["ready_for_export"])

        snapshot["jobs"][0]["metadata"]["artifact_sha256"] = "regenerated"
        stale = content_review_summary(snapshot)
        # Re-encoded bytes do not invalidate decoded-visual QA, but they do
        # invalidate the artifact-bound human/release approvals below.
        self.assertEqual(stale["automated_qa_passed"], 2)
        self.assertEqual(stale["human_approved"], 1)
        self.assertFalse(stale["release_approved"])
        self.assertFalse(stale["ready_for_export"])

    def test_review_facades_bind_exact_current_artifact_hashes(self):
        service = Mock()
        job_call = Mock(return_value={"approved": True})
        release_call = Mock(return_value={"release": "approved"})

        def approve_job_review(
            ep_id, job_id, *, expected_artifact_sha256, expected_edit_selection_sha256,
        ):
            return job_call(
                ep_id, job_id, expected_artifact_sha256=expected_artifact_sha256,
                expected_edit_selection_sha256=expected_edit_selection_sha256,
            )

        def approve_episode_release(
            ep_id, *, expected_artifact_hashes, expected_edit_selection_hashes, qa_report_hash="",
        ):
            return release_call(
                ep_id,
                expected_artifact_hashes=expected_artifact_hashes,
                expected_edit_selection_hashes=expected_edit_selection_hashes,
                qa_report_hash=qa_report_hash,
            )

        service.approve_job_review = approve_job_review
        service.approve_episode_release = approve_episode_release
        result = approve_job_review_via_facade(
            service, "ep_1", "job_1", "artifact-sha", "selection-sha",
        )
        self.assertEqual(result, {"approved": True})
        job_call.assert_called_once_with(
            "ep_1", "job_1", expected_artifact_sha256="artifact-sha",
            expected_edit_selection_sha256="selection-sha",
        )
        released = approve_release_via_facade(
            service, "ep_1", {"job_1": "artifact-sha"},
            {"job_1": "selection-sha"}, qa_report_hash="qa-sha",
        )
        self.assertEqual(released, {"release": "approved"})
        release_call.assert_called_once_with(
            "ep_1",
            expected_artifact_hashes={"job_1": "artifact-sha"},
            expected_edit_selection_hashes={"job_1": "selection-sha"},
            qa_report_hash="qa-sha",
        )

    def test_review_evidence_requires_existing_first_middle_last_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frames = {}
            for slot in ("first", "middle", "last"):
                path = root / f"{slot}.png"
                path.write_bytes(slot.encode())
                frames[f"{slot}_frame_path"] = str(path)
            job = {
                "job_id": "job_1", "metadata": {
                    "content_qa": {"status": "passed", "evidence": {
                        **frames, "action": "hand reaches phone",
                        "first_last": "hands down -> phone raised",
                    }},
                },
                "output_path": str(root / "clip.mp4"),
            }
            (root / "clip.mp4").write_bytes(b"video")
            evidence = job_review_evidence({"jobs": [job]}, job, [root])
            self.assertTrue(evidence["complete"])
            self.assertEqual(evidence["action"], "hand reaches phone")
            self.assertEqual(evidence["first_last"], "hands down -> phone raised")

    def test_review_evidence_prefers_full_source_samples_over_two_frame_edit_window(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            job = {
                "job_id": "job_1", "output_path": str(video),
                "probe": {"duration_seconds": 5.0},
                "metadata": {"content_qa": {
                    "passed": True,
                    "analysis": {"sample_frame_sha256": ["a", "b"]},
                    "source_analysis": {
                        "source_path": str(video),
                        "decoded_visual_sha256": "decoded-source",
                        "sample_frame_sha256": ["1", "2", "3", "4", "5"],
                        "metrics": {"first_last_luma_change": 0.08},
                    },
                }, "edit_selection": {"reason": "hand reaches counter"}},
            }
            evidence = job_review_evidence({"jobs": [job]}, job, [root])
            self.assertTrue(evidence["complete"])
            self.assertEqual(evidence["sample_frame_sha256"], ["1", "2", "3", "4", "5"])
            self.assertEqual(evidence["action"], "hand reaches counter")
            self.assertEqual(evidence["action_source"], "edit_selection_fallback")
            self.assertEqual(evidence["first_last"], 0.08)
            self.assertEqual(evidence["first_last_source"], "decoded_visual_luma_metric")

    def test_review_evidence_binds_current_manual_selection_not_full_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            job = {
                "job_id": "job_1", "output_path": str(video),
                "probe": {"duration_seconds": 5.0},
                "metadata": {
                    "artifact_sha256": "a" * 64,
                    "edit_selection": {
                        "in_seconds": 1.5, "out_seconds": 3.1, "duration_seconds": 1.6,
                        "selection_sha256": "s" * 64, "source_artifact_sha256": "a" * 64,
                    },
                    "content_qa": {
                        "passed": True,
                        "analysis": {
                            "source_path": str(video), "decoded_visual_sha256": "selected",
                            "sample_frame_sha256": ["a", "b", "c"],
                        },
                        "source_analysis": {
                            "source_path": str(video), "decoded_visual_sha256": "full",
                            "sample_frame_sha256": ["1", "2", "3", "4", "5"],
                        },
                    },
                },
            }
            evidence = job_review_evidence({"jobs": [job]}, job, [root])
            self.assertTrue(evidence["complete"])
            self.assertEqual(evidence["sample_frame_sha256"], ["a", "b", "c"])
            self.assertEqual(evidence["review_window"]["binding"], "current_edit_selection")
            self.assertEqual(evidence["review_window"]["selection_sha256"], "s" * 64)
            self.assertAlmostEqual(evidence["video_timestamps"]["first"], 1.5)
            self.assertAlmostEqual(evidence["video_timestamps"]["last"], 3.0)

    def test_review_evidence_rejects_hashes_not_bound_to_current_video(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "clip.mp4"
            other = root / "other.mp4"
            video.write_bytes(b"video")
            other.write_bytes(b"other")
            job = {
                "job_id": "job_1", "output_path": str(video),
                "probe": {"duration_seconds": 5.0},
                "metadata": {"content_qa": {
                    "passed": True,
                    "source_analysis": {
                        "source_path": str(other),
                        "decoded_visual_sha256": "decoded-other",
                        "sample_frame_sha256": ["1", "2", "3", "4", "5"],
                    },
                }},
            }
            evidence = job_review_evidence({"jobs": [job]}, job, [root])
            self.assertFalse(evidence["complete"])

    def test_attach_reference_images_is_immutable_and_deduplicates(self):
        episode = {"character_bible": [{"character_id": "char_a", "reference_images": ["a.png"]}]}
        updated = attach_reference_images(episode, "char_a", ["a.png", "b.png"])
        self.assertEqual(episode["character_bible"][0]["reference_images"], ["a.png"])
        self.assertEqual(updated["character_bible"][0]["reference_images"], ["a.png", "b.png"])

    def test_existing_media_paths_finds_nested_relative_asset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            media = root / "clip.mp4"
            media.write_bytes(b"test")
            found = existing_media_paths({"outputs": {"video": "clip.mp4"}}, [root])
            self.assertEqual(found, [media.resolve()])

    def test_preview_media_prioritizes_video_ahead_of_many_references(self):
        refs = [Path(f"char_{index}.png") for index in range(8)]
        video = Path("panel.mp4")
        comfy_duplicate = Path("comfy-panel-copy.mp4")
        ordered = prioritize_preview_media([*refs, video, comfy_duplicate, refs[0], video])
        self.assertEqual(ordered[0], video)
        self.assertEqual(len(ordered), 9)
        self.assertNotIn(comfy_duplicate, ordered)

    def test_rejected_and_retried_jobs_never_expose_stale_output_as_current_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stale = root / "ep01_panel02_00003_.mp4"
            comfy = root / "comfy_old.mp4"
            archived = root / "rejected" / "panel02" / "old.mp4"
            for path in (stale, comfy, archived):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(path.name.encode())
            base = {
                "job_id": "job_02",
                "status": "failed",
                "output_path": str(stale),
                "preview_path": str(stale),
                "comfy_output_path": str(comfy),
                "error": "QA rejected: random burned text",
                "metadata": {"qa_rejection_audit": [{
                    "reason": "random burned text",
                    "archived_files": {"output_path": {"path": str(archived)}},
                }]},
            }
            failed = job_media_for_review(base, [root])
            self.assertTrue(failed["qa_invalidated"])
            self.assertEqual(failed["current"], [])
            self.assertEqual(failed["audit"], [archived.resolve()])

            for retry_status in ("queued", "pending", "submitted", "running"):
                retried = job_media_for_review({**base, "status": retry_status, "error": None}, [root])
                self.assertEqual(retried["current"], [], retry_status)
                self.assertEqual(retried["audit"], [archived.resolve()], retry_status)

            new_output = root / "panel02_regenerated.mp4"
            new_output.write_bytes(b"new accepted clip")
            succeeded = job_media_for_review({
                **base,
                "status": "succeeded",
                "error": None,
                "output_path": str(new_output),
                "preview_path": str(new_output),
                "comfy_output_path": None,
            }, [root])
            self.assertEqual(succeeded["current"], [new_output.resolve()])
            self.assertEqual(succeeded["audit"], [archived.resolve()])

    def test_safe_mode_selects_earliest_failed_qa_rejection(self):
        jobs = [
            {
                "job_id": "job_03", "panel_index": 3, "status": "failed",
                "metadata": {"qa_rejection_audit": [{"reason": "drift"}]},
            },
            {
                "job_id": "job_02", "panel_index": 2, "status": "error",
                "error": "QA rejected: wrong group count", "metadata": {},
            },
            {
                "job_id": "job_01", "panel_index": 1, "status": "queued",
                "metadata": {"qa_rejection_audit": [{"reason": "old rejection"}]},
            },
        ]
        self.assertEqual(earliest_qa_rejected_failed_job(jobs)["job_id"], "job_02")

    def test_safe_anchor_candidates_follow_authority_priority_and_require_project_files(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            project = Path(folder).resolve()
            continuity_tail = project / "continuity" / "panel_02_tail.png"
            panel_tail = project / "panels" / "panel_02_last.png"
            previous_tail = project / "continuity" / "panel_01_tail.png"
            approved_first = project / "group_anchors" / "panel_02_first.png"
            approved_last = project / "group_anchors" / "panel_02_last.png"
            external = Path(outside).resolve() / "external.png"
            for path in (
                continuity_tail, panel_tail, previous_tail,
                approved_first, approved_last, external,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
            previous = {
                "job_id": "job_01", "panel_id": "panel_01", "status": "succeeded",
                "metadata": {"last_frame_path": str(previous_tail)},
            }
            failed = {
                "job_id": "job_02", "panel_id": "panel_02", "status": "failed",
                "metadata": {
                    "continuity_tail_path": str(continuity_tail),
                    "source_anchor": str(external),
                    "approved_group_anchor": {
                        "status": "approved", "path": str(approved_first),
                        "last_path": str(approved_last),
                    },
                    "inputs": {"continuity_dependency": {
                        "previous_job_id": "job_01", "strict": True,
                    }},
                },
            }
            episode = {"panels": [
                {"panel_id": "panel_01", "last_frame_path": str(previous_tail)},
                {"panel_id": "panel_02", "last_frame_path": str(panel_tail)},
            ]}
            candidates = continuity_anchor_candidates(failed, episode, [previous, failed], [project])
            self.assertEqual(
                [item["source"] for item in candidates],
                [
                    "approved_group_anchor_final", "approved_group_anchor_first",
                    "continuity_tail", "panel_last_frame", "previous_succeeded_tail",
                ],
            )
            self.assertEqual(
                [Path(item["path"]) for item in candidates],
                [
                    approved_last.resolve(), approved_first.resolve(), continuity_tail.resolve(),
                    panel_tail.resolve(), previous_tail.resolve(),
                ],
            )
            self.assertNotIn(str(external), [item["path"] for item in candidates])

    def test_safe_mode_facade_approves_anchor_before_exact_job_launch(self):
        service = Mock()
        service.approve_continuity_anchor.return_value = {"approved": True}
        service.start_continuity_safe.return_value = {"started": True}
        approval, launch = start_continuity_safe_via_facade(
            service,
            "ep_01",
            "job_02",
            r"F:\project\anchor.png",
            "approved five-person group anchor",
        )
        self.assertEqual(approval, {"approved": True})
        self.assertEqual(launch, {"started": True})
        self.assertEqual(service.mock_calls, [
            call.approve_continuity_anchor(
                "ep_01", "job_02", r"F:\project\anchor.png",
                reason="approved five-person group anchor",
            ),
            call.start_continuity_safe(
                "ep_01", "job_02",
                preferred_voice="Microsoft Huihui Desktop",
                motion="slow_push",
                burn_subtitles=False,
            ),
        ])

    def test_two_gates_and_project_stage_require_every_asset_approval(self):
        episode = {
            "character_bible": [{"character_id": "char_a", "reference_images": ["char.png"]}],
            "scene_bible": [{"scene_id": "scene_a", "reference_images": ["scene.png"]}],
            "panels": [{"panel_id": "panel_a", "scene_id": "scene_a"}],
        }
        for section in ("story", "characters", "storyboard"):
            episode = with_creative_approval(episode, section, True)
        self.assertTrue(creative_gate_ready(episode))
        self.assertEqual(derive_project_stage(episode, contract_persisted=False), "creative_approved")
        self.assertFalse(asset_gate_ready(episode))
        episode = with_asset_approval(episode, "character", "char_a", True)
        episode = with_asset_approval(episode, "scene", "scene_a", True)
        self.assertTrue(asset_gate_ready(episode))
        self.assertEqual(derive_project_stage(episode, contract_persisted=True), "ready_for_video")

    def test_scene_asset_attachment_updates_matching_panels_only(self):
        episode = {
            "scene_bible": [{"scene_id": "scene_a", "reference_images": []}],
            "panels": [
                {"panel_id": "p1", "scene_id": "scene_a", "reference_images": []},
                {"panel_id": "p2", "scene_id": "scene_b", "reference_images": []},
            ],
        }
        updated = attach_scene_reference_images(episode, "scene_a", ["scene.png"])
        self.assertEqual(updated["scene_bible"][0]["reference_images"], ["scene.png"])
        self.assertEqual(updated["panels"][0]["reference_images"], ["scene.png"])
        self.assertEqual(updated["panels"][1]["reference_images"], [])

    def test_rejected_asset_revokes_approval_and_blocks_video_until_regenerated(self):
        episode = {
            "character_bible": [{"character_id": "char_a", "reference_images": ["old.png"]}],
            "scene_bible": [{"scene_id": "scene_a", "reference_images": ["scene.png"]}],
            "approval_state": {
                "creative": {"story": True, "characters": True, "storyboard": True},
                "assets": {"character_ids": ["char_a"], "scene_ids": ["scene_a"]},
            },
        }
        self.assertTrue(asset_gate_ready(episode))
        rejected = with_asset_review_status(
            episode, "character", "char_a", "rejected", reason="wrong gender"
        )
        self.assertFalse(asset_gate_ready(rejected))
        self.assertNotIn("char_a", rejected["approval_state"]["assets"]["character_ids"])
        self.assertEqual(rejected["character_bible"][0]["asset_review_status"], "rejected")
        self.assertEqual(rejected["character_bible"][0]["asset_rejection_reason"], "wrong gender")
        self.assertEqual(rejected["character_bible"][0]["reference_images"], ["old.png"])
        regenerating = with_asset_review_status(rejected, "character", "char_a", "regenerating")
        self.assertFalse(asset_gate_ready(regenerating))

    def test_asset_polling_preserves_only_approvals_for_unchanged_images(self):
        local = {
            "character_bible": [{
                "character_id": "char_a", "asset_hash": "hash-a",
                "reference_images": ["a.png"], "asset_review_status": "approved",
            }],
            "scene_bible": [{
                "scene_id": "scene_a", "asset_hash": "hash-s",
                "reference_images": ["s.png"], "asset_review_status": "approved",
            }],
            "approval_state": {
                "creative": {"story": True, "characters": True, "storyboard": True},
                "assets": {"character_ids": ["char_a"], "scene_ids": ["scene_a"]},
            },
        }
        latest = {
            **local,
            "character_bible": [{
                "character_id": "char_a", "asset_hash": "hash-a",
                "reference_images": ["a.png"], "asset_status": "succeeded",
            }],
            "scene_bible": [{
                "scene_id": "scene_a", "asset_hash": "hash-s",
                "reference_images": ["s.png"], "asset_status": "succeeded",
            }],
        }
        merged = merge_episode_asset_review_state(latest, local)
        self.assertEqual(merged["approval_state"]["assets"]["character_ids"], ["char_a"])
        self.assertEqual(merged["approval_state"]["assets"]["scene_ids"], ["scene_a"])
        changed = {**latest, "character_bible": [{
            "character_id": "char_a", "asset_hash": "hash-new",
            "reference_images": ["new.png"], "asset_status": "succeeded",
        }]}
        refreshed = merge_episode_asset_review_state(changed, local)
        self.assertEqual(refreshed["approval_state"]["assets"]["character_ids"], [])
        self.assertEqual(refreshed["approval_state"]["assets"]["scene_ids"], ["scene_a"])

    def test_backend_approved_assets_restore_only_exact_hash_and_reference_bundle(self):
        latest = {
            "character_bible": [{
                "character_id": "char_a", "asset_hash": "hash-a",
                "reference_images": ["refs/char_a_front.png", "refs/char_a_side.png"],
                "asset_status": "succeeded",
            }],
            "scene_bible": [{
                "scene_id": "scene_a", "asset_hash": "hash-s",
                "reference_images": ["refs/scene_a.png"], "asset_status": "succeeded",
            }],
        }
        snapshot = {
            "pipeline": {"assets_status": "approved"},
            "assets": {"items": [
                {
                    "asset_type": "character", "source_id": "char_a",
                    "content_hash": "hash-a", "approved": True, "status": "succeeded",
                    "reference_images": ["refs/char_a_side.png", "refs/char_a_front.png"],
                },
                {
                    "asset_type": "scene", "source_id": "scene_a",
                    "content_hash": "hash-s", "approved": True, "status": "succeeded",
                    "reference_images": ["refs/scene_a.png"],
                },
            ]},
        }
        restored = merge_episode_asset_review_state(latest, {}, snapshot)
        self.assertEqual(restored["approval_state"]["assets"], {
            "character_ids": ["char_a"], "scene_ids": ["scene_a"],
        })
        self.assertEqual(restored["character_bible"][0]["asset_review_status"], "approved")
        self.assertEqual(restored["scene_bible"][0]["asset_review_status"], "approved")

        wrong_hash = {
            **latest,
            "character_bible": [{**latest["character_bible"][0], "asset_hash": "hash-new"}],
        }
        mismatch = merge_episode_asset_review_state(wrong_hash, {}, snapshot)
        self.assertEqual(mismatch["approval_state"]["assets"]["character_ids"], [])
        self.assertEqual(mismatch["approval_state"]["assets"]["scene_ids"], ["scene_a"])

        snapshot["pipeline"]["assets_status"] = "asset_review"
        not_approved = merge_episode_asset_review_state(latest, {}, snapshot)
        self.assertEqual(not_approved["approval_state"]["assets"], {
            "character_ids": [], "scene_ids": [],
        })

    def test_queued_backend_asset_hides_stale_review_references(self):
        latest = {
            "character_bible": [{
                "character_id": "char_a", "asset_hash": "old-hash",
                "reference_images": ["old.png"], "asset_status": "succeeded",
            }],
            "scene_bible": [],
        }
        snapshot = {
            "pipeline": {"assets_status": "pending"},
            "assets": {"items": [{
                "asset_type": "character", "source_id": "char_a",
                "status": "queued", "content_hash": None,
                "reference_images": [], "approved": False,
            }]},
        }
        merged = merge_episode_asset_review_state(latest, {}, snapshot)
        card = merged["character_bible"][0]
        self.assertEqual(card["reference_images"], [])
        self.assertIsNone(card["asset_hash"])
        self.assertEqual(card["asset_status"], "queued")

    def test_persisted_delivery_manifest_hides_legacy_unbound_export(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            exports = project / "exports"
            exports.mkdir()
            video = exports / "episode_vertical.mp4"
            package = exports / "episode_vertical.delivery.zip"
            video.write_bytes(b"mp4")
            package.write_bytes(b"zip")
            manifest_path = exports / "episode_vertical.manifest.json"
            manifest_path.write_text(
                '{"output_path":"episode_vertical.mp4",'
                '"package_path":"episode_vertical.delivery.zip",'
                '"preset":{"canonical_name":"douyin"},'
                '"subtitles":{"burned_in":true,"strict":true}}',
                encoding="utf-8",
            )
            corrupt = exports / "corrupt.manifest.json"
            corrupt.write_text("not-json", encoding="utf-8")
            missing = exports / "missing.manifest.json"
            missing.write_text('{"output_path":"missing.mp4"}', encoding="utf-8")
            snapshot = {"deliveries": [str(corrupt), str(missing), str(manifest_path)]}

            manifests = persisted_delivery_manifests(snapshot, [project])
            self.assertEqual(manifests, [])

    def test_persisted_delivery_manifest_hides_when_current_selection_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            exports = project / "exports"
            exports.mkdir()
            video = exports / "episode.mp4"
            video.write_bytes(b"mp4")
            artifact = hashlib.sha256(video.read_bytes()).hexdigest()
            selection = {
                "in_seconds": 0.0, "out_seconds": 2.0, "duration_seconds": 2.0,
                "source_artifact_sha256": artifact, "selection_sha256": "selection-a",
            }
            job = {
                "job_id": "p01", "status": "succeeded", "output_path": str(video),
                "probe": {"duration_seconds": 2.0},
                "metadata": {
                    "artifact_sha256": artifact, "edit_selection": selection,
                    "content_qa": {"passed": True, "analysis": {
                        "source_path": str(video), "decoded_visual_sha256": "visual-a",
                    }},
                    "editorial_review": {"status": "approved", "artifact_sha256": artifact,
                                          "decoded_visual_sha256": "visual-a", "edit_selection_sha256": "selection-a"},
                    "release": {"status": "approved", "artifact_sha256": artifact,
                                "decoded_visual_sha256": "visual-a", "edit_selection_sha256": "selection-a"},
                },
            }
            qa = exports / "episode.qa-report.json"
            qa.write_text("{}", encoding="utf-8")
            binding = hashlib.sha256(json.dumps({
                "artifact_hashes": {"p01": artifact}, "selection_hashes": {"p01": "selection-a"},
                "visual_hashes": {"p01": "visual-a"},
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            manifest_path = exports / "episode.manifest.json"
            manifest_path.write_text(json.dumps({
                "output_path": str(video), "release_status": "approved", "qa_report_path": str(qa),
                "qa_report_sha256": hashlib.sha256(qa.read_bytes()).hexdigest(),
                "approved_artifact_hashes": {"p01": artifact},
                "approved_edit_selection_hashes": {"p01": "selection-a"},
                "approved_visual_hashes": {"p01": "visual-a"}, "release_binding_sha256": binding,
            }), encoding="utf-8")
            snapshot = {"jobs": [job], "deliveries": [str(manifest_path)], "release_status": "approved"}
            self.assertEqual(len(persisted_delivery_manifests(snapshot, [project])), 1)
            job["metadata"]["edit_selection"]["selection_sha256"] = "selection-b"
            self.assertEqual(persisted_delivery_manifests(snapshot, [project]), [])

    def test_reviewed_asset_hashes_require_displayed_matching_approved_content(self):
        episode = {
            "character_bible": [{
                "character_id": "char_a", "asset_hash": "hash-a",
                "reference_images": ["a.png"],
            }],
            "scene_bible": [{
                "scene_id": "scene_a", "asset_hash": "hash-s",
                "reference_images": ["s.png"],
            }],
            "approval_state": {
                "creative": {"story": True, "characters": True, "storyboard": True},
                "assets": {"character_ids": ["char_a"], "scene_ids": ["scene_a"]},
            },
        }
        snapshot = {"assets": {"items": [
            {
                "asset_id": "ep:character:char_a", "asset_type": "character",
                "source_id": "char_a", "content_hash": "hash-a",
            },
            {
                "asset_id": "ep:scene:scene_a", "asset_type": "scene",
                "source_id": "scene_a", "content_hash": "hash-s",
            },
        ]}}
        self.assertEqual(reviewed_asset_hashes(episode, snapshot), {
            "ep:character:char_a": "hash-a",
            "ep:scene:scene_a": "hash-s",
        })
        snapshot["assets"]["items"][0]["content_hash"] = "hash-regenerated"
        self.assertEqual(reviewed_asset_hashes(episode, snapshot), {
            "ep:scene:scene_a": "hash-s",
        })


if __name__ == "__main__":
    unittest.main()
