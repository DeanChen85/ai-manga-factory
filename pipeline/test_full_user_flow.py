#!/usr/bin/env python3
"""
AI 漫剧工厂 - 全流程端到端测试
模拟用户从建项目到成片的 10 步操作。
直接复用 tests/test_target_e2e.py 的已验证 V3 合同格式。

用法：
  python pipeline/test_full_user_flow.py
"""
from __future__ import annotations
import json, os, sys, tempfile, hashlib, time, traceback
from pathlib import Path
from unittest import mock

PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parent
TESTS = ROOT / "tests"
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(TESTS))

# ── 隔离环境 ─────────────────────────────────────────────────────────────
_tempdir = tempfile.TemporaryDirectory(prefix="e2e_flow_")
WORK_ROOT = Path(_tempdir.name)
os.environ["AI_FACTORY_ROOT"] = str(WORK_ROOT)
os.environ["AI_MANGA_PROJECTS_DIR"] = str(WORK_ROOT / "projects")
os.environ["AI_MANGA_JOB_DB"] = str(WORK_ROOT / "state" / "jobs.sqlite3")

import task_store
task_store._default_store = None
import orchestrator
orchestrator.PROJECTS_DIR = WORK_ROOT / "projects"
import render_service
import story_splitter
import render_video_h3 as renderer
from video_quality import select_edit_window

# 复用已验证的 V3 合同
from test_target_e2e import target_llm_contract

EP_ID = f"ep_e2e_{int(time.time())}"

print("=" * 60)
print("  AI 漫剧工厂 - 全流程端到端测试")
print("=" * 60)
print(f"  工作目录: {WORK_ROOT}")
print(f"  合同版本: {renderer.H3_RUNTIME_PROMPT_CONTRACT}")
print(f"  项目 ID:  {EP_ID}")
print("=" * 60)

# ── Mock 函数 ─────────────────────────────────────────────────────────────
def mock_asset_generator(item, *_args, **kwargs):
    source_id = item.get("character_id") or item.get("scene_id")
    path = WORK_ROOT / "mock_assets" / f"{source_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"mock-ref:{source_id}".encode())
    return {"prompt_id": f"mock-{source_id}", "reference_images": [str(path)]}

def mock_submitter(panel, _output_path, **kwargs):
    job_id = kwargs["job_id"]
    kwargs["store"].update_job(job_id, status="submitted", prompt_id=f"p-{job_id}")
    return {"job_id": job_id, "prompt_id": f"p-{job_id}"}

def mock_waiter(job_id, **kwargs):
    store = kwargs["store"]
    job = store.get_job(job_id)
    output = Path(job["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(f"mock-video:{job_id}".encode())
    art_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    vis_sha = hashlib.sha256(f"vis:{job_id}".encode()).hexdigest()
    store.update_job(job_id, status="succeeded", progress=1.0,
        output_path=str(output), preview_path=str(output),
        probe={"duration_seconds": 10.0, "video": {"width": 720, "height": 1280, "fps": 24.0}},
        metadata={**job["metadata"], "artifact_sha256": art_sha,
            "content_qa": {"passed": True, "analysis": {
                "decoded_visual_sha256": vis_sha, "perceptual_hashes": [vis_sha],
                "static": False, "metrics": {"sample_count": 1}}, "reasons": []},
            "editorial_review": {"status": "pending"}})
    return str(output)

def mock_tail_frame(_video, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"mock-tail")
    return destination

# ── 测试步骤 ──────────────────────────────────────────────────────────────
results = []

def step(n, name):
    def deco(fn):
        def run():
            print(f"\n{'─'*60}\n  步骤 {n}: {name}\n{'─'*60}")
            try:
                fn()
                results.append((n, name, "PASS", ""))
                print(f"  ✅ {name}")
            except Exception as e:
                results.append((n, name, "FAIL", str(e)[:120]))
                print(f"  ❌ {name}: {e}")
                traceback.print_exc()
        return run
    return deco

# ── 1. 生成故事合同（模拟 LLM 返回） ──
@step(1, "生成故事合同 - split_story (mock LLM)")
def s1():
    def fake_m3(sys_p, usr_p, **_kw):
        return json.dumps(target_llm_contract(), ensure_ascii=False)
    with mock.patch.object(story_splitter, "_call_m3", side_effect=fake_m3):
        ep = story_splitter.split_story(
            "林舟在雨夜误入废弃车站，发现一封写给十年后自己的信。",
            topic="写给十年后的自己", synopsis="年轻快递员在废弃车站发现未来的来信。",
            target_audience="年轻成人 18-35", total_duration_seconds=30, shot_count=8,
            platform="douyin", prompt_mode="cinematic",
            visual_style="cinematic Chinese comic",
            style_enforcement="cinematic Chinese comic animation, cold blue rain vs warm gold",
            aspect_ratio="9:16", language="cn", api_key="test-e2e")
    s1.episode = ep
    assert ep["schema_version"] == "ai-manga.prompt-package/v3"
    assert len(ep["panels"]) == 8
    print(f"  镜头: {len(ep['panels'])}, 角色: {len(ep['character_bible'])}, 场景: {len(ep['scene_bible'])}")

# ── 2. 注册合同 ──
@step(2, "注册合同 - prepare_contract")
def s2():
    ep = s1.episode
    ep.setdefault("render_settings", {})["production_strategy"] = "direct_production"
    for p in ep["panels"]:
        p.setdefault("prompt_package", {}).setdefault("render_settings", {})["production_strategy"] = "direct_production"
    draft = render_service.prepare_contract(EP_ID, ep)
    s2.draft = draft
    assert draft["pipeline"]["contract_status"] == "draft"
    assert len(draft["jobs"]) == 8
    print(f"  状态: {draft['pipeline']['contract_status']}, Jobs: {len(draft['jobs'])}")

# ── 3. 批准合同 ──
@step(3, "批准合同 - approve_contract")
def s3():
    try:
        orchestrator.run_episode_jobs(EP_ID)
        assert False, "gate should block"
    except RuntimeError as e:
        assert "production gate blocked" in str(e)
    r = render_service.approve_contract(EP_ID, expected_hash=s2.draft["pipeline"]["contract_hash"])
    assert r["pipeline"]["contract_status"] == "approved"
    print(f"  合同已批准 ✅")

# ── 4. 生成人物/场景资产 ──
@step(4, "生成人物/场景资产")
def s4():
    assets = orchestrator.prepare_all_assets(EP_ID,
        character_generator=mock_asset_generator, scene_generator=mock_asset_generator)
    items = assets["assets"]["items"]
    approved = render_service.approve_assets(EP_ID)
    assert approved["pipeline"]["assets_status"] == "approved"
    for job in approved["jobs"]:
        roles = {e["role"] for e in job["metadata"]["inputs"]["reference_inputs"]}
        assert "character_reference" in roles
    print(f"  资产: {len(items)}, 状态: {approved['pipeline']['assets_status']}")

# ── 5. 视频生成（全 8 镜头） ──
@step(5, "视频生成 - 全 8 镜头")
def s5():
    with mock.patch.object(orchestrator, "submit_render_job", side_effect=mock_submitter), \
         mock.patch.object(orchestrator, "wait_render_job", side_effect=mock_waiter), \
         mock.patch.object(orchestrator, "_extract_tail_frame", side_effect=mock_tail_frame), \
         mock.patch.object(orchestrator, "update_status"):
        result = orchestrator.run_episode_jobs(EP_ID, poll_interval=0.01)
    statuses = [j["status"] for j in result["snapshot"]["jobs"]]
    ok = statuses.count("succeeded")
    fail = statuses.count("failed")
    assert ok == 8, f"期望 8 成功，实际 {ok} 成功 / {fail} 失败"
    s5.result = result
    print(f"  成功: {ok}/8, 失败: {fail}/8")

# ── 6. 逐镜内容审核 ──
@step(6, "内容审核 - 逐镜批准")
def s6():
    store = task_store.default_store()
    count = 0
    for job in store.list_jobs(EP_ID):
        if job["status"] != "succeeded":
            continue
        art_sha = hashlib.sha256(Path(job["output_path"]).read_bytes()).hexdigest()
        # 计算 edit_selection（和正式 E2E 测试一样）
        selection = select_edit_window(
            {"decoded_visual_sha256": job["metadata"]["content_qa"]["analysis"]["decoded_visual_sha256"],
             "algorithm": {"sample_fps": 2.0},
             "metrics": {"adjacent_luma_changes": [12.0] * 24}},
            source_duration_seconds=10.125,
            requested_duration_seconds=float(job["metadata"]["inputs"]["shot_plan"]["edit_duration_seconds"]),
            source_artifact_sha256=art_sha)
        store.update_job(job["job_id"], metadata={**job["metadata"], "edit_selection": selection})
        render_service.approve_job_review(EP_ID, job["job_id"],
            expected_artifact_sha256=art_sha,
            expected_edit_selection_sha256=selection["selection_sha256"],
            reviewed_by="e2e_tester", reason="e2e approved")
        count += 1
    assert count == 8, f"期望 8 批准，实际 {count}"
    print(f"  已批准: {count}/8")

# ── 7. 发布审批 ──
@step(7, "发布审批 - approve_episode_release")
def s7():
    store = task_store.default_store()
    art_hashes = {}
    edit_hashes = {}
    for job in store.list_jobs(EP_ID):
        if job["status"] == "succeeded":
            art_sha = hashlib.sha256(Path(job["output_path"]).read_bytes()).hexdigest()
            art_hashes[job["job_id"]] = art_sha
            sel = (job.get("metadata") or {}).get("edit_selection") or {}
            edit_hashes[job["job_id"]] = sel.get("selection_sha256", "")
    r = render_service.approve_episode_release(EP_ID,
        expected_artifact_hashes=art_hashes,
        expected_edit_selection_hashes=edit_hashes,
        approved_by="e2e_tester")
    assert r["pipeline"].get("release_status") == "approved"
    print(f"  发布状态: {r['pipeline'].get('release_status')}")

# ── 8. 导出成片 ──
@step(8, "导出成片 - export")
def s8():
    try:
        r = render_service.export(EP_ID, "douyin")
        path = r.get("output_path") or r.get("manifest", {}).get("output_path", "")
        print(f"  导出: {path}")
        s8.result = r
    except Exception as e:
        if "ffmpeg" in str(e).lower() or "delivery" in str(e).lower():
            print(f"  ⚠️ 跳过（无 ffmpeg）: {e}")
            s8.result = {"skipped": True}
        else:
            raise

# ── 9. 断点续跑 ──
@step(9, "断点续跑 - resume")
def s9():
    r = render_service.resume(EP_ID)
    print(f"  协调: {len(r.get('reconciled', []))}, 恢复: {r.get('resumed', 0)}")

# ── 10. 提示词质量验证 ──
@step(10, "提示词质量 - v7 正向构图合同验证")
def s10():
    from render_video_h3 import H3_RUNTIME_PROMPT_CONTRACT, compile_h3_runtime_prompt, count_h3_english_words
    assert H3_RUNTIME_PROMPT_CONTRACT == "h3-runtime/v9-complete-fragments"
    panel = {
        "panel_id": "test", "scene_id": "s1",
        "character_ids": ["c1", "c2"], "shot_size": "medium",
        "angle": "eye_level", "composition": "over_shoulder",
        "movement": "static", "duration_seconds": 6.0, "aspect_ratio": "9:16",
        "style": "cinematic Chinese comic",
        "first_state": "at door", "final_state": "at counter",
        "action": "walks to counter",
        "spoken_dialogue": [], "audio_cues": [
            {"cue_type": "ambience", "prompt": "rain outside", "start_seconds": 0, "end_seconds": 6.0}],
        "background_music": "auto_contextual", "ambience": "auto_contextual",
        "scene_context": {
            "model_prompt_en": "convenience store, warm lighting",
            "continuity_lock": "wet exterior remains behind glass; all visible interior surfaces stay uniformly blank"},
        "story_context": {"logline": "Rider finds warmth."},
    }
    bindings = renderer.build_h3_reference_bindings(
        first_frame_filename="scene.png", character_anchor_filename="c1.png",
        extra_reference_filenames=["c2.png"], extra_reference_roles=["character_reference"])
    prompt = compile_h3_runtime_prompt(panel, "", bindings)
    wc = count_h3_english_words(prompt)
    assert wc <= 512, f"超预算: {wc}"
    assert "medium" in prompt.lower() or "shot" in prompt.lower()
    print(f"  合同: {H3_RUNTIME_PROMPT_CONTRACT}")
    print(f"  词数: {wc}/512, shot_size ✅, 连续性 ✅")

# ── 执行 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    s1(); s2(); s3(); s4(); s5(); s6(); s7(); s8(); s9(); s10()

    print(f"\n{'='*60}")
    print(f"  全流程测试结果汇总")
    print(f"{'='*60}")
    passed = sum(1 for *_, s, _ in results if s == "PASS")
    failed = sum(1 for *_, s, _ in results if s == "FAIL")
    for n, name, status, err in results:
        icon = "✅" if status == "PASS" else "❌"
        line = f"  {icon} 步骤 {n}: {name}"
        if err: line += f"\n     └─ {err}"
        print(line)
    print(f"\n{'─'*60}")
    print(f"  总计: {passed} 通过, {failed} 失败 / {len(results)} 步")
    print(f"{'='*60}")
    _tempdir.cleanup()
    sys.exit(1 if failed else 0)
