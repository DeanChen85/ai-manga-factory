from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

import stable_promotion


class StablePromotionTests(unittest.TestCase):
    def _fixture(self, root: Path):
        previews = root / "previews"
        previews.mkdir(parents=True)
        proof = previews / "p01.mp4"
        proof.write_bytes(b"approved-proof")
        artifact_sha = hashlib.sha256(proof.read_bytes()).hexdigest()
        decoded_sha = "d" * 64
        selection_sha = "s" * 64
        prompt_sha = "p" * 64
        reference_sha = "r" * 64
        (previews / "p01.artifact.json").write_text(json.dumps({
            "job_id": "ep:0001:p01", "artifact_sha256": artifact_sha,
            "content_qa": {"passed": True, "analysis": {"decoded_visual_sha256": decoded_sha}},
            "edit_selection": {"selection_sha256": selection_sha},
        }), encoding="utf-8")
        graph = previews / "p01.graph.json"
        graph.write_text(json.dumps({"settings": {
            "render_profile": "proof", "delivery_eligible": False,
            "prompt_sha256": prompt_sha, "reference_bundle_sha256": reference_sha,
        }}), encoding="utf-8")
        promotion = {
            "status": "approved", "output_path": str(proof),
            "artifact_sha256": artifact_sha, "decoded_visual_sha256": decoded_sha,
            "edit_selection_sha256": selection_sha, "prompt_sha256": prompt_sha,
            "reference_bundle_sha256": reference_sha, "graph_path": str(graph),
        }
        return proof, {"job_id": "ep:0001:p01", "metadata": {"preview_promotion": promotion}}

    def test_approved_proof_contract_binds_artifact_qa_prompt_and_references(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proof, job = self._fixture(root)
            resolved, artifact, graph, promotion = stable_promotion._approved_proof_contract(job, root)
            self.assertEqual(resolved, proof.resolve())
            self.assertTrue(artifact["content_qa"]["passed"])
            self.assertEqual(graph["settings"]["prompt_sha256"], promotion["prompt_sha256"])

    def test_approved_proof_contract_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proof, job = self._fixture(root)
            proof.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "artifact hash changed"):
                stable_promotion._approved_proof_contract(job, root)

    def test_content_qa_is_bound_to_installed_output_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "videos" / "p01.mp4"
            qa = {
                "passed": True,
                "analysis": {"source_path": "temporary.mp4", "decoded_visual_sha256": "a" * 64},
                "source_analysis": {"source_path": "temporary.mp4", "decoded_visual_sha256": "b" * 64},
            }
            rebound = stable_promotion._bind_content_qa_output_path(qa, output)
            self.assertEqual(rebound["analysis"]["source_path"], str(output))
            self.assertEqual(rebound["source_analysis"]["source_path"], str(output))
            self.assertEqual(qa["analysis"]["source_path"], "temporary.mp4")


if __name__ == "__main__":
    unittest.main()
