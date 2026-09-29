#!/usr/bin/env python3
"""4.17.0 render-check findings (real GPU renders on the integration tree).

1. Retake a moment (VA-26) shares Extend's size cap, so on a capped Mac a
   720x1280 clip came back 416x768 while the modal said everything outside
   the stretch was "kept exactly as it was". The modal now says the cap
   before the render. The note's text is a pure function, executed in node.

(The live-preview finding from the same pass — a2v/keyframe pipelines drop
the monitor, so the Now card sat on "starting" — is pinned in
test_lipsync_mode.LivePreviewOnKeyframeAndA2v.)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from extract_panel_js import extract_function  # noqa: E402

NODE = shutil.which("node")
QUEUE_JS = (ROOT / "webapp" / "js" / "queue.js").read_text(encoding="utf-8")
INDEX = (ROOT / "webapp" / "index.html").read_text(encoding="utf-8")


def _node(script: str):
    if NODE is None:
        raise unittest.SkipTest("node not on PATH")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        r = subprocess.run([NODE, path], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise AssertionError(r.stderr[-3000:])
        return json.loads(r.stdout.strip().splitlines()[-1])
    finally:
        Path(path).unlink(missing_ok=True)


class RetakeSaysItsSizeCap(unittest.TestCase):
    def _note(self, tier):
        fn = extract_function("retakeClampNoteText", QUEUE_JS)
        return _node(fn + f"\nconsole.log(JSON.stringify(retakeClampNoteText({json.dumps(tier)})));")

    def test_a_capped_mac_is_told_before_the_render(self):
        t = self._note({"extend_max_dim": 768, "label": "Comfortable"})
        self.assertIn("768px", t)
        self.assertIn("Comfortable", t)
        self.assertIn("smaller", t)
        self.assertIn("original stays", t)

    def test_an_uncapped_mac_gets_no_note(self):
        self.assertEqual(self._note({"extend_max_dim": 0, "label": "Studio"}), "")
        self.assertEqual(self._note(None), "")

    def test_the_modal_shows_it_and_no_longer_promises_an_identical_clip(self):
        self.assertIn('id="retakeClampNote"', INDEX)
        self.assertNotIn("everything outside it is kept exactly as it was", INDEX)
        body = extract_function("openRetakeModal", QUEUE_JS)
        self.assertIn("retakeClampNoteText(", body)
        self.assertIn("retakeClampNote", body)


class PlayerMetaWrapsBetweenItems(unittest.TestCase):
    """A portrait clip's player header wrapped each item inside itself
    ("22 / min / ago", the lip-sync verdict on three lines)."""

    def test_meta_row_wraps_between_items_not_inside_them(self):
        css = (ROOT / "webapp" / "style" / "panel.css").read_text(encoding="utf-8")
        start = css.index(".po-meta {")
        block = css[start:css.index("}", start)]
        self.assertIn("flex-wrap: wrap", block)
        self.assertIn(".po-meta > span:not(.po-dot) { white-space: nowrap; }", css)


class WindowsPartialSaysWhatItHolds(unittest.TestCase):
    """VA-38 render check: the partial published after window 2 of 3 failed
    kept params.frames = 241 (the ask), so the gallery called a 5 s file a
    "10s clip". The sidecar now records the frames the file holds."""

    def test_partial_sidecar_records_the_delivered_frames(self):
        import mlx_ltx_panel as P
        from unittest import mock
        ff = str(P.FFMPEG)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pieces = []
            for i in range(2):
                out = td / f"w{i}.mp4"
                subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i",
                                "testsrc=size=64x64:rate=24:duration=0.5", "-f", "lavfi",
                                "-i", "anullsrc=r=48000:cl=stereo", "-t", "0.5",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                                str(out)], check=True)
                pieces.append(str(out))
            raw_out = td / "clip.mp4"
            p = {"frames": 241, "frame_rate": 24.0, "prompt": "a lighthouse", "mode": "t2v"}
            with mock.patch.object(P, "OUTPUT", td), mock.patch.object(P, "push", lambda *a, **k: None):
                P._publish_windows_partial({"id": "j-test"}, p, {"count": 3}, pieces, 1,
                                           raw_out, {"pix_fmt": "yuv420p", "crf": "18"}, 24.0)
            side = json.loads((td / "clip_windows_partial_2of3.mp4.json").read_text())
            self.assertEqual(side["params"]["frames"], 24)
            self.assertEqual(side["params"]["frames_asked"], 241)
            self.assertEqual(side["windows_partial"]["completed"], 2)
            self.assertEqual(p["frames"], 241)          # the job's own params untouched


class TheGradeSurvivesSave(unittest.TestCase):
    """FILM-14 render check: Match colour applied, Save pressed, and the
    saved edit.json had `adjust: null` on all three clips — sbeCleanClip
    rebuilt `adjust` as brightness only, so no per-clip grade ever reached
    the render, the backup or an NLE export. Executed in node through the
    editor's own harness, then through the server's save/load/validate."""

    def _clean(self, clips):
        import test_codex_4170_editor as ed
        return ed.run("out.clips = %s.map(sbeCleanClip);" % json.dumps(clips))["clips"]

    def test_a_graded_clip_keeps_every_grade_key(self):
        c = {"id": "a", "path": "/x/a.mp4", "start": 0.0, "end": 5.0,
             "film_start": 0.0, "film_end": 5.0, "duration": 5.0,
             "source": "human", "locked": False,
             "adjust": {"brightness": 0.1, "exposure": 0.2, "temp": 0.3,
                        "tint": -0.1, "contrast": 1.1, "saturation": 0.9}}
        out = self._clean([c])[0]
        self.assertEqual(out["adjust"], {"brightness": 0.1, "exposure": 0.2, "temp": 0.3,
                                         "tint": -0.1, "contrast": 1.1, "saturation": 0.9})

    def test_a_neutral_grade_is_absent(self):
        c = {"id": "a", "path": "/x/a.mp4", "start": 0.0, "end": 5.0,
             "film_start": 0.0, "film_end": 5.0, "duration": 5.0,
             "source": "human", "locked": False,
             "adjust": {"exposure": 0, "contrast": 1, "saturation": 1}}
        self.assertNotIn("adjust", self._clean([c])[0])

    def test_the_cleaned_grade_saves_validates_and_loads_back(self):
        import storyboard_editor as sedit
        import test_codex_4170_editor as ed
        c = {"id": "a", "path": "/x/a.mp4", "start": 0.0, "end": 5.0,
             "film_start": 0.0, "film_end": 5.0, "duration": 5.0,
             "source": "human", "locked": False,
             "adjust": {"exposure": 0.11234, "temp": 0.67438, "tint": 0.113021}}
        clean = self._clean([c])
        doc = ed._doc(clean)
        self.assertEqual(sedit.blocking_errors(sedit.validate_edit(doc)), [])
        with tempfile.TemporaryDirectory() as d:
            sedit.save_edit(d, doc)
            back = sedit.load_edit(d)
        self.assertEqual(back["clips"][0].get("adjust"),
                         {"exposure": 0.11234, "temp": 0.67438, "tint": 0.113021})


class TheCompactCapOutranksTheSilentTail(unittest.TestCase):
    """VA-30 render check on a 32 GB (Compact) panel: a 12.9 s voice defaulted
    Duration to 13 s, the silent-tail note ("the last 0.1 s ...") returned
    first, and the cap message with its Split link never appeared."""

    def _warn(self, cap, file_len, sec):
        src = (ROOT / "webapp" / "js" / "characters.js").read_text(encoding="utf-8")
        js = """
var A2V_AREA_KNEE = 0.45e6, A2V_KNEE_FRAMES = 450, BOOT = {a2v_max_frames: %d};
var AUDIO_STUDIO = {audioDuration: %s};
const els = {};
function el(id, v) { return els[id] = {id, value: v, style: {}, innerHTML: '', textContent: ''}; }
const document = { getElementById: id => els[id] || null };
el('audioStudioDuration', '%d'); el('audioStudioDurationVal');
el('audioStudioDurationWarn'); el('audioStudioWidth', '448'); el('audioStudioHeight', '768');
el('audioStudioStart', '0');
""" % (cap, json.dumps(file_len), sec) + "\n".join(
            extract_function(n, src) for n in ("_a2vFramesForSeconds", "audioStudioDurationChanged")) + """
audioStudioDurationChanged('%d');
console.log(JSON.stringify(els.audioStudioDurationWarn.style.display === 'none' ? '' : els.audioStudioDurationWarn.innerHTML));
""" % sec
        return _node(js)

    def test_a_compact_mac_sees_the_cap_and_the_split_link(self):
        w = self._warn(241, 12.8675, 13)
        self.assertIn("Split into 10 s clips", w)

    def test_a_blink_of_tail_is_not_a_warning(self):
        self.assertEqual(self._warn(0, 12.8675, 13), "")

    def test_a_real_silent_tail_still_warns(self):
        self.assertIn("render against silence", self._warn(0, 12.8675, 15))

    def test_the_default_duration_never_adds_a_second_of_silence(self):
        src = (ROOT / "webapp" / "js" / "characters.js").read_text(encoding="utf-8")
        self.assertIn("Math.floor(AUDIO_STUDIO.audioDuration + 0.25)", src)
        self.assertNotIn("Math.round(AUDIO_STUDIO.audioDuration)", src)


if __name__ == "__main__":
    unittest.main()
