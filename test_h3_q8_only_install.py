"""An install that built the Q8 pack and then deleted the 41 GB bf16 master is still an installed H3."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import hostinfo
import mlx_ltx_panel as P


def _touch(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class Q8OnlyInstall(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="phos-h3-q8only-"))
        m = self.tmp / "models"
        for f in P.H3_COMPACT_FILES:
            _touch(m / "ddalcu-q8" / f)
        _touch(m.joinpath("upstream-meta", *P.H3_TEXT_CONFIG_REL))
        pack = m / P.H3_DIT_Q8_DIRNAME
        _touch(pack / "config.json", b"{}")
        _touch(pack / "quant_config.json", b"{}")
        _touch(pack / ".built_ok")
        self.models = self.tmp / "models"

    def _paths(self):
        with mock.patch.object(P, "H3_MODELS", self.tmp), \
                mock.patch.object(P, "_h3_q8_shards_complete", lambda d: True):
            return P.h3_paths(), P.h3_dit_choice()

    def test_q8_pack_alone_counts_as_installed(self):
        paths, (kind, pack) = self._paths()
        self.assertEqual([m for m in paths["missing"] if "pruned bf16" in m], [])
        self.assertEqual(kind, "q8")
        self.assertEqual(pack, self.models / P.H3_DIT_Q8_DIRNAME)

    def test_no_pack_and_no_master_is_still_missing(self):
        (self.models / P.H3_DIT_Q8_DIRNAME / ".built_ok").unlink()
        paths, _ = self._paths()
        self.assertTrue(any("pruned bf16" in m for m in paths["missing"]))


class RunnerEnvironment(unittest.TestCase):
    def test_linux_decodes_the_vae_tile_by_tile(self):
        with mock.patch.object(hostinfo, "IS_MAC", False):
            self.assertEqual(hostinfo.h3_env_defaults(), {"H3_VAE_BATCH": "1"})

    def test_macos_keeps_the_runner_defaults(self):
        with mock.patch.object(hostinfo, "IS_MAC", True):
            self.assertEqual(hostinfo.h3_env_defaults(), {})


if __name__ == "__main__":
    unittest.main()
