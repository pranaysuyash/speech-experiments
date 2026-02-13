# Run Receipts

Paste raw terminal transcripts here. No interpretation paragraphs.

---

## GLM-TTS bundle receipts

### Local procedure run

```bash
rm -rf models/glm_tts/venv models/glm_tts/repo
cd models/glm_tts && ./install.sh
```

```text
=== GLM-TTS Installation ===
Target commit: c5dc7aecc3b4032032d631b271e767893984f821

[0/5] Checking system dependencies...
  OpenFST headers: found

[1/5] Setting up repo...
Cloning into '/Users/pranay/Projects/speech_experiments/model-lab/models/glm_tts/repo'...
Note: switching to 'c5dc7aecc3b4032032d631b271e767893984f821'.

HEAD is now at c5dc7ae feat: add Tech Report part in README.md
HEAD is now at c5dc7ae feat: add Tech Report part in README.md
Repo ready at c5dc7aecc3b4032032d631b271e767893984f821

[2/5] Setting up virtual environment...
venv: /Users/pranay/Projects/speech_experiments/model-lab/models/glm_tts/venv

[3/5] Installing Python dependencies...
Dependencies installed

[4/5] Applying device-support patches...
  Applying: 0001-frontend.patch
  Applying: 0002-cosyvoice-file-utils.patch
  Applying: 0003-utils-file-utils.patch
  Applying: 0004-yaml-util.patch
4 patches applied

[5/5] Installing pynini (requires OpenFST)...
pynini installed

=== Installation wrap-up ===
Environment frozen to venv.freeze.txt

=== GLM-TTS Installation: end ===
```

```bash
huggingface-cli download zai-org/GLM-TTS --local-dir ckpt
```

```text
/Users/pranay/Projects/speech_experiments/model-lab/models/glm_tts/ckpt
```

```bash
cd ../..
./scripts/verify_glm_tts.sh
```

```text
=== GLM-TTS Observation Log ===
[OBSERVATION] venv_status: present
[OBSERVATION] repo_dir: present
[OBSERVATION] repo_git: present
[OBSERVATION] repo_commit: c5dc7aecc3b4032032d631b271e767893984f821
[OBSERVATION] expected_commit: c5dc7aecc3b4032032d631b271e767893984f821
[OBSERVATION] patches_dir: present
[OBSERVATION] patches_total: 4
[OBSERVATION] patches_reverse_apply_ok: 4
[OBSERVATION] weights_receipt: present
[OBSERVATION] ckpt_dir: present
[OBSERVATION] ckpt_size_kb: 8680596
[OBSERVATION] ckpt_file_count: 53
[OBSERVATION] imports_attempted: attempted
[OBSERVATION] imports_exit_code: 0
[OBSERVATION] torch_version: 2.3.1
[OBSERVATION] bench_attempted: not_attempted
[OBSERVATION] bench_exit_code: unknown
```

### Claim-leak gate run

```bash
./scripts/claim_leak_gate_glm_tts.sh
```

```text
```
