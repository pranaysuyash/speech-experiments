# EchoPanel v0.2 - Status & Roadmap

**Provenance**: Ported from `EchoPanel/docs/STATUS_AND_ROADMAP.md` on 2026-02-05. Treat as a reference for feature decomposition; not a Model-Lab roadmap.

## ✅ Completed (v0.2)

### Core Features

- Multi-source audio capture (System/Mic/Both)
- ASR provider abstraction (FasterWhisperProvider)
- 10-minute sliding window analysis
- Entity tracking with counts & recency
- Card deduplication & rolling summary
- Speaker diarization (batch at session end)
- Session storage with auto-save (30s)
- Crash recovery support
- First-run onboarding wizard
- Embedded backend (auto-start/stop)

### User Experience & UI

- **Source-tagged Audio**: Internal JSON protocol for separate System/Mic processing
- **Level Meters**: Dual meters for System and Mic in Side Panel
- **Recovery UI**: "Recover/Discard" options in main menu
- **Diarization Config**: Token input in Onboarding
- **Transcript Persistence**: Real-time append-to-disk logic

### De-risking & Quality

- **Pseudo-diarization**: Live labels "You" vs "System" based on source
- **Self-test**: "Test Audio" button in onboarding
- **Trust**: "Needs review" labels for low-confidence
- **Silence Detection**: Banner after 10s of no audio
- **Backend Error UI**: Onboarding alerts if server fails to start

---

## 🔧 Pending Items

### Distribution (Launch Blockers)

| Item                      | Description                                                                | Effort |
| ------------------------- | -------------------------------------------------------------------------- | ------ |
| **Bundle Python runtime** | Package Python + deps for distribution (PyInstaller)                       | 4h     |
| **Model Preloading UI**   | Download progress bar on first launch (currently just checks availability) | 2h     |

### Feature Backlog (v0.3 Candidates)

| Item                    | Description                           | Effort |
| ----------------------- | ------------------------------------- | ------ |
| Cloud ASR provider      | Implement OpenAI Whisper API provider | 4h     |
| Export to Notion/Slack  | Push summary to integrations          | 8h     |
| Custom entity detection | Allow user-defined entity patterns    | 4h     |
| Multi-language UI       | Localization support                  | 4h     |

### Low Priority

| Item                     | Description                        | Effort |
| ------------------------ | ---------------------------------- | ------ |
| Keyboard shortcuts guide | Show shortcuts in Settings         | 30m    |
| Export to Notion/Slack   | Push summary to integrations       | 8h     |
| Custom entity detection  | Allow user-defined entity patterns | 4h     |
| Multi-language UI        | Localization support               | 4h     |

---

## ❓ Decision Items

### 1. Distribution Strategy

**Options:**

- A) **PyInstaller**: Bundle server as single executable (~200MB)
- B) **Bundled venv**: Include Python + uv pip install in Resources (~500MB)
- C) **Cloud-only**: No local server, require internet

**Recommendation:** Option A (PyInstaller) for smallest bundle size.

### 2. Model Download Strategy

**Options:**

- A) Pre-bundle `base` model, download larger on-demand
- B) Download on first launch with progress UI
- C) Let user choose in Settings, download then

**Recommendation:** Option A with Option C for power users.

### 3. Default Audio Source

**Options:**

- A) Default to "System Audio" (meeting capture)
- B) Default to "Both" (system + mic)
- C) Ask in onboarding (current)

**Recommendation:** Keep C (onboarding choice).

### 4. Diarization Token

**Issue:** Requires `ECHOPANEL_HF_TOKEN` for pyannote model.
**Options:**

- A) User provides own HuggingFace token in Settings
- B) Bundle a shared token (license issue)
- C) Make diarization optional/disabled by default

**Recommendation:** Option C, enable with user-provided token.

---

## 🚀 v0.3 Ideas

- **Real-time speaker labels** (streaming diarization)
- **Meeting templates** (standup, 1:1, retrospective)
- **AI-powered action owner detection**
- **Calendar integration** (link to meeting events)
- **Team sharing** (share summaries with attendees)
- **Custom prompts** for summary generation

---

## 📋 Pre-Launch Checklist

- [ ] Test on clean macOS install
- [ ] Test with no internet (graceful degradation)
- [ ] Test with denied permissions
- [ ] Bundle Python runtime
- [ ] Create DMG installer
- [ ] App icon design
- [ ] App Store metadata
- [ ] Privacy policy for audio capture
