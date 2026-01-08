# Tweet Drafts - January 8, 2026

**Context**: Posting findings from 1-day experiment with speech models on Apple Silicon  
**Tone**: Casual, informative, sharing learnings

---

## 🎯 Tweet 1: CUDA Issue & MPS Solution

### Version A (Concise):

```
Spent today testing LiquidAI/LFM2.5-Audio-1.5B on Apple Silicon 🍎

Hit a CUDA issue right away - model tried to load CUDA even on M3 chip.

Quick fix:
• Load processor on CPU first
• Move to MPS after
• Handle audio tensor shapes properly

Now running smooth on Apple Silicon!

#MachineLearning #AppleSilicon
```

### Version B (Problem-Solution):

```
Day 1 with LiquidAI/LFM2.5-Audio-1.5B on M3 MacBook 💻

❌ Problem: Model hardcoded CUDA, failed on Apple Silicon
✅ Solution: Load processor CPU-first, then migrate to MPS

Small fix, big difference. Model now runs native on MPS.

Sometimes the best debugging is just understanding the defaults 🔧

#MLOps #AppleSilicon
```

### Version C (Story Format):

```
Tried running LiquidAI/LFM2.5-Audio-1.5B on my M3 Mac today.

Immediate crash: CUDA not available 🤔

Turns out the processor was hardcoded to CUDA by default. Solution was simple - load on CPU, then move to MPS device.

One day, one model, one fix. That's the fun of experimentation! 🚀

#ML #MachineLearning
```

---

## 🎯 Tweet 2: LFM-2.5-Audio Findings

### Version A (Research Findings):

```
Finished testing LiquidAI/LFM2.5-Audio-1.5B for speech recognition 🎤

Key finding: It's NOT optimized for ASR (automatic speech recognition)

On 163s audio:
• 137.8% WER (vs 24% for specialized models)
• Incomplete long-form transcription (10% output on 15min audio)

LFM is multi-modal & conversational - different use case entirely 🎯
```

### Version B (Comparison Focus):

```
Compared 4 speech models today: Whisper, Faster-Whisper, LiquidAI/LFM2.5-Audio-1.5B, and SeamlessM4T

LFM2.5-Audio findings:
• Built for conversation, not pure transcription
• 137.8% WER on technical audio
• Only completes ~10% of long-form content

Great multi-modal model, but use Whisper for ASR 🎙️

Different tools for different jobs!
```

### Version C (Balanced Perspective):

```
Tested LiquidAI/LFM2.5-Audio-1.5B alongside Whisper models today 🔊

Reality check: LFM isn't for ASR transcription.
• High error rates (138% WER)
• Struggles with long audio
• But that's not what it's built for!

LFM2.5 does conversation & multi-modal - totally different mission.

Right tool for right job 🛠️

@maximelabonne
```

### Version D (Community Value):

```
PSA for anyone testing LiquidAI/LFM2.5-Audio-1.5B for speech-to-text 🎤

My findings after 1 day:
❌ Not suitable for pure ASR (137.8% WER vs 24% for Whisper)
❌ Incomplete on long audio (10% output)
✅ Built for conversational AI instead

Save yourself debugging time - use Whisper/Faster-Whisper for transcription!
```

---

## 🎯 Combined Thread Option

### Tweet 1:

```
24-hour experiment with LiquidAI/LFM2.5-Audio-1.5B on Apple Silicon M3 🧪

First challenge: CUDA hardcoding broke on MPS. Fixed by loading processor on CPU first, then migrating to device.

Then benchmarked for speech recognition... 🧵
```

### Tweet 2:

```
Key finding: LFM2.5-Audio isn't optimized for pure ASR transcription.

Results on real audio:
• 137.8% WER (Whisper: 24%)
• Incomplete long-form (10% output on 15min)

But that's OK! LFM is built for conversational AI, not transcription.
```

### Tweet 3:

```
Bottom line:
✅ Got LFM2.5 running on Apple Silicon (MPS fix needed)
✅ Benchmarked vs Whisper, Faster-Whisper, SeamlessM4T
✅ Found sweet spot for each model

LFM = Conversation
Whisper = Transcription

Full results in model-lab repo 📊

@maximelabonne
```

---

## 🎯 Recommendations

**For Tweet 1 (CUDA/MPS)**: Use **Version B** - clear problem/solution format that's helpful for others

**For Tweet 2 (LFM findings)**: Use **Version C** - balanced, tags Maxime, acknowledges different use cases

**Alternative**: Use the **Combined Thread** if you want to tell the full story in one narrative

---

## 📝 Notes

- All tweets acknowledge LFM2.5 is NOT an ASR-first model
- Findings are constructive, not critical
- Actual model name used: `LiquidAI/LFM2.5-Audio-1.5B`
- MPS solution is practical and helpful for community
- WER numbers provided for context but not overemphasized
- Tone stays positive and appreciative of all models tested
