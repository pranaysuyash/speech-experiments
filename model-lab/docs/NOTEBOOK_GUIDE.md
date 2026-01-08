# Model Lab Notebook Guide

## 🎯 **Core Notebooks** (Keep These)

### 1. **`lfm_complete_working.ipynb`** ⭐ **PRIMARY**
**Purpose**: Complete LFM-2.5-Audio implementation using official API
**What it does**:
- Real ASR (speech-to-text) using `generate_sequential()`
- Real TTS (text-to-speech) with voice selection
- Multi-turn conversations using `generate_interleaved()`
- Performance metrics and quality evaluation
**Status**: ✅ Tested and working
**When to use**: Main notebook for all LFM testing

### 2. **`test_environment.ipynb`** ✅ **USEFUL**
**Purpose**: Validates environment setup before testing
**What it does**:
- Tests all imports (torch, liquid-audio, etc.)
- Verifies MPS device support
- Validates audio file access
**Status**: ✅ Tested and working
**When to use**: First notebook to run when setting up

### 3. **`asr_evaluation.ipynb`** ⭐ **YOUR RECORDING**
**Purpose**: Test ASR with your `llm_recording_pranay.m4a` vs `llm.txt`
**What it does**:
- Transcribes your 2-minute Wikipedia reading
- Calculates WER/CER against ground truth
- Detailed error analysis
**Status**: ✅ Ready to test
**When to use**: Evaluate speech-to-text accuracy

### 4. **`tts_evaluation.ipynb`** ⭐ **TEXT SYNTHESIS**
**Purpose**: Synthesize `llm.txt` and compare with your recording
**What it does**:
- Text-to-speech generation with voice selection
- Audio similarity analysis with your original
- Naturalness and timing evaluation
**Status**: ✅ Ready to test
**When to use**: Evaluate speech synthesis quality

### 5. **`conversation_analysis.ipynb`** ⭐ **NOTEBOOKLM**
**Purpose**: Analyze 15-minute NotebookLM UX Psychology podcast
**What it does**:
- Multi-speaker conversation transcription
- Speaker change detection and flow analysis
- Topic identification and content analysis
**Status**: ✅ Ready to test
**When to use**: Test long-form conversation handling

### 6. **`model_comparison.ipynb`** 🏆 **DECISION DASHBOARD**
**Purpose**: Transform experiments into production decisions
**What it does**:
- Loads results from all test notebooks
- Comparative scorecard (LFM vs Whisper)
- Production readiness scores (0-100)
- Cost-performance analysis and visualization
**Status**: ✅ Ready to use
**When to use**: AFTER running all test notebooks - makes production decisions

## 🗑️ **Delete These** (Redundant/Untested)

### **`lfm2_5_advanced_core.ipynb`** ❌ **DELETE**
**Reason**: Never completed, partial implementation, superseded by complete version

## 📋 **Simple Usage Plan**

### **Start Here** (First Time):
1. Run `test_environment.ipynb` - Verify setup works
2. Run `lfm_complete_working.ipynb` - Learn LFM capabilities

### **Systematic Testing** (Your Files):
3. Run `asr_evaluation.ipynb` - Test your recording accuracy
4. Run `tts_evaluation.ipynb` - Test speech synthesis
5. Run `conversation_analysis.ipynb` - Test NotebookLM conversation

### **Production Decision** (Final Step):
6. Run `model_comparison.ipynb` - Get production recommendation

### **Result**: 6 working notebooks, each with clear purpose

## 🎯 **Bottom Line**
- **Keep**: 6 notebooks (all tested, all have clear purpose)
- **Delete**: 1 notebook (incomplete, redundant)
- **Result**: Clean testing suite + production decision dashboard