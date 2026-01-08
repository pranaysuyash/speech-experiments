#!/usr/bin/env python3
"""
Add conversation/multi-speaker audio tests to the test suite.
This extends the canonical test framework for conversation analysis.
"""

import json
from pathlib import Path
import soundfile as sf


def add_conversation_tests(test_manifest_path: Path):
    """Add conversation test files to the existing test manifest."""
    
    # Load existing manifest
    with open(test_manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Define conversation test files
    conversation_tests = {
        'conversation_2ppl_10s.wav': {
            'path': 'data/audio/conversation_2ppl_10s.wav',
            'description': '10s two-person conversation segment (1:30-1:40 from UX Psychology podcast)',
            'duration': 10.0,
            'type': 'conversation',
            'speakers': 2,
            'source': 'UX_Psychology_From_Miller_s_Law_to_AI.m4a',
            'time_range': '1:30-1:40',
            'language': 'en',
            'test_focus': ['speaker_diarization', 'conversation_flow', 'multi_speaker_transcription']
        },
        'conversation_2ppl_30s.wav': {
            'path': 'data/audio/conversation_2ppl_30s.wav',
            'description': '30s two-person conversation segment (1:30-2:00 from UX Psychology podcast)',
            'duration': 30.0,
            'type': 'conversation',
            'speakers': 2,
            'source': 'UX_Psychology_From_Miller_s_Law_to_AI.m4a',
            'time_range': '1:30-2:00',
            'language': 'en',
            'test_focus': ['speaker_diarization', 'conversation_topics', 'speaker_change_detection']
        }
    }
    
    # Add to manifest
    manifest['files'].update(conversation_tests)
    manifest['conversation_tests_added'] = True
    manifest['total_conversation_files'] = len(conversation_tests)
    
    # Save updated manifest
    with open(test_manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Added {len(conversation_tests)} conversation test files to manifest")
    return conversation_tests


def create_conversation_test_metadata():
    """Create metadata files for conversation tests."""
    
    metadata = {
        'conversation_2ppl_10s': {
            'description': '10-second conversation segment for multi-speaker testing',
            'expected_characteristics': [
                'Two distinct speakers',
                'Natural conversation flow',
                'Speaker transitions',
                'UX/AI topic discussion'
            ],
            'test_objectives': [
                'Speaker diarization accuracy',
                'Multi-speaker transcription',
                'Speaker change detection',
                'Conversation structure analysis'
            ],
            'challenges': [
                'Overlapping speech',
                'Speaker similarity',
                'Background noise',
                'Natural speech patterns'
            ]
        },
        'conversation_2ppl_30s': {
            'description': '30-second conversation segment for extended multi-speaker analysis',
            'expected_characteristics': [
                'Multiple speaker exchanges',
                'Topic development',
                'Natural dialogue rhythm',
                'Professional discussion tone'
            ],
            'test_objectives': [
                'Extended conversation tracking',
                'Topic identification',
                'Speaker consistency',
                'Long-form transcription accuracy'
            ],
            'challenges': [
                'Maintaining speaker identity',
                'Context preservation',
                'Topic coherence',
                'Natural language understanding'
            ]
        }
    }
    
    # Save metadata
    metadata_file = Path('data/text/conversation_test_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created conversation test metadata: {metadata_file}")
    return metadata


def validate_conversation_files():
    """Validate that conversation files exist and are properly formatted."""
    
    conversation_files = [
        'data/audio/conversation_2ppl_10s.wav',
        'data/audio/conversation_2ppl_30s.wav',
        'data/text/conversation_2ppl_30s.txt'
    ]
    
    print("Validating conversation test files:")
    
    for file_path in conversation_files:
        path = Path(file_path)
        if path.exists():
            if file_path.endswith('.wav'):
                # Validate audio file
                try:
                    import soundfile as sf
                    audio, sr = sf.read(path)
                    duration = len(audio) / sr
                    print(f"  ✓ {path.name}: {duration:.1f}s, {sr}Hz")
                except Exception as e:
                    print(f"  ⚠️  {path.name}: Audio validation failed - {e}")
            else:
                print(f"  ✓ {path.name}: File exists")
        else:
            print(f"  ✗ {path.name}: Missing")
    
    return all(Path(f).exists() for f in conversation_files)


def main():
    """Main function to add conversation tests to the suite."""
    
    print("🎙️ Adding Conversation Tests to Model Testing Suite")
    print("=" * 60)
    
    # Paths
    test_manifest = Path('data/audio/test_manifest.json')
    
    # Step 1: Validate files exist
    print("\n1. Validating conversation files...")
    files_valid = validate_conversation_files()
    
    if not files_valid:
        print("❌ Some conversation files are missing")
        print("Ensure you have:")
        print("  - data/audio/conversation_2ppl_10s.wav")
        print("  - data/audio/conversation_2ppl_30s.wav") 
        print("  - data/text/conversation_2ppl_30s.txt")
        return 1
    
    # Step 2: Add to test manifest
    print("\n2. Adding to test manifest...")
    conversation_tests = add_conversation_tests(test_manifest)
    
    # Step 3: Create metadata
    print("\n3. Creating conversation test metadata...")
    metadata = create_conversation_test_metadata()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Conversation tests added successfully!")
    print(f"\nNew test files:")
    for filename, info in conversation_tests.items():
        print(f"  - {filename}: {info['description']}")
    
    print(f"\nTest capabilities added:")
    print(f"  - Multi-speaker transcription")
    print(f"  - Speaker diarization") 
    print(f"  - Conversation flow analysis")
    print(f"  - Speaker change detection")
    
    print(f"\nNext steps:")
    print(f"  1. Update test notebooks to include conversation tests")
    print(f"  2. Add conversation-specific evaluation metrics")
    print(f"  3. Test models on multi-speaker audio")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())