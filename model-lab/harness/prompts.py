"""
Clean prompt management for model testing.
Handles prompt templates, validation, and version control.
No model logic, just prompt engineering utilities.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PromptTemplate:
    """Clean prompt template with metadata."""

    name: str
    template: str
    version: str = "1.0.0"
    description: str = ""
    variables: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Extract variables from template
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> list[str]:
        """Extract variable names from template string."""
        # Find ${variable} patterns
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, self.template)

        # Find $variable patterns (simple template)
        simple_pattern = r"\$([a-zA-Z_][a-zA-Z0-9_]*)"
        simple_matches = re.findall(simple_pattern, self.template)

        # Combine and deduplicate
        all_vars = list(set(matches + simple_matches))
        return sorted(all_vars)

    def render(self, **kwargs) -> str:
        """Render template with provided variables."""
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing template variables: {missing_vars}")

        # Try string.Template first (for $variable syntax)
        try:
            from string import Template

            template = Template(self.template)
            return template.substitute(**kwargs)
        except KeyError:
            # Fall back to f-string style (for {variable} syntax)
            try:
                return self.template.format(**kwargs)
            except KeyError as e:
                raise ValueError(f"Template variable error: {e}") from e

    def validate(self) -> bool:
        """Validate template syntax."""
        try:
            # Test with dummy variables
            dummy_vars = {var: f"test_{var}" for var in self.variables}
            self.render(**dummy_vars)
            return True
        except Exception:
            return False


class PromptLibrary:
    """Clean prompt library with version control and validation."""

    def __init__(self, library_path: Path | None = None):
        self.library_path = Path(library_path) if library_path else None
        self.templates: dict[str, PromptTemplate] = {}

        if self.library_path and self.library_path.exists():
            self.load_library()

    def add_template(self, template: PromptTemplate):
        """Add template to library."""
        key = f"{template.name}_v{template.version}"
        self.templates[key] = template

    def get_template(self, name: str, version: str | None = None) -> PromptTemplate | None:
        """Get template by name and version."""
        if version:
            key = f"{name}_v{version}"
            return self.templates.get(key)
        else:
            # Find latest version
            candidates = [k for k in self.templates.keys() if k.startswith(f"{name}_v")]
            if candidates:
                latest = sorted(candidates)[-1]
                return self.templates[latest]
        return None

    def list_templates(self) -> list[str]:
        """List all available template names."""
        names = set()
        for key in self.templates.keys():
            name = key.split("_v")[0]
            names.add(name)
        return sorted(names)

    def save_library(self, filepath: Path | None = None):
        """Save library to file."""
        filepath = filepath or self.library_path
        if not filepath:
            raise ValueError("No library path specified")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            name: {
                "name": template.name,
                "template": template.template,
                "version": template.version,
                "description": template.description,
                "variables": template.variables,
                "metadata": template.metadata,
            }
            for name, template in self.templates.items()
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_library(self, filepath: Path | None = None):
        """Load library from file."""
        filepath = filepath or self.library_path
        if not filepath or not filepath.exists():
            raise FileNotFoundError(f"Library file not found: {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        self.templates.clear()
        for name, template_data in data.items():
            template = PromptTemplate(**template_data)
            self.templates[name] = template

    def validate_all(self) -> dict[str, bool]:
        """Validate all templates in library."""
        results = {}
        for name, template in self.templates.items():
            results[name] = template.validate()
        return results


# Pre-defined prompt templates for common audio tasks
AUDIO_PROMPTS = {
    "transcription": PromptTemplate(
        name="audio_transcription",
        template="Transcribe the following audio content:\n\nAudio: ${audio_description}\n\nProvide only the transcription text.",
        description="Basic audio transcription prompt",
        version="1.0.0",
    ),
    "translation": PromptTemplate(
        name="audio_translation",
        template="Translate the following audio from ${source_language} to ${target_language}:\n\nAudio: ${audio_description}\n\nProvide only the translation.",
        description="Audio translation prompt",
        version="1.0.0",
    ),
    "summarization": PromptTemplate(
        name="audio_summarization",
        template="Summarize the following audio content in ${max_words} words:\n\nAudio: ${audio_description}\n\nSummary:",
        description="Audio summarization prompt",
        version="1.0.0",
    ),
    "analysis": PromptTemplate(
        name="audio_analysis",
        template="Analyze the following audio for ${analysis_type}:\n\nAudio: ${audio_description}\n\nAnalysis:",
        description="Audio analysis prompt",
        version="1.0.0",
    ),
}


def create_prompt_library(library_path: Path | None = None) -> PromptLibrary:
    """Create a prompt library with default audio prompts."""
    library = PromptLibrary(library_path)

    for prompt in AUDIO_PROMPTS.values():
        library.add_template(prompt)

    return library


def render_prompt(template_name: str, library: PromptLibrary, **kwargs) -> str:
    """Quick function to render a prompt from library."""
    template = library.get_template(template_name)
    if not template:
        raise ValueError(f"Template '{template_name}' not found")

    return template.render(**kwargs)
