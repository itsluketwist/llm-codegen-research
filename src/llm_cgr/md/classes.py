"""Classes for handling markdown responses from LLMs."""

from dataclasses import dataclass

from llm_cgr.md.regex import CODE_BLOCK_REGEX


@dataclass
class CodeBlock:
    """
    A class to represent a block of code with it's language.
    """

    language: str | None
    text: str

    def __init__(self, language: str | None, text: str):
        self.language = language.strip().lower() if language else None
        self.text = text.strip()

    def __repr__(self):
        _language = f"language={self.language or 'unspecified'}"
        _lines = f"lines={len(self.text.splitlines())}"
        return f"{self.__class__.__name__}({_language}, {_lines})"

    def __str__(self):
        return self.text

    @property
    def markdown(self):
        return f"```{self.language or ''}\n{self.text}\n```"


@dataclass
class Markdown:
    """
    A class to hold a markdown response from an LLM as a series of text and code blocks.
    """

    text: str
    code_blocks: list[CodeBlock]
    languages: list[str]

    def __init__(self, text: str):
        self.text = text
        self.code_blocks = self.extract_code_blocks(text)
        self.languages = sorted(
            list({bl.language for bl in self.code_blocks if bl.language})
        )

    def __repr__(self):
        _lines = f"lines={len(self.text.splitlines())}"
        _code_blocks = f"code_blocks={len(self.code_blocks)}"
        _languages = f"languages={','.join(self.languages)}"
        return f"{self.__class__.__name__}({_lines}, {_code_blocks}, {_languages})"

    def __str__(self):
        return self.text

    @staticmethod
    def extract_code_blocks(response: str) -> list[CodeBlock]:
        """
        Extract the code blocks from the markdown formatted text.
        """
        matches = CODE_BLOCK_REGEX.findall(response)
        blocks = []
        for match in matches:
            language, text = match
            blocks.append(CodeBlock(language=language, text=text))
        return blocks
