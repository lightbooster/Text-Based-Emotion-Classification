import re
import abc
import itertools
from typing import Any, List
import inflect
from spellchecker import SpellChecker
import contractions
from emot.emo_unicode import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
from flashtext import KeywordProcessor

replacement_CASED = {
    "Edit:": " ",
    "Translation:": " ",
    "[deleted]": " ",
    "[NAME]": "person",
    "100%": "full",
    "\r": " ", "\\s": " ", "\\u": " "
}

delete_symbols = '%/^<>#/}{:"~\[\]'

replacement_UNCASED = {
    "lol": "lots of laughs",
    "lmao": "laughing",
    "lmfao": "laughing hard",
    "stfu": "shut the fuck up",
    "lmk": "let me know",
    "wtf": "what the fuck",
    "af": "as fuck",
    "fk": "fuck",
    "smh": "somehow",
    "nvm": "never mind",
    "ofc": "of course",
    "tf": "what the fuck",
    "thx": "thanks",
    "idk": "i do not know",
    "omg": "oh my god",
    "u": "you",
    "n": "and",
    "b": "be",
}

EMOTICONS_EMO = {
    **EMOTICONS_EMO,
    **{ 
        '-_-': 'suspicious',
        '<3': 'love',
        "'\(^^)/'": 'happy',
    }
}


class TextProcessingPipe(abc.ABC):
    def __init__(self):
        pass

    def forward(self, text: str, *args, **kwds) -> str:
        return text
    
    def __call__(self, text: str, *args: Any, **kwds: Any) -> str:
        return self.forward(text, *args, **kwds)


class UncasePipe(TextProcessingPipe):
    def __init__(self):
        super().__init__()
    
    def forward(self, text: str,) -> str:
        return text.lower()


class NumbersPipe(TextProcessingPipe):
    def __init__(self, replace_with=None):
        super().__init__()
        self.regex_compiled = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
        self.engine = inflect.engine()
        self.replace_with = replace_with
    
    def _callback(self, match) -> str:
        group = match.group()
        if '.' in group:
            number = float(group)
        else:
            number = int(group)
        return self.engine.number_to_words(number)

    def forward(self, text: str) -> str:
        if self.replace_with is not None and isinstance(self.replace_with, str):
            return self.regex_compiled.sub(self.replace_with, text)
        else:
            return self.regex_compiled.sub(self._callback, text)


class SpellingPipe(TextProcessingPipe):
    def __init__(self):
        super().__init__()
        self.regex_compiled = re.compile(r"(\w+)")
        self.spell = SpellChecker()
    
    def _callback(self, match) -> str:
        self.spell.correction(match.group())

    def forward(self, text: str) -> str:
        return self.regex_compiled.sub(self._callback, text)


class ContractionsPipe(TextProcessingPipe):
    def __init__(self):
        super().__init__()
    
    def forward(self, text: str) -> str:
        return contractions.fix(text)


class RepeatedSymbolsPipe(TextProcessingPipe):
    def __init__(self, max_repeat=2, regex_symbols=r"a-zA-Z"):
        super().__init__()
        self.regex_compiled = re.compile(r"([" + regex_symbols + r"])\1{" + str(max_repeat) + r",}")

    def _callback(self, match) -> str:
        return match.group()[:1]

    def forward(self, text: str) -> str:
        return self.regex_compiled.sub(self._callback, text)


class SymbolReplacePipe(TextProcessingPipe):
    def __init__(self, regex_symbols, replace_with=""):
        super().__init__()
        self.regex_compiled = re.compile(r"([" + regex_symbols + "])")
        self.replace_with = replace_with

    def forward(self, text: str) -> str:
        return self.regex_compiled.sub(self.replace_with, text)


class WordReplacePipe(TextProcessingPipe):
    def __init__(self, replace_dict: dict):
        super().__init__()
        self.key_processor = KeywordProcessor(case_sensitive=True)
        for key, value in replace_dict.items():
            self.key_processor.add_keyword(key, value)

    def forward(self, text: str) -> str:
        return self.key_processor.replace_keywords(text)


class SubstringReplacePipe(TextProcessingPipe):
    def __init__(self, replace_dict: dict, add_spacing=False):
        super().__init__()
        self.replace_dict = replace_dict
        pattern = '|'.join(sorted(re.escape(key) for key in replace_dict))
        self.regex_compiled = re.compile(pattern)
        self.add_spacing = add_spacing
    
    def _callback(self, match) -> str:
        return (' ' if self.add_spacing else '') + \
              self.replace_dict[(match.group())] + \
               (' ' if self.add_spacing else '')

    def forward(self, text):
        return self.regex_compiled.sub(self._callback, text)


class EmojiToTextPipe(SubstringReplacePipe):
    def __new__(cls):
        def short_emoticon(text):
            or_index = text.find('or')
            cm_index = text.find(',')
            if or_index == -1 and cm_index == -1:
                return text
            elif or_index == -1:
                return text[:cm_index].strip()
            elif cm_index == -1:
                return text[:or_index].strip()
            else:
                return text[:min(or_index, cm_index)].strip()
        
        emoji = { 
            **UNICODE_EMOJI,
            **UNICODE_EMOJI_ALIAS 
        }
        emoji = { key: value.replace(":","").replace("_"," ").strip()
                            for key, value in emoji.items() }
        emoticons = {
            **EMOTICONS_EMO,
        }                
        emoticons = { key: short_emoticon(value) for key, value in emoticons.items() }

        return SubstringReplacePipe({ **emoji,  **emoticons }, add_spacing=True)


class TextProcessingPipeline(TextProcessingPipe):
    def __init__(self, pipes: List[TextProcessingPipe]) -> None:
        self.pipes = pipes

    def forward(self, text: str) -> str:
        text = str(text)
        for pipe in self.pipes:
            text = pipe(text)
        return text

    @classmethod
    def get_standard_pipeline(cls):
        return cls([
                EmojiToTextPipe(),
                SubstringReplacePipe(replacement_CASED, add_spacing=True),
                SymbolReplacePipe(delete_symbols, ' '),
                UncasePipe(),
                RepeatedSymbolsPipe(max_repeat=2, regex_symbols=r"a-zA-Z"),
                WordReplacePipe(replacement_UNCASED),
                SymbolReplacePipe("$€", replace_with=" money "),
                NumbersPipe(replace_with=" "),
                RepeatedSymbolsPipe(max_repeat=1, regex_symbols=r"?!,* "),
                ContractionsPipe(),
            ])