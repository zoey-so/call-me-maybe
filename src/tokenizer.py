"""Complete tokenizer for BPE models."""
import json
import regex
from collections import defaultdict

from .models import TokenizerFile, SplitPreTokenizer, SequencePreTokenizer


def do_bytes_to_unicode() -> dict[int, str]:
    """Convertion to OpenAI's 'pretty' unicode chars"""
    _bytes = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    _chars = [chr(c) for c in _bytes]
    next_free = 256
    for b in range(256):
        if b not in _bytes:
            _bytes.append(b)
            _chars.append(chr(next_free))
            next_free += 1
    return dict(zip(_bytes, _chars))


class Tokenizer:
    """All logic for encoding and decoding with additional
    fast mapping dictionaries:
    uni_txt_to_id - just model vocab dict
    id_to_uni_txt - reversed vocab with ids as keys
    byte_to_unicode - mapping of char conversion
    unicode_to_byte - reversed conversion
    id_to_byte_txt - mapping from id to standart text
    first_char_ids - all normal chars have a list of all ids that
                    start with them
    merges - dict[tuple[str, str], int] merge pairs with ranking
    regex - extracted regex for pretokanization split
    Parameeters
    ---------
    tf_path: str
        path to tokenizer.json file to be used.
    """
    def __init__(self, tf_path: str) -> None:
        with open(
                tf_path,
                'r', encoding="utf-8") as f:
            content = json.load(f)
        tf = TokenizerFile.model_validate(content)
        uni_txt_to_id: dict[str, int] = tf.model.vocab
        if tf.added_tokens:
            for at in tf.added_tokens:
                uni_txt_to_id.setdefault(at.content, at.id)
        id_to_uni_txt: dict[int, str] = {
            v: k for k, v in uni_txt_to_id.items()}
        byte_to_unicode = do_bytes_to_unicode()
        unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}
        id_to_byte_txt: dict[int, str] = {}
        for word, id in uni_txt_to_id.items():
            try:
                raw_bytes = bytes(unicode_to_byte[ch] for ch in word)
                id_to_byte_txt[id] = raw_bytes.decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                id_to_byte_txt[id] = word

        first_char_ids: dict[str, list[int]] = defaultdict(list)
        for id, word in id_to_byte_txt.items():
            if word:
                first_char_ids[word[0]].append(id)

        self.merges = {el: i for i, el in enumerate(tf.model.merges)}
        self.uni_txt_to_id: dict[str, int] = uni_txt_to_id
        self.id_to_uni_txt: dict[int, str] = id_to_uni_txt
        self.id_to_byte_txt: dict[int, str] = id_to_byte_txt
        self.unicode_to_byte: dict[str, int] = unicode_to_byte
        self.byte_to_unicode: dict[int, str] = byte_to_unicode
        self.first_char_ids: dict[str, list[int]] = first_char_ids
        self.regex = regex.compile(self._extract_regex(tf))
        with open("id_to_byte_txt.json", 'w') as f:
            json.dump(self.id_to_byte_txt, f)

    def _extract_regex(self, tf: TokenizerFile) -> str:
        pt = tf.pre_tokenizer
        if isinstance(pt, SplitPreTokenizer) and pt.pattern.Regex is not None:
            return pt.pattern.Regex
        if isinstance(pt, SequencePreTokenizer):
            for member in pt.pretokenizers:
                if isinstance(member, SplitPreTokenizer) and\
                        member.pattern.Regex is not None:
                    return member.pattern.Regex
        return "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|" +\
            "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|" +\
            "\\s+(?!\\S)|\\s+"

    def _pretokenize(self, text: str) -> list[str]:
        """Splits text into chunks and maps character to unicode."""
        words: list[str] = self.regex.findall(text)
        words = [
            "".join(self.byte_to_unicode[ord(c)] for c in word)
            for word in words]
        return words

    def _tokenize_part(self, text: str) -> list[int]:
        """Encodes given text without special tokens to tokens."""
        words = self._pretokenize(text)
        tokens: list[int] = []
        for word in words:
            parts = list(word)
            while len(parts) > 1:
                pairs = list(zip(parts, parts[1:]))
                min_pair = min(
                    pairs,
                    key=lambda pair:
                    self.merges.get(pair, float('inf')))
                if min_pair not in self.merges:
                    break
                first, second = min_pair
                new_parts = []
                i = 0
                while i < len(parts):
                    if i < len(parts) - 1 and\
                            (parts[i], parts[i + 1]) == min_pair:
                        new_parts.append(first + second)
                        i += 2
                    else:
                        new_parts.append(parts[i])
                        i += 1
                parts = new_parts
            for part in parts:
                tokens.append(self.uni_txt_to_id[part])
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decodes given ids to standard text.
        Parameters
        ----------
        ids: list[int] - list of ids
        Returns
        ---------
        str - decoded utf-8 text.
        """
        uni_chars = "".join(self.id_to_uni_txt[i] for i in ids)
        raw_bytes = bytes(self.unicode_to_byte[ch] for ch in uni_chars)
        return raw_bytes.decode("utf-8")

    def encode(self, text: str) -> list[int]:
        """"Main input to encode text including special tokens.
        Parameters
        ----------
        text: str - text to encode
        Returns
        ---------
        list[int] - list of token ids.
        """
        tokens: list[int] = []
        specials = regex.findall(r"(?s)<\|.*?\|>", text)[::-1]
        rest = regex.split(r"(?s)<\|.*?\|>", text)[::-1]
        while specials or rest:
            if rest:
                tokens += self._tokenize_part(rest.pop())
            if specials:
                s = specials.pop()
                tok_id = self.uni_txt_to_id.get(s)
                if not tok_id:
                    tokens += self._tokenize_part(s)
                    continue
                tokens.append(tok_id)
        return tokens

    def ids_starting_with_chars(self, chars: str) -> list[int]:
        """Returns all ids starting with any of the given chars."""
        ids = []
        for c in chars:
            ids.extend(self.first_char_ids.get(c, []))
        return ids
