from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, RootModel, field_validator, ConfigDict, Field
from typing import Annotated, Literal, Any


class VocabFile(RootModel[dict[str, int]]):
    @field_validator("root")
    @classmethod
    def _check_ids(cls, v: dict[str, int]) -> dict[str, int]:
        if not v:
            raise ValueError("vocab file is empty")
        if any(i < 0 for i in v.values()):
            raise ValueError("vocab file contains negative token ids")
        return v


class _ExtraIgnore(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SplitPattern(_ExtraIgnore):
    Regex: str | None = None


class SplitPreTokenizer(_ExtraIgnore):
    type: Literal["Split"]
    pattern: SplitPattern
    behavior: str
    invert: bool = False


class OtherPreTokenizer(_ExtraIgnore):
    type: str


class SequencePreTokenizer(_ExtraIgnore):
    type: Literal["Sequence"]
    pretokenizers: list[SplitPreTokenizer | OtherPreTokenizer]


class BPEModel(_ExtraIgnore):
    type: Literal["BPE"]
    merges: list[tuple[str, str]]
    vocab: dict[str, int]

    @field_validator("merges", mode="before")
    @classmethod
    def _tuplise_merges(cls, raw: list[Any]) -> list[tuple[str, str]]:
        res: list[tuple[str, str]] = []
        for el in raw:
            try:
                if isinstance(el, str):
                    parts = el.split(" ", 1)
                    if len(parts) != 2:
                        raise ValueError(
                            f"Invalid element of merges: {el!r}")
                    res.append((parts[0], parts[1]))
                elif isinstance(el, (list, tuple)):
                    if len(el) != 2:
                        raise ValueError(
                            f"Invalid element of merges: {el!r}")
                    res.append((str(el[0]), str(el[1])))
                else:
                    raise ValueError(
                        f"Invalid element of merges: {el!r}")
            except Exception as e:
                raise ValueError(
                    f"Invalid shape of merges in tokenizer.json - {e!r}")
        return res


class AddedTok(_ExtraIgnore):
    id: int
    content: str


class TokenizerFile(_ExtraIgnore):
    version: str
    added_tokens: list[AddedTok]
    pre_tokenizer: Annotated[SequencePreTokenizer |
                             SplitPreTokenizer | None,
                             Field(discriminator="type")] = None
    model: BPEModel


# arguments
# ===== Function Definitions
class ParamType(Enum):
    BOOL = "boolean"
    STR = "string"
    NB = "number"
    OBJ = "object"
    ARR = "array"
    NULL = "null"


class ParamDef(BaseModel):
    type: ParamType
    properties: dict[str, ParamDef] | None = None
    items: ParamDef | None = None


ParamDef.model_rebuild()


class FuncDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParamDef]


class FuncDefs(RootModel[list[FuncDef]]):
    @property
    def all(self) -> list[FuncDef]:
        return self.root


# ===== Test prompts
class Prompts(RootModel[list[dict[Literal["prompt"], str]]]):
    @property
    def all(self) -> list[str]:
        """Return the list of input prompts."""
        return [x["prompt"] for x in self.root]


