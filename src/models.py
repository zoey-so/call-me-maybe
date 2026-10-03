from __future__ import annotations

from pydantic import BaseModel, RootModel, \
            field_validator, ConfigDict, Field
from typing import Annotated, Literal, Any


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

class FlatParamDef(BaseModel):
    type: Literal["boolean", "string", "number", "null", "integer"]


class ObjParamDef(BaseModel):
    type: Literal["object"]
    properties: dict[str, ParamDef]


class ArrParamDef(BaseModel):
    type: Literal["array"]
    items: ParamDef


ParamDef = FlatParamDef | ObjParamDef | ArrParamDef


class RootParamDef(BaseModel):
    root: ParamDef = Field(..., discriminator="type")

ObjParamDef.model_rebuild()
ArrParamDef.model_rebuild()


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
