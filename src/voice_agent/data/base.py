"""Abstract contracts for recognition data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Occupation:
    """A German occupation mapping candidate."""

    esco_id: str
    label_de: str
    label_en: str
    kldb_code: str | None
    confidence: float


@dataclass(frozen=True)
class Authority:
    """A local recognition authority or advice centre."""

    name: str
    city: str
    email: str | None
    phone: str
    website: str


@dataclass(frozen=True)
class Document:
    """A document usually needed for a recognition application."""

    name_de: str
    name_en: str
    notes: str


@dataclass(frozen=True)
class LabourMarketStatus:
    """A rough labour-market signal for one profession and region."""

    profession: str
    region: str
    shortage: bool
    shortage_level: str
    open_positions: int
    applicants_per_opening: float
    note: str


class DataProvider(ABC):
    """Abstract interface for occupation, authority, documents, and labour lookups."""

    @abstractmethod
    async def find_german_occupation(
        self,
        description: str,
        source_lang: str,
    ) -> list[Occupation]:
        """Return German occupation candidates for a caller's description."""

    @abstractmethod
    async def get_recognition_authority(
        self,
        profession: str,
        city: str = "Nürnberg",
    ) -> Authority | None:
        """Return the best local authority for a profession and city."""

    @abstractmethod
    async def get_required_documents(self, profession: str) -> list[Document]:
        """Return the usual required documents for a profession."""

    @abstractmethod
    async def get_labour_market_status(
        self,
        profession: str,
        region: str = "Bayern",
    ) -> LabourMarketStatus:
        """Return a rough labour-market status for a profession and region."""


__all__ = [
    "Authority",
    "DataProvider",
    "Document",
    "LabourMarketStatus",
    "Occupation",
]
