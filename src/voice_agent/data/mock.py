"""Mock recognition data for the Nürnberg voice-agent demo."""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from dataclasses import asdict, dataclass

from .base import Authority, DataProvider, Document, Occupation


@dataclass(frozen=True)
class _ProfessionRecord:
    """Immutable bundle for one mocked profession."""

    key: str
    occupation: Occupation
    aliases: tuple[str, ...]
    authority: Authority
    documents: tuple[Document, ...]


_COMMON_DOCUMENTS: tuple[Document, ...] = (
    Document(
        name_de="Identitätsnachweis",
        name_en="Proof of identity",
        notes="Reisepass oder Personalausweis als gut lesbare Kopie.",
    ),
    Document(
        name_de="Tabellarischer Lebenslauf",
        name_en="Curriculum vitae",
        notes="Auf Deutsch, mit Ausbildung, Berufserfahrung und aktuellen Kontaktdaten.",
    ),
    Document(
        name_de="Abschlusszeugnis und Diplom",
        name_en="Diploma and final certificate",
        notes="Mit vereidigter deutscher Übersetzung, falls das Original nicht deutsch ist.",
    ),
    Document(
        name_de="Fächer- und Stundenübersicht",
        name_en="Transcript of subjects and hours",
        notes="Hilft der Stelle, Inhalte und Umfang der Ausbildung zu vergleichen.",
    ),
    Document(
        name_de="Arbeitszeugnisse",
        name_en="Employment references",
        notes="Nachweise über einschlägige Berufserfahrung, möglichst mit Aufgabenbeschreibung.",
    ),
)


_RECORDS: tuple[_ProfessionRecord, ...] = (
    _ProfessionRecord(
        key="nurse",
        occupation=Occupation(
            esco_id="http://data.europa.eu/esco/occupation/healthcare-nurse",
            label_de="Krankenschwester",
            label_en="Registered nurse",
            kldb_code="81302",
            confidence=0.96,
        ),
        aliases=(
            "nurse",
            "registered nurse",
            "staff nurse",
            "pflegefachfrau",
            "pflegefachmann",
            "pflegefachkraft",
            "krankenschwester",
            "krankenpfleger",
        ),
        authority=Authority(
            name="Beratungsstelle für Anerkennung ausländischer Berufsqualifikationen Nürnberg (Marienstraße 23)",
            city="Nürnberg",
            email="anerkennung-pflege@nuernberg.de",
            phone="+49 911 231-10420",
            website="https://www.nuernberg.de/internet/anerkennung/",
        ),
        documents=(
            *_COMMON_DOCUMENTS,
            Document(
                name_de="Berufszulassung oder Registrierung",
                name_en="Professional licence or registration",
                notes="Nachweis, dass Sie im Herkunftsland als Pflegekraft arbeiten dürfen.",
            ),
            Document(
                name_de="Deutsch-Sprachnachweis B2",
                name_en="German language certificate B2",
                notes="Für die Berufserlaubnis wird in der Regel mindestens B2 verlangt.",
            ),
            Document(
                name_de="Ärztliches Attest und Führungszeugnis",
                name_en="Medical certificate and certificate of good conduct",
                notes="Meist erst kurz vor der endgültigen Erlaubnis erforderlich.",
            ),
        ),
    ),
    _ProfessionRecord(
        key="engineer",
        occupation=Occupation(
            esco_id="http://data.europa.eu/esco/occupation/professional-engineer",
            label_de="Ingenieur",
            label_en="Engineer",
            kldb_code="25104",
            confidence=0.93,
        ),
        aliases=(
            "engineer",
            "engineering",
            "civil engineer",
            "mechanical engineer",
            "electrical engineer",
            "ingenieur",
            "bauingenieur",
            "maschinenbauingenieur",
            "elektroingenieur",
        ),
        authority=Authority(
            name="Bayerische Ingenieurekammer-Bau - Anerkennungsberatung Nürnberg (Äußere Sulzbacher Straße 159)",
            city="Nürnberg",
            email="anerkennung@bayika.de",
            phone="+49 911 941-4800",
            website="https://www.bayika.de/de/anerkennung/",
        ),
        documents=(
            *_COMMON_DOCUMENTS,
            Document(
                name_de="Modulhandbuch oder Studienplan",
                name_en="Module handbook or study plan",
                notes="Besonders wichtig, wenn die Studieninhalte nicht klar aus dem Zeugnis hervorgehen.",
            ),
            Document(
                name_de="Nachweis der Berufspraxis",
                name_en="Proof of professional practice",
                notes="Projektlisten, Arbeitsverträge oder Referenzen können die Bewertung stützen.",
            ),
            Document(
                name_de="Nachweis über die Führung der Berufsbezeichnung",
                name_en="Proof of right to use the professional title",
                notes="Falls Sie im Herkunftsland bereits eine geschützte Ingenieurbezeichnung geführt haben.",
            ),
        ),
    ),
    _ProfessionRecord(
        key="teacher",
        occupation=Occupation(
            esco_id="http://data.europa.eu/esco/occupation/school-teacher",
            label_de="Lehrer",
            label_en="Teacher",
            kldb_code="84114",
            confidence=0.92,
        ),
        aliases=(
            "teacher",
            "school teacher",
            "secondary teacher",
            "primary teacher",
            "lehrer",
            "lehrerin",
            "grundschullehrer",
            "gymnasiallehrer",
        ),
        authority=Authority(
            name="Regierung von Mittelfranken - Anerkennung Lehramt (Keßlerplatz 12)",
            city="Nürnberg",
            email="lehramt-anerkennung@reg-mfr.bayern.de",
            phone="+49 911 235-1850",
            website="https://www.regierung.mittelfranken.bayern.de/",
        ),
        documents=(
            *_COMMON_DOCUMENTS,
            Document(
                name_de="Nachweis der Lehramtsbefähigung",
                name_en="Proof of teaching qualification",
                notes="Urkunden oder Bescheinigungen über die volle Lehrbefähigung im Herkunftsland.",
            ),
            Document(
                name_de="Praxisnachweise aus Schulen",
                name_en="Proof of school teaching practice",
                notes="Stundenumfang, Schulart, Fächer und Klassenstufen sollten erkennbar sein.",
            ),
            Document(
                name_de="Deutsch-Sprachnachweis C1",
                name_en="German language certificate C1",
                notes="Für den Schuldienst wird meist ein sehr gutes Deutschniveau erwartet.",
            ),
        ),
    ),
    _ProfessionRecord(
        key="doctor",
        occupation=Occupation(
            esco_id="http://data.europa.eu/esco/occupation/medical-doctor",
            label_de="Arzt",
            label_en="Medical doctor",
            kldb_code="81404",
            confidence=0.95,
        ),
        aliases=(
            "doctor",
            "medical doctor",
            "physician",
            "general practitioner",
            "arzt",
            "ärztin",
            "mediziner",
            "approbation arzt",
        ),
        authority=Authority(
            name="Approbationsstelle Mittelfranken für Ärztinnen und Ärzte (Sulzbacher Straße 11)",
            city="Nürnberg",
            email="approbation@reg-mfr.bayern.de",
            phone="+49 911 235-3500",
            website="https://www.regierung.mittelfranken.bayern.de/aufgaben/40068/40080/leistung/leistung_12515/",
        ),
        documents=(
            *_COMMON_DOCUMENTS,
            Document(
                name_de="Ärztliche Berufszulassung aus dem Herkunftsland",
                name_en="Medical licence from country of origin",
                notes="Approbation, Registrierung oder vergleichbarer Nachweis der Berufsausübung.",
            ),
            Document(
                name_de="Certificate of Good Standing",
                name_en="Certificate of good standing",
                notes="Aktuelle Bescheinigung der zuständigen Ärztekammer oder Behörde.",
            ),
            Document(
                name_de="Fachsprachprüfung oder Deutsch C1 Medizin",
                name_en="Medical German language proof",
                notes="Für Ärztinnen und Ärzte wird zusätzlich zur Alltagssprache Fachsprache geprüft.",
            ),
            Document(
                name_de="Ärztliches Attest und Führungszeugnis",
                name_en="Medical certificate and certificate of good conduct",
                notes="Die Unterlagen dürfen bei Antragstellung oft nur wenige Monate alt sein.",
            ),
        ),
    ),
    _ProfessionRecord(
        key="it-admin",
        occupation=Occupation(
            esco_id="http://data.europa.eu/esco/occupation/ict-system-administrator",
            label_de="Fachinformatiker Systemintegration",
            label_en="IT systems administrator",
            kldb_code="43102",
            confidence=0.94,
        ),
        aliases=(
            "it admin",
            "it administrator",
            "systems administrator",
            "system administrator",
            "network administrator",
            "ict system administrator",
            "fachinformatiker systemintegration",
            "fachinformatikerin systemintegration",
            "systemadministrator",
            "netzwerkadministrator",
        ),
        authority=Authority(
            name="IHK Nürnberg für Mittelfranken - Anerkennungsberatung / IHK FOSA (Ulmenstraße 52g)",
            city="Nürnberg",
            email="anerkennung@nuernberg.ihk.de",
            phone="+49 911 1335-112",
            website="https://www.ihk-nuernberg.de/anerkennung",
        ),
        documents=(
            *_COMMON_DOCUMENTS,
            Document(
                name_de="Ausbildungsordnung oder Lehrplan",
                name_en="Training regulation or curriculum",
                notes="Beschreibt, welche technischen Inhalte Ihre Ausbildung abgedeckt hat.",
            ),
            Document(
                name_de="Projekt- und Tätigkeitsnachweise",
                name_en="Project and task evidence",
                notes="Zum Beispiel Serveradministration, Netzwerke, Ticketsysteme oder Cloud-Betrieb.",
            ),
            Document(
                name_de="IHK-FOSA-Antragsformular",
                name_en="IHK FOSA application form",
                notes="Für duale Ausbildungsberufe ist die IHK FOSA häufig die zentrale Stelle.",
            ),
        ),
    ),
)


class MockDataProvider(DataProvider):
    """Hardcoded mock provider for the hackathon demo."""

    def __init__(self) -> None:
        self._records = _RECORDS

    async def find_german_occupation(
        self,
        description: str,
        source_lang: str,
    ) -> list[Occupation]:
        """Return plausible German occupation candidates for free text."""
        _ = source_lang
        query = _normalize(description)
        if not query:
            return []

        ranked: list[tuple[float, Occupation]] = []
        for record in self._records:
            score = _match_score(query, record)
            if score > 0:
                ranked.append((score, record.occupation))

        ranked.sort(key=lambda item: (-item[0], -item[1].confidence, item[1].label_de))
        return [occupation for _, occupation in ranked]

    async def get_recognition_authority(
        self,
        profession: str,
        city: str = "Nürnberg",
    ) -> Authority | None:
        """Return the local Nürnberg-area authority for a known profession."""
        _ = city
        record = self._best_match(profession)
        if record is None:
            return None
        return record.authority

    async def get_required_documents(self, profession: str) -> list[Document]:
        """Return the document checklist for a known profession."""
        record = self._best_match(profession)
        if record is None:
            return []
        return list(record.documents)

    def _best_match(self, profession: str) -> _ProfessionRecord | None:
        query = _normalize(profession)
        if not query:
            return None

        best_record: _ProfessionRecord | None = None
        best_score = 0.0
        for record in self._records:
            score = _match_score(query, record)
            if score > best_score:
                best_score = score
                best_record = record
        return best_record


def _normalize(text: str) -> str:
    text = text.replace("ß", "ss").replace("ẞ", "SS")
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    words_only = re.sub(r"[^a-z0-9]+", " ", ascii_text.lower())
    return re.sub(r"\s+", " ", words_only).strip()


def _match_score(query: str, record: _ProfessionRecord) -> float:
    aliases = (record.occupation.label_de, record.occupation.label_en, *record.aliases)
    query_tokens = set(query.split())
    best_score = 0.0

    for alias in aliases:
        normalized_alias = _normalize(alias)
        if not normalized_alias:
            continue

        if query == normalized_alias:
            best_score = max(best_score, 3.0 + record.occupation.confidence)
        elif normalized_alias in query:
            best_score = max(
                best_score,
                2.0 + record.occupation.confidence + min(len(normalized_alias), 80) / 100,
            )
        elif query in normalized_alias:
            best_score = max(
                best_score,
                1.0 + record.occupation.confidence + min(len(query), 80) / 100,
            )
        else:
            alias_tokens = set(normalized_alias.split())
            overlap = query_tokens & alias_tokens
            if alias_tokens and overlap == alias_tokens:
                best_score = max(best_score, 1.5 + record.occupation.confidence)
            elif len(overlap) >= 2:
                best_score = max(
                    best_score,
                    0.5 + record.occupation.confidence + len(overlap) / max(len(alias_tokens), 1),
                )

    return best_score


async def _sample_payload() -> dict[str, object]:
    provider = MockDataProvider()
    samples: dict[str, object] = {}
    for query in (
        "I am a nurse",
        "mechanical engineer",
        "school teacher",
        "medical doctor",
        "IT systems administrator",
    ):
        occupations = await provider.find_german_occupation(query, "en")
        profession = occupations[0].label_de if occupations else query
        authority = await provider.get_recognition_authority(profession)
        documents = await provider.get_required_documents(profession)
        samples[query] = {
            "occupations": [asdict(occupation) for occupation in occupations[:2]],
            "authority": asdict(authority) if authority is not None else None,
            "documents": [asdict(document) for document in documents[:3]],
        }
    return samples


def main() -> None:
    """Print smoke-test sample data for manual verification."""
    print(json.dumps(asyncio.run(_sample_payload()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["MockDataProvider"]
