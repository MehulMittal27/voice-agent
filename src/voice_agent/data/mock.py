"""Mock recognition data for the Nürnberg voice-agent demo."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Final, NamedTuple

from .base import Authority, DataProvider, Document, LabourMarketStatus, Occupation

logger = logging.getLogger(__name__)

_CSV_PATH = Path(__file__).parent / "nuremberg_authorities.csv"
_PROFESSION_FIELDS: Final[tuple[str, ...]] = (
    "profession_de",
    "profession_en",
    "profession_uk",
)
_FIELD_LANG: Final[dict[str, str]] = {
    "profession_de": "de",
    "profession_en": "en",
    "profession_uk": "uk",
}
_BASELINE_DOCUMENTS: Final[tuple[Document, ...]] = (
    Document(
        name_de="Identitätsnachweis",
        name_en="Proof of identity",
        notes="Reisepass oder Personalausweis als gut lesbare Kopie.",
    ),
    Document(
        name_de="Tabellarischer Lebenslauf",
        name_en="Curriculum vitae",
        notes="Auf Deutsch, mit Ausbildung, Berufserfahrung und Kontaktdaten.",
    ),
    Document(
        name_de="Abschlusszeugnis oder Diplom",
        name_en="Diploma or final certificate",
        notes="Mit vereidigter deutscher Übersetzung, wenn das Original nicht deutsch ist.",
    ),
    Document(
        name_de="Fächer- und Stundenübersicht",
        name_en="Transcript of subjects and hours",
        notes="Zeigt der Stelle Inhalt und Umfang der Ausbildung.",
    ),
    Document(
        name_de="Arbeitsnachweise",
        name_en="Employment evidence",
        notes="Arbeitszeugnisse oder Verträge mit möglichst genauer Aufgabenbeschreibung.",
    ),
)
_DOCUMENTS_BY_CATEGORY: Final[dict[str, tuple[Document, ...]]] = {
    "Approbationsberuf": (
        Document(
            name_de="Identitätsnachweis",
            name_en="Proof of identity",
            notes="Reisepass oder Personalausweis als Kopie.",
        ),
        Document(
            name_de="Ausbildungsabschluss und Fächerübersicht",
            name_en="Degree certificate and transcript",
            notes="Diplom, Stundenumfang und Inhalte mit deutscher Übersetzung.",
        ),
        Document(
            name_de="Berufszulassung im Herkunftsland",
            name_en="Professional licence in country of origin",
            notes="Nachweis, dass Sie dort selbstständig im Beruf arbeiten dürfen.",
        ),
        Document(
            name_de="Certificate of Good Standing",
            name_en="Certificate of good standing",
            notes="Aktuelle Bescheinigung der zuständigen Kammer oder Behörde.",
        ),
        Document(
            name_de="Deutsch- und Fachsprachennachweis",
            name_en="German and professional language proof",
            notes="Meist B2 Deutsch plus Fachsprachprüfung oder C1 Medizin.",
        ),
        Document(
            name_de="Ärztliches Attest und Führungszeugnis",
            name_en="Medical certificate and certificate of good conduct",
            notes="Oft erst kurz vor der endgültigen Erlaubnis nötig.",
        ),
    ),
    "Approbationsberuf + Facharztanerkennung": (
        Document(
            name_de="Approbation oder ärztliche Berufserlaubnis",
            name_en="Medical licence or temporary professional permit",
            notes="Die Facharztanerkennung baut auf der ärztlichen Zulassung auf.",
        ),
        Document(
            name_de="Facharztdiplom",
            name_en="Specialist medical certificate",
            notes="Urkunde über die abgeschlossene Spezialisierung im Herkunftsland.",
        ),
        Document(
            name_de="Weiterbildungslogbuch",
            name_en="Specialist training logbook",
            notes="Rotationen, Eingriffe, Dauer und Inhalte möglichst genau aufführen.",
        ),
        Document(
            name_de="Arbeitszeugnisse aus Kliniken",
            name_en="Clinical employment references",
            notes="Mit Fachgebiet, Zeitraum, Wochenstunden und Aufgaben.",
        ),
        Document(
            name_de="Certificate of Good Standing",
            name_en="Certificate of good standing",
            notes="Aktuelle Bescheinigung der ärztlichen Aufsichtsbehörde.",
        ),
        Document(
            name_de="Deutsch-Fachsprachennachweis",
            name_en="Medical German language proof",
            notes="Fachsprachprüfung oder gleichwertiger Nachweis.",
        ),
    ),
    "Gesundheitsfachberuf": (
        Document(
            name_de="Identitätsnachweis",
            name_en="Proof of identity",
            notes="Reisepass oder Personalausweis als gut lesbare Kopie.",
        ),
        Document(
            name_de="Berufsabschluss im Gesundheitsberuf",
            name_en="Health profession diploma",
            notes="Diplom oder Zeugnis mit deutscher Übersetzung.",
        ),
        Document(
            name_de="Stunden- und Fächerübersicht",
            name_en="Transcript of hours and subjects",
            notes="Theorie, Praxis, Einsätze und Gesamtstunden sind wichtig.",
        ),
        Document(
            name_de="Berufszulassung oder Registrierung",
            name_en="Professional licence or registration",
            notes="Nachweis aus dem Herkunftsland, falls dort vorgeschrieben.",
        ),
        Document(
            name_de="Deutsch-Sprachnachweis B2",
            name_en="German language certificate B2",
            notes="Für Patientenkontakt wird in der Regel B2 erwartet.",
        ),
        Document(
            name_de="Führungszeugnis und ärztliches Attest",
            name_en="Certificate of good conduct and medical certificate",
            notes="Die Behörde sagt Ihnen, wann diese Unterlagen aktuell sein müssen.",
        ),
    ),
    "Handwerksberuf": (
        Document(
            name_de="Identitätsnachweis und Lebenslauf",
            name_en="Proof of identity and CV",
            notes="Mit Ausbildung, Gesellenzeit und Berufspraxis.",
        ),
        Document(
            name_de="Ausbildungszeugnis",
            name_en="Vocational training certificate",
            notes="Abschluss der Berufsausbildung mit deutscher Übersetzung.",
        ),
        Document(
            name_de="Ausbildungsinhalte oder Lehrplan",
            name_en="Training curriculum",
            notes="Hilft der Handwerkskammer beim Vergleich mit dem deutschen Beruf.",
        ),
        Document(
            name_de="Arbeitszeugnisse und Tätigkeitsnachweise",
            name_en="Employment references and task evidence",
            notes="Zeigen Sie möglichst konkrete Maschinen, Anlagen oder Techniken.",
        ),
        Document(
            name_de="Nachweise über Weiterbildungen",
            name_en="Further training certificates",
            notes="Zum Beispiel Elektro, SHK, Kfz-Diagnose oder Schweißscheine.",
        ),
    ),
    "IHK-Beruf": (
        Document(
            name_de="IHK-FOSA-Antragsformular",
            name_en="IHK FOSA application form",
            notes="Für duale IHK-Berufe ist IHK FOSA die zentrale Stelle.",
        ),
        Document(
            name_de="Identitätsnachweis und Lebenslauf",
            name_en="Proof of identity and CV",
            notes="Bitte mit aktueller Adresse und beruflichem Verlauf.",
        ),
        Document(
            name_de="Ausbildungsabschluss",
            name_en="Vocational qualification certificate",
            notes="Zeugnis oder Diplom der abgeschlossenen Ausbildung.",
        ),
        Document(
            name_de="Ausbildungsplan oder Fächerübersicht",
            name_en="Training plan or transcript",
            notes="Beschreibt Dauer, Inhalte und Prüfungen.",
        ),
        Document(
            name_de="Arbeitszeugnisse",
            name_en="Employment references",
            notes="Berufserfahrung kann Unterschiede teilweise ausgleichen.",
        ),
    ),
    "Reglementierter Titel": (
        Document(
            name_de="Identitätsnachweis",
            name_en="Proof of identity",
            notes="Reisepass oder Personalausweis als Kopie.",
        ),
        Document(
            name_de="Hochschulabschluss",
            name_en="University degree certificate",
            notes="Diplom, Bachelor oder Master mit Übersetzung.",
        ),
        Document(
            name_de="Modulhandbuch oder Studienplan",
            name_en="Module handbook or study plan",
            notes="Wichtig für den fachlichen Vergleich.",
        ),
        Document(
            name_de="Nachweis zur Berufsbezeichnung",
            name_en="Proof of right to use the title",
            notes="Falls Sie den geschützten Titel im Herkunftsland führen durften.",
        ),
        Document(
            name_de="Berufserfahrung und Projektliste",
            name_en="Professional experience and project list",
            notes="Vor allem bei Ingenieurberufen hilfreich.",
        ),
    ),
    "Zeugnisanerkennung": (
        Document(
            name_de="Schulabschlusszeugnis",
            name_en="School-leaving certificate",
            notes="Original oder beglaubigte Kopie des Abschlusszeugnisses.",
        ),
        Document(
            name_de="Fächer- und Notenübersicht",
            name_en="Subjects and grades overview",
            notes="Mit Schuljahren, Noten und Abschlussdatum.",
        ),
        Document(
            name_de="Deutsche Übersetzung",
            name_en="German translation",
            notes="Von vereidigten Übersetzerinnen oder Übersetzern.",
        ),
        Document(
            name_de="Identitätsnachweis",
            name_en="Proof of identity",
            notes="Reisepass, Personalausweis oder Aufenthaltstitel.",
        ),
        Document(
            name_de="Antragsformular der Zeugnisanerkennungsstelle",
            name_en="Certificate recognition application form",
            notes="Die Zeugnisanerkennungsstelle stellt das Formular bereit.",
        ),
    ),
    "Nicht reglementiert (Führerschein)": (
        Document(
            name_de="Ausländischer Führerschein",
            name_en="Foreign driving licence",
            notes="Original und Kopie des gültigen Führerscheins.",
        ),
        Document(
            name_de="Übersetzung oder internationaler Führerschein",
            name_en="Translation or international driving permit",
            notes="Wenn die Fahrerlaubnisbehörde es verlangt.",
        ),
        Document(
            name_de="Identitätsnachweis und Aufenthaltstitel",
            name_en="Proof of identity and residence permit",
            notes="Für die Umschreibung bei der Führerscheinstelle.",
        ),
        Document(
            name_de="Biometrisches Passfoto",
            name_en="Biometric passport photo",
            notes="Aktuelles Foto für den deutschen Führerschein.",
        ),
        Document(
            name_de="Sehtest und Erste-Hilfe-Nachweis",
            name_en="Eye test and first-aid proof",
            notes="Je nach Klasse und Herkunftsland kann das nötig sein.",
        ),
    ),
}


class _RowMatch(NamedTuple):
    row_index: int
    row: dict[str, str]
    field: str
    confidence: float
    exact: bool


@lru_cache(maxsize=1)
def _load_authorities() -> list[dict[str, str]]:
    """Load the verified Nürnberg authority CSV once."""
    with _CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    logger.info("Loaded %d authority rows from nuremberg_authorities.csv", len(rows))
    return rows


class MockDataProvider(DataProvider):
    """CSV-backed mock provider for the hackathon demo."""

    def __init__(self) -> None:
        self._authority_rows = _load_authorities()

    async def find_german_occupation(
        self,
        description: str,
        source_lang: str,
    ) -> list[Occupation]:
        """Return German occupation candidates from the verified CSV."""
        query = _normalize(description)
        if not query:
            logger.info(
                "Occupation lookup description=%r source_lang=%s matched none",
                description,
                source_lang,
            )
            return []

        matches = [
            match
            for match in _iter_profession_matches(query, self._authority_rows)
            if _normalize(match.row.get("category", "")) != "beratung"
        ]
        matches.sort(key=lambda match: (not match.exact, match.row_index))

        occupations: list[Occupation] = []
        for match in matches[:3]:
            logger.info(
                "Occupation lookup description=%r source_lang=%s matched_lang=%s "
                "matched_field=%s matched_row=%s profession_de=%r",
                description,
                source_lang,
                _FIELD_LANG[match.field],
                match.field,
                match.row_index,
                match.row.get("profession_de", ""),
            )
            occupations.append(
                Occupation(
                    esco_id=f"MOCK-{match.row_index}",
                    label_de=match.row.get("profession_de", ""),
                    label_en=match.row.get("profession_en", ""),
                    kldb_code=None,
                    confidence=match.confidence,
                )
            )

        if not occupations:
            logger.info(
                "Occupation lookup description=%r source_lang=%s matched none",
                description,
                source_lang,
            )
        return occupations

    async def get_recognition_authority(
        self,
        profession: str,
        city: str = "Nürnberg",
    ) -> Authority | None:
        """Return the recognition authority for a profession, falling back to KuBB."""
        match = _first_profession_match(profession, self._authority_rows)
        row = match.row if match is not None else _fallback_row(self._authority_rows)
        if row is None:
            logger.info(
                "Authority lookup profession=%r matched_row=None confidence=None",
                profession,
            )
            return None

        logger.info(
            "Authority lookup profession=%r matched_row=%r confidence=%s",
            profession,
            row.get("profession_de", ""),
            row.get("confidence", ""),
        )
        return _authority_from_row(row, city)

    async def get_required_documents(self, profession: str) -> list[Document]:
        """Return a category-aware document checklist for a profession."""
        match = _first_profession_match(profession, self._authority_rows)
        if match is None:
            return list(_BASELINE_DOCUMENTS)

        category = match.row.get("category", "").strip()
        return list(_documents_for_category(category))

    async def get_labour_market_status(
        self,
        profession: str,
        region: str = "Bayern",
    ) -> LabourMarketStatus:
        """Return plausible labour-market shortage data for the profession."""
        match = _first_profession_match(profession, self._authority_rows)
        row = match.row if match is not None else None
        status = _labour_market_status_for_row(profession, region, row)
        logger.info(
            "Labour market lookup profession=%r matched_row=%r category=%r shortage=%s",
            profession,
            row.get("profession_de", "") if row is not None else None,
            row.get("category", "") if row is not None else None,
            status.shortage,
        )
        return status


def _normalize(text: str | None) -> str:
    if text is None:
        return ""
    normalized = re.sub(r"\s+", " ", text.casefold().strip())
    return normalized.strip(" .,;:!?()[]{}\"'")


def _iter_profession_matches(
    query: str,
    rows: list[dict[str, str]],
) -> list[_RowMatch]:
    matches: list[_RowMatch] = []
    for row_index, row in enumerate(rows, start=1):
        match = _match_row(query, row_index, row)
        if match is not None:
            matches.append(match)
    return matches


def _first_profession_match(
    profession: str,
    rows: list[dict[str, str]],
) -> _RowMatch | None:
    query = _normalize(profession)
    if not query:
        return None

    first_substring_match: _RowMatch | None = None
    for row_index, row in enumerate(rows, start=1):
        match = _match_row(query, row_index, row)
        if match is None:
            continue
        if match.exact:
            return match
        if first_substring_match is None:
            first_substring_match = match
    return first_substring_match


def _match_row(query: str, row_index: int, row: dict[str, str]) -> _RowMatch | None:
    for field in _PROFESSION_FIELDS:
        value = _normalize(row.get(field, ""))
        if not value:
            continue
        if query == value:
            return _RowMatch(row_index, row, field, 0.95, True)
        if query in value or value in query:
            return _RowMatch(row_index, row, field, 0.75, False)
    return None


def _fallback_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if _normalize(row.get("category", "")) == "beratung":
            return row
    return None


def _authority_from_row(row: dict[str, str], fallback_city: str) -> Authority:
    address = row.get("authority_address", "").strip()
    city = fallback_city
    if address:
        last_token = address.split()[-1].strip().strip(",")
        if last_token:
            city = last_token

    email = row.get("authority_email", "").strip() or None
    return Authority(
        name=row.get("authority_name", "").strip(),
        city=city,
        email=email,
        phone=row.get("authority_phone", "").strip(),
        website=row.get("authority_website", "").strip(),
    )


def _documents_for_category(category: str) -> tuple[Document, ...]:
    if category in _DOCUMENTS_BY_CATEGORY:
        return _DOCUMENTS_BY_CATEGORY[category]
    if category.startswith("Gesundheitsfachberuf"):
        return _DOCUMENTS_BY_CATEGORY["Gesundheitsfachberuf"]
    return _BASELINE_DOCUMENTS


def _labour_market_status_for_row(
    profession: str,
    region: str,
    row: dict[str, str] | None,
) -> LabourMarketStatus:
    profession_label = profession.strip() or "Unbekannte Profession"
    category = ""
    haystack = _normalize(profession)
    if row is not None:
        profession_label = row.get("profession_de", "").strip() or profession_label
        category = row.get("category", "").strip()
        haystack = _normalize(
            " ".join(
                [
                    profession,
                    row.get("profession_de", ""),
                    row.get("profession_en", ""),
                    row.get("profession_uk", ""),
                    category,
                ]
            )
        )

    if _contains_any(haystack, ("pflege", "nurse", "медсестра", "медична сестра")):
        return LabourMarketStatus(
            profession=profession_label,
            region=region,
            shortage=True,
            shortage_level="hoch",
            open_positions=12000,
            applicants_per_opening=0.4,
            note="Rund 12.000 offene Stellen in Bayern; Pflege ist ein klarer Mangelberuf.",
        )
    if _contains_any(haystack, ("arzt", "ärztin", "doctor", "лікар")):
        return LabourMarketStatus(
            profession=profession_label,
            region=region,
            shortage=True,
            shortage_level="hoch",
            open_positions=5500,
            applicants_per_opening=0.5,
            note="Ärztinnen und Ärzte werden gesucht; die Approbation ist meist der Engpass.",
        )
    if _contains_any(haystack, ("kfz", "car mechanic", "автомеханік")):
        return LabourMarketStatus(
            profession=profession_label,
            region=region,
            shortage=True,
            shortage_level="hoch",
            open_positions=3800,
            applicants_per_opening=0.6,
            note="Kfz-Mechatronik ist in Bayern knapp; praktische Erfahrung hilft sehr.",
        )
    if category == "Handwerksberuf" and _contains_any(
        haystack,
        (
            "elektroniker",
            "electrician",
            "електрик",
            "anlagenmechaniker",
            "plumber",
            "shk",
            "сантехнік",
        ),
    ):
        return LabourMarketStatus(
            profession=profession_label,
            region=region,
            shortage=True,
            shortage_level="hoch",
            open_positions=6200,
            applicants_per_opening=0.7,
            note="Elektro- und SHK-Handwerk sind Mangelbereiche mit vielen offenen Stellen.",
        )
    if _contains_any(haystack, ("fachinformatiker", "it specialist", "it-спеціаліст")):
        return LabourMarketStatus(
            profession=profession_label,
            region=region,
            shortage=True,
            shortage_level="mittel",
            open_positions=7400,
            applicants_per_opening=0.9,
            note="IT-Fachkräfte sind gefragt; Anerkennung ist hilfreich, aber oft nicht die größte Hürde.",
        )
    if _contains_any(haystack, ("erzieher", "kindergarten teacher", "вихователь")):
        return LabourMarketStatus(
            profession=profession_label,
            region=region,
            shortage=True,
            shortage_level="hoch",
            open_positions=6900,
            applicants_per_opening=0.5,
            note="Kitas suchen stark; mit Anerkennung oder Anpassungsqualifizierung steigen die Chancen.",
        )
    if _contains_any(haystack, ("einzelhandel", "retail", "продавець")):
        return LabourMarketStatus(
            profession=profession_label,
            region=region,
            shortage=False,
            shortage_level="normal",
            open_positions=9000,
            applicants_per_opening=2.8,
            note="Es gibt viele Stellen im Verkauf, aber es ist kein klarer Mangelberuf.",
        )
    if category in {"Reglementierter Titel", "Reglementierter Beruf"}:
        return LabourMarketStatus(
            profession=profession_label,
            region=region,
            shortage=False,
            shortage_level="regional unterschiedlich",
            open_positions=1800,
            applicants_per_opening=2.1,
            note="Die Chancen hängen stark vom Fachgebiet ab; die Anerkennung des Titels hilft beim Einstieg.",
        )
    return LabourMarketStatus(
        profession=profession_label,
        region=region,
        shortage=False,
        shortage_level="normal",
        open_positions=2200,
        applicants_per_opening=2.4,
        note="Es gibt Stellen, aber die Lage ist eher normal; Berufserfahrung und Deutsch helfen stark.",
    )


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(_normalize(needle) in text for needle in needles)


async def _sample_payload() -> dict[str, object]:
    provider = MockDataProvider()
    samples: dict[str, object] = {}
    for query in (
        "I am a nurse",
        "mechanical engineer",
        "school teacher",
        "medical doctor",
        "IT specialist",
    ):
        occupations = await provider.find_german_occupation(query, "en")
        profession = occupations[0].label_de if occupations else query
        authority = await provider.get_recognition_authority(profession)
        documents = await provider.get_required_documents(profession)
        labour_market = await provider.get_labour_market_status(profession)
        samples[query] = {
            "occupations": [asdict(occupation) for occupation in occupations[:2]],
            "authority": asdict(authority) if authority is not None else None,
            "documents": [asdict(document) for document in documents[:3]],
            "labour_market": asdict(labour_market),
        }
    return samples


def main() -> None:
    """Print smoke-test sample data for manual verification."""
    print(json.dumps(asyncio.run(_sample_payload()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["MockDataProvider"]
