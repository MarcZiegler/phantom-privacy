"""Synthetic distractor generators for the limited-knowledge PII attack."""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

# Small built-in name lists for synthetic name distractors. These are common
# public-domain given names and surnames; no project-specific PII is used.
_SYNTHETIC_FIRST_NAMES = (
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph",
    "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy",
    "Daniel", "Lisa", "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra",
    "Donald", "Ashley", "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna",
    "Joshua", "Michelle", "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda",
    "George", "Melissa", "Timothy", "Deborah", "Ronald", "Stephanie", "Edward",
    "Rebecca", "Jason", "Sharon", "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob",
    "Kathleen", "Gary", "Amy", "Nicholas", "Shirley", "Eric", "Angela", "Jonathan",
    "Helen", "Stephen", "Anna", "Larry", "Brenda", "Justin", "Pamela", "Scott",
    "Nicole", "Brandon", "Emma", "Benjamin", "Samantha", "Samuel", "Katherine",
    "Gregory", "Christine", "Alexander", "Debra", "Frank", "Rachel", "Patrick",
    "Catherine", "Raymond", "Carolyn", "Jack", "Janet", "Dennis", "Ruth",
)

_SYNTHETIC_LAST_NAMES = (
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
    "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
    "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
    "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson", "Watson",
    "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes",
    "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers", "Long", "Ross",
    "Foster", "Jimenez",
)

_PHONE_FORMATS = (
    "{area}-{prefix}-{line}",
    "({area}) {prefix}-{line}",
    "{area}.{prefix}.{line}",
    "{area} {prefix}-{line}",
    "{area}) {prefix}-{line}",
)


class SyntheticDistractorGenerator:
    """Generate plausible synthetic distractor values for PII extraction attacks.

    The generator uses only public reference lists and statistics derived from the
    ground-truth values themselves (e.g., the area-code distribution of phone
    numbers). It does not use any other user's true PII value as a distractor.
    """

    def __init__(
        self,
        all_ground_truth: dict[str, dict[str, dict[str, object]]],
        pii_types: Iterable[str],
        seed: int = 42,
        job_titles_file: str | Path = "configs/pii_graph_job_titles.txt",
    ) -> None:
        """Initialize the generator.

        Args:
            all_ground_truth: Mapping from user to PII-type metadata. Used only to
                estimate aggregate statistics such as phone area-code frequencies.
            pii_types: PII types that may be requested.
            seed: Random seed for reproducibility.
            job_titles_file: Path to a newline-delimited list of job titles.
        """
        self.rng = random.Random(seed)
        self.pii_types = set(pii_types)
        self.phone_distribution = self._build_phone_distribution(all_ground_truth)
        self.email_domain_distribution = self._build_email_domain_distribution(all_ground_truth)
        self.job_titles = self._load_job_titles(job_titles_file)

    def _build_phone_distribution(
        self,
        all_ground_truth: dict[str, dict[str, dict[str, object]]],
    ) -> list[tuple[str, float]] | None:
        """Return a weighted list of (area_code, probability) or None if no phones."""
        area_codes: list[str] = []
        for gt in all_ground_truth.values():
            for pii_type, meta in gt.items():
                if pii_type not in ("phone", "fax"):
                    continue
                value = meta.get("value")
                if not value:
                    continue
                digits = re.sub(r"\D", "", str(value))
                if len(digits) >= 10:
                    area_codes.append(digits[:3])
        if not area_codes:
            return None
        counts = Counter(area_codes)
        total = sum(counts.values())
        return [(ac, counts[ac] / total) for ac in sorted(counts)]

    def _build_email_domain_distribution(
        self,
        all_ground_truth: dict[str, dict[str, dict[str, object]]],
    ) -> list[tuple[str, float]] | None:
        """Return a weighted list of (domain, probability) or None if no emails."""
        domains: list[str] = []
        for gt in all_ground_truth.values():
            for pii_type, meta in gt.items():
                if pii_type != "email":
                    continue
                value = meta.get("value")
                if not value:
                    continue
                m = re.search(r"@(.+)$", str(value))
                if m:
                    domains.append(m.group(1).lower())
        if not domains:
            return None
        counts = Counter(domains)
        total = sum(counts.values())
        return [(d, counts[d] / total) for d in sorted(counts)]

    def _load_job_titles(self, path: str | Path) -> list[str]:
        """Load a list of job titles from a file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Job-title list not found: {path}\n"
                "Download or provide the file and see README.md Setup step 5 for sources "
                "and licensing notes."
            )
        lines = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines

    @staticmethod
    def _canonical(value: str, pii_type: str) -> str:
        """Return a normalized form of ``value`` for equality checks."""
        if pii_type in ("phone", "fax"):
            digits = re.sub(r"\D", "", str(value))
            if len(digits) == 11 and digits.startswith("1"):
                return digits[1:]
            return digits
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def _random_phone(self, exclude_canonical: set[str]) -> str:
        """Generate a random phone number, avoiding excluded canonical values."""
        if self.phone_distribution is not None:
            area_codes, weights = zip(*self.phone_distribution, strict=True)
            area = self.rng.choices(area_codes, weights=weights, k=1)[0]
        else:
            area = f"{self.rng.randint(200, 999):03d}"

        for _ in range(1000):
            prefix = f"{self.rng.randint(200, 999):03d}"
            line = f"{self.rng.randint(0, 9999):04d}"
            fmt = self.rng.choice(_PHONE_FORMATS)
            value = fmt.format(area=area, prefix=prefix, line=line)
            if self._canonical(value, "phone") not in exclude_canonical:
                return value
        raise RuntimeError("Failed to generate a unique synthetic phone number")

    def _random_email(self, exclude_canonical: set[str]) -> str:
        """Generate a random email address, avoiding excluded canonical values."""
        if self.email_domain_distribution is not None:
            domains, weights = zip(*self.email_domain_distribution, strict=True)
            domain = self.rng.choices(domains, weights=weights, k=1)[0]
        else:
            domain = "example.com"

        for _ in range(1000):
            first = self.rng.choice(_SYNTHETIC_FIRST_NAMES).lower()
            last = self.rng.choice(_SYNTHETIC_LAST_NAMES).lower()
            fmt = self.rng.choice(
                (
                    f"{first}.{last}@{domain}",
                    f"{first}{last[0]}@{domain}",
                    f"{first[0]}.{last}@{domain}",
                    f"{last}.{first}@{domain}",
                )
            )
            if self._canonical(fmt, "email") not in exclude_canonical:
                return fmt
        raise RuntimeError("Failed to generate a unique synthetic email address")

    def _random_job_title(self, exclude_canonical: set[str]) -> str:
        """Sample a random job title, avoiding excluded values."""
        if not self.job_titles:
            raise RuntimeError("No job-title reference list available")
        for _ in range(1000):
            value = self.rng.choice(self.job_titles)
            value = value.title()
            if self._canonical(value, "job_title") not in exclude_canonical:
                return value
        raise RuntimeError("Failed to generate a unique synthetic job title")

    def _random_name(self, exclude_canonical: set[str]) -> str:
        """Generate a random "Last, First" name, avoiding excluded values."""
        for _ in range(1000):
            first = self.rng.choice(_SYNTHETIC_FIRST_NAMES)
            last = self.rng.choice(_SYNTHETIC_LAST_NAMES)
            value = f"{last}, {first}"
            if self._canonical(value, "name") not in exclude_canonical:
                return value
        raise RuntimeError("Failed to generate a unique synthetic name")

    def sample(self, pii_type: str, target_value: str, n: int) -> list[str]:
        """Return ``n`` synthetic distractors for ``pii_type`` excluding ``target_value``.

        Args:
            pii_type: One of the supported PII types.
            target_value: The true value to exclude from distractors.
            n: Number of distractors to generate.

        Returns:
            A list of ``n`` distinct synthetic distractor strings.
        """
        exclude = {self._canonical(target_value, pii_type)}
        distractors: list[str] = []

        generator = {
            "phone": self._random_phone,
            "fax": self._random_phone,
            "email": self._random_email,
            "job_title": self._random_job_title,
            "name": self._random_name,
        }.get(pii_type)

        if generator is None:
            raise ValueError(f"No synthetic distractor generator for PII type {pii_type!r}")

        while len(distractors) < n:
            value = generator(exclude)
            canonical = self._canonical(value, pii_type)
            if canonical not in exclude:
                exclude.add(canonical)
                distractors.append(value)

        return distractors
