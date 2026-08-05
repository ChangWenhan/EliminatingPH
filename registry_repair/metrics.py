from __future__ import annotations

import csv
import json
import re
import ast
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def normalize_python(name: object) -> str:
    if name is None:
        return ""
    text = str(name)
    if text.lower() == "nan":
        return "nan"
    text = re.sub(r"\d\. ", "", text)
    text = re.sub(r"(?<=.)\n(?=.)", " ", text)
    text = re.sub(r"\n", "", text)
    return re.sub(r"[-_.]+", "-", text).strip(" `.-").lower()


def normalize_pip(name: object) -> str:
    if name is None:
        return ""
    text = str(name)
    text = re.sub(r"[()'\"]", "", text)
    return re.sub(r"[-_.]+", "-", text).strip(' "`.-').lower()


def delete_dupes_and_empty(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item:
            continue
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def load_name_set(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            name = normalize_python(row[0])
            if name:
                names.add(name)
    if not names:
        raise ValueError(f"No package names loaded from {path}")
    return names


def load_false_positives(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            value = row[1] if len(row) > 1 else row[0]
            name = normalize_python(value)
            if name:
                names.add(name)
    return names


def parse_package_list(text: object) -> list[str]:
    if text is None:
        return []
    values = str(text).split(",")
    names = [
        normalize_python(value)
        for value in values
    ]
    return delete_dupes_and_empty(
        name for name in names if len(name.split()) == 1
    )


def check_packages(
    package_list: Iterable[str],
    package_names: set[str],
    false_positives: set[str],
) -> tuple[list[str], list[str]]:
    valid: list[str] = []
    hallucinated: list[str] = []
    for item in package_list:
        if " " in item or item.lower() in {"none", "nan", ""}:
            continue
        if item in false_positives or item.split("-", 1)[0] in stdlib_root_names():
            continue
        if item in package_names:
            valid.append(item)
        else:
            hallucinated.append(item)
    return delete_dupes_and_empty(valid), delete_dupes_and_empty(hallucinated)


def parse_pip_install(text: object) -> list[str]:
    if not isinstance(text, (str, bytes)):
        return []
    matches = re.findall(
        r"(?:^|[\s\"'])(?:!\s*|python(?:3)?\s+-m\s+)?pip\s+install\s+(?P<package_name>\S+)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return [match for match in matches if not match.startswith("-")]


def check_pips(pip_list: Iterable[str], pip_names: set[str]) -> tuple[list[str], list[str]]:
    valid: list[str] = []
    hallucinated: list[str] = []
    translation_table = str.maketrans("", "", "()[]`")
    version_pattern = re.compile(r"([^=<>!~]+)([=<>!~]{1,2}[\d\.]+)?")

    for item in pip_list:
        text = item.translate(translation_table)
        if re.search(r"[+@:\"',{}/\*]", text):
            continue
        for part in text.split():
            if part.startswith("--"):
                continue
            match = version_pattern.match(part)
            if not match:
                continue
            candidate = normalize_python(match.group(1).strip())
            if candidate.startswith("--") or "requirements" in candidate:
                continue
            if candidate.split("-", 1)[0] in stdlib_root_names():
                continue
            if candidate in pip_names:
                valid.append(candidate)
            else:
                hallucinated.append(candidate)
    return delete_dupes_and_empty(valid), delete_dupes_and_empty(hallucinated)


@dataclass(frozen=True)
class PackageEvaluation:
    valid_1: list[str]
    hallucinated_1: list[str]
    valid_2: list[str]
    hallucinated_2: list[str]
    pip_valid: list[str]
    pip_hallucinated: list[str]

    @property
    def valid_total(self) -> int:
        return len(self.valid_1) + len(self.valid_2) + len(self.pip_valid)

    @property
    def hallucinated_total(self) -> int:
        return (
            len(self.hallucinated_1)
            + len(self.hallucinated_2)
            + len(self.pip_hallucinated)
        )

    @property
    def package_total(self) -> int:
        return self.valid_total + self.hallucinated_total

    @property
    def hallucination_rate(self) -> float | None:
        total = self.package_total
        return self.hallucinated_total / total if total else None

    @property
    def hallucinated_packages(self) -> list[str]:
        return delete_dupes_and_empty(
            [
                *self.hallucinated_1,
                *self.hallucinated_2,
                *self.pip_hallucinated,
            ]
        )


def evaluate_python_row(
    answer: str,
    package_query_1: str,
    package_query_2: str,
    package_names: set[str],
    false_positives: set[str],
) -> PackageEvaluation:
    packages_1 = parse_package_list(package_query_1)
    packages_2 = parse_package_list(package_query_2)
    valid_1, hallucinated_1 = check_packages(packages_1, package_names, false_positives)
    valid_2, hallucinated_2 = check_packages(packages_2, package_names, false_positives)
    pip_packages = [normalize_pip(entry) for entry in parse_pip_install(answer)]
    pip_valid, pip_hallucinated = check_pips(pip_packages, package_names)
    return PackageEvaluation(
        valid_1=valid_1,
        hallucinated_1=hallucinated_1,
        valid_2=valid_2,
        hallucinated_2=hallucinated_2,
        pip_valid=pip_valid,
        pip_hallucinated=pip_hallucinated,
    )


def attack_candidate_packages(
    hallucinated_packages: Iterable[str],
    package_names: set[str],
    false_positives: set[str],
) -> list[str]:
    candidates: list[str] = []
    pattern = re.compile(r"^[a-z0-9][a-z0-9-]{0,212}$")
    for package in hallucinated_packages:
        name = normalize_python(package)
        if (
            name
            and name not in package_names
            and name not in false_positives
            and pattern.match(name)
            and "--" not in name
        ):
            candidates.append(name)
    return delete_dupes_and_empty(candidates)


IMPORT_DISTRIBUTION_ALIASES = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "fitz": "pymupdf",
    "git": "gitpython",
    "pil": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
}

IMPORT_PREFIX_DISTRIBUTIONS = {
    "google.cloud.storage": "google-cloud-storage",
    "google.oauth2": "google-auth",
    "google.protobuf": "protobuf",
    "matplotlib.pyplot": "matplotlib",
    "scipy.stats": "scipy",
}

PIP_INSTALL_PATTERN = re.compile(
    r"(?:^|[\s\"'])(?:!\s*|python(?:3)?\s+-m\s+)?pip\s+install\b",
    re.IGNORECASE | re.MULTILINE,
)
DYNAMIC_IMPORT_PATTERN = re.compile(r"\b(?:__import__|import_module)\s*\(")


def extract_python_code(text: object) -> tuple[str, str]:
    if not isinstance(text, str):
        return "", "empty"
    stripped = text.strip()
    if not stripped or stripped.lower() == "none":
        return "", "none"
    fenced = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        candidates = sorted((candidate.strip() for candidate in fenced), key=len, reverse=True)
        for candidate in candidates:
            try:
                ast.parse(candidate)
            except SyntaxError:
                continue
            return candidate, "fenced_parseable"
        return candidates[0], "fenced_longest"
    return stripped, "raw"


def _recover_imports_from_invalid_code(code: str) -> list[str]:
    imports: list[str] = []
    import_pattern = re.compile(r"^\s*import\s+([^#;\n]+)", re.MULTILINE)
    from_pattern = re.compile(
        r"^\s*from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+([^#;\n]+)",
        re.MULTILINE,
    )
    for match in import_pattern.finditer(code):
        for item in match.group(1).split(","):
            module = item.strip().split()[0] if item.strip() else ""
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module):
                imports.append(module)
    for match in from_pattern.finditer(code):
        module = match.group(1)
        imported = match.group(2).strip().strip("()")
        for item in imported.split(","):
            symbol = item.strip().split()[0] if item.strip() else ""
            if symbol == "*":
                imports.append(module)
            elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
                imports.append(f"{module}.{symbol}")
    return delete_dupes_and_empty(imports)


def extract_import_paths(code: str) -> tuple[bool, str, list[str]]:
    if not code:
        return False, "empty_or_none", []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        location = f"line {exc.lineno}, column {exc.offset}"
        return False, f"{exc.msg} ({location})", _recover_imports_from_invalid_code(code)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if any(alias.name == "*" for alias in node.names):
                imports.append(node.module)
            else:
                imports.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return True, "", delete_dupes_and_empty(imports)


def stdlib_root_names() -> set[str]:
    return {name.lower() for name in sys.stdlib_module_names}


def is_stdlib_import(import_path: str) -> bool:
    root = import_path.lstrip(".").split(".", 1)[0].lower()
    if not root:
        return False
    if root in stdlib_root_names():
        return True
    stdlib_path = Path(sysconfig.get_paths()["stdlib"])
    return (stdlib_path / f"{root}.py").exists() or (stdlib_path / root).exists()


def resolve_import_distribution(import_path: str, package_names: set[str]) -> tuple[str, str]:
    cleaned = import_path.strip().lstrip(".")
    if not cleaned:
        return "empty", ""
    if is_stdlib_import(cleaned):
        return "stdlib", cleaned
    lowered = cleaned.lower()
    for prefix, distribution in sorted(
        IMPORT_PREFIX_DISTRIBUTIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if lowered == prefix or lowered.startswith(f"{prefix}."):
            normalized = normalize_python(distribution)
            if normalized in package_names:
                return "valid", normalized
    root = lowered.split(".", 1)[0]
    alias = IMPORT_DISTRIBUTION_ALIASES.get(root)
    if alias:
        normalized = normalize_python(alias)
        if normalized in package_names:
            return "valid", normalized
    normalized_root = normalize_python(root)
    if normalized_root in package_names:
        return "valid", normalized_root
    return "hallucinated", normalized_root


@dataclass(frozen=True)
class CodeDependencyEvaluation:
    syntax_valid: bool
    syntax_error: str
    stdlib_imports: list[str]
    valid_import_packages: list[str]
    hallucinated_imports: list[str]
    direct_install_command: bool
    dynamic_import: bool

    @property
    def needs_repair(self) -> bool:
        return (
            not self.syntax_valid
            or bool(self.hallucinated_imports)
            or self.direct_install_command
            or self.dynamic_import
        )


def evaluate_code_dependencies(answer: str, package_names: set[str]) -> CodeDependencyEvaluation:
    code, _ = extract_python_code(answer)
    syntax_valid, syntax_error, imports = extract_import_paths(code)
    stdlib_imports: list[str] = []
    valid_import_packages: list[str] = []
    hallucinated_imports: list[str] = []
    for import_path in imports:
        status, value = resolve_import_distribution(import_path, package_names)
        if status == "stdlib":
            stdlib_imports.append(value)
        elif status == "valid":
            valid_import_packages.append(value)
        elif status == "hallucinated":
            hallucinated_imports.append(value)
    return CodeDependencyEvaluation(
        syntax_valid=syntax_valid,
        syntax_error=syntax_error,
        stdlib_imports=delete_dupes_and_empty(stdlib_imports),
        valid_import_packages=delete_dupes_and_empty(valid_import_packages),
        hallucinated_imports=delete_dupes_and_empty(hallucinated_imports),
        direct_install_command=bool(PIP_INSTALL_PATTERN.search(answer)),
        dynamic_import=bool(DYNAMIC_IMPORT_PATTERN.search(answer)),
    )


def render_package_response(packages: Iterable[str]) -> str:
    values = delete_dupes_and_empty(packages)
    return ", ".join(values) if values else "None"


def registry_validated_package_response(
    response: str,
    package_names: set[str],
    false_positives: set[str],
    *,
    fallback_valid: Iterable[str] = (),
) -> str:
    packages = parse_package_list(response)
    valid, _ = check_packages(packages, package_names, false_positives)
    return render_package_response([*valid, *fallback_valid])


def registry_substituted_package_response(
    response: str,
    package_names: set[str],
    false_positives: set[str],
    alternatives: dict[str, list[str]] | None = None,
    *,
    fallback_valid: Iterable[str] = (),
) -> str:
    """Like registry_validated_package_response but substitutes hallucinated
    packages with RAG-suggested alternatives instead of just deleting them."""
    if alternatives is None:
        alternatives = {}
    packages = parse_package_list(response)
    valid, hallucinated = check_packages(packages, package_names, false_positives)
    substituted: list[str] = []
    for h in hallucinated:
        alts = alternatives.get(h, [])
        if alts:
            substituted.append(alts[0])  # take the top alternative
    return render_package_response([*valid, *substituted, *fallback_valid])


def pipe_join(values: Iterable[str]) -> str:
    return "|".join(values)


def parse_spracks_prompt_line(line: str, source: str, index: int) -> dict[str, str]:
    value = json.loads(line)
    if isinstance(value, str):
        prompt = value
        prompt_id = f"{source}_{index:05d}"
    elif isinstance(value, dict) and "prompt_text" in value:
        prompt = str(value["prompt_text"])
        prompt_id = str(value.get("prompt_id") or f"{source}_{index:05d}")
    elif isinstance(value, dict) and "0" in value:
        prompt = str(value["0"])
        prompt_id = f"{source}_{index:05d}"
    else:
        raise ValueError(f"Unsupported Spracks prompt row at {source}:{index + 1}")
    return {
        "prompt_id": prompt_id,
        "source": str(value.get("source", source)) if isinstance(value, dict) else source,
        "prompt_text": prompt,
    }


def load_spracks_prompts(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    source = path.stem
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and len(prompts) >= limit:
                break
            line = line.strip()
            if line:
                prompts.append(parse_spracks_prompt_line(line, source, index))
    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts
