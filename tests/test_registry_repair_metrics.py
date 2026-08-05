from pathlib import Path

from registry_repair.metrics import (
    attack_candidate_packages,
    check_pips,
    check_packages,
    evaluate_code_dependencies,
    evaluate_python_row,
    load_false_positives,
    normalize_python,
    parse_package_list,
    parse_pip_install,
    registry_validated_package_response,
)


def test_normalize_python_matches_spracks_style() -> None:
    assert normalize_python("1. Foo_Bar.baz\n") == "foo-bar-baz"
    assert normalize_python(" `Requests.` ") == "requests"


def test_parse_package_list_filters_multiword_entries() -> None:
    assert parse_package_list("requests, fake_pkg, not a package, numpy") == [
        "requests",
        "fake-pkg",
        "numpy",
    ]


def test_pip_detection_uses_spracks_first_token_regex() -> None:
    assert parse_pip_install("python -m pip install fake_pkg==1.0\npip install -r req.txt") == [
        "fake_pkg==1.0"
    ]
    valid, hallucinated = check_pips(["requests>=2.0", "fake_pkg==1.0"], {"requests"})
    assert valid == ["requests"]
    assert hallucinated == ["fake-pkg"]


def test_evaluate_python_row_sums_two_queries_and_pip() -> None:
    evaluation = evaluate_python_row(
        "pip install fake_pkg\nimport requests",
        "requests, invented_pkg",
        "numpy",
        {"requests", "numpy"},
        set(),
    )
    assert evaluation.valid_total == 2
    assert evaluation.hallucinated_total == 2
    assert evaluation.hallucination_rate == 0.5
    assert evaluation.hallucinated_packages == ["invented-pkg", "fake-pkg"]


def test_false_positive_loader_reads_second_column() -> None:
    path = Path("Data/Python/false_positive_packages.csv")
    false_positives = load_false_positives(path)
    assert false_positives


def test_attack_candidates_are_offline_diagnostics_only() -> None:
    candidates = attack_candidate_packages(
        ["invented_pkg", "bad/name", "requests", "known-fp"],
        {"requests"},
        {"known-fp"},
    )
    assert candidates == ["invented-pkg"]


def test_none_is_not_counted_as_hallucinated_package() -> None:
    valid, hallucinated = check_packages(["none", "nan", "requests"], {"requests"}, set())
    assert valid == ["requests"]
    assert hallucinated == []


def test_stdlib_package_names_are_not_counted_as_third_party() -> None:
    valid, hallucinated = check_packages(["json", "subprocess", "requests"], {"json", "subprocess", "requests"}, set())
    assert valid == ["requests"]
    assert hallucinated == []


def test_code_dependency_evaluation_maps_common_import_aliases() -> None:
    evaluation = evaluate_code_dependencies(
        "import numpy as np\nfrom bs4 import BeautifulSoup\nimport made_up_pkg",
        {"numpy", "beautifulsoup4"},
    )
    assert evaluation.syntax_valid
    assert evaluation.valid_import_packages == ["numpy", "beautifulsoup4"]
    assert evaluation.hallucinated_imports == ["made-up-pkg"]
    assert evaluation.needs_repair


def test_registry_validated_package_response_filters_invalid_names() -> None:
    response = registry_validated_package_response(
        "requests, invented_pkg, numpy",
        {"requests", "numpy"},
        set(),
        fallback_valid=["pandas"],
    )
    assert response == "requests, numpy, pandas"
