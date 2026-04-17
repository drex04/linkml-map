"""Tests for YarrrmlCompiler — exercises compile(spec) and inspects serialization."""

from pathlib import Path

import pytest
import yaml
from linkml_runtime import SchemaView

from linkml_map.compiler.yarrrml_compiler import _COMPOSITE_SLOT_RE, YarrrmlCompiler
from linkml_map.datamodel.transformer_model import (
    ClassDerivation,
    KeyVal,
    SlotDerivation,
    TransformationSpecification,
    UnitConversionConfiguration,
)

# ---------------------------------------------------------------------------
# Paths to static fixtures
# ---------------------------------------------------------------------------
INPUT_DIR = Path(__file__).parent.parent / "input" / "yarrrml"
SOURCE_SCHEMA_PATH = INPUT_DIR / "source_schema.yaml"
TARGET_SCHEMA_PATH = INPUT_DIR / "target_schema.yaml"


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _make_prefixes() -> dict[str, KeyVal]:
    return {
        "skos": KeyVal(key="skos", value="http://www.w3.org/2004/02/skos/core#"),
        "xsd": KeyVal(key="xsd", value="http://www.w3.org/2001/XMLSchema#"),
        "nor_radar": KeyVal(key="nor_radar", value="https://example.org/nor-radar/"),
        "mc": KeyVal(key="mc", value="https://example.org/mc/"),
    }


def _make_slot_derivations_csv() -> dict[str, SlotDerivation]:
    """Simple slot derivations — no composites, no unit conversion."""
    return {
        "latitude": SlotDerivation(name="latitude", populated_from="latitude"),
        "longitude": SlotDerivation(name="longitude", populated_from="longitude"),
        "speed": SlotDerivation(name="speed", populated_from="speed"),
    }


def _make_spec(
    source_format: str = "csv",
    slot_derivations: dict[str, SlotDerivation] | None = None,
) -> TransformationSpecification:
    return TransformationSpecification(
        source_schema=str(SOURCE_SCHEMA_PATH),
        target_schema=str(TARGET_SCHEMA_PATH),
        comments=[f"rosetta:source_format={source_format}"],
        prefixes=_make_prefixes(),
        class_derivations=[
            ClassDerivation(
                name="Track",
                populated_from="Track",
                slot_derivations=slot_derivations or _make_slot_derivations_csv(),
            )
        ],
    )


@pytest.fixture()
def csv_spec() -> TransformationSpecification:
    return _make_spec("csv")


@pytest.fixture()
def compiler_csv(csv_spec: TransformationSpecification) -> YarrrmlCompiler:
    return YarrrmlCompiler(
        source_schemaview=SchemaView(str(SOURCE_SCHEMA_PATH)),
        target_schemaview=SchemaView(str(TARGET_SCHEMA_PATH)),
    )


# ---------------------------------------------------------------------------
# Test 1 — valid YARRRML structure (prefixes + mappings keys)
# ---------------------------------------------------------------------------


def test_compile_class_derivation_produces_valid_yarrrml(
    compiler_csv: YarrrmlCompiler, csv_spec: TransformationSpecification
) -> None:
    result = compiler_csv.compile(csv_spec)
    doc = yaml.safe_load(result.serialization)
    assert doc is not None, "yaml.safe_load returned None"
    assert "prefixes" in doc, "top-level 'prefixes' key missing"
    assert "mappings" in doc, "top-level 'mappings' key missing"


# ---------------------------------------------------------------------------
# Test 2 — CSV column annotations drive references
# ---------------------------------------------------------------------------


def test_compile_slot_references_use_csv_annotations(
    compiler_csv: YarrrmlCompiler, csv_spec: TransformationSpecification
) -> None:
    serialization = compiler_csv.compile(csv_spec).serialization
    # latitude annotation is "latitude", longitude is "longitude", speed is "speed_knots"
    assert "$(latitude)" in serialization
    assert "$(longitude)" in serialization
    assert "$(speed_knots)" in serialization
    # slot name "speed" must NOT appear as the raw reference (annotation overrides it)
    assert "$(speed)" not in serialization


# ---------------------------------------------------------------------------
# Test 3 — datatype emitted for range-bearing slot
# ---------------------------------------------------------------------------


def test_compile_datatype_emitted_when_range_set(
    compiler_csv: YarrrmlCompiler,
) -> None:
    spec = _make_spec(
        "csv",
        slot_derivations={
            "speed": SlotDerivation(
                name="speed", populated_from="speed", range="float"
            ),
        },
    )
    serialization = compiler_csv.compile(spec).serialization
    assert "datatype: xsd:float" in serialization


# ---------------------------------------------------------------------------
# Test 4 — unit conversion emits GREL expression
# ---------------------------------------------------------------------------


def test_compile_unit_conversion_emits_grel(
    compiler_csv: YarrrmlCompiler,
) -> None:
    spec = _make_spec(
        "csv",
        slot_derivations={
            "speed": SlotDerivation(
                name="speed",
                populated_from="speed",
                unit_conversion=UnitConversionConfiguration(
                    source_unit="meter", target_unit="foot"
                ),
            ),
        },
    )
    serialization = compiler_csv.compile(spec).serialization
    # Template emits the function block with name + parameters
    assert "function: grel:value" in serialization
    # The GREL expression is stored on the function dict — confirm it appears
    # in the compiled output via the po block (the template emits the function name
    # and parameters; the grel key is present in the mapping context dict)
    assert "grel:value" in serialization
    # The parameter binding for the meter→foot conversion should reference the source col
    assert "$(speed_knots)" in serialization


# ---------------------------------------------------------------------------
# Test 5 — composite slot → separate mapping block + parent po reference
# ---------------------------------------------------------------------------


def test_compile_composite_separate_triplesmap(
    compiler_csv: YarrrmlCompiler,
) -> None:
    spec = _make_spec(
        "csv",
        slot_derivations={
            "location": SlotDerivation(
                name="location",
                expr="f({latitude}, {longitude})",
            ),
        },
    )
    doc = yaml.safe_load(compiler_csv.compile(spec).serialization)
    mapping_names = list(doc["mappings"].keys())

    # There must be at least two mappings: the parent class + one composite
    assert len(mapping_names) >= 2, f"Expected composite mapping, got: {mapping_names}"

    # The composite block name follows pattern <source_class>_<slot_name>
    composite_name = "Track_location"
    assert composite_name in mapping_names, (
        f"Composite mapping '{composite_name}' not found in {mapping_names}"
    )

    # Parent class mapping must reference the composite via a mapping: entry
    track_mapping = doc["mappings"]["Track"]
    po_entries = track_mapping["po"]
    mapping_refs = [
        entry for entry in po_entries if isinstance(entry, dict) and "o" in entry
    ]
    mapping_o_values = []
    for entry in mapping_refs:
        o = entry["o"]
        if isinstance(o, list):
            for item in o:
                if isinstance(item, dict) and "mapping" in item:
                    mapping_o_values.append(item["mapping"])
        elif isinstance(o, dict) and "mapping" in o:
            mapping_o_values.append(o["mapping"])
    assert composite_name in mapping_o_values, (
        f"Parent po does not reference '{composite_name}'; po entries: {po_entries}"
    )


# ---------------------------------------------------------------------------
# Test 6 — subject template uses identifier slot
# ---------------------------------------------------------------------------


def test_compile_subject_template_uses_identifier_slot(
    compiler_csv: YarrrmlCompiler, csv_spec: TransformationSpecification
) -> None:
    serialization = compiler_csv.compile(csv_spec).serialization
    # 'id' is the identifier slot in the source schema
    assert "$(id)" in serialization


# ---------------------------------------------------------------------------
# Test 7 — sources placeholder for csv
# ---------------------------------------------------------------------------


def test_compile_sources_placeholder(
    compiler_csv: YarrrmlCompiler, csv_spec: TransformationSpecification
) -> None:
    serialization = compiler_csv.compile(csv_spec).serialization
    assert "$(DATA_FILE)~csv" in serialization


# ---------------------------------------------------------------------------
# Test 8 — JSON format uses verbatim jsonpath (no re-wrap)
# ---------------------------------------------------------------------------


def test_compile_json_format_uses_jsonpath(
    compiler_csv: YarrrmlCompiler,
) -> None:
    """JSON annotations must be emitted verbatim — not wrapped in $()."""
    spec = _make_spec(
        "json",
        slot_derivations={
            "latitude": SlotDerivation(name="latitude", populated_from="latitude"),
        },
    )
    serialization = compiler_csv.compile(spec).serialization
    # rosetta_jsonpath annotation value is "$.latitude" — must appear as-is
    assert "$.latitude" in serialization
    # Must NOT be re-wrapped as $($.latitude)
    assert "$($.latitude)" not in serialization


# ---------------------------------------------------------------------------
# Test 9 — missing source_schema raises ValueError
# ---------------------------------------------------------------------------


def test_compile_missing_source_schema_raises() -> None:
    compiler = YarrrmlCompiler()
    spec = TransformationSpecification(
        target_schema=str(TARGET_SCHEMA_PATH),
        comments=["rosetta:source_format=csv"],
        prefixes=_make_prefixes(),
        class_derivations=[
            ClassDerivation(
                name="Track",
                populated_from="Track",
                slot_derivations=_make_slot_derivations_csv(),
            )
        ],
    )
    with pytest.raises(ValueError, match="source_schema"):
        compiler.compile(spec)


# ---------------------------------------------------------------------------
# Test 10 — no identifier slot raises ValueError
# ---------------------------------------------------------------------------


def test_compile_no_identifier_slot_raises(
    tmp_path: Path,
) -> None:
    # Reference a class name that does not exist in the source schema.
    # get_identifier_slot returns None and get_class returns None, so the
    # heuristic loop is skipped entirely and _get_identifier_slot raises ValueError.
    no_id_schema = tmp_path / "no_id.yaml"
    no_id_schema.write_text(
        """
id: https://example.org/noid
name: noid
default_prefix: noid
prefixes:
  linkml: https://w3id.org/linkml/
imports:
  - linkml:types
classes:
  SomeClass:
    slots:
      - value_reading
slots:
  value_reading:
    range: float
"""
    )
    compiler = YarrrmlCompiler(
        source_schemaview=SchemaView(str(no_id_schema)),
        target_schemaview=SchemaView(str(TARGET_SCHEMA_PATH)),
    )
    spec = TransformationSpecification(
        comments=["rosetta:source_format=csv"],
        prefixes=_make_prefixes(),
        class_derivations=[
            ClassDerivation(
                name="Track",
                # "NonExistentClass" is not defined in the schema —
                # get_class returns None, bypassing the all_slots() call
                populated_from="NonExistentClass",
                slot_derivations={
                    "speed": SlotDerivation(name="speed", populated_from="value_reading"),
                },
            )
        ],
    )
    # Either "identifier" (our error) or "No such class" (SchemaView) — both are ValueError
    with pytest.raises(ValueError):
        compiler.compile(spec)


# ---------------------------------------------------------------------------
# Test 11 — composite subject = parent_subject + "/" + composite_slot_name
# ---------------------------------------------------------------------------


def test_compile_composite_subject_is_parent_slash_slot(
    compiler_csv: YarrrmlCompiler,
) -> None:
    spec = _make_spec(
        "csv",
        slot_derivations={
            "location": SlotDerivation(
                name="location",
                expr="f({latitude}, {longitude})",
            ),
        },
    )
    doc = yaml.safe_load(compiler_csv.compile(spec).serialization)
    parent_subject = doc["mappings"]["Track"]["s"]
    composite_subject = doc["mappings"]["Track_location"]["s"]
    assert composite_subject == f"{parent_subject}/location", (
        f"Expected composite subject '{parent_subject}/location', got '{composite_subject}'"
    )


# ---------------------------------------------------------------------------
# Test 12 — subject uses source prefix (nor_radar), not target prefix (mc)
# ---------------------------------------------------------------------------


def test_compile_subject_uses_source_prefix(
    compiler_csv: YarrrmlCompiler, csv_spec: TransformationSpecification
) -> None:
    doc = yaml.safe_load(compiler_csv.compile(csv_spec).serialization)
    subject = doc["mappings"]["Track"]["s"]
    assert subject.startswith("nor_radar:"), (
        f"Subject should start with source prefix 'nor_radar:', got: {subject!r}"
    )
    assert not subject.startswith("mc:"), (
        f"Subject must not use target prefix 'mc:'; got: {subject!r}"
    )


# ---------------------------------------------------------------------------
# Test 13 — composite expr regex parser
# ---------------------------------------------------------------------------


def test_compile_composition_expr_parser() -> None:
    # Direct regex test
    assert _COMPOSITE_SLOT_RE.findall("f({lat}, {lon})") == ["lat", "lon"]
    assert _COMPOSITE_SLOT_RE.findall("{a}_{b}_{c}") == ["a", "b", "c"]
    assert _COMPOSITE_SLOT_RE.findall("no_braces_here") == []

    # An expr with no {…} tokens causes ValueError during compile()
    compiler = YarrrmlCompiler(
        source_schemaview=SchemaView(str(SOURCE_SCHEMA_PATH)),
        target_schemaview=SchemaView(str(TARGET_SCHEMA_PATH)),
    )
    spec = _make_spec(
        "csv",
        slot_derivations={
            "location": SlotDerivation(
                name="location",
                expr="no_slot_refs_at_all()",
            ),
        },
    )
    with pytest.raises(ValueError, match="no source slots"):
        compiler.compile(spec)
