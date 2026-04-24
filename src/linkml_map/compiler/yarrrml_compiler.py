"""YARRRML compiler for linkml-map TransformationSpecifications.

Emits YARRRML (https://rml.io/yarrrml/spec/) consumable by morph-kgc.

Composite slot derivations (slot_deriv.expr is not None) are emitted as separate
YARRRML mapping blocks. Member source-slot names are parsed from slot_deriv.expr
using the regex r"\\{([a-zA-Z_][a-zA-Z0-9_]*)\\}".

Example:
    >>> import re
    >>> re.findall(r"\\{([a-zA-Z_][a-zA-Z0-9_]*)\\}", "f({lat}, {lon})")
    ['lat', 'lon']
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from linkml_runtime import SchemaView

from linkml_map.compiler.compiler import CompiledSpecification, Compiler
from linkml_map.datamodel.transformer_model import (
    ClassDerivation,
    TransformationSpecification,
)

YARRRML_TEMPLATE_DIR = str(Path(__file__).parent / "templates")

# Linear-conversion function IDs — stable IRIs under the rosetta UDF namespace.
# The compiler emits unit_conversion as a FnML function reference by IRI. The
# downstream engine (morph-kgc) must register each IRI as a user-defined
# function (UDF) at materialize time. rosetta-cli's rml_runner writes a Python
# UDF file to work_dir and passes `udfs=<path>` in morph-kgc's INI config;
# see rosetta/core/rml_runner.py::_write_udf_file.
_ROSETTA_UDF_NS = "https://rosetta.interop/udf/"
LINEAR_CONVERSION_FUN_IDS: dict[tuple[str, str], str] = {
    ("meter", "foot"): _ROSETTA_UDF_NS + "meter_to_foot",
    ("foot", "meter"): _ROSETTA_UDF_NS + "foot_to_meter",
    ("kilogram", "pound"): _ROSETTA_UDF_NS + "kilogram_to_pound",
    ("celsius", "fahrenheit"): _ROSETTA_UDF_NS + "celsius_to_fahrenheit",
    ("kelvin", "celsius"): _ROSETTA_UDF_NS + "kelvin_to_celsius",
}

_COMPOSITE_SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

_VALID_FORMATS = {"csv", "json", "xml"}

_STRING_TYPES = {"xsd:string", "xsd:anyURI", "xsd:uriorcurie"}


def _resolve_xsd_datatype(slot_name: str, target_view: SchemaView) -> str | None:
    """Return the XSD datatype CURIE for a target slot, or None for string/class ranges."""
    slot = target_view.get_slot(slot_name)
    if slot is None or not slot.range:
        return None
    type_def = target_view.get_type(slot.range)
    if type_def is None or not type_def.uri:
        return None
    uri = str(type_def.uri)
    if uri in _STRING_TYPES:
        return None
    return uri


@dataclass
class YarrrmlCompiler(Compiler):
    """Compiles a TransformationSpecification to YARRRML."""

    source_schemaview: SchemaView | None = field(default=None)  # type: ignore[assignment]
    target_schemaview: SchemaView | None = field(default=None)

    def _resolve_schemas(
        self, spec: TransformationSpecification
    ) -> tuple[SchemaView, SchemaView]:
        """Resolve source and target SchemaView instances.

        Prefers constructor overrides; falls back to spec fields.
        Raises ValueError if neither is available.
        """
        source_view: SchemaView | None = self.source_schemaview
        if source_view is None:
            if spec.source_schema:
                source_view = SchemaView(spec.source_schema)
            else:
                raise ValueError(
                    "source_schema not set on TransformSpec and not provided via constructor"
                )

        target_view: SchemaView | None = self.target_schemaview
        if target_view is None:
            if spec.target_schema:
                target_view = SchemaView(spec.target_schema)
            else:
                raise ValueError(
                    "target_schema not set on TransformSpec and not provided via constructor"
                )

        return source_view, target_view

    def _resolve_source_format(self, spec: TransformationSpecification) -> str:
        """Scan spec.comments for rosetta:source_format=<fmt> and return <fmt>.

        Raises ValueError if not found.
        """
        prefix = "rosetta:source_format="
        for comment in spec.comments or []:
            if comment.startswith(prefix):
                fmt = comment[len(prefix):]
                return fmt
        raise ValueError("No rosetta:source_format=<fmt> in spec.comments")

    def _get_identifier_slot(self, class_name: str, source_view: SchemaView) -> str:
        """Return identifier slot name for class_name.

        Steps:
        a. Use get_identifier_slot if available.
        b. Heuristic: look for id / identifier / {class_name}_id / {class_name_lower}_id.
        c. Raise ValueError if nothing found.
        """
        id_slot = source_view.get_identifier_slot(class_name)
        if id_slot is not None:
            return id_slot.name  # type: ignore[union-attr]

        # Heuristic fallback
        candidates = {
            "id",
            "identifier",
            f"{class_name}_id",
            f"{class_name.lower()}_id",
        }
        cls = source_view.get_class(class_name)
        if cls is not None:
            for slot_name in source_view.class_slots(class_name):
                if slot_name.lower() in {c.lower() for c in candidates}:
                    sys.stderr.write(
                        f"[YarrrmlCompiler] WARNING: using heuristic identifier slot "
                        f"'{slot_name}' for class '{class_name}'\n"
                    )
                    return slot_name

        raise ValueError(f"No identifier slot found for class {class_name}")

    def _fun_id_for_linear(self, source_unit: str, target_unit: str) -> str:
        """Return stable rosetta UDF IRI for a linear unit-conversion pair.

        The IRI references a user-defined function the downstream engine
        (morph-kgc) must have registered. rosetta-cli's rml_runner writes a
        Python UDF file at materialize time whose @udf-decorated functions
        use matching fun_ids.
        """
        key = (source_unit, target_unit)
        if key in LINEAR_CONVERSION_FUN_IDS:
            return LINEAR_CONVERSION_FUN_IDS[key]
        raise ValueError(
            f"No linear conversion function registered for {source_unit} → {target_unit}"
        )

    def _sources_entry(self, fmt: str) -> list[str]:
        """Return YARRRML sources list for the given format."""
        if fmt == "csv":
            return ["$(DATA_FILE)~csv"]
        elif fmt == "json":
            return ["$(DATA_FILE)~jsonpath", "$.[*]"]
        elif fmt == "xml":
            return ["$(DATA_FILE)~xpath", "/*/*"]
        else:
            raise ValueError(f"Unsupported source format: {fmt!r}")

    def _resolve_reference(
        self, slot_name: str, source_view: SchemaView, fmt: str
    ) -> str:
        """Look up the annotation-based reference for a slot in the given format.

        For csv: reads annotations.rosetta_csv_column → wraps as $(column).
        For json: reads annotations.rosetta_jsonpath → returns verbatim.
        For xml: reads annotations.rosetta_xpath → returns verbatim.

        Falls back with stderr warning for csv/json; raises for xml.
        """
        slot = source_view.get_slot(slot_name)
        annotations: Any = getattr(slot, "annotations", None) if slot else None

        def _get_ann(key: str) -> str | None:
            if not annotations:
                return None
            ann = annotations.get(key) if isinstance(annotations, dict) else None
            return getattr(ann, "value", None)

        if fmt == "csv":
            col = _get_ann("rosetta_csv_column")
            if col:
                return f"$({col})"
            sys.stderr.write(
                f"[YarrrmlCompiler] WARNING: no rosetta_csv_column annotation for "
                f"slot '{slot_name}'; falling back to $({slot_name})\n"
            )
            return f"$({slot_name})"

        elif fmt == "json":
            path = _get_ann("rosetta_jsonpath")
            if path:
                return path  # verbatim
            sys.stderr.write(
                f"[YarrrmlCompiler] WARNING: no rosetta_jsonpath annotation for "
                f"slot '{slot_name}'; falling back to $.{slot_name}\n"
            )
            return f"$.{slot_name}"

        elif fmt == "xml":
            xpath = _get_ann("rosetta_xpath")
            if xpath:
                return xpath  # verbatim
            raise ValueError(
                f"No rosetta_xpath annotation for slot '{slot_name}' and no safe "
                f"XPath fallback exists"
            )

        else:
            raise ValueError(f"Unsupported format for reference resolution: {fmt!r}")

    def _build_mapping_context(
        self,
        class_derivation: ClassDerivation,
        source_view: SchemaView,
        target_view: SchemaView,
        source_format: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Build the YARRRML mapping dict for a ClassDerivation.

        Returns (class_mapping, composite_mappings).
        """
        target_class_name = class_derivation.name
        # populated_from is the source class name; fall back to target name
        source_class_name = class_derivation.populated_from or target_class_name

        # Determine prefixes
        source_schema = source_view.schema
        target_schema = target_view.schema
        source_prefix = getattr(source_schema, "default_prefix", None) or (
            source_schema.name or "source"
        )
        target_prefix = getattr(target_schema, "default_prefix", None) or (
            target_schema.name or "target"
        )

        id_slot = self._get_identifier_slot(source_class_name, source_view)
        subject_template = f"{source_prefix}:{source_class_name}/$({id_slot})"
        sources = self._sources_entry(source_format)

        # rdf:type — use target class URI if available
        target_cls = target_view.get_class(target_class_name)
        rdf_type: str | None = None
        if target_cls is not None:
            class_uri = target_view.get_uri(target_cls, expand=False)
            if class_uri:
                rdf_type = str(class_uri)
            else:
                rdf_type = f"{target_prefix}:{target_class_name}"

        predicateobjects: list[dict[str, Any]] = []
        composite_mappings: list[dict[str, Any]] = []

        slot_derivations = class_derivation.slot_derivations or {}

        for slot_name, slot_deriv in slot_derivations.items():
            # Determine target predicate URI
            target_slot = target_view.get_slot(slot_deriv.name)
            if target_slot is not None and target_slot.slot_uri:
                predicate = str(target_slot.slot_uri)
            else:
                predicate = f"{target_prefix}:{slot_deriv.name}"

            if slot_deriv.expr is not None:
                # --- Composite slot ---
                members = _COMPOSITE_SLOT_RE.findall(slot_deriv.expr)
                if not members:
                    raise ValueError(
                        f"composite slot '{slot_deriv.name}' has expr that references "
                        f"no source slots: {slot_deriv.expr!r}"
                    )

                composite_name = f"{source_class_name}_{slot_deriv.name}"
                composite_subject = f"{subject_template}/{slot_deriv.name}"
                composite_pos: list[dict[str, Any]] = []

                for member in members:
                    member_ref = self._resolve_reference(
                        member, source_view, source_format
                    )
                    member_predicate = f"{source_prefix}:{slot_deriv.name}_{member}"
                    composite_pos.append(
                        {"predicate": member_predicate, "reference": member_ref}
                    )

                composite_mappings.append(
                    {
                        "name": composite_name,
                        "sources": sources,
                        "subject": composite_subject,
                        "predicateobjects": composite_pos,
                    }
                )

                # Add a mapping-reference po entry to the parent class
                predicateobjects.append(
                    {"predicate": predicate, "mapping": composite_name}
                )

            else:
                # --- Simple slot ---
                source_slot_name = slot_deriv.populated_from or slot_deriv.name
                reference = self._resolve_reference(
                    source_slot_name, source_view, source_format
                )

                po: dict[str, Any] = {"predicate": predicate, "reference": reference}

                xsd_type = _resolve_xsd_datatype(slot_deriv.name, target_view)

                if slot_deriv.unit_conversion is not None:
                    uc = slot_deriv.unit_conversion
                    src_unit = uc.source_unit or ""
                    tgt_unit = uc.target_unit or ""
                    try:
                        fun_id = self._fun_id_for_linear(src_unit, tgt_unit)
                        po = {
                            "predicate": predicate,
                            "reference": reference,
                            "function": {
                                # morph-kgc resolves this as a full IRI when
                                # it contains "://" — no angle-bracket wrap.
                                "name": fun_id,
                                "parameters": [
                                    {
                                        "name": "grel:valueParameter",
                                        "value": reference,
                                    }
                                ],
                            },
                        }
                        if xsd_type is not None:
                            po["function"]["datatype"] = xsd_type
                    except ValueError:
                        sys.stderr.write(
                            f"[YarrrmlCompiler] WARNING: no conversion function "
                            f"registered for {src_unit!r} → {tgt_unit!r}; "
                            f"emitting plain reference\n"
                        )

                if xsd_type is not None and "function" not in po:
                    po["datatype"] = xsd_type

                predicateobjects.append(po)

        class_mapping: dict[str, Any] = {
            "name": source_class_name,
            "sources": sources,
            "subject": subject_template,
            "predicateobjects": predicateobjects,
        }
        if rdf_type:
            class_mapping["rdf_type"] = rdf_type

        return class_mapping, composite_mappings

    def compile(self, specification: TransformationSpecification) -> CompiledSpecification:
        """Compile a TransformationSpecification to a YARRRML document."""
        source_view, target_view = self._resolve_schemas(specification)
        source_format = self._resolve_source_format(specification)

        mappings: list[dict[str, Any]] = []
        for cd in specification.class_derivations or []:
            class_mapping, composite_mappings = self._build_mapping_context(
                cd, source_view, target_view, source_format
            )
            mappings.append(class_mapping)
            mappings.extend(composite_mappings)

        prefixes = {
            k: getattr(v, "value", str(v))
            for k, v in (specification.prefixes or {}).items()
        }
        context: dict[str, Any] = {"prefixes": prefixes, "mappings": mappings}

        env = Environment(
            loader=FileSystemLoader(YARRRML_TEMPLATE_DIR),
            autoescape=False,
        )
        rendered = env.get_template("yarrrml.j2").render(**context)
        return CompiledSpecification(serialization=rendered)
