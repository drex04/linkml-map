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

# Minimal linear-conversion table. Extend as needed.
LINEAR_GREL_CONVERSIONS: dict[tuple[str, str], str] = {
    ("meter", "foot"): "value.toNumber() * 3.28084",
    ("foot", "meter"): "value.toNumber() * 0.3048",
    ("kilogram", "pound"): "value.toNumber() * 2.20462",
    ("celsius", "fahrenheit"): "value.toNumber() * 1.8 + 32",
    ("kelvin", "celsius"): "value.toNumber() - 273.15",
}

_COMPOSITE_SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

_VALID_FORMATS = {"csv", "json", "xml"}


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

    def _grel_for_linear(self, source_unit: str, target_unit: str) -> str:
        """Return GREL expression for linear unit conversion.

        Raises ValueError for unknown pairs.
        """
        key = (source_unit, target_unit)
        if key in LINEAR_GREL_CONVERSIONS:
            return LINEAR_GREL_CONVERSIONS[key]
        raise ValueError(
            f"No linear GREL conversion known for {source_unit} → {target_unit}"
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

                if slot_deriv.unit_conversion is not None:
                    uc = slot_deriv.unit_conversion
                    src_unit = uc.source_unit or ""
                    tgt_unit = uc.target_unit or ""
                    try:
                        grel_expr = self._grel_for_linear(src_unit, tgt_unit)
                        po = {
                            "predicate": predicate,
                            "reference": reference,
                            "function": {
                                "name": "grel:value",
                                "parameters": [
                                    {"name": "value", "value": reference}
                                ],
                                "grel": grel_expr,
                            },
                        }
                    except ValueError:
                        sys.stderr.write(
                            f"[YarrrmlCompiler] WARNING: no GREL conversion for "
                            f"{src_unit!r} → {tgt_unit!r}; emitting plain reference\n"
                        )

                if slot_deriv.range is not None and "function" not in po:
                    po["datatype"] = f"xsd:{slot_deriv.range}"

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
