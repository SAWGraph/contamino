from __future__ import annotations

import argparse
import hashlib
import html
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Iterable

from owlrl import DeductiveClosure, OWLRL_Semantics
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, SKOS


@dataclass
class RestrictionInfo:
    source: str
    property_uri: URIRef | BNode | None
    property_label: str
    kind: str
    value_label: str
    value_uri: URIRef | None = None
    raw_value: str | None = None


@dataclass
class EquivalentExpression:
    source: str
    operator: str
    members: list[str]


@dataclass
class NodeInfo:
    uri: URIRef
    label: str
    sources: set[str] = field(default_factory=set)
    parents: set[URIRef] = field(default_factory=set)
    children: set[URIRef] = field(default_factory=set)
    alignment_targets: DefaultDict[str, set[URIRef]] = field(default_factory=lambda: defaultdict(set))
    outgoing_rules: DefaultDict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    incoming_rules: DefaultDict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    incoming_instances: DefaultDict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    restrictions: DefaultDict[str, list[RestrictionInfo]] = field(default_factory=lambda: defaultdict(list))
    equivalent_expressions: DefaultDict[str, list[EquivalentExpression]] = field(default_factory=lambda: defaultdict(list))


def local_name(term: URIRef | BNode | Literal) -> str:
    if isinstance(term, Literal):
        return str(term)
    text = str(term)
    for separator in ("#", "/", ":"):
        if separator in text:
            text = text.rsplit(separator, 1)[-1]
    return text


def label_for(graph: Graph, node: URIRef | BNode) -> str:
    for predicate in (RDFS.label, SKOS.prefLabel):
        for value in graph.objects(node, predicate):
            if isinstance(value, Literal):
                return str(value)
    return local_name(node)


MATERIAL_ENTITY_IRI = URIRef("http://purl.obolibrary.org/obo/BFO_0000040")
COSO_MATERIAL_SAMPLE_IRI = URIRef("http://w3id.org/coso/v1/contaminoso#MaterialSample")


def namespace_prefix_for_uri(uri: URIRef) -> str:
    text = str(uri).lower()
    if "foodon_" in text:
        return "FOODON"
    if "egad" in text:
        return "EGAD"
    if "wqp" in text or "us-wqp" in text:
        return "WQP"
    if "contaminoso" in text or "coso" in text:
        return "COSO"
    return local_name(uri).split("_")[0].upper() or "NS"


def namespace_class_for_uri(uri: URIRef) -> str:
    return namespace_prefix_for_uri(uri).lower()


def source_namespace_prefix(source: str) -> str:
    if source in {"merged", "merged_reasoned", "reasoned", "inferred"}:
        return ""
    if source == "egad_alignment":
        return "EGAD"
    if source == "wqp_alignment":
        return "WQP"
    if source == "contaminoso_materialSample_ext":
        return "COSO"
    if source == "foodon_lite_updated":
        return "FOODON"
    return source.split("_")[0].upper() or "NS"


def source_namespace_for_uri(uri: URIRef) -> str:
    text = str(uri).lower()
    if "us-wqp" in text or "wqp" in text:
        return "WQP"
    if "egad" in text:
        return "EGAD"
    if "contaminoso" in text or "coso" in text:
        return "COSO"
    if "foodon" in text:
        return "FOODON"
    return "NS"


def namespace_prefixes_in_view(nodes: dict[URIRef, NodeInfo], selected: set[URIRef]) -> list[str]:
    prefixes: set[str] = set()
    visited: set[URIRef] = set()

    def visit(uri: URIRef) -> None:
        if uri in visited or uri not in nodes:
            return
        visited.add(uri)
        node = nodes[uri]
        prefixes.add(namespace_prefix_for_uri(uri))
        prefixes.update(source_namespace_prefix(source) for source in node.incoming_instances)
        prefixes.update(source_namespace_prefix(source) for source in node.restrictions)
        for child in node.children & selected:
            visit(child)

    for root in choose_roots(nodes, selected):
        visit(root)

    return sorted(prefixes)


def merge_graphs(graphs: Iterable[Graph]) -> Graph:
    merged = Graph()
    for graph in graphs:
        for triple in graph:
            merged.add(triple)
    return merged


def property_label(graph: Graph, predicate: URIRef | BNode | None) -> str:
    if isinstance(predicate, URIRef):
        return label_for(graph, predicate)
    return local_name(predicate) if predicate is not None else "related to"


CARDINALITY_PREDICATES = {
    OWL.minCardinality: "min",
    OWL.maxCardinality: "max",
    OWL.cardinality: "exactly",
    OWL.minQualifiedCardinality: "min qualified",
    OWL.maxQualifiedCardinality: "max qualified",
    OWL.qualifiedCardinality: "exactly qualified",
}

VALUE_PREDICATES = {
    OWL.someValuesFrom: "some",
    OWL.allValuesFrom: "only",
    OWL.hasValue: "value",
    OWL.onClass: "onClass",
}


def expression_label(
    structure_graph: Graph,
    label_graph: Graph,
    node: URIRef | BNode | Literal | None,
) -> str:
    if node is None:
        return "?"
    if isinstance(node, Literal):
        return str(node)
    if isinstance(node, URIRef):
        return label_for(label_graph, node)

    union_list = structure_graph.value(node, OWL.unionOf)
    if isinstance(union_list, BNode):
        members = [
            expression_label(structure_graph, label_graph, item)
            for item in Collection(structure_graph, union_list)
        ]
        members = [m for m in members if m and m != "?"]
        if members:
            return "(" + " or ".join(members) + ")"

    intersection_list = structure_graph.value(node, OWL.intersectionOf)
    if isinstance(intersection_list, BNode):
        members = [
            expression_label(structure_graph, label_graph, item)
            for item in Collection(structure_graph, intersection_list)
        ]
        members = [m for m in members if m and m != "?"]
        if members:
            return "(" + " and ".join(members) + ")"

    if (node, RDF.type, OWL.Restriction) in structure_graph:
        prop = structure_graph.value(node, OWL.onProperty)
        prop_text = property_label(label_graph, prop)

        for predicate, kind in (
            (OWL.hasValue, "value"),
            (OWL.someValuesFrom, "some"),
            (OWL.allValuesFrom, "only"),
        ):
            value = structure_graph.value(node, predicate)
            if value is not None:
                return f"{prop_text} {kind} {expression_label(structure_graph, label_graph, value)}"

        for predicate, kind in CARDINALITY_PREDICATES.items():
            value = structure_graph.value(node, predicate)
            if value is not None:
                on_class = structure_graph.value(node, OWL.onClass)
                if on_class is not None:
                    return f"{prop_text} {kind} {value} {expression_label(structure_graph, label_graph, on_class)}"
                return f"{prop_text} {kind} {value}"

    return "[anonymous class]"


def object_label(
    structure_graph: Graph,
    label_graph: Graph,
    obj: URIRef | BNode | Literal | None,
) -> str:
    return expression_label(structure_graph, label_graph, obj)


def manchester_expr(
    structure_graph: Graph,
    label_graph: Graph,
    node: URIRef | BNode | Literal | None,
) -> str:
    if node is None:
        return "?"
    if isinstance(node, Literal):
        return str(node)
    if isinstance(node, URIRef):
        return label_for(label_graph, node)

    union_list = structure_graph.value(node, OWL.unionOf)
    if isinstance(union_list, BNode):
        members = [
            manchester_expr(structure_graph, label_graph, item)
            for item in Collection(structure_graph, union_list)
        ]
        members = [m for m in members if m and m != "?"]
        return " or ".join(members)

    intersection_list = structure_graph.value(node, OWL.intersectionOf)
    if isinstance(intersection_list, BNode):
        members = [
            manchester_expr(structure_graph, label_graph, item)
            for item in Collection(structure_graph, intersection_list)
        ]
        members = [m for m in members if m and m != "?"]
        return " and ".join(members)

    if (node, RDF.type, OWL.Restriction) in structure_graph:
        prop = structure_graph.value(node, OWL.onProperty)
        prop_text = property_label(label_graph, prop)

        some_value = structure_graph.value(node, OWL.someValuesFrom)
        if some_value is not None:
            return f"{prop_text} some {manchester_expr(structure_graph, label_graph, some_value)}"

        all_value = structure_graph.value(node, OWL.allValuesFrom)
        if all_value is not None:
            return f"{prop_text} only {manchester_expr(structure_graph, label_graph, all_value)}"

        has_value = structure_graph.value(node, OWL.hasValue)
        if has_value is not None:
            return f"{prop_text} value {manchester_expr(structure_graph, label_graph, has_value)}"

        for predicate, kind in CARDINALITY_PREDICATES.items():
            value = structure_graph.value(node, predicate)
            if value is not None:
                on_class = structure_graph.value(node, OWL.onClass)
                if on_class is not None:
                    return f"{prop_text} {kind} {value} {manchester_expr(structure_graph, label_graph, on_class)}"
                return f"{prop_text} {kind} {value}"

    return "[anonymous class]"


def parse_restrictions(
    label_graph: Graph,
    graph: Graph,
    subject: URIRef,
    expression: URIRef | BNode,
    source: str,
) -> list[RestrictionInfo]:
    restrictions: list[RestrictionInfo] = []

    if isinstance(expression, URIRef) and (expression, RDF.type, OWL.Restriction) not in graph:
        pass
    elif (expression, RDF.type, OWL.Restriction) in graph:
        prop = graph.value(expression, OWL.onProperty)
        prop_text = property_label(label_graph, prop)

        for predicate, kind in VALUE_PREDICATES.items():
            for value in graph.objects(expression, predicate):
                restrictions.append(
                    RestrictionInfo(
                        source=source,
                        property_uri=prop,
                        property_label=prop_text,
                        kind=kind,
                        value_label=manchester_expr(graph, label_graph, value),
                        value_uri=value if isinstance(value, URIRef) else None,
                        raw_value=str(value),
                    )
                )

        for predicate, kind in CARDINALITY_PREDICATES.items():
            for value in graph.objects(expression, predicate):
                on_class = graph.value(expression, OWL.onClass)
                value_text = str(value)
                value_uri: URIRef | None = None
                if on_class is not None:
                    value_uri = on_class if isinstance(on_class, URIRef) else None
                    value_text = f"{value} {label_for(label_graph, on_class)}"
                restrictions.append(
                    RestrictionInfo(
                        source=source,
                        property_uri=prop,
                        property_label=prop_text,
                        kind=kind,
                        value_label=value_text,
                        value_uri=value_uri,
                        raw_value=str(value),
                    )
                )

        for value in graph.objects(expression, OWL.hasSelf):
            restrictions.append(
                RestrictionInfo(
                    source=source,
                    property_uri=prop,
                    property_label=prop_text,
                    kind="self",
                    value_label=str(value),
                    raw_value=str(value),
                )
            )

    union_list = graph.value(expression, OWL.unionOf)
    if isinstance(union_list, BNode):
        union_items = list(Collection(graph, union_list))
        grouped_values: dict[tuple[str, str], list[RestrictionInfo]] = defaultdict(list)

        for item in union_items:
            if isinstance(item, BNode):
                nested_restrictions = parse_restrictions(label_graph, graph, subject, item, source)
                for r in nested_restrictions:
                    if r.kind == "value":
                        grouped_values[(r.property_label, r.kind)].append(r)
                    else:
                        restrictions.append(r)
            elif isinstance(item, URIRef):
                restrictions.append(
                    RestrictionInfo(
                        source=source,
                        property_uri=None,
                        property_label="union",
                        kind="class",
                        value_label=label_for(label_graph, item),
                        value_uri=item,
                        raw_value=str(item),
                    )
                )

        for (prop_label, kind), grouped in grouped_values.items():
            unique_vals = []
            seen_vals = set()
            for r in grouped:
                key = (r.value_label, r.raw_value)
                if key in seen_vals:
                    continue
                seen_vals.add(key)
                unique_vals.append(r)

            if len(unique_vals) == 1:
                restrictions.append(unique_vals[0])
            else:
                restrictions.append(
                    RestrictionInfo(
                        source=source,
                        property_uri=unique_vals[0].property_uri,
                        property_label=prop_label,
                        kind="value",
                        value_label=" or ".join(r.value_label for r in unique_vals),
                        value_uri=None,
                        raw_value="|".join((r.raw_value or r.value_label) for r in unique_vals),
                    )
                )

    intersection_list = graph.value(expression, OWL.intersectionOf)
    if isinstance(intersection_list, BNode):
        for item in Collection(graph, intersection_list):
            if isinstance(item, BNode):
                restrictions.extend(parse_restrictions(label_graph, graph, subject, item, source))
            elif isinstance(item, URIRef):
                restrictions.append(
                    RestrictionInfo(
                        source=source,
                        property_uri=None,
                        property_label="intersection",
                        kind="class",
                        value_label=label_for(label_graph, item),
                        value_uri=item,
                        raw_value=str(item),
                    )
                )

    return restrictions


def collect_restriction_targets(
    graph: Graph,
    node: URIRef | BNode,
    seen: set[URIRef | BNode] | None = None,
) -> set[URIRef]:
    seen = seen or set()
    if node in seen:
        return set()
    seen.add(node)

    targets: set[URIRef] = set()
    for predicate, obj in graph.predicate_objects(node):
        if predicate in {OWL.someValuesFrom, OWL.hasValue, OWL.onClass, OWL.allValuesFrom} and isinstance(obj, URIRef):
            targets.add(obj)
        elif predicate in {OWL.intersectionOf, OWL.unionOf} and isinstance(obj, BNode):
            for item in Collection(graph, obj):
                if isinstance(item, URIRef):
                    targets.add(item)
                elif isinstance(item, BNode):
                    targets.update(collect_restriction_targets(graph, item, seen))
        elif isinstance(obj, BNode):
            targets.update(collect_restriction_targets(graph, obj, seen))
    return targets


def collect_equivalent_expression(
    structure_graph: Graph,
    label_graph: Graph,
    node: URIRef | BNode,
    source: str,
) -> EquivalentExpression | None:
    union_list = structure_graph.value(node, OWL.unionOf)
    if isinstance(union_list, BNode):
        members = [
            manchester_expr(structure_graph, label_graph, item)
            for item in Collection(structure_graph, union_list)
        ]
        members = [m for m in members if m and m != "?"]
        if members:
            return EquivalentExpression(source=source, operator="or", members=members)

    intersection_list = structure_graph.value(node, OWL.intersectionOf)
    if isinstance(intersection_list, BNode):
        members = [
            manchester_expr(structure_graph, label_graph, item)
            for item in Collection(structure_graph, intersection_list)
        ]
        members = [m for m in members if m and m != "?"]
        if members:
            return EquivalentExpression(source=source, operator="and", members=members)

    if (node, RDF.type, OWL.Restriction) in structure_graph:
        text = manchester_expr(structure_graph, label_graph, node)
        if text and text != "[anonymous class]":
            return EquivalentExpression(source=source, operator="", members=[text])

    return None


def node_dom_id(uri: URIRef) -> str:
    digest = hashlib.sha1(str(uri).encode("utf-8")).hexdigest()[:10]
    return f"node-{digest}"


def existing_path(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_node(nodes: dict[URIRef, NodeInfo], graph: Graph, uri: URIRef, source: str) -> NodeInfo:
    node = nodes.get(uri)
    if node is None:
        node = NodeInfo(uri=uri, label=label_for(graph, uri))
        nodes[uri] = node
    node.sources.add(source)
    if node.label == local_name(uri):
        node.label = label_for(graph, uri)
    return node


def class_terms(graph: Graph) -> set[URIRef]:
    terms: set[URIRef] = set()

    for predicate in (
        RDF.type,
        RDFS.subClassOf,
        OWL.equivalentClass,
        OWL.intersectionOf,
        OWL.unionOf,
    ):
        for subject in graph.subjects(predicate, None):
            if isinstance(subject, URIRef):
                terms.add(subject)

    for subject in graph.subjects(RDF.type, OWL.Class):
        if isinstance(subject, URIRef):
            terms.add(subject)

    for subject in graph.subjects(RDF.type, OWL.Restriction):
        if isinstance(subject, URIRef):
            terms.add(subject)

    return terms


def preferred_parent(nodes: dict[URIRef, NodeInfo], uri: URIRef, selected: set[URIRef]) -> URIRef | None:
    parents = list(nodes[uri].parents & selected)
    if not parents:
        return None

    coso_parents = [p for p in parents if namespace_prefix_for_uri(p) == "COSO"]
    if coso_parents:
        return sorted(coso_parents, key=lambda p: nodes[p].label.lower())[0]

    return sorted(parents, key=lambda p: nodes[p].label.lower())[0]


def add_restrictions_to_node(
    nodes: dict[URIRef, NodeInfo],
    graph: Graph,
    label_graph: Graph,
    subject: URIRef,
    expression: URIRef | BNode,
    source: str,
) -> None:
    for r in parse_restrictions(label_graph, graph, subject, expression, source):
        nodes[subject].restrictions[source].append(r)

        if r.property_label and r.value_label:
            rule_text = f"{label_for(label_graph, subject)}: {r.property_label} {r.kind} {r.value_label}"
            nodes[subject].outgoing_rules[source].append(rule_text)
            if r.value_uri is not None:
                ensure_node(nodes, graph, r.value_uri, source)
                nodes[r.value_uri].incoming_rules[source].append(rule_text)


def build_structure(graph: Graph, label_graph: Graph, source: str, nodes: dict[URIRef, NodeInfo]) -> None:
    for subject in class_terms(graph):
        ensure_node(nodes, graph, subject, source)

    for subject in {s for s in graph.subjects(RDF.type, OWL.Restriction) if isinstance(s, URIRef)}:
        ensure_node(nodes, graph, subject, source)
        add_restrictions_to_node(nodes, graph, label_graph, subject, subject, source)

    for subject, parent in graph.subject_objects(RDFS.subClassOf):
        if not isinstance(subject, URIRef):
            continue
        ensure_node(nodes, graph, subject, source)
        if isinstance(parent, URIRef):
            ensure_node(nodes, graph, parent, source)
            nodes[subject].parents.add(parent)
            nodes[parent].children.add(subject)
        elif isinstance(parent, BNode):
            add_restrictions_to_node(nodes, graph, label_graph, subject, parent, source)

    for subject, expression in graph.subject_objects(OWL.equivalentClass):
        if not isinstance(subject, URIRef):
            continue
        ensure_node(nodes, graph, subject, source)
        if isinstance(expression, URIRef):
            ensure_node(nodes, graph, expression, source)
            nodes[subject].alignment_targets[source].add(expression)
            nodes[subject].equivalent_expressions[source].append(
                EquivalentExpression(
                    source=source,
                    operator="",
                    members=[label_for(label_graph, expression)],
                )
            )
        elif isinstance(expression, BNode):
            expr = collect_equivalent_expression(graph, label_graph, expression, source)
            if expr is not None:
                nodes[subject].equivalent_expressions[source].append(expr)
            targets = collect_restriction_targets(graph, expression)
            nodes[subject].alignment_targets[source].update(targets)
            add_restrictions_to_node(nodes, graph, label_graph, subject, expression, source)

    for subject in class_terms(graph):
        for expression in graph.objects(subject, OWL.intersectionOf):
            if not isinstance(expression, BNode):
                continue
            for item in Collection(graph, expression):
                if isinstance(item, URIRef):
                    ensure_node(nodes, graph, item, source)
                    nodes[subject].alignment_targets[source].add(item)
                elif isinstance(item, BNode):
                    targets = collect_restriction_targets(graph, item)
                    nodes[subject].alignment_targets[source].update(targets)
                    add_restrictions_to_node(nodes, graph, label_graph, subject, item, source)

        for expression in graph.objects(subject, OWL.unionOf):
            if not isinstance(expression, BNode):
                continue
            for item in Collection(graph, expression):
                if isinstance(item, URIRef):
                    ensure_node(nodes, graph, item, source)
                elif isinstance(item, BNode):
                    targets = collect_restriction_targets(graph, item)
                    nodes[subject].alignment_targets[source].update(targets)
                    add_restrictions_to_node(nodes, graph, label_graph, subject, item, source)


def add_inferred_instances(graph: Graph, label_graph: Graph, nodes: dict[URIRef, NodeInfo]) -> None:
    class_like_targets = {target for target in graph.subjects(RDF.type, OWL.Class) if isinstance(target, URIRef)}
    type_sets: dict[URIRef, set[URIRef]] = defaultdict(set)

    for subject, target in graph.subject_objects(RDF.type):
        if not isinstance(subject, URIRef) or not isinstance(target, URIRef):
            continue
        if target in {OWL.Class, OWL.Restriction, OWL.Ontology}:
            continue
        if subject in class_like_targets:
            continue
        type_sets[subject].add(target)

    for subject, targets in type_sets.items():
        minimal_targets: set[URIRef] = set()
        for candidate in targets:
            if not any(
                other != candidate and candidate in set(graph.transitive_objects(other, RDFS.subClassOf))
                for other in targets
            ):
                minimal_targets.add(candidate)

        source = source_namespace_for_uri(subject).lower()
        ensure_node(nodes, label_graph, subject, source)
        subject_label = label_for(label_graph, subject)

        for target in minimal_targets:
            ensure_node(nodes, label_graph, target, source)
            nodes[target].incoming_instances[source].add(subject_label)


def collect_alignment_targets(graph: Graph) -> dict[URIRef, set[URIRef]]:
    targets_by_subject: dict[URIRef, set[URIRef]] = {}
    for subject in {s for s in graph.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}:
        targets: set[URIRef] = set()
        for expression in graph.objects(subject, OWL.equivalentClass):
            if isinstance(expression, URIRef):
                targets.add(expression)
            elif isinstance(expression, BNode):
                targets.update(collect_restriction_targets(graph, expression))
        for expression in graph.objects(subject, OWL.intersectionOf):
            if isinstance(expression, BNode):
                for item in Collection(graph, expression):
                    if isinstance(item, URIRef):
                        targets.add(item)
                    elif isinstance(item, BNode):
                        targets.update(collect_restriction_targets(graph, item))
        for expression in graph.objects(subject, OWL.unionOf):
            if isinstance(expression, BNode):
                for item in Collection(graph, expression):
                    if isinstance(item, URIRef):
                        targets.add(item)
                    elif isinstance(item, BNode):
                        targets.update(collect_restriction_targets(graph, item))
        if targets:
            targets_by_subject[subject] = targets
    return targets_by_subject


def choose_roots(nodes: dict[URIRef, NodeInfo], selected: set[URIRef]) -> list[URIRef]:
    preferred = [
        candidate
        for candidate in (MATERIAL_ENTITY_IRI, COSO_MATERIAL_SAMPLE_IRI)
        if candidate in selected
    ]

    disconnected = [
        uri for uri in selected
        if not (nodes[uri].parents & selected)
        and uri not in preferred
        and (
            nodes[uri].incoming_instances
            or nodes[uri].equivalent_expressions
            or nodes[uri].restrictions
            or (nodes[uri].children & selected)
        )
    ]

    return preferred + sorted(disconnected, key=lambda uri: nodes[uri].label.lower())


def render_tree(nodes: dict[URIRef, NodeInfo], selected: set[URIRef], shared_targets: set[URIRef]) -> str:
    
    def visible_children(uri: URIRef) -> list[URIRef]:
        return sorted(nodes[uri].children & selected, key=lambda item: nodes[item].label.lower())

    def is_leaf(uri: URIRef) -> bool:
        return not visible_children(uri)

    def render_node(uri: URIRef, path: set[URIRef]) -> str:
        node = nodes[uri]
        next_path = set(path)
        next_path.add(uri)
        prefix = namespace_prefix_for_uri(uri)
        prefix_class = namespace_class_for_uri(uri)
        overlap_tag = " <span class='tag overlap'>shared</span>" if uri in shared_targets else ""

        restriction_summary = ""
        if node.restrictions:
            lines = []
            for source, restrictions in sorted(node.restrictions.items()):
                source_prefix = source_namespace_prefix(source)
                if not source_prefix:
                    continue
                items = []
                seen = set()
                for r in restrictions:
                    key = (r.property_label, r.kind, r.value_label)
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(
                        f"<span class='rule-pill'>{html.escape(r.property_label)} {html.escape(r.kind)} {html.escape(r.value_label)}</span>"
                    )
                if items:
                    lines.append(
                        f"<div class='mapping-line'>"
                        f"<span class='inline-label'>Restrictions</span>"
                        f"<span class='ns-box ns-{source_prefix.lower()}'>{html.escape(source_prefix)}</span>"
                        f"{''.join(items)}"
                        f"</div>"
                    )
            if lines:
                restriction_summary = (
                    f"<div class='meta-block'>"
                    f"<div class='alignment-lines'>{''.join(lines)}</div>"
                    f"</div>"
                )

        if node.equivalent_expressions:
            restriction_summary = ""

        instance_summary = ""
        if node.incoming_instances:
            lines = []
            for source, instances in sorted(node.incoming_instances.items()):
                source_prefix = source_namespace_prefix(source)
                source_class = source_prefix.lower()
                instance_markup = " ".join(
                    f"<span class='instance-pill {source_class}'>{html.escape(instance)}</span>"
                    for instance in sorted(instances)
                )
                lines.append(
                    f"<div class='mapping-line instances'>"
                    f"<span class='inline-label'>CV terms</span>"
                    f"<span class='ns-box ns-{source_class}'>{html.escape(source_prefix)}</span>"
                    f"{instance_markup}"
                    f"</div>"
                )
            instance_summary = (
                f"<div class='meta-block'>"
                f"<div class='alignment-lines'>{''.join(lines)}</div>"
                f"</div>"
            )

        equivalent_summary = ""
        if node.equivalent_expressions:
            lines = []
            for source, expressions in sorted(node.equivalent_expressions.items()):
                source_prefix = source_namespace_prefix(source)
                if not source_prefix:
                    continue
                expr_chunks = []
                for expr in expressions:
                    if not expr.members:
                        continue
                    parts = [f"<span class='inline-label'>equivalent to</span>"]
                    for i, member in enumerate(expr.members):
                        if i > 0 and expr.operator:
                            parts.append(f"<span class='op-pill'>{html.escape(expr.operator)}</span>")
                        parts.append(f"<span class='rule-pill'>{html.escape(member)}</span>")
                    expr_chunks.append("".join(parts))

                if expr_chunks:
                    lines.append(
                        f"<div class='mapping-line'>"
                        f"<span class='ns-box ns-{source_prefix.lower()}'>{html.escape(source_prefix)}</span>"
                        f"{''.join(expr_chunks)}"
                        f"</div>"
                    )

            if lines:
                equivalent_summary = (
                    f"<div class='meta-block'>"
                    f"<div class='alignment-lines'>{''.join(lines)}</div>"
                    f"</div>"
                )

        child_markup = []
        for child in visible_children(uri):
            if child in next_path:
                continue

            child_markup.append(render_node(child, next_path))

        toggle_button = (
            "<span class='toggle-marker' aria-hidden='true'></span>"
            if child_markup
            else "<span class='toggle-marker hidden' aria-hidden='true'></span>"
        )

        if child_markup:
            return (
                "<li>"
                f"<details class='node ns-{prefix_class}' open>"
                f"<summary class='node-header'>{toggle_button}"
                f"<span class='ns-box ns-{prefix_class}'>{html.escape(prefix)}</span>"
                f"<span class='label'>{html.escape(node.label)}</span>"
                f"<span class='uri'>{html.escape(str(uri))}</span>{overlap_tag}</summary>"
                f"{equivalent_summary}{restriction_summary}{instance_summary}"
                f"<div class='children'><ul>{''.join(child_markup)}</ul></div>"
                "</details>"
                "</li>"
            )

        return (
            "<li>"
            f"<div class='node ns-{prefix_class}'>"
            f"<div class='node-header'>{toggle_button}"
            f"<span class='ns-box ns-{prefix_class}'>{html.escape(prefix)}</span>"
            f"<span class='label'>{html.escape(node.label)}</span>"
            f"<span class='uri'>{html.escape(str(uri))}</span>{overlap_tag}</div>"
            f"{equivalent_summary}{restriction_summary}{instance_summary}"
            f"</div>"
            "</li>"
        )

    return "".join(render_node(root, set()) for root in choose_roots(nodes, selected))


def build_html(nodes: dict[URIRef, NodeInfo], selected: set[URIRef], shared_targets: set[URIRef], title: str) -> str:
    tree_markup = render_tree(nodes, selected, shared_targets)
    namespaces = namespace_prefixes_in_view(nodes, selected)

    return f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #fafafa;
      --panel: #ffffff;
      --panel-soft: #f4f6f8;
      --text: #1f2937;
      --muted: #5b6472;
      --border: #d7dde5;
      --tree-line: #dbe4ee;

      --foodon-bg: #dbeafe;
      --foodon-fg: #1e3a5f;
      --foodon-border: #93c5fd;

      --coso-bg: #f3e8d8;
      --coso-fg: #6b4f2a;
      --coso-border: #d6b98b;

      --egad-bg: #f3d9eb;
      --egad-fg: #7a1f5c;
      --egad-border: #d38dbd;

      --wqp-bg: #d9f0e8;
      --wqp-fg: #195c49;
      --wqp-border: #86c7ae;

      --rule-bg: #fff4cc;
      --rule-fg: #5e4b00;
      --rule-border: #dcc777;

      --instance-bg: #ffffff;
      --instance-fg: #111827;
      --instance-border: #111827;

      --shared-bg: #fde7b0;
      --shared-fg: #6b4e00;
    }}

    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      color: var(--text);
    }}

    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 20px 18px 28px;
    }}

    h1 {{
      margin: 0 0 6px;
      font-size: 1.45rem;
      letter-spacing: -0.02em;
    }}

    p {{
      color: var(--muted);
      line-height: 1.35;
      margin: 6px 0 0;
    }}

    .legend,
    .summary {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      margin: 12px 0;
    }}

    .tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}

    .tag {{
      display: inline-flex;
      align-items: center;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #eef2f7;
      color: #334155;
      font-size: 0.74rem;
      line-height: 1.2;
    }}

    .tag.overlap {{
      background: var(--shared-bg);
      color: var(--shared-fg);
      border-color: #e3c86b;
    }}

    .ns-box {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 3.3rem;
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      border: 1px solid var(--border);
      font-weight: 700;
      letter-spacing: 0.04em;
      font-size: 0.76rem;
    }}

    .ns-foodon {{ background: var(--foodon-bg); color: var(--foodon-fg); border-color: var(--foodon-border); }}
    .ns-coso {{ background: var(--coso-bg); color: var(--coso-fg); border-color: var(--coso-border); }}
    .ns-egad {{ background: var(--egad-bg); color: var(--egad-fg); border-color: var(--egad-border); }}
    .ns-wqp {{ background: var(--wqp-bg); color: var(--wqp-fg); border-color: var(--wqp-border); }}

    .tree {{
      margin-top: 10px;
    }}

    .tree > ul {{
      border-left: 0;
      padding-left: 0;
    }}

    ul {{
      list-style: none;
      margin: 0;
      padding-left: 18px;
      border-left: 1px solid var(--tree-line);
    }}

    li {{
      margin: 5px 0;
    }}

    .node {{
      padding: 7px 9px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--panel-soft);
      margin: 4px 0;
    }}

    .node.ns-foodon {{ border-color: var(--foodon-border); }}
    .node.ns-coso {{ border-color: var(--coso-border); }}
    .node.ns-egad {{ border-color: var(--egad-border); }}
    .node.ns-wqp {{ border-color: var(--wqp-border); }}

    details.node {{
      padding: 0;
    }}

    details.node > summary {{
      list-style: none;
      cursor: pointer;
      margin: 0;
    }}

    details.node > summary::-webkit-details-marker {{
      display: none;
    }}

    details.node > .children {{
      margin-top: 4px;
    }}

    .node-header {{
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }}

    .label {{
      font-weight: 650;
      color: #111827;
    }}

    .uri {{
      color: var(--muted);
      font-size: 0.72rem;
      margin-left: 8px;
      word-break: break-all;
    }}

    .toggle-marker {{
      display: inline-flex;
      width: 1rem;
      justify-content: center;
      color: #64748b;
      font-size: 0.95rem;
      flex: 0 0 auto;
    }}

    details[open] > summary .toggle-marker::before {{
      content: "▾";
    }}

    details:not([open]) > summary .toggle-marker::before {{
      content: "▸";
    }}

    .toggle-marker.hidden {{
      visibility: hidden;
    }}

    .meta-block {{
      margin-top: 7px;
      padding-top: 6px;
      border-top: 1px dashed var(--border);
    }}

    .alignment-lines {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    .mapping-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      align-items: center;
    }}

    .inline-label {{
      display: inline-flex;
      align-items: center;
      padding: 0.12rem 0.45rem;
      border-radius: 999px;
      background: #eef2f7;
      color: #475569;
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      white-space: nowrap;
    }}

    .rule-pill {{
      display: inline-flex;
      align-items: center;
      padding: 0.18rem 0.55rem;
      border-radius: 4px;
      border: 1px solid var(--rule-border);
      background: var(--rule-bg);
      color: var(--rule-fg);
      font-style: italic;
      font-size: 0.82rem;
      line-height: 1.25;
    }}

    .instance-pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 1.55rem;
      padding: 0.08rem 0.58rem;
      border-radius: 999px;
      border: 2px solid var(--instance-border);
      background: var(--instance-bg);
      color: var(--instance-fg);
      font-weight: 700;
      font-size: 0.8rem;
      line-height: 1.1;
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.9) inset;
    }}

    .instance-pill.legend-instance {{
      background: #ffffff;
      border: 1px solid #111827;
      color: #111827;
      box-shadow: none;
    }}

    .op-pill {{
      display: inline-flex;
      align-items: center;
      padding: 0.12rem 0.42rem;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #f8fafc;
      color: #475569;
      font-size: 0.72rem;
      font-weight: 700;
      line-height: 1.1;
      text-transform: lowercase;
    }}

    .mapping-line.instances .ns-box.ns-egad + .instance-pill,
    .mapping-line.instances .instance-pill.egad {{
      border-color: var(--egad-fg);
    }}

    .mapping-line.instances .ns-box.ns-wqp + .instance-pill,
    .mapping-line.instances .instance-pill.wqp {{
      border-color: var(--wqp-fg);
    }}

    @media print {{
      body {{ background: #fff; }}
      main {{ max-width: none; padding: 0; }}
      .legend, .summary, .node {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <p>Hierarchy extracted from the ontology sources, with FOODON/COSO classes, EGAD and WQP controlled vocabulary terms, and class restrictions shown inline.</p>
    <section class='legend'>
      <strong>Namespaces</strong>
      <div class='tags'>
        {''.join(f"<span class='ns-box ns-{ns.lower()}'>{html.escape(ns)}</span>" for ns in namespaces)}
        <span class='instance-pill legend-instance'>instances</span>
      </div>
    </section>
    <section class='tree'>
      <ul>{tree_markup}</ul>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a hierarchy visualization from ContaminOSO, FOODON, and alignment TTL files."
    )

    ontology_dir = (
        Path(__file__).resolve().parents[2]
        / "pfas-kg"
        / "datasets"
        / "federal"
        / "us-wqp"
        / "ontology"
    )

    parser.add_argument(
        "--foodon",
        type=Path,
        default=Path(__file__).resolve().parent / "reuse" / "foodon_lite_updated.ttl",
        help="Path to the FOODON lite TTL file.",
    )
    parser.add_argument(
        "--contaminoso",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "v1" / "contaminoso_materialSample_ext.ttl",
        help="Path to the ContaminOSO material sample extension TTL file.",
    )
    parser.add_argument(
        "--egad-alignment",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "pfas-kg" / "datasets" / "maine" / "egad" / "ontology" / "egad-controlledVocab-alignment.ttl",
        help="Path to the EGAD controlled vocabulary alignment TTL file.",
    )
    parser.add_argument(
        "--wqp-alignment",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "pfas-kg" / "datasets" / "federal" / "us-wqp" / "ontology" / "wqp_alignment.ttl",
        help="Path to the WQP alignment TTL file.",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "hierarchy_visualization.html",
        help="Output HTML file.",
    )
    args = parser.parse_args()

    foodon_path = existing_path([args.foodon])
    contaminoso_path = existing_path([
        args.contaminoso,
        args.contaminoso.with_name("contaminoso_matrialSample_ext.ttl"),
    ])
    egad_path = existing_path([args.egad_alignment])
    wqp_path = existing_path([args.wqp_alignment])

    missing: list[str] = []
    if foodon_path is None:
        missing.append(str(args.foodon))
    if contaminoso_path is None:
        missing.append(str(args.contaminoso))
    if egad_path is None:
        missing.append(str(args.egad_alignment))
    if missing:
        raise FileNotFoundError(f"Required input file(s) not found: {', '.join(missing)}")

    sources: list[tuple[str, Graph]] = []
    for source, path in (
        ("foodon_lite_updated", foodon_path),
        ("contaminoso_materialSample_ext", contaminoso_path),
        ("egad_alignment", egad_path),
    ):
        graph = Graph()
        graph.parse(path)
        sources.append((source, graph))

    if wqp_path is not None:
        graph = Graph()
        graph.parse(wqp_path)
        sources.append(("wqp_alignment", graph))

    merged_graph = merge_graphs(graph for _, graph in sources)
    reasoned_graph = Graph()
    for triple in merged_graph:
        reasoned_graph.add(triple)
    DeductiveClosure(OWLRL_Semantics, axiomatic_triples=False, datatype_axioms=False).expand(reasoned_graph)

    nodes: dict[URIRef, NodeInfo] = {}
    for source, graph in sources:
        build_structure(graph, merged_graph, source, nodes)

    add_inferred_instances(reasoned_graph, reasoned_graph, nodes)

    alignment_targets: dict[str, dict[URIRef, set[URIRef]]] = {
        source: collect_alignment_targets(graph)
        for source, graph in sources
        if source in {"egad_alignment", "wqp_alignment"}
    }

    shared_targets: set[URIRef] = set()
    if {"egad_alignment", "wqp_alignment"}.issubset(alignment_targets):
        egad_targets = {target for targets in alignment_targets["egad_alignment"].values() for target in targets}
        wqp_targets = {target for targets in alignment_targets["wqp_alignment"].values() for target in targets}
        shared_targets = egad_targets & wqp_targets
    elif "wqp_alignment" not in alignment_targets:
        print("WQP alignment file not found; overlap highlighting will be omitted.")

    def descendants(start: URIRef, allowed: set[URIRef]) -> set[URIRef]:
        seen = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in allowed:
                continue
            seen.add(cur)
            stack.extend(nodes[cur].children & allowed)
        return seen

    selected: set[URIRef] = set()
    for root in (MATERIAL_ENTITY_IRI, COSO_MATERIAL_SAMPLE_IRI):
        if root in nodes:
            selected.update(descendants(root, set(nodes)))
    
    if shared_targets:
        selected.update(shared_targets & set(nodes))

    org_substance = URIRef("http://purl.obolibrary.org/obo/UBERON_0000463")
    if org_substance in nodes:
        print("organism substance children:")
        for child in sorted(nodes[org_substance].children, key=lambda u: nodes[u].label.lower()):
            print(" -", nodes[child].label, child)
            print("   parents:", [nodes[p].label for p in sorted(nodes[child].parents, key=lambda u: nodes[u].label.lower())])
            print("   preferred:", nodes[preferred_parent(nodes, child, selected)].label if preferred_parent(nodes, child, selected) else None)

    html_output = build_html(nodes, selected, shared_targets, "ContaminOSO hierarchy visualization")
    args.output.write_text(html_output, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())