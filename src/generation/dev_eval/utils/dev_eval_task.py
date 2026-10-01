from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional


@dataclass(frozen=True)
class DevEvalTask:
    namespace: str
    completion_path: str
    signature_position: Tuple[int, int]
    requirement_text: str
    dependency: list[str] = None
    indent: Optional[int] = None


def build_requirement_text(requirement: Any) -> str:
    functionality = requirement.get("Functionality")
    arguments = requirement.get("Arguments")
    return functionality + "\n" + arguments


def parse_task(record: Dict[str, Any]) -> DevEvalTask:
    namespace = record.get("namespace")
    completion_path = record.get("completion_path")
    signature_position = record.get("signature_position")

    task_dependency = record.get('dependency', {})
    dependency = []
    dependency.extend(task_dependency.get('intra_class', []))
    dependency.extend(task_dependency.get('intra_file', []))
    dependency.extend(task_dependency.get('cross_file', []))

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("Task missing valid 'namespace'")
    if not isinstance(completion_path, str) or not completion_path:
        raise ValueError(f"Task {namespace} missing valid 'completion_path'")
    if (
        not isinstance(signature_position, (list, tuple))
        or len(signature_position) != 2
        or not all(isinstance(x, int) for x in signature_position)
    ):
        raise ValueError(f"Task {namespace} missing valid 'signature_position'")

    requirement_text = build_requirement_text(record.get("requirement"))
    indent = record.get("indent")
    indent = indent if isinstance(indent, int) and indent >= 0 else None

    start, end = int(signature_position[0]), int(signature_position[1])
    if start <= 0 or end <= 0 or end < start:
        raise ValueError(
            f"Task {namespace} has invalid signature_position: {signature_position}"
        )

    return DevEvalTask(
        namespace=namespace,
        completion_path=completion_path,
        signature_position=(start, end),
        requirement_text=requirement_text,
        dependency=dependency,
        indent=indent,
    )

