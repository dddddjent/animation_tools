from typing import List
import numpy as np
import uuid
from .thirdParty.Quaternions import Quaternions


class SkeletonJoint:
    """A skeleton joint class for a skeleton structure.

    Represents a single joint in a skeleton hierarchy with parent-child relationships.
    """

    def __init__(
        self,
        name: str,
        children: List["SkeletonJoint"] = [],
        parent: "SkeletonJoint" = None,
        uuid_val: uuid.UUID = uuid.uuid4(),
        offset: np.ndarray = np.zeros(3),
        rotations: Quaternions = Quaternions.identity(),
        positions: np.ndarray = np.array([]),
    ) -> None:
        """Initialize a joint node.

        Args:
            name: The name of the joint
            children: List of child joints (defaults to empty list)
            parent: Parent joint node (defaults to None)
            uuid_val: Unique identifier for the joint (defaults to generated UUID)
            offset: Offset from parent as numpy array (defaults to zero vector)
            rotations: Quaternion rotations (defaults to identity quaternion)
            positions: Position data as numpy array (defaults to empty array)
        """
        self.name = name
        self.children = children
        self.parent = parent
        self.uuid = uuid_val
        self.offset = offset
        self.rotations = rotations
        self.positions = positions

    def add_child(self, child: "SkeletonJoint") -> None:
        """Add a child joint to this joint.

        Args:
            child: The child joint to add
        """
        if child not in self.children:
            self.children.append(child)
            child.parent = self

    def remove_child(self, child: "SkeletonJoint") -> None:
        """Remove a child joint from this joint.

        Args:
            child: The child joint to remove
        """
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def __repr__(self) -> str:
        """String representation of the joint node."""
        return self._to_dict_str()
    
    def _to_dict_str(self, indent: int = 0) -> str:
        """Convert to dictionary-like string representation recursively."""
        indent_str = "  " * indent
        next_indent_str = "  " * (indent + 1)
        
        lines = [f"{indent_str}{{"]
        lines.append(f"{next_indent_str}\"name\": \"{self.name}\",")
        lines.append(f"{next_indent_str}\"uuid\": \"{str(self.uuid)[:8]}...\",")
        parent_name = f'"{self.parent.name}"' if self.parent else 'null'
        lines.append(f"{next_indent_str}\"parent\": {parent_name},")
        lines.append(f"{next_indent_str}\"offset\": [{', '.join(f'{x:.3f}' for x in self.offset)}],")
        
        if self.children:
            lines.append(f"{next_indent_str}\"children\": [")
            for i, child in enumerate(self.children):
                child_str = child._to_dict_str(indent + 2)
                if i < len(self.children) - 1:
                    lines.append(f"{child_str},")
                else:
                    lines.append(child_str)
            lines.append(f"{next_indent_str}]")
        else:
            lines.append(f"{next_indent_str}\"children\": []")
        
        lines.append(f"{indent_str}}}")
        return "\n".join(lines)
