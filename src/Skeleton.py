from .thirdParty.Animation import Animation
from .thirdParty import BVH
from .SkeletonJoint import SkeletonJoint
from .thirdParty.Quaternions import Quaternions
import numpy as np
import uuid


class Skeleton:
    """
    A Skeleton class that represents a hierarchical bone structure for animation.

    This class can be initialized with the returns from BVH.load() function.
    """

    # Maps a canonical body label to an unordered set of possible names/aliases
    # that may appear in various skeleton sources (e.g., BVH exports)
    body_label_map: dict[str, set[str]] = {
        "RightArm": {
            "RightArm", "right_arm", "RightForearm", "RightForeArm", "RightHand",
            "RArm", "RForeArm", "RHand", "RightWrist", "RWrist", "RightElbow", "RElbow",
            "right_hand", "right_forearm", "right_elbow", "right_wrist"
        },
        "LeftArm": {
            "LeftArm", "left_arm", "LeftForearm", "LeftForeArm", "LeftHand",
            "LArm", "LForeArm", "LHand", "LeftWrist", "LWrist", "LeftElbow", "LElbow",
            "left_hand", "left_forearm", "left_elbow", "left_wrist"
        },
        "Head": {
            "Head", "head", "HEAD", "HeadTop", "HeadTop_End", "Neck", "neck"
        },
        "Spine": {
            "Spine", "spine", "Spine1", "Spine2", "Spine3", "Spine4",
            "Spine01", "Chest", "UpperChest", "Torso", "Hips", "hips"
        },
        "RightLeg": {
            "RightLeg", "right_leg", "RightUpLeg", "RightThigh", "RightFoot",
            "RightToeBase", "RLeg", "RUpLeg", "RThigh", "RFoot", "RToeBase", "RightAnkle", "RAnkle",
            "right_foot", "right_thigh", "right_leg", "right_ankle", "right_toe"
        },
        "LeftLeg": {
            "LeftLeg", "left_leg", "LeftUpLeg", "LeftThigh", "LeftFoot",
            "LeftToeBase", "LLeg", "LUpLeg", "LThigh", "LFoot", "LToeBase", "LeftAnkle", "LAnkle",
            "left_foot", "left_thigh", "left_leg", "left_ankle", "left_toe"
        },
    }

    def __init__(self, animation: Animation, names: list, frametime: float):
        """
        Initialize the Skeleton with data from BVH.load().

        Parameters
        ----------
        animation : Animation
            Animation object containing rotations, positions, orients, offsets, and parents
        names : list
            List of joint names
        frametime : float
            Frame time in seconds
        """
        self.frametime = frametime

        # Build the joint tree
        self.skeleton = self._build_joint_tree(animation, names)

        self.remove_unwanted_joints()
        self.remove_redundant_root()
        self._label_skeleton()
        self.orientation = self.guess_orientations()

    @staticmethod
    def load(filename):
        """
        Load a skeleton from a BVH file.

        Parameters
        ----------
        filename : str
            Path to the BVH file

        Returns
        -------
        Skeleton
            A new Skeleton instance loaded from the BVH file
        """
        animation, names, frametime = BVH.load(filename)
        return Skeleton(animation, names, frametime)

    def _build_joint_tree(self, animation: Animation, names: list) -> SkeletonJoint:
        """
        Build a tree of SkeletonJoint objects from the animation data.

        Returns
        -------
        SkeletonJoint
            The root joint of the skeleton tree
        """
        # Create all joint objects first
        joints = []
        num_joints = len(names)

        for i in range(num_joints):
            # Extract rotations for this joint across all timesteps
            # animation.rotations has shape (frames, joints) where each element is a quaternion
            joint_rotations = animation.rotations[:, i]  # Shape: (frames,)

            # Copy offset for this joint
            joint_offset = animation.offsets[i].copy()  # Shape: (3,)

            # Positions are only for the root joint (parent index -1)
            joint_positions = np.array([])
            if animation.parents[i] == -1:  # This is the root joint
                # Shape: (frames, 3)
                joint_positions = animation.positions[:, i].copy()

            # Create the joint
            joint = SkeletonJoint(
                name=names[i],
                children=[],  # Will be set up below
                parent=None,  # Will be set up below
                uuid_val=uuid.uuid4(),
                offset=joint_offset,
                rotations=joint_rotations,
                positions=joint_positions
            )

            joints.append(joint)

        # Set up parent-child relationships
        root_joint = None

        for i in range(num_joints):
            parent_idx = animation.parents[i]

            if parent_idx == -1:  # This is the root joint
                root_joint = joints[i]
            else:  # This joint has a parent
                parent_joint = joints[parent_idx]
                child_joint = joints[i]

                # Set parent-child relationship
                parent_joint.add_child(child_joint)

        return root_joint

    def remove_unwanted_joints(self):
        """
        Remove unwanted joints from the skeleton.

        Unwanted joints include:
        - fingers
        - Jaw
        - eyes
        - toe (but toe base is preserved)

        Joints are identified by:
        1. Name matching unwanted patterns
        2. Structural analysis (joints with too many children, indicating finger branches)
        """
        if not self.skeleton:
            return

        # Define patterns for unwanted joints
        unwanted_patterns = [
            # Finger patterns
            'finger', 'thumb', 'index', 'middle', 'ring', 'pinky', 'fingers',
            # Jaw patterns
            'jaw', 'mouth', 'teeth',
            # Eye patterns
            'eye', 'eyelid', 'eyebrow', 'pupil', 'cornea',
            # Toe patterns (but preserve toe base)
            'toe_tip', 'toe_end', 'toe_distal', 'toe1', 'toe2', 'toe3', 'toe4', 'toe5',
            # Additional finger/toe patterns
            'fing', 'phalange', 'digit', 'metacarpal',
            # Face patterns that might include eyes/jaw
            'face', 'facial'
        ]

        def is_unwanted_by_name(name: str) -> bool:
            """Check if joint name matches unwanted patterns."""
            name_lower = name.lower()
            # Check for unwanted patterns
            for pattern in unwanted_patterns:
                if pattern in name_lower:
                    return True
            # Special case: toe but not toe base
            if 'toe' in name_lower and 'base' not in name_lower:
                return True
            return False

        def has_too_many_children(joint: SkeletonJoint) -> bool:
            """Check if joint has too many children (indicating finger branches)."""
            # Normal limb joints usually have 0 or 1 child (end effector)
            # If a joint has 3+ children, it's likely a finger branching point
            return len(joint.children) >= 3

        def should_remove_joint(joint: SkeletonJoint) -> bool:
            """Determine if a joint should be removed."""
            # Check by name
            if is_unwanted_by_name(joint.name):
                return True

            # Check by structure (too many children)
            if has_too_many_children(joint):
                # Additional check: only remove if it looks like a finger joint
                # Check if children have finger-like names
                finger_like_children = 0
                for child in joint.children:
                    child_name_lower = child.name.lower()
                    if any(pattern in child_name_lower for pattern in
                          ['finger', 'thumb', 'index', 'middle', 'ring', 'pinky', 'phalange', 'digit']):
                        finger_like_children += 1

                # If majority of children are finger-like, remove this joint
                if finger_like_children >= len(joint.children) / 2:
                    return True

            return False

        def remove_joint(joint: SkeletonJoint):
            """Remove a joint and reassign its children to its parent."""
            if not joint.parent:
                return  # Don't remove root joint

            parent = joint.parent

            # Collect children to reassign
            children_to_reassign = joint.children.copy()

            # Remove joint from parent's children
            parent.remove_child(joint)

            # Reassign children to parent
            for child in children_to_reassign:
                # Transfer the removed joint's rotation to the child
                child.rotations = joint.rotations * child.rotations
                # Update child's offset to be relative to new parent
                child.offset = joint.offset + child.offset
                parent.add_child(child)

        def traverse_and_remove(joint: SkeletonJoint):
            """Recursively traverse and remove unwanted joints."""
            if not joint:
                return

            # First, process children recursively (post-order traversal)
            # We need to copy the children list because it might be modified
            children_copy = joint.children.copy()
            for child in children_copy:
                traverse_and_remove(child)

            # Then check if current joint should be removed
            if should_remove_joint(joint):
                remove_joint(joint)

        # Start removal from root
        traverse_and_remove(self.skeleton)

    def remove_redundant_root(self):
        """
        Remove redundant root joint above hips and transfer its motion to hips.

        This method identifies cases where there's an unwanted root joint above the hips
        and removes it, transferring any root motion data to the hips joint.
        """
        if not self.skeleton:
            return

        root = self.skeleton

        # Check if this looks like a redundant root scenario:
        # 1. Root has only one child
        # 2. The child is likely a hips/lower body joint
        # 3. Root has position data (motion) and the child doesn't

        if (len(root.children) == 1 and
            root.positions.size > 0 and
            root.children[0].positions.size == 0):

            child = root.children[0]

            # Check if the child looks like hips based on name or structure
            child_name = child.name.lower()
            is_hips_like = any(keyword in child_name for keyword in [
                'hip', 'hips', 'pelvis', 'spine', 'torso', 'root'
            ])

            # Also check if child has multiple children (typical for hips)
            has_multiple_children = len(child.children) >= 2

            if is_hips_like or has_multiple_children:
                print(f"Removing redundant root '{root.name}' and moving motion to '{child.name}'")

                # Check if root has meaningful translation (not all zeros)
                root_has_translation = root.positions.size > 0 and not np.allclose(root.positions, 0)

                # Only transfer root motion if it has meaningful translation
                if root_has_translation:
                    child.positions = root.positions.copy()
                    # Add child's own offset to maintain correct world positioning
                    child.positions = child.positions + child.offset
                else:
                    # If root has no translation, keep child's original positions
                    # but still add child's offset to maintain rest pose relationship
                    if child.positions.size == 0:
                        # If child has no positions, create positions array with child's offset
                        num_frames = child.rotations.qs.shape[0] if hasattr(child.rotations, 'qs') else 0
                        if num_frames > 0:
                            child.positions = np.tile(child.offset, (num_frames, 1))
                    else:
                        # If child has positions, add child's offset
                        child.positions = child.positions + child.offset

                # Update child's offset to include root's offset
                child.offset = root.offset + child.offset

                # Make child the new root
                child.parent = None
                self.skeleton = child

                # Transfer the root's rotation to the child
                # The root's rotation should be combined with the child's rotation
                if root.rotations.qs.shape == child.rotations.qs.shape:
                    # Combine root rotation with child rotation
                    child.rotations = root.rotations * child.rotations
                # If shapes don't match, keep child's original rotation

    def store(self, filename: str, order: str = 'zyx', save_positions: bool = False) -> None:
        """
        Save the current skeleton animation to a BVH file.

        Parameters
        ----------
        filename : str
            Output BVH file path
        order : str, optional
            Euler order used for saving rotations, by default 'zyx'
        save_positions : bool, optional
            Whether to save per-joint positions for all joints (BVH with 6 channels).
            By default False, which saves only root positions and rotations for other joints.
        frametime : float | None, optional
            Frame time to encode in the BVH. Defaults to the Skeleton's frametime when None.
        """
        # Flatten the joint tree into arrays compatible with Animation
        joints_order: list[SkeletonJoint] = []
        parents = []

        def traverse(node: SkeletonJoint, parent_index: int):
            index = len(joints_order)
            joints_order.append(node)
            parents.append(parent_index)
            for child in node.children:
                traverse(child, index)

        traverse(self.skeleton, -1)

        # Names in traversal order
        out_names = [j.name for j in joints_order]

        # Offsets (J,3)
        offsets = np.vstack([j.offset for j in joints_order])

        # Orients (J) Quaternions (identity)
        orients = Quaternions.id(len(joints_order))

        # Rotations (F,J,4) as Quaternions
        # Infer number of frames from the first joint's rotations
        num_frames = joints_order[0].rotations.qs.shape[0]
        rotations_qs = np.stack([j.rotations.qs for j in joints_order], axis=1)
        rotations = Quaternions(rotations_qs)

        # Positions (F,J,3). Use root positions if available, zeros otherwise
        positions = np.zeros((num_frames, len(joints_order), 3), dtype=float)
        if joints_order[0].positions.size != 0:
            positions[:, 0, :] = joints_order[0].positions

        anim = Animation(rotations, positions, orients,
                         offsets, np.array(parents, dtype=int))

        BVH.save(
            filename=filename,
            anim=anim,
            names=out_names,
            frametime=self.frametime,
            order=order,
            positions=save_positions,
            orients=True,
        )

    def _label_skeleton(self):
        """
        Label the skeleton with the body_label_map.
        """
        # Build fast alias->canonical map
        alias_to_canonical: dict[str, str] = {}
        for canonical, aliases in self.body_label_map.items():
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical

        def find_spine_path_from_root(root: SkeletonJoint) -> list[SkeletonJoint]:
            # Find a path from root through successive single-child joints up to
            # the first joint that has exactly 3 children (the chest)
            path: list[SkeletonJoint] = [root]
            current = root
            # We need to ignore root's 3 children rule; root often has 3 children (hips)
            while True:
                children = current.children
                if not children:
                    break
                if len(children) == 1:
                    current = children[0]
                    path.append(current)
                    continue
                if len(children) == 3 and current is not root:
                    # We reached the chest-like branching
                    break
                # If there are 2 or more but not the branching we want, choose the child whose
                # name best resembles spine/hips
                preferred = None
                for child in children:
                    name_l = child.name.lower()
                    if any(k in name_l for k in ["spine", "hip", "hips", "pelvis", "torso", "chest"]):
                        preferred = child
                        break
                if preferred is None:
                    # fallback: choose first to avoid getting stuck
                    preferred = children[0]
                current = preferred
                path.append(current)
            return path

        def collect_limbs_from_chest(chest: SkeletonJoint) -> tuple[SkeletonJoint, SkeletonJoint, SkeletonJoint]:
            # chest is the first joint with exactly 3 children after spine
            # We must decide which is head, left arm, right arm by name heuristics.
            assert len(chest.children) == 3
            head_candidate = None
            left_candidate = None
            right_candidate = None
            for child in chest.children:
                name_l = child.name.lower()
                if any(k in name_l for k in ["head", "neck"]):
                    head_candidate = child
                elif any(k in name_l for k in ["left", "l_"]):
                    left_candidate = child
                elif any(k in name_l for k in ["right", "r_"]):
                    right_candidate = child
            # Fallbacks if names are ambiguous: use offsets X to disambiguate if available
            missing = [x is None for x in [
                head_candidate, left_candidate, right_candidate]]
            if any(missing):
                # Sort by x offset sign: left has x>0 for many rigs, right x<0; head y> others
                # Use absolute heuristics carefully; fallback to arbitrary but consistent ordering
                sorted_children = sorted(
                    chest.children,
                    key=lambda j: (-(j.offset[1] if hasattr(j.offset,
                                   '__iter__') else 0.0), j.offset[0]),
                )
                # Highest Y likely head
                head_candidate = head_candidate or sorted_children[0]
                rem = [c for c in chest.children if c is not head_candidate]
                if len(rem) == 2:
                    if rem[0].offset[0] >= rem[1].offset[0]:
                        left_candidate = left_candidate or rem[0]
                        right_candidate = right_candidate or rem[1]
                    else:
                        left_candidate = left_candidate or rem[1]
                        right_candidate = right_candidate or rem[0]
                else:
                    # If structure is unexpected, assign arbitrarily
                    for c in rem:
                        if left_candidate is None:
                            left_candidate = c
                        elif right_candidate is None:
                            right_candidate = c
            return head_candidate, left_candidate, right_candidate

        def traverse_chain_down(joint: SkeletonJoint) -> list[SkeletonJoint]:
            # Follow the longest single-child path until termination or split
            chain = [joint]
            current = joint
            while True:
                if not current.children:
                    break
                if len(current.children) != 1:
                    # stop at branching
                    break
                current = current.children[0]
                chain.append(current)
            return chain

        def traverse_branch_until_end(joint: SkeletonJoint) -> list[SkeletonJoint]:
            # Depth-first collect until leaves
            nodes = []
            stack = [joint]
            while stack:
                node = stack.pop()
                nodes.append(node)
                for c in node.children:
                    stack.append(c)
            return nodes

        def assign_label_to_group(joints: list[SkeletonJoint], default_canonical: str) -> None:
            # If any joint in the group matches an alias, use that canonical key; else use default
            canonical = default_canonical
            for j in joints:
                name_l = j.name.lower()
                if name_l in alias_to_canonical:
                    canonical = alias_to_canonical[name_l]
                    break
            for j in joints:
                j.body_part_label = canonical

        root = self.skeleton
        if root is None:
            return

        # Find spine path from root to first branching (3-children) after root
        spine_path = find_spine_path_from_root(root)
        # The chest is the last in path if it has 3 children; otherwise try to find next node with 3 children
        chest = spine_path[-1]
        if len(chest.children) != 3:
            # Search below for the first node with 3 children
            candidates = traverse_branch_until_end(chest)
            for n in candidates:
                if len(n.children) == 3:
                    chest = n
                    break

        # Assign spine label (including hips/root)
        assign_label_to_group(spine_path, "Spine")

        # Legs: from hips/root, two branches that end with feet/toes
        leg_candidates = [c for c in root.children if c not in spine_path]
        # If root is hips and has 3 children, two non-spine branches are legs
        legs = []
        for c in leg_candidates:
            # Heuristic: choose branches that go downward in Y first segment
            if c is chest:
                continue
            legs.append(c)
        if len(legs) > 2 and chest in legs:
            legs.remove(chest)
        legs = legs[:2]

        if len(legs) == 2:
            left_leg_root, right_leg_root = None, None
            # Decide left/right by x offset of first joint
            if legs[0].offset[0] >= legs[1].offset[0]:
                left_leg_root, right_leg_root = legs[0], legs[1]
            else:
                left_leg_root, right_leg_root = legs[1], legs[0]

            left_leg_nodes = traverse_branch_until_end(left_leg_root)
            right_leg_nodes = traverse_branch_until_end(right_leg_root)
            assign_label_to_group(left_leg_nodes, "LeftLeg")
            assign_label_to_group(right_leg_nodes, "RightLeg")

        # Arms and head from chest
        if len(chest.children) == 3:
            head_node, left_arm_root, right_arm_root = collect_limbs_from_chest(
                chest)
            if head_node is not None:
                head_nodes = traverse_branch_until_end(head_node)
                assign_label_to_group(head_nodes, "Head")
            if left_arm_root is not None:
                left_arm_nodes = traverse_branch_until_end(left_arm_root)
                assign_label_to_group(left_arm_nodes, "LeftArm")
            if right_arm_root is not None:
                right_arm_nodes = traverse_branch_until_end(right_arm_root)
                assign_label_to_group(right_arm_nodes, "RightArm")

    def guess_orientations(self) -> dict[str, str]:
        """
        Guess the orientations of the skeleton.
        Returns: {"forward": 'x', "up": 'y}
        It can be x, y, z, or -x, -y, -z.
        """
        # We need 2 vectors, the spine and left end to right end (not necessarily the hands)
        # The spine determines the up direction. the other vector and up determine the forward direction.

        def find_joints_by_label(label: str) -> list[SkeletonJoint]:
            """Find all joints with a specific body part label."""
            joints = []

            def traverse(node: SkeletonJoint):
                if hasattr(node, 'body_part_label') and node.body_part_label == label:
                    joints.append(node)
                for child in node.children:
                    traverse(child)

            if self.skeleton:
                traverse(self.skeleton)
            return joints

        def get_joint_world_position(joint: SkeletonJoint) -> np.ndarray:
            """Calculate the world position of a joint by accumulating offsets."""
            position = np.zeros(3)
            current = joint
            while current:
                position += np.asarray(current.offset).flatten()
                current = current.parent
            return position

        def vector_to_axis_name(vector: np.ndarray) -> str:
            """Convert a direction vector to the closest axis name."""
            # Normalize the vector
            vector = vector / np.linalg.norm(vector)

            # Find the axis with maximum absolute component
            abs_components = np.abs(vector)
            max_idx = np.argmax(abs_components)

            # Determine if it's positive or negative
            sign = '-' if vector[max_idx] < 0 else ''
            axis_names = ['x', 'y', 'z']
            return f"{sign}{axis_names[max_idx]}"

        # Find spine joints to determine up direction
        spine_joints = find_joints_by_label("Spine")
        if not spine_joints:
            return {"forward": "x", "up": "y"}  # fallback

        # Calculate spine direction (from bottom to top)
        if len(spine_joints) >= 2:
            # Use the bottom and top spine joints
            spine_bottom = min(spine_joints, key=lambda j: get_joint_world_position(j)[1])
            spine_top = max(spine_joints, key=lambda j: get_joint_world_position(j)[1])
            spine_vector = get_joint_world_position(spine_top) - get_joint_world_position(spine_bottom)
        else:
            # Use the single spine joint's offset as direction
            spine_vector = spine_joints[0].offset

        # Determine up direction from spine
        up_axis = vector_to_axis_name(spine_vector)

        # Find left and right endpoints for forward direction
        # Try arms first, then legs
        left_endpoints = []
        right_endpoints = []

        # Check arms
        left_arm_joints = find_joints_by_label("LeftArm")
        right_arm_joints = find_joints_by_label("RightArm")

        if left_arm_joints and right_arm_joints:
            # Use the farthest joints in each arm (typically hands/wrists)
            left_endpoints = [max(left_arm_joints, key=lambda j: np.linalg.norm(get_joint_world_position(j)))]
            right_endpoints = [max(right_arm_joints, key=lambda j: np.linalg.norm(get_joint_world_position(j)))]
        else:
            # Fall back to legs
            left_leg_joints = find_joints_by_label("LeftLeg")
            right_leg_joints = find_joints_by_label("RightLeg")

            if left_leg_joints and right_leg_joints:
                # Use the farthest joints in each leg (typically feet/toes)
                left_endpoints = [max(left_leg_joints, key=lambda j: np.linalg.norm(get_joint_world_position(j)))]
                right_endpoints = [max(right_leg_joints, key=lambda j: np.linalg.norm(get_joint_world_position(j)))]

        # Convert up_axis to a vector for cross product calculation
        up_components = {'x': 0, 'y': 1, 'z': 2, '-x': 0, '-y': 1, '-z': 2}

        if left_endpoints and right_endpoints:
            # Calculate left-to-right vector
            left_pos = get_joint_world_position(left_endpoints[0])
            right_pos = get_joint_world_position(right_endpoints[0])
            left_right_vector = right_pos - left_pos

            up_axis_clean = up_axis.lstrip('-')
            up_idx = up_components[up_axis_clean]
            up_vector = np.zeros(3)
            up_vector[up_idx] = 1 if not up_axis.startswith('-') else -1

            # Calculate forward direction as cross product of up and left-right vectors
            # This gives us a direction perpendicular to both up and left-right
            forward_vector = np.cross(up_vector, left_right_vector)

            # If forward vector is zero (vectors are parallel), fall back to projection method
            if np.linalg.norm(forward_vector) < 1e-6:
                # Zero out the up component to get forward direction
                forward_vector = left_right_vector.copy()
                forward_vector[up_idx] = 0
                # If still zero, use the original left-right vector
                if np.linalg.norm(forward_vector) < 1e-6:
                    forward_vector = left_right_vector

            forward_axis = vector_to_axis_name(forward_vector)
        else:
            # If no suitable endpoints found, use a default forward direction
            # perpendicular to the up direction
            up_idx = up_components[up_axis.lstrip('-')]
            available_axes = [i for i in range(3) if i != up_idx]
            forward_axis = ['x', 'y', 'z'][available_axes[0]]

        return {"forward": forward_axis, "up": up_axis}

    def align_orientation(self, new_orientation: dict[str, str]):
        """
        Align the skeleton to a new orientation by transforming bone directions,
        rotations, and root translations.

        Parameters
        ----------
        new_orientation : dict[str, str]
            Dictionary specifying the new forward and up axes, e.g., {"forward": "x", "up": "y"}
            Values can be "x", "y", "z", "-x", "-y", "-z"
        """
        if not self.skeleton:
            return

        # Get current orientation
        current_orientation = self.orientation

        # Calculate transformation quaternion from current to new orientation
        transform_quat = self._calculate_orientation_transform(current_orientation, new_orientation)

        # Apply transformation to all components
        self._transform_root_positions(transform_quat)
        self._transform_all_rotations(transform_quat)
        self._transform_all_offsets(transform_quat)

        # Update default orientation
        self.orientation = new_orientation.copy()

    def _calculate_orientation_transform(self, current_orientation: dict[str, str],
                                        new_orientation: dict[str, str]) -> Quaternions:
        """
        Calculate the quaternion transformation from current to new orientation.
        """
        # Convert axis strings to unit vectors
        def axis_to_vector(axis: str) -> np.ndarray:
            vectors = {
                'x': np.array([1, 0, 0]), '-x': np.array([-1, 0, 0]),
                'y': np.array([0, 1, 0]), '-y': np.array([0, -1, 0]),
                'z': np.array([0, 0, 1]), '-z': np.array([0, 0, -1])
            }
            return vectors[axis]

        def normalize(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v)
            if n < 1e-8:
                return v
            return v / n

        def build_basis(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
            # Orthonormal, right-handed basis: [right, up, forward]
            up_n = normalize(up)
            f_n = normalize(forward)
            right = normalize(np.cross(up_n, f_n))
            # Recompute forward to ensure orthogonality
            forward_ortho = normalize(np.cross(right, up_n))
            basis = np.column_stack([right, up_n, forward_ortho])
            return basis

        # Get current coordinate system vectors
        current_forward = axis_to_vector(current_orientation['forward'])
        current_up = axis_to_vector(current_orientation['up'])

        # Get new coordinate system vectors
        new_forward = axis_to_vector(new_orientation['forward'])
        new_up = axis_to_vector(new_orientation['up'])

        # Build orthonormal right-handed bases
        current_basis = build_basis(current_forward, current_up)
        new_basis = build_basis(new_forward, new_up)

        # Calculate transformation matrix (active rotation mapping current -> new)
        # For orthonormal bases, inverse is transpose
        transform_matrix = new_basis @ current_basis.T

        # Check if this is essentially an identity transformation
        if np.allclose(transform_matrix, np.eye(3), atol=1e-6):
            # Return identity quaternion
            return Quaternions.identity()

        # Convert to quaternion
        transform_quat = Quaternions.from_transforms(transform_matrix.reshape(1, 3, 3))

        return transform_quat

    def _transform_root_positions(self, transform_quat: Quaternions):
        """Apply transformation to root joint positions/translations."""
        if self.skeleton and self.skeleton.positions.size > 0:
            # Transform positions for each frame
            self.skeleton.positions = transform_quat * self.skeleton.positions

    def _transform_all_rotations(self, transform_quat: Quaternions):
        """Apply transformation to all joint rotations."""
        # Calculate inverse transformation for rotations
        transform_quat_inv = -transform_quat

        def transform_joint_rotations(joint: SkeletonJoint):
            if joint.rotations.qs.size > 0:
                # For coordinate system transformation, we need to apply:
                # new_rotation = transform_quat * old_rotation * transform_quat_inv
                joint.rotations = transform_quat * joint.rotations * transform_quat_inv

            # Recursively transform children
            for child in joint.children:
                transform_joint_rotations(child)

        if self.skeleton:
            transform_joint_rotations(self.skeleton)

    def _transform_all_offsets(self, transform_quat: Quaternions):
        """Apply transformation to joint offsets to align bone directions."""
        def transform_joint_offsets(joint: SkeletonJoint):
            if joint.offset.size > 0:
                # Transform offset vector
                transformed_offset = transform_quat * joint.offset
                joint.offset = transformed_offset

            # Recursively transform children
            for child in joint.children:
                transform_joint_offsets(child)

        if self.skeleton:
            transform_joint_offsets(self.skeleton)

    def get_height(self) -> float:
        pass