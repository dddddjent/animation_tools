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
