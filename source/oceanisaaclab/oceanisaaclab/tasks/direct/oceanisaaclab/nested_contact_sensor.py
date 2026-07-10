"""PhysX contact sensor support for rigid bodies nested by the URDF importer."""

from __future__ import annotations

from isaaclab.sensors.contact_sensor import BaseContactSensor
from isaaclab_physx.physics import PhysxManager as SimulationManager
from isaaclab_physx.sensors.contact_sensor import ContactSensor as PhysXContactSensor


class NestedBodyContactSensor(PhysXContactSensor):
    """Contact sensor for one rigid body at an exact, possibly nested, prim path.

    Isaac Lab's PhysX contact sensor discovers nested rigid bodies, but rebuilds their
    paths as if every body were a direct child of the articulation root. The URDF
    importer used by this project preserves the kinematic hierarchy, so that path
    reconstruction produces invalid paths such as ``base_link/base_link``. This
    variant keeps the configured full path and otherwise uses the standard PhysX
    buffers, timing, and contact kernels.
    """

    def _initialize_impl(self) -> None:
        BaseContactSensor._initialize_impl(self)
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        if "/Geometry/" not in self.cfg.prim_path:
            raise ValueError(
                "NestedBodyContactSensor expects a path below the articulation Geometry prim."
            )
        articulation_path, nested_body_path = self.cfg.prim_path.split("/Geometry/", 1)
        body_name = nested_body_path.rsplit("/", 1)[-1]
        # PhysX articulation views select links by name below the articulation's
        # Geometry container, even though USD keeps the full kinematic hierarchy.
        body_path_glob = f"{articulation_path.replace('.*', '*')}/Geometry/({body_name})"
        filter_prim_paths_glob = [
            expression.replace(".*", "*") for expression in self.cfg.filter_prim_paths_expr
        ]
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_path_glob)
        self._contact_view = self._physics_sim_view.create_rigid_contact_view(
            body_path_glob,
            filter_patterns=filter_prim_paths_glob,
            max_contact_data_count=self.cfg.max_contact_data_count_per_prim * self._num_envs,
        )

        self._num_sensors = self.body_physx_view.count // self._num_envs
        if self._num_sensors != 1:
            raise RuntimeError(
                "NestedBodyContactSensor requires one exact rigid-body path per environment."
                f"\n\tInput prim path : {self.cfg.prim_path}"
                f"\n\tPhysX glob      : {body_path_glob}"
                f"\n\tMatched bodies  : {self.body_physx_view.count} across {self._num_envs} environments"
            )

        if self.cfg.track_contact_points or self.cfg.track_friction_forces:
            if not self.cfg.filter_prim_paths_expr:
                raise ValueError(
                    "Contact-point and friction tracking require filter_prim_paths_expr."
                )
            if self.cfg.max_contact_data_count_per_prim < 1:
                raise ValueError("max_contact_data_count_per_prim must be greater than zero.")

        self._create_buffers()
