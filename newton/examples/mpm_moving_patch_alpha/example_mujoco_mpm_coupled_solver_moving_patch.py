# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dynamic box on alpha moving terrain with fresh particle recycling."""

import warp as wp

import newton
import newton.examples
from newton.examples.mpm_moving_patch_alpha.impl.base_example import MovingTerrainExample


class Example(MovingTerrainExample):
    """Track an autonomous dynamic box using alpha's shared terrain lifecycle."""

    def _emit_rigid_bodies(self, builder: newton.ModelBuilder):
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0), q=wp.quat_identity()),
            mass=75.0,
        )
        builder.add_shape_box(body, hx=float(0.25), hy=float(0.35), hz=float(0.25))
        return builder.body_label[0]


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
