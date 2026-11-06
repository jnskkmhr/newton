# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Author static desert geometry, procedural materials, and lighting as USD."""

from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from .terrain import build_rock, sample_heights


def _textures(directory: Path, *, seed: int):
    """Generate repeatable sand and rock color/normal textures without external assets."""
    rng = np.random.default_rng(seed)
    size = 1024
    y, x = np.mgrid[:size, :size].astype(np.float32) / size
    grain = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    ripple = np.sin(2 * np.pi * (38 * x + 5 * y) + 0.65 * np.sin(2 * np.pi * 3 * y))
    for name, base in [("sand", (0.72, 0.55, 0.34)), ("rock", (0.43, 0.36, 0.28))]:
        if name == "sand":
            variation = 0.025 * grain + 0.012 * ripple
            height = 0.0008 * ripple + 0.0004 * grain
        else:
            variation = 0.06 * grain
            for frequency, amplitude in ((3, 0.07), (11, 0.035), (37, 0.025)):
                variation += amplitude * np.sin(2 * np.pi * (frequency * x + y)) * np.cos(2 * np.pi * frequency * y)
            height = variation * 0.025
        color = np.clip(np.asarray(base)[None, None, :] + variation[..., None], 0.0, 1.0)
        Image.fromarray(np.uint8(color * 255)).save(directory / f"{name}_color.png")
        dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) / (16.0 / size)
        dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) / (16.0 / size)
        normal = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        Image.fromarray(np.uint8(np.clip(normal * 0.5 + 0.5, 0.0, 1.0) * 255)).save(directory / f"{name}_normal.png")


def _sky_texture(path: Path):
    """Write a small linear HDR sky gradient using the Radiance RGBE format."""
    width, height = 512, 256
    elevation = np.cos(np.linspace(0, np.pi, height))[:, None, None]
    horizon = np.array([0.48, 0.66, 0.86])[None, None, :]
    zenith = np.array([0.08, 0.25, 0.56])[None, None, :]
    sky = horizon + (zenith - horizon) * np.maximum(elevation, 0.0) ** 0.5
    sky = np.where(elevation >= 0, sky, np.array([0.28, 0.22, 0.15])[None, None, :])
    sky = np.repeat(sky, width, axis=1)
    mantissa, exponent = np.frexp(np.max(sky, axis=-1))
    scale = mantissa * 256 / np.maximum(np.max(sky, axis=-1), 1.0e-10)
    rgbe = np.zeros((height, width, 4), np.uint8)
    rgbe[..., :3] = np.uint8(sky * scale[..., None])
    rgbe[..., 3] = np.uint8(exponent + 128)
    with path.open("wb") as stream:
        stream.write(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n".encode())
        for row in rgbe:
            stream.write(bytes([2, 2, width >> 8, width & 255]))
            for channel in range(4):
                for start in range(0, width, 128):
                    chunk = row[start : start + 128, channel].tobytes()
                    stream.write(bytes([len(chunk)]))
                    stream.write(chunk)


def _material(stage, path: str, directory: Path, name: str):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82 if name == "sand" else 0.92)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    st = UsdShade.Shader.Define(stage, path + "/UV")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for suffix, input_name in [("color", "diffuseColor"), ("normal", "normal")]:
        texture = UsdShade.Shader.Define(stage, path + "/" + suffix)
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(directory / f"{name}_{suffix}.png"))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if suffix == "color" else "raw")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        if suffix == "normal":
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.0, -1.0, -1.0, 0.0))
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput(
            input_name, Sdf.ValueTypeNames.Normal3f if suffix == "normal" else Sdf.ValueTypeNames.Color3f
        ).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mesh(stage, path, points, uvs, indices, material, *, normals=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices) // 3, 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([points.min(axis=0), points.max(axis=0)])))
    if normals is not None:
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        mesh.SetNormalsInterpolation("vertex")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
        Vt.Vec2fArray.FromNumpy(uvs)
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def write_scene(
    directory: Path,
    terrain,
    *,
    seed: int,
    rock_count: int,
    device,
    sun_elevation: float,
    g1_asset: Path | None = None,
    include_terrain: bool = True,
):
    """Write a static USD desert and return its path and geometry statistics."""
    directory.mkdir(parents=True, exist_ok=True)
    _textures(directory, seed=seed)
    _sky_texture(directory / "sky.hdr")
    path = directory / "desert.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Desert")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    sand = _material(stage, "/Desert/Materials/Sand", directory, "sand")
    rock = _material(stage, "/Desert/Materials/Rock", directory, "rock")
    points, normals, uvs, indices = terrain
    if include_terrain:
        _mesh(stage, "/Desert/Dunes", points, uvs, indices, sand, normals=normals)
    rng = np.random.default_rng(seed)
    instancer = UsdGeom.PointInstancer.Define(stage, "/Desert/Stones")
    prototypes = []
    for i in range(6):
        p, uv, idx, n = build_rock(seed=seed + i)
        mesh = _mesh(stage, f"/Desert/Stones/Rock{i}", p, uv, idx, rock, normals=n)
        prototypes.append(mesh.GetPath())
    instancer.CreatePrototypesRel().SetTargets(prototypes)
    # Concentrate stones around the basin, with a sparse scattering farther out.
    angle = rng.uniform(0, 2 * np.pi, rock_count)
    radius = rng.uniform(6.0, 17.0, rock_count)
    far = np.arange(rock_count) % 4 == 0
    radius[far] = rng.uniform(18, 85, far.sum())
    xy = np.stack([11 + radius * np.cos(angle), 19 + radius * np.sin(angle)], axis=-1).astype(np.float32)
    heights = sample_heights(xy, seed=seed, device=device)
    size = rng.lognormal(-2.4, 0.65, rock_count).clip(0.025, 0.6)
    size[: min(45, rock_count)] = rng.uniform(0.5, 1.8, min(45, rock_count))
    scales = np.stack(
        [size, size * rng.uniform(0.65, 1.3, rock_count), size * rng.uniform(0.5, 0.9, rock_count)], axis=-1
    ).astype(np.float32)
    positions = np.column_stack([xy, heights + scales[:, 2] * 0.35]).astype(np.float32)
    orientations = [Gf.Quath(float(np.cos(a / 2)), Gf.Vec3h(0, 0, float(np.sin(a / 2)))) for a in angle]
    instancer.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(rng.integers(0, 6, rock_count, dtype=np.int32)))
    instancer.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(positions))
    instancer.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(scales))
    instancer.CreateOrientationsAttr(Vt.QuathArray(orientations))
    dome = UsdLux.DomeLight.Define(stage, "/Desert/Sky")
    dome.CreateTextureFileAttr(str(directory / "sky.hdr"))
    dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(350.0)
    sun = UsdLux.DistantLight.Define(stage, "/Desert/Sun")
    sun.CreateIntensityAttr(2600.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.91, 0.77))
    sun.CreateAngleAttr(0.7)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(0, 90 - sun_elevation, -35))
    statistics = {"terrain_vertices": len(points), "terrain_triangles": len(indices) // 3, "rock_instances": rock_count}
    if g1_asset is not None:
        placement = UsdGeom.Xform.Define(stage, "/Desert/G1Placement")
        robot = stage.DefinePrim("/Desert/G1Placement/Robot")
        robot.GetReferences().AddReference(str(g1_asset))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        bounds = cache.ComputeWorldBound(robot).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise ValueError("G1 asset has no visible geometry")
        floor = float(sample_heights(np.array([[0, -3]], np.float32), seed=seed, device=device)[0])
        placement.AddTranslateOp().Set(Gf.Vec3d(0, -3, floor + 0.015 - bounds.GetMin()[2]))
        placement.AddRotateZOp().Set(90.0)
        cache.Clear()
        placed = cache.ComputeWorldBound(robot).ComputeAlignedRange()
        statistics["g1_height_m"] = placed.GetMax()[2] - placed.GetMin()[2]
        statistics["g1_ground_clearance_m"] = placed.GetMin()[2] - floor
    stage.GetRootLayer().Save()
    return path, statistics
