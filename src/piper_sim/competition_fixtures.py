"""Physical desk fixtures; all dimensions are in the robot base frame.

Additional desks deliberately share the pickup surface height. Existing motion
and perception consumers assume a level work surface; unequal desks need a
separate calibration and controller change, not just a visual offset.
"""

import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


def _values(values):
    return " ".join(str(float(v)) for v in values)


def validate_tables(cfg):
    tables = {"pickup": cfg["table"], **cfg.get("additional_tables", {})}
    if "pickup" in cfg.get("additional_tables", {}):
        raise ValueError("Additional tables cannot replace pickup")
    top = cfg["table"]["top_m"]
    if not np.isfinite(top):
        raise ValueError("Invalid table top")
    rectangles = []
    for name, table in tables.items():
        center, half = np.asarray(table["center_xy_m"]), np.asarray(table["half_size_xy_m"])
        thickness = table.get("thickness_m", top)
        if (
            center.shape != (2,)
            or half.shape != (2,)
            or not np.isfinite([center, half]).all()
            or np.any(half <= 0)
            or not np.isfinite(thickness)
            or thickness <= 0
        ):
            raise ValueError(f"Invalid table geometry: {name}")
        if not np.isclose(table.get("top_m", top), top, atol=1e-9, rtol=0):
            raise ValueError("All competition tables must share the same surface height")
        lo, hi = center - half, center + half
        if any(np.all(hi > a) and np.all(lo < b) for a, b in rectangles):
            raise ValueError("Competition tables overlap")
        rectangles.append((lo, hi))
        ground = cfg.get("environment", {}).get("ground_z_m")
        if ground is not None and (not np.isfinite(ground) or ground >= top - thickness):
            raise ValueError("Ground must be below the table underside")
    return tables


def _wood_texture(path, rgb):
    """Low-contrast deterministic grain: avoid colored perception distractors."""
    rng = np.random.default_rng(1206)
    y, x = np.mgrid[:512, :512]
    warp = 1.8 * np.sin(y / 100) + 0.8 * np.sin(y / 37)
    grain = 0.018 * np.sin((x + warp) * 1.9) + 0.012 * np.sin(x / 13 + y / 170)
    grain += rng.normal(0, 0.004, x.shape)
    pixels = np.clip((np.asarray(rgb) + grain[..., None]) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def build_tables(root, cfg, output):
    tables = validate_tables(cfg)
    world, assets = root.find("worldbody"), root.find("asset")
    environment = cfg.get("environment", {})
    if environment.get("soft_lighting", False):
        ET.SubElement(
            root.find("visual"),
            "headlight",
            ambient=".4 .4 .4",
            diffuse=".6 .6 .6",
            specular=".1 .1 .1",
        )
        ET.SubElement(
            world,
            "light",
            name="desk_fill",
            pos="1 1 2",
            dir="0 0 -1",
            diffuse=".45 .45 .45",
            castshadow="false",
        )
    ground = environment.get("ground_z_m")
    floor = world.find("geom[@name='floor']")
    if ground is not None:
        floor.set("pos", _values([0, 0, ground]))
        floor.set("rgba", _values(environment.get("floor_rgba", [0.56, 0.53, 0.46, 1])))
    if "robot_rgba" in environment:
        for geom in world.find("body[@name='dummy_link']").iter("geom"):
            if geom.get("type") == "mesh":
                geom.set("rgba", _values(environment["robot_rgba"]))
    top = cfg["table"]["top_m"]
    for name, table in tables.items():
        geom_name = "competition_table" if name == "pickup" else "competition_table_" + name
        center, half = np.asarray(table["center_xy_m"]), np.asarray(table["half_size_xy_m"])
        thickness = table.get("thickness_m", top)
        appearance = {"rgba": _values(table.get("rgba", [0.72, 0.73, 0.70, 1]))}
        if "wood_rgb" in table:
            path = output / f"table_{name}_wood.png"
            _wood_texture(path, table["wood_rgb"])
            texture, material = f"desk_{name}_texture", f"desk_{name}_material"
            ET.SubElement(assets, "texture", name=texture, type="2d", file=str(path.resolve()))
            ET.SubElement(
                assets,
                "material",
                name=material,
                texture=texture,
                texrepeat="2 2",
                texuniform="false",
                specular=".12",
                shininess=".15",
            )
            appearance = {"material": material}
        ET.SubElement(
            world,
            "geom",
            name=geom_name,
            type="box",
            pos=_values([*center, top - thickness / 2]),
            size=_values([*half, thickness / 2]),
            **appearance,
        )
        if ground is not None:
            height = top - thickness - ground
            leg_half = min(0.02, min(half) / 5)
            for i, signs in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
                xy = center + np.asarray(signs) * (half - 2 * leg_half)
                ET.SubElement(
                    world,
                    "geom",
                    name=f"{geom_name}_leg_{i}",
                    type="box",
                    pos=_values([*xy, ground + height / 2]),
                    size=_values([leg_half, leg_half, height / 2]),
                    rgba=".16 .17 .18 1",
                )
