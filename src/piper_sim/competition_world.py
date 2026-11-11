"""Configurable competition fixtures. Procedural props are not official fruit assets.

Observations contain sensors/state only. Semantic ground truth lives in evaluator exports.
"""

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from piper_sim.backend import MujocoBackend
from piper_sim.calibration import R_OPTICAL_MUJOCO, ROOT, load_config
from piper_sim.competition_fixtures import build_tables, validate_tables
from piper_sim.task_scene import configure_manipulator


def numbers(values):
    return " ".join(str(float(x)) for x in values)


def part_bounds(part):
    if "fromto" in part:
        ends = np.asarray(part["fromto"]).reshape(2, 3)
        return ends.min(0) - part["size"][0], ends.max(0) + part["size"][0]
    size = np.array(part["size"])
    kind = part["type"]
    ext = (
        size
        if kind in ("box", "ellipsoid")
        else (
            np.repeat(size[0], 3) if kind == "sphere" else np.array([size[0], size[0], sum(size)])
        )
    )
    return np.array(part["pos"]) - ext, np.array(part["pos"]) + ext


def look_rotation(position, target):
    z = np.asarray(position) - np.asarray(target)  # MuJoCo camera looks along -Z.
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class ScenePackingError(ValueError):
    """Retryable random placement failure; distinct from invalid user configuration."""


class CompetitionWorld:
    def __init__(self, task="exploration", config=None, output=None, render=False):
        self.cfg = yaml.safe_load(
            Path(config or ROOT / "config/competition/scene.yaml").read_text()
        )
        if self.cfg.get("camera_config"):
            calibration = load_config(ROOT / self.cfg["camera_config"])
            sensor, hand_eye = calibration["camera"], calibration["hand_eye"]
            cam = self.cfg["camera"]
            cam.update(
                width=sensor["width"],
                height=sensor["height"],
                intrinsics=copy.deepcopy(sensor["intrinsics"]),
                depth_range_m=[sensor["depth"]["min_valid_m"], sensor["depth"]["max_valid_m"]],
                calibration_status=calibration["calibration"]["status"],
                calibration_source=self.cfg["camera_config"],
                model=sensor["model"],
            )
            if cam["mount"] == "wrist":
                cam.update(
                    parent_body=hand_eye["parent_body"],
                    T_parent_optical=copy.deepcopy(hand_eye["T_end_camera"]),
                    visual=copy.deepcopy(calibration["simulation"]["camera_visual"]),
                )
        self._validate_regions()
        if task not in self.cfg["tasks"]:
            raise ValueError(f"Unknown task {task}")
        self.catalog = yaml.safe_load((ROOT / self.cfg["object_catalog"]).read_text())["objects"]
        self.task = task
        self.output = Path(output or ROOT / "artifacts/competition" / task)
        self.output.mkdir(parents=True, exist_ok=True)
        self.render = render
        self.backend = None
        self.active_request = None
        self.time_origin_s = 0.0

    def _validate_regions(self):
        tables = validate_tables(self.cfg)
        regions = []
        for name, zone in self.cfg["trays"].items():
            support = zone.get("support_table", "pickup")
            if support not in tables:
                raise ValueError(f"Unknown support table {support} for target region {name}")
            table = tables[support]
            center = np.array(table["center_xy_m"])
            half = np.array(table["half_size_xy_m"])
            point, size = np.array(zone["center_xy_m"]), np.array(zone["half_size_xy_m"])
            if (
                point.shape != (2,)
                or size.shape != (2,)
                or not np.isfinite([point, size]).all()
                or np.any(size <= 0)
            ):
                raise ValueError(f"Invalid target region {name}")
            lo, hi = point - size, point + size
            if np.any(lo < center - half - 1e-9) or np.any(hi > center + half + 1e-9):
                raise ValueError(f"Target region {name} extends beyond the table")
            drop = np.array(zone.get("placement_xy_m", point))
            if (
                drop.shape != (2,)
                or not np.isfinite(drop).all()
                or np.any(drop < lo)
                or np.any(drop > hi)
            ):
                raise ValueError(f"Placement point outside target region {name}")
            if any(np.all(hi > a) and np.all(lo < b) for a, b in regions):
                raise ValueError("Target regions overlap")
            regions.append((lo, hi))
        for task in self.cfg["tasks"].values():
            if any(color not in self.cfg["trays"] for color in task.get("mapping", {}).values()):
                raise ValueError("Task mapping references a missing target region")

    def reset(self, seed=0):
        self.close()
        self.seed = int(seed)
        self.active_request = None
        for attempt in range(32):
            rng = np.random.default_rng(np.random.SeedSequence([seed, attempt]))
            try:
                self._build(rng)
            except ScenePackingError:
                continue
            camera = load_config(ROOT / self.cfg.get("camera_config", "config/eye_in_hand.yaml"))
            cam = self.cfg["camera"]
            if cam["mount"] == "wrist":
                camera["hand_eye"]["parent_body"] = cam["parent_body"]
                camera["hand_eye"]["T_end_camera"] = copy.deepcopy(cam["T_parent_optical"])
            camera["simulation"]["initial_qpos"] = self.cfg["robot_initial_qpos"]
            camera["camera"].update(
                name="competition_camera",
                width=cam["width"],
                height=cam["height"],
                intrinsics=copy.deepcopy(cam["intrinsics"]),
            )
            camera["camera"]["depth"].update(
                min_valid_m=cam["depth_range_m"][0], max_valid_m=cam["depth_range_m"][1]
            )
            b = MujocoBackend(camera=camera, render=False, model_path=self.output / "scene.xml")
            self._apply_colors(b.model)
            b.reset()
            for _ in range(round(self.cfg["spawn"]["settle_s"] / b.model.opt.timestep)):
                mujoco.mj_step(b.model, b.data)
            mujoco.mj_forward(b.model, b.data)
            self.backend = b
            if self._valid_reset():
                break
            self.close()
        else:
            raise RuntimeError("Could not obtain a separated, collision-free initial scene")
        b.data.time = 0
        self.time_origin_s = 0.0
        if self.render:
            b.renderer = mujoco.Renderer(b.model, height=cam["height"], width=cam["width"])
        self.events = {
            o["id"]: {"lift_s": 0.0, "grasped": False, "stable_s": {}, "placed": []}
            for o in self.objects
        }
        self.initial_camera_pose = self.camera_pose()
        self.initial_centers = {
            o["id"]: b.data.xpos[b.model.body(o["id"]).id].copy() for o in self.objects
        }
        self.reset_attempt = attempt
        self.export_metadata()
        return self.observe()

    def _build(self, rng):
        robot_scene = ROOT / "assets/piper/scene.xml"
        root = ET.parse(robot_scene).getroot()
        for mesh in root.findall("./asset/mesh"):
            mesh.set("file", str(robot_scene.parent / mesh.attrib["file"]))
        world = root.find("worldbody")
        for light in world.findall("light"):
            light.set("ambient", ".25 .25 .25")
            light.set("diffuse", ".8 .8 .8")
        # Remove previous demonstration props/camera markers, retaining the PiPER model.
        for parent in root.iter():
            for child in list(parent):
                name = child.get("name", "")
                if name.startswith("target_") or name in (
                    "demo_table",
                    "camera_visual_marker",
                    "camera_bracket",
                    "wrist_camera",
                ):
                    parent.remove(child)
        configure_manipulator(root, 0.122, 800, 50, 1500, 4)
        root.find("option").set("cone", self.cfg["physics"]["cone"])
        root.find("option").set("impratio", str(self.cfg["physics"]["impratio"]))
        top = self.cfg["table"]["top_m"]
        table = self.cfg["table"]
        build_tables(root, self.cfg, self.output)
        tg = self.cfg["tray_geometry"]
        for color, tray in self.cfg["trays"].items():
            hx, hy = tray["half_size_xy_m"]
            t, f, h = (tg[k] for k in ("wall_thickness_m", "floor_thickness_m", "wall_height_m"))
            body = ET.SubElement(
                world, "body", name="tray_" + color, pos=numbers([*tray["center_xy_m"], top])
            )
            pieces = [
                ("floor", [0, 0, f / 2], [hx + t, hy + t, f / 2]),
                ("left", [-hx - t / 2, 0, f + h / 2], [t / 2, hy + t, h / 2]),
                ("right", [hx + t / 2, 0, f + h / 2], [t / 2, hy + t, h / 2]),
                ("front", [0, -hy - t / 2, f + h / 2], [hx, t / 2, h / 2]),
                ("back", [0, hy + t / 2, f + h / 2], [hx, t / 2, h / 2]),
            ]
            # A colored target area is a thin mat, not a container with walls.
            if tg.get("kind", "tray") == "flat_zone":
                pieces = [("floor", [0, 0, f / 2], [hx, hy, f / 2])]
            for name, pos, size in pieces:
                ET.SubElement(
                    body,
                    "geom",
                    name=f"tray_{color}_{name}",
                    type="box",
                    pos=numbers(pos),
                    size=numbers(size),
                    rgba=numbers(tray["rgba"]),
                )
        self.objects = []
        boxes = []
        task = self.cfg["tasks"][self.task]
        spawn = {**self.cfg["spawn"], **task.get("spawn_override", {})}
        if "object_pool" in task:
            pool = task["object_pool"]
            count = task["object_count"]
            if len(set(pool)) != len(pool) or not 1 <= count <= len(pool):
                raise ValueError("Object pool must be unique and count within pool size")
            category_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 994]))
            categories = category_rng.choice(pool, count, replace=False).tolist()
        else:
            categories = list(task["objects"])
        for i, category in enumerate(categories):
            asset = self.catalog[category]
            name = f"object_{i + 1}"
            variation = self.cfg.get("object_variation", {})
            enabled = variation.get("enabled", False)
            scale_range = (
                variation.get("scale_ranges", {}).get(category, spawn["scale_range"])
                if enabled
                else spawn["scale_range"]
            )
            override = task.get("instance_overrides", {}).get(name, {}) if enabled else {}
            if len(scale_range) != 2 or not 0 < scale_range[0] <= scale_range[1]:
                raise ValueError(f"Invalid size range for {category}")
            scale = float(override.get("scale", rng.uniform(*scale_range)))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid size scale for {name}")
            palette = (
                variation.get("color_palettes", {}).get(category, ["original"])
                if enabled
                else ["original"]
            )
            color_rng = np.random.default_rng(np.random.SeedSequence([self.seed, i, 510]))
            color = str(override.get("color", color_rng.choice(palette)))
            if color not in palette:
                raise ValueError(f"Color {color} is not allowed for {category}: {palette}")
            if color != "original" and color not in variation.get("colors", {}):
                raise ValueError(f"Unknown color variant {color}")
            parts = copy.deepcopy(asset.get("parts", []))
            if "visual_mesh" in asset:
                low, high = np.asarray(asset["visual_mesh"]["bounds_m"]) * scale
            else:
                for part in parts:
                    for key in ("size", "pos", "fromto"):
                        if key in part:
                            part[key] = (np.array(part[key]) * scale).tolist()
                bounds = [part_bounds(p) for p in parts]
                low = np.min([x[0] for x in bounds], axis=0)
                high = np.max([x[1] for x in bounds], axis=0)
            yaw = rng.uniform(-np.pi, np.pi)
            rotation = Rotation.from_euler("z", yaw)
            # Rotated bounding boxes allow elongated fruit to share the workspace
            # without the excessive empty circles used for small round fruit.
            corners = np.array(
                [
                    [x, y, z]
                    for x in (low[0], high[0])
                    for y in (low[1], high[1])
                    for z in (low[2], high[2])
                ]
            )
            rotated = rotation.apply(corners)
            local_lo, local_hi = rotated[:, :2].min(0), rotated[:, :2].max(0)
            for _ in range(spawn["max_attempts"]):
                xy = np.array([rng.uniform(*spawn["x_m"]), rng.uniform(*spawn["y_m"])])
                lo, hi = xy + local_lo, xy + local_hi
                occupied = boxes + [
                    (
                        np.array(t["center_xy_m"]) - t["half_size_xy_m"],
                        np.array(t["center_xy_m"]) + t["half_size_xy_m"],
                    )
                    for t in self.cfg["trays"].values()
                ]
                table_center, table_half = (
                    np.array(table["center_xy_m"]),
                    np.array(table["half_size_xy_m"]),
                )
                if (
                    np.all(lo >= table_center - table_half)
                    and np.all(hi <= table_center + table_half)
                    and all(
                        np.any(hi + spawn["separation_m"] < other_lo)
                        or np.any(lo - spawn["separation_m"] > other_hi)
                        for other_lo, other_hi in occupied
                    )
                ):
                    break
            else:
                raise ScenePackingError("Spawn region too small for configured objects/separation")
            boxes.append((lo, hi))
            body = ET.SubElement(
                world,
                "body",
                name=name,
                pos=numbers([*xy, top - low[2] + 0.003]),
                quat=numbers(rotation.as_quat()[[3, 0, 1, 2]]),
            )
            ET.SubElement(body, "freejoint", name=name + "_free")
            mass = rng.uniform(*asset["mass_kg_range"])
            if "visual_mesh" in asset:
                self._add_scanned_object(root, body, name, category, asset, scale, mass)
            for j, part in enumerate(parts):
                attrs = {k: numbers(v) if isinstance(v, list) else v for k, v in part.items()}
                if "fromto" in attrs:
                    attrs.pop("pos")
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{name}_part_{j}",
                    **attrs,
                    mass=str(mass / len(parts)),
                    condim="6",
                    priority="1",
                    friction=f"{self.cfg['physics']['friction']} .01 .001",
                    solref=".005 1",
                )
            self.objects.append(
                {
                    "id": name,
                    "category": category,
                    "scale": scale,
                    "color_variant": color,
                    "mass_kg": mass,
                    "initial_local_extents_m": (high - low).tolist(),
                }
            )
        cam = self.cfg["camera"]
        if cam["mount"] == "fixed":
            parent, pos = world, cam["position_m"]
            rotation = look_rotation(pos, cam["look_at_m"])
        elif cam["mount"] == "wrist":
            parent = root.find(f".//body[@name='{cam['parent_body']}']")
            if parent is None:
                raise ValueError("Unknown camera parent")
            transform = np.asarray(cam["T_parent_optical"], dtype=float)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("Invalid camera transform")
            np.testing.assert_allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
            np.testing.assert_allclose(
                transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6
            )
            if not np.isclose(np.linalg.det(transform[:3, :3]), 1):
                raise ValueError("Camera rotation must be right-handed")
            pos, rotation = transform[:3, 3], transform[:3, :3] @ R_OPTICAL_MUJOCO
        else:
            raise ValueError("Camera mount must be fixed or wrist")
        k = cam["intrinsics"]
        ET.SubElement(
            parent,
            "camera",
            name="competition_camera",
            pos=numbers(pos),
            quat=numbers(Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]),
            resolution=f"{cam['width']} {cam['height']}",
            sensorsize="1 1",
            focalpixel=numbers([k["fx"], k["fy"]]),
            principalpixel=numbers([cam["width"] / 2 - k["cx"], cam["height"] / 2 - k["cy"]]),
        )
        if cam["mount"] == "wrist" and "visual" in cam:
            visual = cam["visual"]
            marker = ET.SubElement(
                parent,
                "body",
                name="wrist_sensor_housing",
                pos=numbers(pos),
                quat=numbers(Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]),
            )
            width, height, thickness = visual["housing_size_m"]
            attrs = {"contype": "0", "conaffinity": "0", "mass": "0", "group": "2"}
            ET.SubElement(
                marker,
                "geom",
                type="box",
                name="wrist_sensor_shell",
                pos=numbers([0, 0, thickness / 2]),
                size=numbers([width / 2, height / 2, thickness / 2]),
                rgba=".85 .87 .88 1",
                **attrs,
            )
            for i, x in enumerate([-width * 0.35, 0, width * 0.35]):
                ET.SubElement(
                    marker,
                    "geom",
                    type="cylinder",
                    name=f"wrist_sensor_lens_{i}",
                    pos=numbers([x, 0, -0.001]),
                    size=".005 .001",
                    rgba=".02 .03 .04 1",
                    **attrs,
                )
            ET.SubElement(
                parent,
                "geom",
                type="box",
                name="wrist_sensor_mount",
                pos=numbers(visual["bracket_position_m"]),
                size=numbers(np.array(visual["bracket_size_m"]) / 2),
                rgba=".12 .13 .15 1",
                **attrs,
            )
        root.find("visual/global").set("offwidth", str(cam["width"]))
        root.find("visual/global").set("offheight", str(cam["height"]))
        ET.indent(root)
        ET.ElementTree(root).write(self.output / "scene.xml")

    def _add_scanned_object(self, root, body, name, category, asset, scale, mass):
        assets = root.find("asset")
        # Unique texture per instance: two apples can have different colors.
        material = "scan_" + name
        if assets.find(f"material[@name='{material}']") is None:
            ET.SubElement(
                assets,
                "texture",
                name=material,
                type="2d",
                file=str(ROOT / asset["visual_mesh"]["texture"]),
            )
            ET.SubElement(
                assets,
                "material",
                name=material,
                texture=material,
                specular=str(asset["visual_mesh"].get("material", {}).get("specular", .12)),
                shininess=str(asset["visual_mesh"].get("material", {}).get("shininess", .15)),
                rgba="1 1 1 1",
            )
        ET.SubElement(
            assets,
            "mesh",
            name=name + "_visual",
            file=str(ROOT / asset["visual_mesh"]["file"]),
            scale=numbers([scale] * 3),
        )
        ET.SubElement(
            body,
            "geom",
            name=name + "_visual",
            type="mesh",
            mesh=name + "_visual",
            material=material,
            mass="0",
            contype="0",
            conaffinity="0",
            group="1",
        )
        volume = sum(p["volume_m3"] for p in asset["collision_meshes"])
        for i, piece in enumerate(asset["collision_meshes"]):
            mesh_name = f"{name}_collision_{i}"
            ET.SubElement(
                assets,
                "mesh",
                name=mesh_name,
                file=str(ROOT / piece["file"]),
                scale=numbers([scale] * 3),
            )
            ET.SubElement(
                body,
                "geom",
                name=mesh_name,
                type="mesh",
                mesh=mesh_name,
                mass=str(mass * piece["volume_m3"] / volume),
                group="3",
                rgba=".3 .8 .3 .3",
                condim="6",
                priority="1",
                friction=f"{self.cfg['physics']['friction']} .01 .001",
                solref=".005 1",
            )

    def _apply_colors(self, model):
        """Runtime texture augmentation; source scan PNGs are never changed."""
        import cv2

        for obj in self.objects:
            if (
                obj["color_variant"] == "original"
                or "visual_mesh" not in self.catalog[obj["category"]]
            ):
                continue
            settings = self.cfg["object_variation"]["colors"][obj["color_variant"]]
            hue, sat, value = (
                float(settings[k]) for k in ("hue_deg", "saturation_scale", "value_scale")
            )
            if not np.isfinite([hue, sat, value]).all() or not (
                0 <= hue < 360 and sat > 0 and value > 0
            ):
                raise ValueError(
                    "Color hue must be in [0,360); saturation/value scales must be positive"
                )
            tid = model.texture("scan_" + obj["id"]).id
            start = model.tex_adr[tid]
            height, width, channels = (
                int(a[tid]) for a in (model.tex_height, model.tex_width, model.tex_nchannel)
            )
            pixels = model.tex_data[start : start + height * width * channels].reshape(
                height, width, channels
            )
            hsv = cv2.cvtColor(pixels[:, :, :3].copy(), cv2.COLOR_RGB2HSV)
            mask = (hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 25)
            if mask.any():
                histogram = np.bincount(hsv[:, :, 0][mask], minlength=180)
                dominant = int(histogram.argmax())
                window = float(self.cfg["object_variation"].get("main_color_hue_window_deg", 36))
                if not np.isfinite(window) or not 0 < window <= 180:
                    raise ValueError("Main color hue window must be in (0,180] degrees")
                difference = np.abs(hsv[:, :, 0].astype(np.int16) - dominant)
                difference = np.minimum(difference, 180 - difference)
                mask &= difference <= window / 2
                shifted = (hsv[:, :, 0].astype(np.int16) + round(hue / 2) - dominant) % 180
                hsv[:, :, 0][mask] = shifted[mask].astype(np.uint8)
                hsv[:, :, 1][mask] = np.clip(hsv[:, :, 1][mask].astype(float) * sat, 0, 255).astype(
                    np.uint8
                )
                hsv[:, :, 2][mask] = np.clip(
                    hsv[:, :, 2][mask].astype(float) * value, 0, 255
                ).astype(np.uint8)
                pixels[:, :, :3][mask] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[mask]

    def bounds(self, object_id):
        b = self.backend
        bid = b.model.body(object_id).id
        lows, highs = [], []
        for g in range(
            b.model.body_geomadr[bid], b.model.body_geomadr[bid] + b.model.body_geomnum[bid]
        ):
            r, size = b.data.geom_xmat[g].reshape(3, 3), b.model.geom_size[g]
            kind = b.model.geom_type[g]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                # Hull pieces cover scan vertices; skip the separate non-contact visual mesh.
                if b.model.geom_contype[g] == 0:
                    continue
                mid = b.model.geom_dataid[g]
                start, count = b.model.mesh_vertadr[mid], b.model.mesh_vertnum[mid]
                points = b.model.mesh_vert[start : start + count] @ r.T + b.data.geom_xpos[g]
                lows.append(points.min(axis=0))
                highs.append(points.max(axis=0))
                continue
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                ext = np.sqrt((r * r) @ (size * size))
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                ext = np.repeat(size[0], 3)
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                ext = abs(r[:, 2]) * size[1] + size[0]
            else:
                ext = abs(r) @ size
            lows.append(b.data.geom_xpos[g] - ext)
            highs.append(b.data.geom_xpos[g] + ext)
        return np.min(lows, 0), np.max(highs, 0)

    def _valid_reset(self):
        b = self.backend
        if not np.isfinite(b.data.qpos).all() or not np.isfinite(b.data.qvel).all():
            return False
        table = self.cfg["table"]
        center, half = np.array(table["center_xy_m"]), np.array(table["half_size_xy_m"])
        ids = {b.model.body(o["id"]).id for o in self.objects}
        robot_ids = {
            b.model.body(n).id
            for n in ["base_link", "gripper_base"] + [f"link{i}" for i in range(1, 9)]
        }
        for obj in self.objects:
            lo, hi = self.bounds(obj["id"])
            if np.any(lo[:2] < center - half) or np.any(hi[:2] > center + half):
                return False
            if lo[2] < table["top_m"] - 0.001:
                return False
        for c in b.data.contact[: b.data.ncon]:
            bodies = {int(b.model.geom_bodyid[g]) for g in (c.geom1, c.geom2)}
            if bodies & ids:
                if len(bodies & ids) > 1 or bodies & robot_ids or c.dist < -0.001:
                    return False
                if any(
                    (b.model.geom(g).name or "").startswith("tray_") for g in (c.geom1, c.geom2)
                ):
                    return False
        return True

    def camera_pose(self):
        b = self.backend
        i = b.model.camera("competition_camera").id
        t = np.eye(4)
        t[:3, :3] = b.data.cam_xmat[i].reshape(3, 3) @ R_OPTICAL_MUJOCO
        t[:3, 3] = b.data.cam_xpos[i]
        return t

    def observe(self):
        obs = self.backend.observe()
        return {
            "rgb": obs.rgb,
            "depth_m": obs.depth_m,
            "joint_position_rad": obs.joint_position_rad,
            "joint_velocity_rad_s": obs.joint_velocity_rad_s,
            "gripper_width_m": obs.gripper_width_m,
            "T_base_end": obs.T_base_end,
            "timestamp_s": obs.timestamp_s,
            "intrinsics": copy.deepcopy(self.cfg["camera"]["intrinsics"]),
            "T_base_camera": self.camera_pose(),
            "instruction": self.active_request["text"] if self.active_request else None,
        }

    def expected_selection(self, request):
        """Evaluator ONLY. Never pass this result to a visual actor/model."""
        candidates = [o["id"] for o in self.objects if o["category"] == request["category"]]
        if not candidates:
            return None
        if request["relation"] == "unique":
            return candidates[0] if len(candidates) == 1 else None
        if request["relation"] in ("largest", "smallest"):
            sizes = {o["id"]: np.prod(o["initial_local_extents_m"]) for o in self.objects}
            ordered = sorted(candidates, key=sizes.get)
            if len(ordered) > 1 and np.isclose(sizes[ordered[0]], sizes[ordered[-1]], rtol=0.02):
                return None
            return ordered[-1 if request["relation"] == "largest" else 0]
        inverse = np.linalg.inv(self.initial_camera_pose)

        def screen_x(name):
            p = inverse @ np.r_[self.initial_centers[name], 1]
            return p[0] / p[2]

        if request["relation"] not in ("leftmost", "rightmost"):
            raise ValueError("Unknown relation")
        return sorted(candidates, key=screen_x)[0 if request["relation"] == "leftmost" else -1]

    def select_instruction(self, index):
        requests = self.cfg["tasks"]["exploration"]["instructions"]
        if self.task != "exploration":
            raise ValueError("Instructions only apply to exploration scenes")
        if self.backend.data.time > 1e-9:
            raise ValueError("Reset before selecting another independent instruction test")
        self.active_request = copy.deepcopy(requests[index])
        self.request_start_s = self.backend.data.time
        self.request_target = self.expected_selection(self.active_request)
        self.request_place_s = 0.0
        self.request_complete = False
        return self.active_request["text"]

    def begin_attempt(self):
        """New interactive attempt budget/judging, retaining all physical state."""
        self.time_origin_s = float(self.backend.data.time)
        self.active_request = None
        self.events = {
            o["id"]: {"lift_s": 0.0, "grasped": False, "stable_s": {}, "placed": []}
            for o in self.objects
        }

    @property
    def elapsed_s(self):
        return float(self.backend.data.time) - self.time_origin_s

    def step(self, action, observe=True):
        b = self.backend
        from piper_sim.control import action_to_target

        if self.elapsed_s >= self.cfg["tasks"][self.task]["time_limit_s"]:
            return self.observe() if observe else None
        target = action_to_target(action, b.measured(), b.previous_target, b.interface)
        b.previous_target = target
        b._set_target(target)
        for _ in range(b.substeps):
            if self.elapsed_s >= self.cfg["tasks"][self.task]["time_limit_s"] - 1e-9:
                break
            mujoco.mj_step(b.model, b.data)
            mujoco.mj_forward(b.model, b.data)
            self._judge()
        return self.observe() if observe else None

    def _judge(self):
        b, cfg = self.backend, self.cfg["provisional_judging"]
        dt = b.model.opt.timestep
        finger_ids = {b.model.body("link7").id, b.model.body("link8").id}
        robot_ids = {
            b.model.body(n).id
            for n in ["base_link", "gripper_base"] + [f"link{i}" for i in range(1, 9)]
        }
        for obj in self.objects:
            name, bid = obj["id"], b.model.body(obj["id"]).id
            contacts = set()
            for c in b.data.contact[: b.data.ncon]:
                pair = {int(b.model.geom_bodyid[g]) for g in (c.geom1, c.geom2)}
                if bid in pair:
                    contacts |= pair - {bid}
            lo, hi = self.bounds(name)
            event = self.events[name]
            held = bool(
                lo[2] >= self.cfg["table"]["top_m"] + cfg["lift_height_m"] and contacts & finger_ids
            )
            event["lift_s"] = event["lift_s"] + dt if held else 0.0
            event["grasped"] |= event["lift_s"] >= cfg["lift_hold_s"]
            dof = b.model.joint(name + "_free").dofadr[0]
            velocity = b.data.qvel[dof : dof + 6]
            still = (
                np.linalg.norm(velocity[:3]) < cfg["linear_speed_m_s"]
                and np.linalg.norm(velocity[3:]) < cfg["angular_speed_rad_s"]
                and not contacts & robot_ids
            )
            for color, tray in self.cfg["trays"].items():
                center, half = np.array(tray["center_xy_m"]), np.array(tray["half_size_xy_m"])
                floor = self.cfg["table"]["top_m"] + self.cfg["tray_geometry"]["floor_thickness_m"]
                supported = b.model.body("tray_" + color).id in contacts
                inside = (
                    np.all(lo[:2] >= center - half)
                    and np.all(hi[:2] <= center + half)
                    and lo[2] >= floor - 0.001
                    and supported
                    and still
                )
                event["stable_s"][color] = event["stable_s"].get(color, 0.0) + dt if inside else 0.0
                if (
                    event["grasped"]
                    and event["stable_s"][color] >= cfg["stable_hold_s"]
                    and color not in event["placed"]
                ):
                    event["placed"].append(color)
                if (
                    self.active_request
                    and name == self.request_target
                    and color == self.active_request["destination"]
                ):
                    self.request_place_s = self.request_place_s + dt if inside else 0.0
                    self.request_complete |= (
                        event["grasped"] and self.request_place_s >= cfg["stable_hold_s"]
                    )

    def evaluation(self):
        mapping = self.cfg["tasks"][self.task].get("mapping", {})
        complete = [
            o["id"]
            for o in self.objects
            if mapping.get(o["category"]) in self.events[o["id"]]["placed"]
        ]
        return {
            "time_s": self.elapsed_s,
            "time_limit_s": self.cfg["tasks"][self.task]["time_limit_s"],
            "timed_out": self.elapsed_s >= self.cfg["tasks"][self.task]["time_limit_s"] - 1e-9,
            "correctly_placed": complete,
            "events": copy.deepcopy(self.events),
            "instruction_complete": self.request_complete if self.active_request else None,
            "official_score": None,
            "judging_status": "provisional_development_metrics",
        }

    def export_metadata(self):
        public = {
            "task": self.task,
            "seed": self.seed,
            "status": self.cfg["status"],
            "requirements": copy.deepcopy(self.cfg.get("requirements", {})),
            "target_regions": copy.deepcopy(self.cfg["trays"]),
            "mapping_status": self.cfg["tasks"][self.task].get(
                "mapping_status", "development_only"
            ),
            "time_limit_s": self.cfg["tasks"][self.task]["time_limit_s"],
            "camera": self.cfg["camera"],
            "instructions": [
                {"id": r["id"], "text": r["text"]}
                for r in self.cfg["tasks"]["exploration"]["instructions"]
                + self.cfg["tasks"]["exploration"]["negative_tests"]
            ]
            if self.task == "exploration"
            else [],
            "classification_rule": self.cfg["tasks"][self.task].get("mapping"),
            "relation_frame": "initial_camera_image_x",
        }
        (self.output / "task.json").write_text(json.dumps(public, ensure_ascii=False, indent=2))
        private = {
            "objects": self.objects,
            "reset_attempt": self.reset_attempt,
            "positions": {k: v.tolist() for k, v in self.initial_centers.items()},
            "answers": [
                {
                    "id": r["id"],
                    "target": self.expected_selection(r),
                    "destination": r["destination"],
                }
                for r in self.cfg["tasks"]["exploration"]["instructions"]
                + self.cfg["tasks"]["exploration"]["negative_tests"]
            ],
        }
        (self.output / "evaluator_private.json").write_text(json.dumps(private, indent=2))
        (self.output / "resolved_config.yaml").write_text(
            yaml.safe_dump(self.cfg, allow_unicode=True)
        )

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None
