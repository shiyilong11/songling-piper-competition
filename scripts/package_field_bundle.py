"""Assemble an editable Ubuntu 22.04/Humble L1 and L2 deployment directory."""

import argparse
import hashlib
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "__init__",
    "active_view",
    "calibration",
    "competition_controller",
    "competition_segmentation",
    "competition_sensor",
    "control",
    "demo_skill",
    "field_calibration",
    "field_cli",
    "field_dashboard",
    "field_motion",
    "field_process",
    "field_ros",
    "grasp_feedback",
    "l1_rgb",
    "l2_progress",
    "l2_qwen_categories",
    "l2_sensor_controller",
    "mask_grasp",
    "observation",
    "qwen_adapter",
    "sam_adapter",
    "sam31_adapter",
    "scene_observation",
    "segmentation_grounding",
    "segmentation_lab",
    "sensor_ik",
    "sensor_pick_place",
    "sensor_robot",
    "traditional",
    "vision",
    "yoloe_mask",
)
CONFIGS = (
    "config/interface.yaml",
    "config/interactive_demo.yaml",
    "config/scene_observation.yaml",
    "config/competition/l1_rgb.yaml",
    "config/competition/mask_grasp.yaml",
    "config/competition/sensor_tasks.yaml",
    "config/segmentation/sam2.yaml",
    "config/segmentation/sam31.yaml",
    "config/vlm/qwen.yaml",
)
SAM_PATH = "models/sam2/sam2.1_hiera_small.pt"
SAM_SHA = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
YOLOE_PATH = "models/yoloe/yoloe26n-visual-640x480.torchscript"
YOLOE_SHA = "bf3acda64a91c6bbc2e273e294d7a27f4929471b4e4908d40f3230ce1fce8b97"
QWEN_PATH = "models/Qwen3-VL-4B-Instruct"


def digest(path):
    with path.open("rb") as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        return hasher.hexdigest()


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def robot_model(output):
    source = ROOT / "assets/piper/scene.xml"
    tree = ET.parse(source)
    root = tree.getroot()
    world = root.find("worldbody")
    for node in list(world):
        if node.tag != "body" or node.attrib.get("name") != "dummy_link":
            world.remove(node)
    target = output / "assets/piper/scene.xml"
    for mesh in root.findall("./asset/mesh"):
        path = source.parent / mesh.attrib["file"]
        relative = Path("meshes") / path.name
        copy(path, target.parent / relative)
        mesh.set("file", relative.as_posix())
    tree.write(target, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--qwen-wheelhouse", type=Path)
    parser.add_argument("--yoloe-license", type=Path)
    parser.add_argument(
        "--calibration", type=Path, default=ROOT / "deploy/field/field_calibration.yaml"
    )
    args = parser.parse_args()
    wheels = sorted(args.wheelhouse.glob("*.whl"))
    if not any(p.name.startswith("torch-2.7.1+cpu-cp310") for p in wheels):
        parser.error("Wheelhouse must contain the pinned Python 3.10 CPU PyTorch wheels")
    qwen_wheels = sorted(args.qwen_wheelhouse.glob("*.whl")) if args.qwen_wheelhouse else []
    if args.qwen_wheelhouse and not any(
        p.name.startswith("torch-2.11.0+cu128-cp310") for p in qwen_wheels
    ):
        parser.error("Qwen wheelhouse must contain the pinned Python 3.10 CUDA PyTorch wheels")
    if digest(ROOT / SAM_PATH) != SAM_SHA:
        raise ValueError("SAM2.1 checkpoint checksum mismatch")
    if digest(ROOT / YOLOE_PATH) != YOLOE_SHA:
        raise ValueError("YOLOE visual export checksum mismatch")
    yoloe_license = args.yoloe_license
    if yoloe_license is None:
        yoloe_license = next(
            (ROOT / ".local/yoloe-runtime/lib").glob(
                "python*/site-packages/ultralytics-8.4.0.dist-info/licenses/LICENSE"
            ),
            None,
        )
    if yoloe_license is None or not yoloe_license.is_file():
        parser.error("Supply the license accompanying the YOLOE export via --yoloe-license")
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    for module in MODULES:
        path = f"src/piper_sim/{module}.py"
        copy(ROOT / path, out / path)
    for path in (
        *CONFIGS,
        SAM_PATH,
        YOLOE_PATH,
        "scripts/sam_worker.py",
        "scripts/qwen_segmentation_worker.py",
        "assets/ui/field_dashboard.html",
    ):
        copy(ROOT / path, out / path)
    copy(args.calibration, out / "field_calibration.yaml")
    for path in ("requirements.txt", "run.sh", "setup.sh", "现场测试.md"):
        copy(ROOT / "deploy/field" / path, out / path)
    for path in ("run.sh", "setup.sh"):
        (out / path).chmod(0o755)
    for wheel in wheels:
        copy(wheel, out / "wheelhouse" / wheel.name)
    if qwen_wheels:
        shutil.copytree(
            ROOT / QWEN_PATH,
            out / QWEN_PATH,
            ignore=shutil.ignore_patterns(".cache", ".gitattributes"),
        )
        for wheel in qwen_wheels:
            copy(wheel, out / "wheelhouse-qwen" / wheel.name)
        for path in ("requirements-qwen.txt", "setup-qwen.sh"):
            copy(ROOT / "deploy/field" / path, out / path)
        (out / "setup-qwen.sh").chmod(0o755)
        copy(Path("/usr/share/common-licenses/Apache-2.0"), out / "licenses/Qwen-Apache-2.0")
    else:
        cfg_path = out / "config/interactive_demo.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["demos"]["l2_category_backend"] = "rules"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    sam = ROOT / ".local/sam2/source"
    shutil.copytree(
        sam / "sam2", out / "vendor/sam2", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    copy(sam / "LICENSE", out / "licenses/SAM2-LICENSE")
    copy(yoloe_license, out / "licenses/Ultralytics-YOLOE-LICENSE")
    copy(ROOT / "licenses/PiPER-MIT.txt", out / "licenses/PiPER-model-LICENSE")
    robot_model(out)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "field_calibration.yaml")
    (out / "SHA256SUMS").write_text(
        "".join(f"{digest(p)}  {p.relative_to(out).as_posix()}\n" for p in files)
    )
    print(out.resolve())


if __name__ == "__main__":
    main()
