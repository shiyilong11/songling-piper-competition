# 松灵杯 PiPER 竞赛工程

仓库包含腕部 RGB-D 仿真、L1/L2 的 ROS 2 现场测试源码，以及 L3 的 IK 仿真闭环。L3 支持提交一到三条正式指令，按物体逐件复查，复合指令的所有动作通过后才算完成。

| 任务 | 仿真 | ROS 2 现场入口 |
|---|---|---|
| L1 方块分类 | RGB-D 颜色识别与抓放 | `run.sh l1` |
| L2 果蔬分拣 | 实例分割、本地千问分类与抓放 | `run.sh l2` |
| L3 探索赛 | 多视角识别、固定指令解释和 IK 抓放 | 接入待完成 |

任务台并排显示腕部 RGB 和本次任务的实例分割叠图。仿真使用标称腕部相机。正式工位使用单独提供的彩色内参、配准深度参数和手眼矩阵。试提依据夹爪反馈，放置依据可见足迹和连续两秒稳定观测。仿真结果供工程测试，实机和正式计分由现场核验。

## 启动仿真

在 Linux 安装 Python 3.12 和 OSMesa，例如系统包 `libosmesa6`。PiPER 运动学和仿真网格随仓库提供。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
./scripts/run_interactive_demo.sh --task l1
```

打开终端显示的本机地址。L1 可在 CPU 上运行。L2/L3 的默认果蔬识别需要本地千问和 SAM2.1。

```bash
./scripts/setup_sam.sh
python3.12 -m venv .local/qwen-runtime
.local/qwen-runtime/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
.local/qwen-runtime/bin/python -m pip install -r requirements-model.txt
.local/qwen-runtime/bin/python scripts/download_model.py
./scripts/run_interactive_demo.sh --task l3
```

千问模型固定版本见 `config/vlm/qwen.yaml`。模型下载和初始安装在联网环境完成，任务推理使用本地权重。YOLOE 视觉提示导出文件放在 `models/yoloe/yoloe26n-visual-640x480.torchscript` 时可加速分割，缺少该文件时继续使用 SAM2.1。模型、离线轮子和运行日志随各自部署环境保存。

仿真相机参数在 `config/cameras/competition_sim_nominal.yaml`，仿真裁判左右参考在 `config/interactive_demo.yaml`。每次运行保存传感器记录、抓放事件及复查结果。

## L3 学习策略接口

L3 默认使用 IK。后续可在 Python 中通过 `Demo(l3_transfer_executor=...)` 或 `ExplorationController(transfer_executor=...)` 接入单件抓放执行器。执行器接收已确认的目标实例、同帧观测、目标 mask 和实测目标区，释放后返回物体在基座坐标系中的预计中心（米）。任务控制器仍独立复查物体入区和两秒稳定性。只有确认抓空且已恢复时，执行器才能通过 `RetryableEmptyTransfer` 请求再次观测。

离线加载本地 SB3 权重时，安装可选依赖 `pip install -e '.[learning]'`，使用 [MultiRatePolicyPreview](src/piper_sim/competition_policy.py)。每次视觉更新提供对齐深度、目标与目标区 mask、原始帧时间和拍摄位姿。每次预测提供当前关节与夹爪反馈和上一次实际下发的动作。模型返回六个归一化关节增量及一个归一化夹爪开口。频率和时效要求见 [多频率配置](config/learning/piper_multirate.yaml)。

只读预测仅用于开发验证。BC 权重、训练数据和实测标定单独保存在本地。实机学习执行器和 200 Hz ROS 2 下发尚未验证。

## 准备现场部署

使用组委会提供的 Ubuntu 22.04 x86_64 / ROS 2 Humble 镜像和现场驱动。队内离线目录包含指定的 Python 3.10 依赖、SAM2.1、YOLOE 导出及可选千问权重。源码入口为 `src/piper_sim/field_cli.py`，标定模板和安装脚本在 `deploy/field/`。

重建目录时，提供已校验的模型文件、安装好的 SAM2 源码、YOLOE 导出的许可证来源和离线 wheelhouse：

```bash
.venv/bin/python scripts/package_field_bundle.py \
  --output field-build \
  --wheelhouse /path/to/cpu-wheelhouse \
  --qwen-wheelhouse /path/to/qwen-wheelhouse \
  --yoloe-license /path/to/yoloe-LICENSE \
  --calibration /path/to/field_calibration.yaml
```

现场话题、单位、只读检查及点动流程见 [现场测试说明](deploy/field/现场测试.md)。当前部署使用组委会镜像，容器交付需在目标主机另行构建并验证。

## 测试与资产

```bash
.venv/bin/python -m pytest -q
```

PiPER 模型由 AgileX 官方描述转换，使用 [MIT 许可证](licenses/PiPER-MIT.txt)。YCB 扫描的出处和授权见 [资产说明](assets/competition/ycb/NOTICE.md)。六类比赛果蔬的仿真模型使用原创参数化网格和程序化纹理，尺寸和接触属性是开发估计。[模型说明](assets/competition/video_reference_20261111/NOTICE.md)
