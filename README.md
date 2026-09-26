# robot-ik-solver-python

机器人多关节逆运动学（IK）求解与奇异位形诊断服务。给定末端执行器目标位姿与机械臂
运动学参数，返回满足关节限位与连续性约束的可行解；在目标不可达、限位冲突、迭代不
收敛或超时时返回**可区分的诊断**，绝不静默返回近似错误解。所有计算在本地完成，固定
随机种子下结果完全可复现。

## 运行

```bash
uv sync                 # 安装依赖
uv run uvicorn main:app # 启动 HTTP 服务
uv run pytest           # 运行 53 个自动化测试
```

也可以不走 HTTP，直接在内存中使用 `robot_ik.IKService`（见下文示例）。

## 运动学模型约定

机械臂为串联旋转关节链。每个关节 `JointSpec` 贡献

```
T(q) = Π_i [ T_fixed_i · Rot(axis_i, q_i) ] · T_tool
```

- `axis`：关节转轴，表达在本关节坐标系内（自动归一化）；
- `translation` / `rotation`：父关节系到本关节系的**固定**变换；
- `lower_limit` / `upper_limit`：关节限位（弧度）；
- `tool_transform`：末关节之后的常量工具变换（4x4 齐次矩阵）。

该表示可表达任意串联旋转臂（DH 参数可直接换算）。配置校验
（`validate_config`）拒绝：空关节链、重名关节、零/非有限转轴、非法旋转矩阵
（非正交或行列式 ≠ +1）、下限 ≥ 上限、非法工具变换、空版本号等不一致配置。

## 数值方法

求解器（`robot_ik.solver.IKSolver`）采用**阻尼最小二乘**（Levenberg–Marquardt 风格）
迭代：

```
dq = Jᵀ (J Jᵀ + λ²I)⁻¹ · err,   q ← clip(q + α·dq, limits)
```

- `err` 为 6 维位姿误差（位置 + 旋转矢量，旋转矢量由 `R_err = R_target R_currentᵀ`
  的对数映射得到，180° 附近做了数值稳定处理）；
- 每步迭代后将关节角**钳制**到限位内，保证返回的解必然满足限位；
- 收敛判据：`‖err‖ < tolerance`；
- **多种子重启**：种子序列 = [上一解（若提供）, 零位形, 限位中点, 由
  `np.random.default_rng(seed)` 生成的 `num_restarts` 个限位内均匀随机种子]。
  每个求解器实例持有独立 RNG，因此并发交错不影响复现性；
- **多解选择**：所有收敛解中选取与 `previous_joints` 欧氏距离最近者（未提供时取
  离零位形最近者），平局按残差与种子序确定，保证轨迹跟踪时不发生大幅跳变且行为
  完全确定；
- **超时**：每次迭代检查墙钟预算，超时立即返回 `timeout`，不携带任何部分结果。

## 诊断语义

`IKResult.status` 取值（互不重叠，可区分）：

| 状态 | 含义 |
|---|---|
| `success` | 在限位内收敛到 `tolerance` 以内，`joint_angles` 有效 |
| `unreachable` | 目标在工作空间外：超出最大臂展（解析判定），或**去掉限位**的诊断性求解仍以残差停滞 |
| `joint_limit_conflict` | 去掉限位可收敛、带限位不收敛：运动学可达但限位禁止 |
| `not_converged` | 迭代预算耗尽时误差仍在下降（增大 `max_iterations` 可确认） |
| `timeout` | 墙钟预算耗尽，无部分结果 |
| `invalid_request` | 请求非法（NaN、四元数为零、维度不符、非法选项等） |
| `invalid_config` | 配置非法（构造/更新服务时校验） |

失败时 `joint_angles` 恒为 `None`——不会把未收敛的中间迭代值当作解返回；
`pose_error` 给出最优迭代点的残差供诊断。

### 奇异性报告

对解（或失败时的最优迭代点）处的 6×n 几何雅可比做 SVD，设可配置阈值为 `t`
（`SolverOptions.singularity_threshold`）：

- `min_singular_value`：最小奇异值（自由度 < 6 时补零，因此低自由度臂在 6 维任务
  空间恒有退化方向）；
- `manipulability`：可操作度（奇异值乘积）；
- `is_near_singular`：`min_singular_value < t`；
- `is_singular`：`min_singular_value < t × 10⁻³`（数值秩亏），蕴含前者；
- `degenerate_directions`：奇异值低于 `t` 的左奇异向量，即末端难以运动的任务空间
  方向（6 维：线速度 xyz + 角速度 xyz）。

## 并发、复现与配置热更新

- **并发隔离**：`IKService` 持有的配置快照不可变；每次 `solve` 创建独立求解器
  （独立 RNG 与缓冲区），同配置下的并发请求互不影响；
- **可复现**：固定 `seed` 与相同输入 ⇒ 输出逐位一致，与线程调度无关；
- **热更新与版本校验**：`update_config(config, expected_version)` 先完整校验新配置
  再原子切换；`expected_version` 与当前版本不符时拒绝（HTTP 层返回 409）。已开始的
  求解继续使用入口处快照的版本，新旧配置绝不混用；结果中的 `config_version` 标识
  该解实际使用的配置版本。

## 可调参数（`SolverOptions`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_iterations` | 200 | 每个种子的最大迭代数 |
| `tolerance` | 1e-6 | 6 维位姿误差收敛阈值 |
| `damping` | 1e-3 | 阻尼因子 λ |
| `step_size` | 1.0 | 步长缩放 α ∈ (0, 1] |
| `num_restarts` | 8 | 随机重启次数（另有 3 个确定性种子） |
| `seed` | 0 | 随机种子 |
| `timeout_ms` | None | 墙钟超时（毫秒），None 为不限 |
| `singularity_threshold` | 1e-3 | 奇异接近阈值 |

## 内存接口示例

```python
import numpy as np
from robot_ik import (
    ArmConfig, IKRequest, IKService, JointSpec, Pose, SolverOptions,
)

config = ArmConfig(
    name="arm6", version="v1",
    joints=(
        JointSpec(name="j1", axis=(0, 0, 1), translation=(0, 0, 0.2)),
        JointSpec(name="j2", axis=(0, 1, 0), translation=(0, 0, 0.1)),
        JointSpec(name="j3", axis=(0, 1, 0), translation=(0.3, 0, 0)),
        JointSpec(name="j4", axis=(1, 0, 0), translation=(0.2, 0, 0)),
        JointSpec(name="j5", axis=(0, 1, 0), translation=(0.1, 0, 0)),
        JointSpec(name="j6", axis=(1, 0, 0), translation=(0.05, 0, 0)),
    ),
)
service = IKService(config, SolverOptions(seed=7))

result = service.solve(IKRequest(
    target=Pose(position=(0.4, 0.1, 0.3), orientation_wxyz=(1, 0, 0, 0)),
    previous_joints=None,          # 传入上一解可保证连续性
))
print(result.status, result.joint_angles, result.singularity)

# 热更新（带版本校验）
service.update_config(
    ArmConfig(name="arm6", version="v2", joints=config.joints),
    expected_version="v1",
)
```

## HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/config` | 当前配置、自由度与默认求解选项 |
| PUT | `/config` | 热更新配置（可带 `expected_version`；非法 422 / 冲突 409） |
| PUT | `/options` | 更新默认求解选项 |
| POST | `/solve` | IK 求解，返回状态、关节角、误差、奇异报告与配置版本 |

## 测试覆盖

`tests/` 共 53 个用例，覆盖：配置/请求校验、FK 与雅可比（对有限差分验证）、
多解选择与轨迹连续性、不可达（臂展外与工作空间空洞内）、限位冲突、迭代不收敛、
超时无部分结果、奇异检测与阈值可配置性、并发隔离、并发下复现性、配置热更新与
版本校验、HTTP 端到端流程。
