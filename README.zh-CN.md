# Mean-field kernel / MVNN

本仓库对应论文 **MVNN: A Measure-Valued Neural Network for Learning McKean-Vlasov Dynamics from Particle Data**，作者为 Liyao Lyu、Xinyue Yu 和 Hayden Schaeffer。

- [最终版论文](paper/mean-field-kernel.pdf)
- [论文图表与代码对应关系](docs/PAPER_MAP.md)
- [数据内容、抽取方式及来源](docs/DATA.md)
- [复现步骤及尚未补齐的部分](docs/REPRODUCTION.md)

仓库从指定的 amd20 与 Anvil 目录中整理而来。已收录一阶模型的数据生成、训练和推演代码，修订中的数据量实验、非线性 McKean–Vlasov 比较实验，以及二阶实验现存的绘图代码和数据。原服务器文件未修改。

## 快速使用

```bash
git clone git@github.com:Lyuliyao/mean-field-kernel-code.git
cd mean-field-kernel-code
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/verify_repository.py
python scripts/replot.py --figures 1 5 9 --output-dir runs/figures
python scripts/plot_snapshots.py --case aggregation_2d --scenario ring
```

图 1 使用原实验保存的七种训练数据量、三个独立模型的统计结果；图 5 使用保存的汇总耗时；图 9 使用冻结的确认集绘图缓存和在线参数估计曲线。重绘这些结果不需要 GPU、MPI 或 TeX。上述步骤不重新训练模型，也不重新测量耗时。

`data/representative/` 是原始数组的少量时间帧及连续粒子子集，保留原数值和精度；各文件的源路径、形状、所选帧和粒子数都有记录。这些样例不能替代完整训练集或原图使用的所有汇总轨迹。

尚未找到二阶模型的数据生成与训练代码，也未找到 GP 和在线参数估计的原始拟合代码。相关图形、保存的预测轨迹及原始文件清单已经保留。原绘图脚本存在旧路径、时间标签和 binary 图重复读取预测数据等问题，详见复现说明；没有据此宣称所有图已完成端到端复现。

默认保留为私有仓库，暂未自行指定开源许可证。
