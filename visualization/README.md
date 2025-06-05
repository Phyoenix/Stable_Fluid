# Poisson Solver Visualization

这个Python程序用于可视化展示不同数值方法（Jacobi、Gauss-Seidel和Multigrid）在求解泊松方程时的收敛过程。该程序特别关注于Stable Fluid模拟中的压力投影步骤。

## 功能特点

- 在128x128的二维网格上模拟压力投影过程
- 生成包含高频和低频成分的初始散度场
- 实现三种不同的求解方法：
  - Jacobi迭代
  - Gauss-Seidel迭代
  - Multigrid V-cycle
- 可视化展示：
  - 3D残差表面图（x-y平面为网格点，z轴为残差值）
  - 收敛曲线（迭代次数vs残差）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

直接运行主程序：

```bash
python poisson_solver_visualization.py
```

程序会自动：
1. 生成初始散度场
2. 使用三种不同方法求解
3. 显示3D残差图
4. 显示收敛曲线

## 输出说明

- 3D残差图：展示了在网格上残差的分布情况，可以直观地看到不同方法处理高频和低频误差的效果
- 收敛曲线：展示了不同方法随迭代次数的收敛情况，使用对数坐标显示残差变化

## 注意事项

- 程序默认使用128x128的网格
- 初始散度场由70%的低频成分和30%的高频成分组成
- Multigrid方法默认使用3层网格 