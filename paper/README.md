# IEEE 论文源码说明

主文件：

```bash
paper/main.tex
```

参考文献：

```bash
paper/refs.bib
```

推荐编译：

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

波束图生成流程：

1. 先在仓库根目录运行实验，生成波束矩阵：

```bash
python run_experiment.py --methods zf,sdr,vanilla,ec,cse \
  --algorithm cse --K 2 --sinr-db 12 \
  --updates 40 --episodes-per-update 64 --eval-episodes 128
```

2. 使用日志目录绘制图：

```bash
python plot_beampatterns.py --input-dir log/YYYYMMDD-HHMMSS \
  --output-dir paper/figs
```

3. 编译论文。
