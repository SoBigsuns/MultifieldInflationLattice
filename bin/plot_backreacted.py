"""Figures for the exact-realization quadratic Fourier reference."""
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, MaxNLocator

def main(root, homogeneous_path=None):
    with (root/"backreacted_comparison.csv").open() as f:
        rows=list(csv.DictReader(f))
    output=root/"figures"
    output.mkdir(exist_ok=True)
    old_predictions={}
    if homogeneous_path is not None:
        with homogeneous_path.open() as f:
            old_predictions={(int(r["trial"]),r["quantity"],int(r["k_bin"])):r
                             for r in csv.DictReader(f) if r["snapshot"]=="final"}
    report=["# 同じ初期実現・背景への揺らぎの寄与を含めた比較", "",
            "格子の測定値は変更していません。独立なFourier方程式を、実際の初期配列から発展させています。",
            "理論側も各モードの運動・勾配・ポテンシャルエネルギーで背景を発展させます。",
            "これは有限格子・二場分離型二次ポテンシャル・計量揺らぎなしの基準です。",
            "背景が試行に依存するため、曲線を乱数についてのアンサンブル平均と呼ぶことはできません。", "",
            "|試行|量|2～8番ビンの最大残差[%]|全ビンの最大残差[%]|基準計算の刻み半減差[%]|",
            "|---|---|---:|---:|---:|"]
    for trial in sorted({int(r["trial"]) for r in rows}):
        fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
        for col,q in enumerate(("phi","chi","zeta_linear")):
            rr=sorted((r for r in rows if int(r["trial"])==trial and
                       r["snapshot"]=="final" and r["quantity"]==q),key=lambda r:int(r["k_bin"]))
            k=np.array([float(r["k_comoving"]) for r in rr])
            axes[0,col].loglog(k,[float(r["observed"]) for r in rr],color="tab:blue",label="Lattice measurement")
            axes[0,col].loglog(k,[float(r["conditioned_prediction"]) for r in rr],
                               color="tab:orange",ls="--",label="Independent coupled reference")
            if old_predictions:
                old=[old_predictions[(trial,q,int(r["k_bin"]))] for r in rr]
                if any(abs(float(a["efolds"])-float(b["efolds"]))>1e-9 for a,b in zip(rr,old)):
                    raise ValueError("Old and new predictions must refer to the same endpoint")
                axes[0,col].loglog(k,[float(r["conditioned_prediction"]) for r in old],
                                  color="0.5",ls=":",label="Old homogeneous-background prediction")
                axes[1,col].plot(k,[100*float(r["residual_fraction"]) for r in old],
                                color="0.5",ls=":",label="Old reference residual")
            axes[0,col].set_title(q);axes[0,col].set_ylabel("Dimensionless power")
            axes[0,col].xaxis.set_major_locator(FixedLocator(np.geomspace(k.min(),k.max(),4)))
            axes[0,col].xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{value:.1e}"))
            axes[0,col].xaxis.set_minor_formatter(NullFormatter())
            axes[1,col].plot(k,[100*float(r["residual_fraction"]) for r in rr],color="tab:blue",label="Coupled reference residual")
            axes[1,col].axhline(0,color="black",lw=.7)
            axes[1,col].set_ylabel("Lattice / reference - 1 [%]")
            axes[1,col].xaxis.set_major_locator(MaxNLocator(nbins=4))
            for ax in axes[:,col]:
                ax.set_xlabel(r"$k_{\rm eff}$ [reduced Planck units]");ax.grid(alpha=.2)
            band=[r for r in rr if 2<=int(r["k_bin"])<=8]
            report.append(f"|{trial}|{q}|{max(abs(float(r['residual_fraction'])) for r in band)*100:.6g}|"
                          f"{max(abs(float(r['residual_fraction'])) for r in rr)*100:.6g}|"
                          f"{max(float(r['reference_relative_change']) for r in rr)*100:.6g}|")
        axes[0,0].legend(fontsize=8)
        axes[1,0].legend(fontsize=8)
        fig.suptitle(f"Trial {trial}: same finite-lattice physics and actual initial realization")
        for ext in ("png","pdf"):
            fig.savefig(output/f"backreacted_trial_{trial:02d}.{ext}",dpi=180)
        plt.close(fig)
    report+=["", "従来の spectra_comparison.csv は一様背景との比較として残しています。",
             "背景の違いを含む旧図だけで、数値精度を判定しないでください。",
             "この検証は量子的な真空エネルギーの繰り込みや完全な重力摂動を含む物理予測の検証ではありません。"]
    (root/"BACKREACTED_REPORT_JA.md").write_text("\n".join(report)+"\n")
    old=root/"REPORT_JA.md"
    note="\n\n## 2026-10-06以降の精度検証\n\n背景への揺らぎの寄与をそろえた比較は [BACKREACTED_REPORT_JA.md](BACKREACTED_REPORT_JA.md) を参照してください。\n"
    if old.exists() and "## 2026-10-06以降の精度検証" not in old.read_text():
        with old.open("a") as f:
            f.write(note)
    print(root/"BACKREACTED_REPORT_JA.md")

if __name__=="__main__":
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve() if len(sys.argv)>2 else None)
