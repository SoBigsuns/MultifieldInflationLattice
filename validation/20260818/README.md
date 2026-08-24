# 2026-08-18 double-chaotic run validation

対象run:

`two_field_quadratic_20260818T205254Z_6c5330a9ab54`

このフォルダは、二場二次ポテンシャルの格子出力を、解析的slow-roll近似、独立背景ODE、Friedmann拘束、Bunch-Daviesスペクトル、Hubble-exit後の軽い場スペクトル、Parseval恒等式と比較した再現可能な検証一式です。

## 主なファイル

- `compare_double_chaotic.py`: 全比較、図、CSV、JSON、TeX macroを生成
- `validate_background_ode.py`: Julia実装から独立したadaptive Dormand-Prince 5(4) ODE検証
- `report/double_chaotic_validation.tex`: 編集可能な日本語TeX報告書
- `report/double_chaotic_validation.pdf`: 同じ結果から生成・全8ページを視覚確認したPDF companion
- `data/validation_metrics.csv`: 17項目の判定値と許容値
- `data/validation_summary.json`: 設定、測定値、検証範囲、限界の機械可読summary
- `figures/`: 背景、エネルギー、スペクトル比較図
- `ode_validation/`: 独立ODEの時系列、判定表、図

## Python解析の再実行

リポジトリのルートから、PowerShellでプロジェクトのPython仮想環境を使います。

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\validation\20260818\compare_double_chaotic.py' `
  --project-root '.' `
  --run '.\runs\two_field_quadratic_20260818T205254Z_6c5330a9ab54'
```

macOSでは、同じくリポジトリのルートから次を実行します。

```bash
source .venv/bin/activate
python validation/20260818/compare_double_chaotic.py \
  --project-root . \
  --run runs/two_field_quadratic_20260818T205254Z_6c5330a9ab54
```

正常時は `Selected validation checks: PASS` を表示し、`data/`、`figures/`、`ode_validation/`を更新します。

## TeXのコンパイル

日本語LuaLaTeX環境で次を実行します。

```powershell
Set-Location '.\report'
lualatex double_chaotic_validation.tex
lualatex double_chaotic_validation.tex
```

TeX文書は `ltjsarticle`、`luatexja-fontspec`、`siunitx`などを使用します。この作業環境にはTeX engineがなかったため、TeXそのもののコンパイルは実行していません。代わりに、同じJSON値・図を使う `build_pdf_preview.py` でPDF companionを生成し、Popplerで全8ページを画像化して、欠落文字、切れ、重なりがないことを確認しました。

## 結論の範囲

全17検査はPASSしました。初期Hubble率、二段階の場発展、総e-fold、Friedmann拘束、平均場ODE、初期真空振幅、軽い場のfreeze-out power、FFT規格化は理論的期待と整合します。

一方、単一の `32^3` 格子、単一時間刻み、単一seedの検証なので、連続極限、有限体積独立性、seed ensemble、曲率摂動スペクトルの完全な精度までは保証しません。また、不等質量なので、これはU(1)対称な複素場ではなく、二つの正準実場、またはU(1)が破れた異方的複素場です。
