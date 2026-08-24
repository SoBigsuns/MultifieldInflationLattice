# MultifieldInflationLattice

複数の正準実スカラー場を、膨張する平坦 FLRW 時空上の周期 3 次元格子で
非線形発展させる Julia 研究コードです。格子平均から背景を自己無撞着に発展させ、
背景量・エネルギー・スローロール量・場の自己/クロススペクトル・格子配位からの
線形曲率推定量・一様密度面への非摂動的 `delta N` を出力します。

物理・数値仕様は
[InflationEasy](https://github.com/caravangelo/inflation-easy) の commit
`d4f0cfdde0d148fa8ffaa14fe37d705cd79b2366` を基準に、多場化、Julia CPU
スレッド、交換可能なポテンシャル/積分器、ヘッダ付き CSV、完全チェックポイントを
追加したものです。独立した線形モード方程式、計量摂動、GPU、分散メモリ並列は
この版の対象外です。

## 必要環境

- 64-bit Julia（対応版は `Project.toml` の `[compat]` を参照）
- FFT と JLD2 の依存パッケージ
- `32^3`、2 場の標準例には十分なメモリと数時間程度の CPU 時間を見込んでください。
  `delta N` は全格子点で別宇宙発展を行うため、短い動作確認には `lattice.size = 8`
  と小さい終了条件を使うのが実用的です。

初回だけ、リポジトリ直下で環境を構築します。

```sh
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

CPU スレッド数は Julia 起動前に設定します。FFT スレッド数は TOML の
`parallel.fft_threads` で別に指定し、過剰な多重化を避けてください。

```sh
# POSIX shell
JULIA_NUM_THREADS=auto julia --project=. bin/validate_config.jl --config configs/two_field_quadratic.toml

# PowerShell
$env:JULIA_NUM_THREADS = "auto"
julia --project=. bin/validate_config.jl --config configs/two_field_quadratic.toml
```

## クイックスタート

まず設定を検証します。未知キー、型、配列長、有限性、格子条件、CFL 条件、出力範囲
などが不正なら、run ディレクトリを作る前に終了します。

```sh
julia --project=. bin/validate_config.jl --config configs/two_field_quadratic.toml
```

新規計算を開始します。

```sh
julia --project=. bin/simulate.jl --config configs/two_field_quadratic.toml
```

保存済みチェックポイントから再開します。

```sh
julia --project=. bin/simulate.jl --resume runs/<run-id>/checkpoint.jld2
```

保存結果を再解析します。

```sh
julia --project=. bin/analyze.jl --input runs/<run-id>
```

各コマンドは `--help` を持ちます。終了コードは、`0`: 正常、`2`: 設定/引数不正、
`3`: I/O またはチェックポイント異常、`4`: 数値異常、`130`: 通常中断です。

## 標準 2 場設定

[`configs/two_field_quadratic.toml`](configs/two_field_quadratic.toml) は

```text
V(phi,psi) = (M^2 phi^2 + m^2 psi^2)/2
M = 9e-6, m = 1e-6
phi(0) = psi(0) = 13
phidot(0) = 1e-10, psidot(0) = 0
```

を `32^3` 格子で実行します。初期速度を両方 0 にはしていません。線形曲率推定量の
分母 `sum_I phidotbar_I^2` が初期時刻にゼロになるのを避けるためです。

主な設定節は次のとおりです。

- `[model]`: 場名、ポテンシャル、質量、還元 Planck 質量、内部再スケール `B`
- `[lattice]`: 2 の冪かつ 8 以上の格子数、箱長、周期境界、7 点 Laplacian
- `[initial]`: 背景値/速度、乱数 seed、Bunch-Davies 揺らぎ、UV/IR cutoff
- `[evolution]`: `leapfrog` または `rk4`、刻み係数、終了条件、Friedmann 許容差
- `[deltaN]`: 一様密度面、局所刻み、前後向き上限、別宇宙診断
- `[spectra]`: 格子有効運動量と線形曲率の速度閾値
- `[output]`: 出力先/間隔、histogram、2D slice、checkpoint、上書き方針
- `[parallel]`: CPU/FFT スレッドと決定的縮約

`evolution.max_wall_time = 0.0` は壁時計上限なしです。`output.slice_index` と出力 CSV
の格子 index は Julia と同じ 1 始まりです。`output.root` の相対パスはコマンドを
起動したカレントディレクトリを基準に解決されます。

## 出力と上書き保護

新規実行 ID は
`<run_name>_<UTC timestamp>_<input-config SHA-256 の先頭 12 桁>` です。
既存 ID がある場合、`output.overwrite = false`（既定）なら内容を変更せず終了します。
明示的に `true` にした場合だけ同じ run ディレクトリを置換します。

```text
runs/<run-id>/
├── config.input.toml
├── config.effective.toml
├── metadata.toml
├── data_dictionary.md
├── run.log
├── background.csv
├── energies.csv
├── slowroll.csv
├── diagnostics.csv
├── field_spectra.csv
├── linear_curvature_spectra.csv
├── deltaN_summary.csv
├── deltaN_spectrum.csv
├── histograms/
├── slices/
├── snapshots/                 # 有効時のみ
└── checkpoint.jld2            # 有効時
```

`config.input.toml` は入力をそのまま保存し、`config.effective.toml` は全既定値を展開します。
`metadata.toml` には両 SHA-256、Julia/依存版、OS/CPU/スレッド、seed、Git 状態、格子、
規約 ID、警告、終了理由を保存します。列の意味、単位、Fourier/スペクトル規格化は
各 run の `data_dictionary.md` に記録されます。

CSV は UTF-8、カンマ区切り、ヘッダ付きです。場名から派生する wide 列名は安全な
ASCII 識別子へ正規化し、衝突時は連番を付けます。ユーザー向け元の場名はスペクトル
long 行とメタデータに保持します。未定義値は空欄や 0 で偽装せず `NaN` とします。

### 再開の整合性

チェックポイントは `checkpoint.jld2.tmp` へ完全に書いて再読込した後に置換し、直前
世代を `checkpoint.jld2.bak` に保持します。全配位、共形微分、背景、積分器の半ステップ、
刻み、RNG、出力進行を payload に含めます。再開時には checkpoint の各 CSV 最終
`output_id`、バイト長、SHA-256 と実ファイルを照合します。1回の観測で更新する複数CSVは
`.output-transaction` に先に全追記内容を準備し、全ファイルの確定後には
`.output-transaction.last.toml` receipt を残します。中断時は未変更または完全一致する追記だけを
再適用し、receipt と checkpoint 状態が同じ step の直後1件である場合だけ進行状態を復元します。
欠落、部分行、改変、複数観測分の先行が一つでもあれば、自動追記や自動切り詰めをせず終了します。

## 数値規約

- 空間は一辺 `L` の周期立方格子、場配列順序は `(x,y,z,field)` です。
- Laplacian は 2 次中心差分の 7 点 stencil です。
- 内部時間は共形時間。出力には共形時間 `tau`、宇宙時間 `t`、`efolds=log(a/a0)` を保存します。
- Friedmann 方程式は `H^2=<rho>/3`（還元 Planck 単位）です。
- DFT は `X_k=dx^3 sum_x exp(-ikx)X(x)`、逆変換は `L^-3 sum_k exp(ikx)X_k` です。
- 無次元 cross power は `k_eff^3 <Re(X_k Y_k*)>/(2 pi^2 L^3)` です。
- `zeta_lin=sum_I[-H phidotbar_I delta_phi_I/sum_J phidotbar_J^2]` です。
- `zeta_deltaN=N_local-<N_local>` です。非線形量を場ごとには分解しません。

格子平均 `epsilon_H` が初めて設定終了値へ達した同期状態の平均密度を、`delta N` の
共通一様密度面にします。別宇宙条件 `R_SU=a H dx` を記録し、閾値不足は既定で警告、
`strict_separate_universe=true` なら停止します。

## ライブラリ API

CLI と同じ主要操作はパッケージ API から呼び出せます。

```julia
using MultifieldInflationLattice

validate_config_file("configs/two_field_quadratic.toml")
run_simulation("configs/two_field_quadratic.toml")
resume_simulation("runs/<run-id>/checkpoint.jld2")
analyze_run("runs/<run-id>")
```

ポテンシャルは `potential_value`、`potential_gradient!`、`potential_hessian!`、
`characteristic_mass` の共通 interface を実装し、registry に明示登録します。任意の Julia
コードを設定から読み込む仕組みではないため、追加模型はレビュー可能なソース変更として
管理してください。時間積分器と観測量はポテンシャルの具体型へ依存しません。

## 再現性と検証

同じソース、Manifest、実効設定、seed、Julia/FFT スレッド条件を保存してください。
並列縮約では加算順序により最下位 bit が変わり得ます。厳密な回帰確認には
`parallel.deterministic_reductions = true` を使い、環境間比較は規定許容誤差で行います。

開発時の試験はリポジトリ直下で実行します。

```sh
julia --project=. -e "using Pkg; Pkg.test()"
```

## ライセンスと引用

本コードは [MIT License](LICENSE) で提供され、無保証です。研究利用時は
[`CITATION.cff`](CITATION.cff) と基準実装 InflationEasy/対応論文を引用してください。
