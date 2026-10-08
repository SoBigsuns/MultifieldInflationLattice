# 二場の揺らぎエネルギーを含む理論比較

## 改善内容

格子の Friedmann 方程式は揺らぎを含む空間平均エネルギー密度を使います。従来の比較理論は一様場だけのエネルギーで背景を発展させていたため、両者の膨張履歴が異なっていました。

追加した `bin/backreacted_theory.jl` は、各試行で保存した初期 Fourier モードを独立に線形発展させ、その運動・勾配・ポテンシャルエネルギーを背景の Friedmann 方程式にも含めます。格子と同じ離散ラプラシアンの実効波数を使い、場間の相関も曲率揺らぎの計算に含めます。計測済みの終状態へのフィットやスペクトルの再規格化は行いません。

対象は V=(m²φ²+M²χ²)/2、還元 Planck 質量1、m=9×10⁻⁶、M=10⁻⁶、φ₀=χ₀=13、初期速度0、a₀=1、コード座標で箱長2です。時空の揺らぎと delta-N は含みません。初期乱数は質量なしの真空モードから生成します。これは時空の揺らぎを含む完全な Mukhanov–Sasaki 系や真空エネルギーの繰り込みではありません。

新しい検証ドライバーは RK4 を既定にします。既存の計算本体 `src/` と通常のシミュレーションの既定設定は変更していません。

## 環境の準備

検証した環境は Julia 1.12.2 と Python 3.11 です。Python 3.11 以上、NumPy、Matplotlib が必要です。以下はリポジトリ直下で実行します。

```sh
git clone https://github.com/SoBigsuns/MultifieldInflationLattice.git
cd MultifieldInflationLattice
git checkout fix/fluctuation-energy-validation
julia --project=. -e 'using Pkg; Pkg.instantiate()'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
```

上の checkout は修正版のブランチを選びます。main に取り込まれた後は省略できます。Windows PowerShell の仮想環境の有効化は `.venv\Scripts\Activate.ps1` です。Julia が PATH にない場合は、下記 Python コマンドに `--julia /path/to/julia` を追加してください。

## 最初の動作確認

```sh
python bin/validate_all.py --smoke --runs 2 --jobs 1 --worker-threads 2 --output runs/validation_smoke
```

8³の小さな格子と短い時間発展で、乱数生成、格子計算、二種類の理論比較、画像出力を確認します。インフレーション終期の精度検証には次の通常実行を使います。

## 32³・128³の検証

32³で10試行を行う例です。各試行は異なるシードを使い、マスターシードを固定すると一連の初期状態を再現できます。

```sh
python bin/validate_all.py --runs 10 --jobs 2 --size 32 --worker-threads 2 --integrator rk4 --timestep-factor 0.01 --master-seed 8449150461266532141 --output runs/validation32_rk4
```

128³で2試行を行う例です。計算時間とメモリ消費が増えるため、まず1ジョブで実行できます。

```sh
python bin/validate_all.py --runs 2 --jobs 1 --size 128 --worker-threads 8 --integrator rk4 --timestep-factor 0.01 --master-seed 8449150461266532141 --output runs/validation128_rk4
```

`--jobs` は同時実行する Julia プロセス数、`--worker-threads` は各プロセスの Julia スレッド数です。CPUで動作します。両者の積が使用可能なコア数を超えないようにします。`--runs 1` なら `--jobs 1` を指定してください。出力先は新しいディレクトリを指定します。既存の出力先への上書きは拒否します。

時間刻みの収束を確認するには、同じ格子・試行数・マスターシードで `--timestep-factor 0.005` と別の出力先を指定して再実行します。同じシードでも格子サイズが変わると同じ初期モードの標本にはなりません。

## 比較画像とデータの場所

出力ディレクトリ内で、まず次を確認してください。

| ファイル | 内容 |
| --- | --- |
| `BACKREACTED_REPORT_JA.md` | エネルギーを合わせた理論との残差・収束の集計 |
| `figures/backreacted_trial_01.png` / `.pdf` | 試行1の場φ・場χ・曲率揺らぎのパワーと理論との差 |
| `figures/backreacted_trial_02.png` / `.pdf` | 試行2の同じ比較（試行数に応じて追加） |
| `backreacted_comparison.csv` | モード別・試行別の格子値、予測値、残差 |
| `backreacted_background.csv` | 理論の背景発展 |
| `backreacted_initial_checks.csv` | 初期エネルギーの一致確認 |
| `backreacted_diagnostics.toml` | 理論積分の精度確認 |
| `REPORT_JA.md` | 初期条件の統計検証と従来の一様背景との比較 |

新しい比較画像の上段はパワースペクトル、下段は相対残差です。青は格子データ、橙はその試行の初期モードと揺らぎエネルギーを使った理論です。従来理論を併記した場合の灰色点線は一様場だけで背景を発展させた予測です。曲率揺らぎは線形推定量を比較しており、完全な時空揺らぎの計算結果ではありません。

## 保存済み試行で理論比較だけ実行する

この検証ドライバーで生成したバッチには `initial_state.bin` が保存されます。その出力先を指定して再計算できます。

```sh
julia --project=. --compiled-modules=existing --threads=4 bin/backreacted_theory.jl runs/validation32_rk4
python bin/plot_backreacted.py runs/validation32_rk4
```

通常の `bin/simulate.jl` の出力だけでは必要な初期状態が揃いません。中断バッチの場合は、完了済みの試行が trial_01 から連続している必要があります。

## 確認済みの精度と残る確認

これまでの手元の検証では、128³の初期揺らぎエネルギーは一様場エネルギーの約27.75%でした。以前の約25%の曲率パワーの上振れは、背景のエネルギー条件の不一致が主な原因でした。

| 検証 | 結果 |
| --- | --- |
| 既存128³の完了済み2試行と新しい理論 | 終期のビン2–8の曲率パワー残差は約+1.63% |
| 32³をRK4で再実行し新しい理論と比較 | 同範囲の曲率パワー残差は0.0001%未満 |

128³を新しいRK4ドライバーで終期まで再実行する検証は未完了です。したがって、128³の残る約1.63%まで解消済みとは判断していません。高波数の格子効果、時間刻み、非線形項の影響は別に確認が必要です。また初期真空モードを古典場として扱うため、解像度によるエネルギー増加自体を取り除く修正ではありません。

## テスト

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. --compiled-modules=existing test/validation_theory.jl
julia --project=. --compiled-modules=existing --threads=2 test/backreacted_reference.jl
```

追加テストは初期エネルギーの実空間・Fourier空間での一致、理論の時間刻み収束、並列・逐次計算の一致、短時間の格子RK4とのパワー比較、揺らぎゼロでの一様背景への帰着を確認します。
