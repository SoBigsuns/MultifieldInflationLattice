# MacBook での GitHub 同期手順

このリポジトリは、シミュレーション本体、設定、テスト、Notebook、検証コードと報告書を複数端末で同期するためのものです。`runs/`、Python仮想環境、Jupyterのチェックポイント、大容量の再生成可能データは同期しません。

## 初回セットアップ

1. ターミナルでGitを利用できるか確認します。

   ```bash
   git --version
   ```

   未導入の場合は、表示される案内に従ってCommand Line Toolsを導入します。

2. GitHubからリポジトリを取得します。

   ```bash
   git clone https://github.com/SoBigsuns/MultifieldInflationLattice.git
   cd MultifieldInflationLattice
   ```

3. Gitの表示名とメールアドレスを未設定の場合のみ登録します。

   ```bash
   git config --global user.name "YOUR NAME"
   git config --global user.email "YOUR_GITHUB_EMAIL"
   ```

4. Julia依存関係を復元します。

   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

5. Python検証コードを使う場合は仮想環境を作成します。

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install numpy pandas scipy matplotlib
   ```

6. Jupyter Notebookを使う場合は同じ仮想環境へ追加します。

   ```bash
   python -m pip install jupyterlab ipykernel
   jupyter lab
   ```

## 日常の更新

作業を始める前に、GitHub上の最新版を取得します。

```bash
git pull --ff-only
```

編集後は差分を確認し、コミットしてGitHubへ送ります。

```bash
git status
git add <更新したファイル>
git commit -m "変更内容の要約"
git push
```

Windows側でも同様に、作業前の `git pull --ff-only` と作業後の `git push` を習慣にすると競合を避けやすくなります。

## 実行結果について

`runs/` は計算量とファイルサイズが大きくなりやすいためGitHubには保存しません。MacBookへ結果データも移したい場合は、クラウドストレージ、外付けSSD、または研究データ用リポジトリを別途利用してください。
