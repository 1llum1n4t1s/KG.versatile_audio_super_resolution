# Repository working agreement

このファイルはリポジトリ全体に適用する。システム構造と設計判断は
[DESIGN.md](DESIGN.md)、利用者向け手順は[README.md](README.md)、リリース履歴は
[CHANGELOG.md](CHANGELOG.md)を正本とする。

## 環境と依存管理

- 対応Pythonは3.10以上3.15未満。通常の開発にはPython 3.12または3.13を使う。
- PyTorch、TorchVision、TorchAudioはOS・アクセラレータに合う公式wheelを先に導入する。
  固定したCUDA indexを通常の`requirements.txt`へ追加しない。
- 通常環境の直接依存は`requirements.txt`と`setup.py`の`REQUIRED`を同期する。
  テスト依存は`setup.py`の`EXTRAS["test"]`が正本である。
- PyPI配布名は`kagayoi-audiosr`、Python import名とCLI名は`audiosr`である。
  配布名を変更するときは`setup.py`、READMEの導入手順、package metadataテストを同期する。
- LibrosaはPython 3.10 / 3.11で0.11系、Python 3.12以降で1系を選ぶ環境マーカーを
  `requirements.txt`と`setup.py`で同期し、対応Python範囲全体を解決可能に保つ。
- 製品バージョンの正本は`setup.py`の`VERSION`。明示的なバージョン依頼時だけ更新し、
  READMEの現行版表記、CHANGELOG、メタデータテストを同期する。
- lock fileは現在コミットしていない。新しいpackage managerやlock形式へ移行するときは、
  Python 3.10〜3.14とCPU/CUDA/macOSの解決方法を同時に定義する。

## 実装上の不変条件

- 推論の作業・出力サンプルレートは48 kHz。入力は読み込み時に48 kHzへ変換する。
- 公開パイプラインの生成波形は`[batch, channels, samples]`、単一チャンネル処理は
  1次元`[samples]`を基本契約とする。時間軸は常に最後の軸として扱う。
- モノラル化を暗黙に行わない。CLI、長尺処理、Gradio、保存処理でチャンネル数と順序を保つ。
- 特徴抽出前の長さは245,760サンプル単位に切り上げ、生成後に入力の正確な長さへ戻す。
- 無音、NaN、Infinity、短い最終チャンク、ゼロoverlapを有限値のまま処理する。
- DDIM stepsは1〜1000。CLIだけでなくサンプラー境界でも検証する。
- `.bin` / `.ckpt`は`torch.load(..., weights_only=True)`、`.safetensors`は
  `safetensors.torch.load_file`で読む。安全でないpickle fallbackを追加しない。
- CLIの`--help`や`import audiosr`では、モデル構築、checkpoint download、
  `from_pretrained`を実行しない。CLAP・学習用依存は実際に使う経路まで遅延する。
- Gradioは同時に1モデルだけをキャッシュする。モデル切替時は旧モデルを解放し、
  バッチとチャンネルには再現可能で衝突しない派生seedを渡す。
- 通常の音声I/OはSoundFileを使い、resamplingにはTorchaudioを使う。通常推論へ
  `ffprobe`、一時WAV、外部変換プロセスを再導入しない。

## 変更時の確認範囲

- パイプライン変更は`audiosr/pipeline.py`、`audiosr/utils.py`、公開export、CLI、Gradioの
  呼び出し契約を確認する。
- 音声shape、chunk、seed、checkpoint、CLI引数を変更したら対応する`tests/`へ回帰テストを追加する。
- 約6.18 GBの実checkpointは通常の単体テストでdownloadしない。モデル生成をmockし、
  特徴抽出・shape・trim・呼び出し引数は実コードで検証する。
- READMEは利用者向けのインストール、使い方、主要機能、制約に限定する。
  内部構造の説明はDESIGN.mdへ記載する。
- 変更前後に`git status --short --branch`を確認し、無関係な作業ツリー差分を保持する。

## 必須検証コマンド

通常変更では、対象テストに続けて全テストを実行する。

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run --isolated --no-project --python 3.13 `
  --with-requirements requirements.txt --with pytest==9.1.1 pytest -q
```

Python・依存・パッケージ境界を変更した場合は、同じコマンドを`--python 3.10`と
`--python 3.14`でも実行する。

```powershell
uvx ruff check --select E9,F63,F7,F82 app.py setup.py audiosr tests
uv run --isolated --no-project --python 3.14 python -m compileall -q `
  app.py setup.py audiosr tests
git diff --check
```

配布物またはmanifestを変更した場合は、wheelとsdistを一時ディレクトリへ生成し、
`twine check`、wheelの新規環境インストール、`audiosr --help`、sdistの必須ファイルを確認する。
PyPI公開時は公開ファイルのSHA-256をローカル成果物と照合し、PyPIから新規インストールする。
生成された`build/`、`dist/`、`*.egg-info/`はコミットしない。

## 配布ファイル

- `MANIFEST.in`はREADME、CHANGELOG、LICENSE、可視化画像、example資料、モデル設定・語彙を含める。
- `.github/dependabot.yml`はrootのpip ecosystemを毎週監視し、minor/patchをまとめる。
