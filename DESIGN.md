# AudioSR fork design

## 目的と範囲

このシステムは、低域通過済み音声を条件として不足する高周波成分を生成し、48 kHzの
音声として出力するAudioSR推論パッケージである。音楽、音声、環境音を同じ推論経路で
扱う。失われた原音の厳密な復元、packet loss修復、低周波ノイズ除去、DirectML実行は
システムの契約に含まれない。

本リポジトリは`haoheliu/versatile_audio_super_resolution`から分離した独立フォークで、
通常インストール、CLI、ローカルGradio、Cog Predictorを同じ推論コアへ接続する。

## 主要コンポーネント

| コンポーネント | 責務 |
| --- | --- |
| `audiosr/__init__.py` | 公開APIを遅延exportし、軽量importを維持する。 |
| `audiosr/__main__.py` | `python -m audiosr`と`audiosr`コマンドの引数検証、モデル共有、ファイル反復、保存を行う。 |
| `audiosr/pipeline.py` | device選択、checkpoint読込、特徴batch生成、標準・長尺・batch推論を統括する。 |
| `audiosr/utils.py` | SoundFile読込、48 kHz resampling、正規化、STFT/mel、低域条件、checkpoint取得、出力保存を担う。 |
| `audiosr/lowpass.py` | 推論条件用の低域通過フィルタと出力長整列を行う。 |
| `audiosr/latent_diffusion/` | Latent Diffusion、DDIM/PLMS sampler、VAE、HiFi-GAN、条件encoderを実装する。 |
| `app.py` | Gradio UI、単一モデルcache、チャンネル別chunk batch、crossfade、UI用変換を担う。 |
| `predict.py` | Cogの`setup`でモデルを1回構築し、`predict`でファイル入力からWAVを返す。 |
| `tests/` | 大容量checkpointを使わず、公開契約、shape、境界、package設定を検証する。 |

`inference.py`は補助的なstandalone実装として残っているが、READMEの標準CLIでも
`cog.yaml`のPredictor入口でもない。

## 標準推論データフロー

1. `load_audio`がSoundFileで`[channels, samples]`のfloat32波形を読み、必要なら
   Torchaudioで48 kHzへresampleする。
2. `super_resolution`は各チャンネルを独立した1次元波形として処理する。
3. `_padded_sample_count`が長さを245,760サンプル（5.12秒）単位へ切り上げる。
4. `_prepare_mono_batch`が有限値化、振幅正規化、padding、STFT、mel、低域条件を生成する。
5. `LatentDiffusion.generate_batch`が条件付きlatentをsampleし、VAEとvocoderを通して波形を生成する。
6. 生成波形を入力チャンネルごとの元sample数へtrim/padし、`[1, channels, samples]`へ束ねる。
7. `save_wave`が入力durationを上限としてSoundFileで48 kHz WAVを保存する。

## 長尺・batch・UIの流れ

- `super_resolution_long_audio`はチャンネルごとに入力をchunkへ分割し、overlap区間を
  crossfadeして元の正確な長さへ戻す。chunk durationはoverlapより長くなければならない。
- `super_resolution_batch`は長さの異なる複数のモノラル波形を共通のpadding長へ揃え、
  1回のモデル呼び出しで処理してから個別長へ戻す。
- Gradioは5.1秒chunkと0.5秒overlapを使い、選択batch size単位で
  `super_resolution_batch`を呼ぶ。batchごと・チャンネルごとにseedを派生させる。
- Gradioのcacheは1モデルだけを保持する。同名モデルの連続要求は再利用し、
  `basic` / `speech`を切り替えると旧モデルを解放して新しいモデルを構築する。
- Cogは`Predictor.setup`でモデルを構築し、各`predict`ではそのインスタンスを再利用する。

## 重要な不変条件

- モデル内部、特徴量、出力は48 kHzを前提とする。
- 公開生成結果のshapeはBCT、SoundFileへ渡す直前だけ時間優先の配列へ転置する。
- チャンネル同士を推論前に混合せず、出力チャンネル数と順序を入力に合わせる。
- 短い入力もSTFT前に最低1segmentへpaddingする。生成後のpaddingは利用者へ返さない。
- 低域cutoffが0またはNyquist以上ならfilterを適用せず、入力のcopyを条件として使う。
- 無音と非有限値は0除算やNaNを発生させず有限値として処理する。
- DDIM stepsは1〜1000で、CLIとsamplerの両境界が検証する。
- 同じseedと入力は再現可能にしつつ、Gradioの別batch・別チャンネルは異なる派生seedを使う。
- checkpointの形式を拡張子で判別し、pickle由来checkpointは`weights_only=True`で読む。
- package importとCLI helpはネットワーク、checkpoint、tokenizer、GPU初期化を要求しない。

## 採用済み設計判断

### SoundFileを通常I/O境界にする

SoundFileはモノラル・複数チャンネルを同じshape契約で読み書きでき、duration取得も行える。
これにより通常推論の一時WAV、`ffprobe`、`ffmpeg` subprocessを除去した。resamplingは
PyTorch tensorとdevice契約へ接続しやすいTorchaudioへ残している。

### チャンネル別推論とBCT契約

モデルの条件生成はモノラルを前提とするため、複数チャンネルは混合せずチャンネルごとに
同じモデルへ通す。計算量はチャンネル数に比例するが、stereo情報と公開shapeを失わない。

### 固定segmentと入力長への復元

モデルの時間解像度に合わせて245,760サンプル単位へpaddingする。任意長入力を扱える代わりに
短い入力でも1segment分を生成する計算コストが生じる。出力側で必ず元長へ戻す。

### in-memory batchと単一モデルcache

Gradio chunkを一時ファイルへ書かずbatch tensorへまとめ、I/Oとモデル呼び出し回数を減らす。
batch sizeに比例してaccelerator memoryを使うため既定値は1である。モデルcacheも1件に限定し、
`basic`と`speech`の両方（合計約12.36 GB）を同時常駐させない。

### import副作用を遅延する

公開API、CLI、UI importではモデルstackを読み込まない。SRで使わないCLAPはDDPMの常時構築から
外し、学習用`get_audio_features`とRoBERTaはCLAP実行時まで遅延する。起動時の不要なHub依存と
memory消費を避ける一方、最初の実モデル構築時にはAudioSR checkpoint取得が必要である。

### 通常環境とCog環境を分離する

通常環境はPython 3.10〜3.14と新しい依存rangeを使う。Cogは既存GPU imageとの互換性のため
CUDA 11.7 / PyTorch 2.0系を固定している。両者は同じソースを実行するが、依存更新と
セキュリティ評価は別々に行う。

通常環境のLibrosaは、Python 3.10 / 3.11では0.11系、Python 3.12以降では1系を環境マーカーで
選択する。単一バージョン範囲よりmanifestは複雑になるが、対応Pythonの下限を維持しながら
新しいPythonでは現行メジャーを利用できる。

## 検証境界

単体テストはI/O、実特徴抽出、padding、chunk、shape、seed、引数、checkpoint loader、package
metadataを検証する。`basic`と`speech`は各約6.18 GBのため、通常テストは生成モデルをmockし、
実checkpointを使うend-to-end品質評価とGPU memory測定は含めない。
