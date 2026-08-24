# AudioSR fork design

## 目的と範囲

このシステムは、低域通過済み音声を条件として不足する高周波成分を生成し、48 kHzの
音声として出力するAudioSR推論パッケージである。音楽、音声、環境音を同じ推論経路で
扱う。失われた原音の厳密な復元、packet loss修復、低周波ノイズ除去、DirectML実行は
システムの契約に含まれない。

この配布物は推論用途を対象とし、ToyDataset、WebDataset学習backend、phonemeの数値展開は
同梱しない。これらを選択した経路は、欠落シンボルによる偶発的な失敗ではなく、明示的な
非対応エラーを返す。

本リポジトリは`haoheliu/versatile_audio_super_resolution`から分離した独立フォークで、
通常インストール、CLI、ローカルGradioを同じ推論コアへ接続する。

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
| `tests/` | 大容量checkpointを使わず、公開契約、shape、境界、package設定を検証する。 |

## 標準推論データフロー

1. `load_audio`がSoundFileで`[channels, samples]`のfloat32波形を読み、必要なら
   Torchaudioで48 kHzへresampleする。
2. `super_resolution`は各チャンネルを独立した1次元波形として処理する。
3. `_padded_sample_count`が長さを245,760サンプル（5.12秒）単位へ切り上げる。
4. `_prepare_mono_batch`が有限値化、振幅正規化、padding、STFT、melを生成し、記録・チャンネルで
   固定したfilter種類から低域条件を作る。
5. `LatentDiffusion.generate_batch`が条件付きlatentをsampleし、VAEとvocoderを通して波形を生成する。
6. 生成波形を入力チャンネルごとの元sample数へtrim/padし、`[1, channels, samples]`へ束ねる。
7. `save_wave`が生成済みの正確なsample数を変えず、SoundFileで48 kHz WAVを保存する。

## 長尺・batch・UIの流れ

- `super_resolution_long_audio`はチャンネルごとに入力をchunkへ分割し、overlap区間を
  crossfadeして元の正確な長さへ戻す。chunk durationはoverlapより長くなければならない。
- `super_resolution_batch`は長さの異なる複数のモノラル波形を共通のpadding長へ揃え、
  1回のモデル呼び出しで処理してから個別長へ戻す。
- Gradioは5.1秒chunkと0.5秒overlapを使い、選択batch size単位で
  `super_resolution_batch`を呼ぶ。生成seedはbatch・チャンネルごとに派生させる一方、
  低域条件用`lowpass_seed`はチャンネル内で固定する。
- Gradioのcacheは1モデルだけを保持する。同名モデルの連続要求は再利用し、
  `basic` / `speech`を切り替えると旧モデルを解放して新しいモデルを構築する。

## 重要な不変条件

- モデル内部、特徴量、出力は48 kHzを前提とする。
- 公開生成結果のshapeはBCT、SoundFileへ渡す直前だけ時間優先の配列へ転置する。
- チャンネル同士を推論前に混合せず、出力チャンネル数と順序を入力に合わせる。
- 短い入力もSTFT前に最低1segmentへpaddingする。生成後のpaddingは利用者へ返さない。
- 低域cutoffが0またはNyquist以上ならfilterを適用せず、入力のcopyを条件として使う。
- 無音と非有限値は0除算やNaNを発生させず有限値として処理する。
- DDIM stepsは1〜1000で、要求数と同数の有効なtimestepを返し、既定50 stepsのscheduleを保つ。
- 同じseedと入力は再現可能にしつつ、Gradioの別batch・別チャンネルは異なる生成seedを使う。
  低域filterの種類は記録・チャンネル単位で固定し、chunk境界では変えない。
- checkpointの形式を拡張子で判別し、pickle由来checkpointは`weights_only=True`で読む。
- 自動取得する`basic`と`speech`のcheckpointは、検証済みHugging Face revisionへ固定する。
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

### 低域条件を記録単位で固定する

推論条件のIIR filter種類はseedから記録・チャンネル単位で1回選び、全chunkとbatchで再利用する。
chunkごとの再抽選を避けることで通過帯域やrippleの差がoverlap境界へ混入しない。Gradioでは
生成の多様性を保つ派生seedと、条件を安定させる`lowpass_seed`を分離する。

### in-memory batchと単一モデルcache

Gradio chunkを一時ファイルへ書かずbatch tensorへまとめ、I/Oとモデル呼び出し回数を減らす。
batch sizeに比例してaccelerator memoryを使うため既定値は1である。モデルcacheも1件に限定し、
`basic`と`speech`の両方（合計約12.36 GB）を同時常駐させない。

### import副作用を遅延する

公開API、CLI、UI importではモデルstackを読み込まない。SRで使わないCLAPはDDPMの常時構築から
外し、学習用`get_audio_features`とRoBERTaはCLAP実行時まで遅延する。起動時の不要なHub依存と
memory消費を避ける一方、最初の実モデル構築時にはAudioSR checkpoint取得が必要である。

### checkpoint取得をrevisionへ固定する

既定の`basic`と`speech`は、検証済みのHugging Face完全長revisionから取得する。package versionが
同じなら取得時期によって重みが変わらないことを優先し、model更新はrevisionと回帰テストを
明示的に同期する。利用者が指定したlocal checkpoint pathはこの固定対象外である。

### 対応Pythonに応じてLibrosaを選択する

通常環境のLibrosaは、Python 3.10 / 3.11では0.11系、Python 3.12以降では1系を環境マーカーで
選択する。単一バージョン範囲よりmanifestは複雑になるが、対応Pythonの下限を維持しながら
新しいPythonでは現行メジャーを利用できる。

### 配布名とPython APIを分離する

PyPIでは本家の`audiosr`配布物と区別するため`kagayoi-audiosr`として公開する。一方、既存の
利用コードとCLI互換性を維持するため、import packageとconsole commandは`audiosr`のままにする。
配布名と実行時名が異なる複雑さは生じるが、フォークの識別性と利用者互換性を両立できる。

## 検証境界

単体テストはI/O、実特徴抽出、padding、chunk、shape、seed、引数、checkpoint loader、package
metadataを検証する。`basic`と`speech`は各約6.18 GBのため、通常テストは生成モデルをmockし、
実checkpointを使うend-to-end品質評価とGPU memory測定は含めない。
