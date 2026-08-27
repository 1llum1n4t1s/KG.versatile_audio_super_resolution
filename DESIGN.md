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
| `audiosr/sampling.py` | sampler名と`ddim_eta`の正規化を、torch非依存でmodel・pipeline・CLIへ供給する。 |
| `audiosr/latent_diffusion/` | Latent Diffusion、DDIM/DPM-Solver++ sampler、VAE、HiFi-GAN、条件encoderを実装する。 |
| `app.py` | Gradio UI、単一モデルcache、チャンネル別chunk batch、crossfade、UI用変換を担う。 |
| `tests/` | 大容量checkpointを使わず、公開契約、shape、境界、package設定を検証する。 |
| `tools/benchmark_samplers.py` | 実checkpointでsampler構成ごとの所要時間と再構成品質を測る開発用ハーネス。 |

## 標準推論データフロー

1. `load_audio`がSoundFileで`[channels, samples]`のfloat32波形を読み、必要なら
   Torchaudioで48 kHzへresampleする。
2. `super_resolution`は各チャンネルを独立した1次元波形として処理する。
3. `_padded_sample_count`が長さを245,760サンプル（5.12秒）単位へ切り上げる。
4. `_prepare_mono_batch`が有限値化、振幅正規化、padding、STFT、melを生成し、記録・チャンネルで
   固定したfilter種類から低域条件を作る。
5. `LatentDiffusion.generate_batch`が条件latentから生成shapeを決め、選択したsamplerでlatentを
   sampleし、VAEとvocoderを通して波形を生成する。生成はnoiseから始まり対象spectrogramのlatentを
   使わないため、first stage encodeは実行しない。
6. 生成波形の低域を入力の低域へ置換する。batch全体を1回のSTFT/ISTFTで処理し、波形を保持している
   deviceの上で完結させる。
7. 生成波形を入力チャンネルごとの元sample数へtrim/padし、`[1, channels, samples]`へ束ねる。
8. `save_wave`が生成済みの正確なsample数を変えず、SoundFileで48 kHz WAVを保存する。

## 長尺・batch・UIの流れ

- `super_resolution_long_audio`はチャンネルごとに入力をchunkへ分割し、指定したbounded
  batch単位で生成してoverlap区間をcrossfadeし、元の正確な長さへ戻す。acceleratorの
  out-of-memory時は同じchunk群を1件ずつ再試行する。chunk durationはoverlapより長くなければならない。
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
- samplerは`ddim` / `dpmpp2m` / `ddpm`のいずれかで、公開入口とsampler境界の両方で検証する。
  `ddim_eta`は有限の非負値で、`ddim`にだけ作用する。
- timestep spacingは`uniform` / `trailing` / `quad`のいずれかで、既定は`uniform`。
  `trailing`はstep数によらず最ノイズtimestepから開始する。
- samplerを変えてもnetwork評価回数はstep数と一致させ、到達する終端SNRも同じalpha列に揃える。
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

長尺APIとGradioのchunkを一時ファイルへ書かずbatch tensorへまとめ、I/Oとモデル呼び出し回数を
減らす。batch sizeに比例してaccelerator memoryを使うためライブラリ既定値は1とし、呼び出し側が
実行環境に合わせて1〜8を指定する。batch生成がout-of-memoryになった場合だけ1件ずつ再試行する。
モデルcacheも1件に限定し、`basic`と`speech`の両方（合計約12.36 GB）を同時常駐させない。

### NVIDIA CUDAだけSDPAを使う

既定モデルの`CrossAttention`は、NVIDIA CUDAではPyTorchのscaled dot product attentionを使い、
対応するfused kernelへ委ねる。ROCm、CPU、MPSは従来の明示的なattention計算を維持する。
ROCmで同じ置換を行うと推論が遅くなる環境があるため一律には適用せず、DDIM steps、CFG、
FP32、maskの意味は変更しない。maskは有限値の加算biasへ変換し、全keyがmaskされた場合も
従来と同じ結果を保つ。

### DDIMの共通コアをdevice非依存で最適化する

通常のDDIM推論では、条件付き・無条件のCFG分岐をbatch軸へ連結し、1ステップにつき1回の
U-Net forwardで計算してから分離する。内部batchとactivationの増加を抑えるため、公開pipelineの
batch sizeが2以下の場合だけ融合し、条件構造を安全に連結できない場合は従来の2回実行へ戻す。
この経路はCPU、CUDA、ROCmで共通であり、GPU名やbackend名による分岐を持たない。

同じmodel、device、steps、eta、scheduleに対するDDIM係数はmodelに1件だけcacheし、変更時は
cache keyで再生成する。timestep行と係数viewもsampling開始時にまとめ、stepごとの小tensor生成を
避ける。公開pipelineは利用しないintermediate latentを保持しない一方、`DDIMSampler`を直接呼ぶ
既存利用者向けの既定値はintermediate返却と逐次CFGを維持する。融合ではkernelの計算順序による
微小な丸め差が生じ得るため、固定seedの有限値、波形相関、高域スペクトル、peak memoryを実機で
確認してから配布物へ採用する。

### samplerを名前で選び、DDIMのalpha列を共有する

`ddim_eta=1.0`の既定はancestral samplingであり、少ないstepでは品質が落ちる。stepと品質の
trade-offを実測できるようにするため、samplerを`ddim` / `dpmpp2m` / `ddpm`から名前で選ぶ。
`ddim`と`use_plms`のflag対は残さず1つの名前へ統合し、tri-stateの分岐を作らない。

PLMSは削除した。公開pipeline、CLI、Gradioのどこからも到達できず、`register_buffer`が
`cuda`を直書きしていたためCPU・ROCm・MPSでは構築時に失敗する状態だった。線形多段法としての
役割はDPM-Solver++が上位互換で置き換える。

`dpmpp2m`は`DDIMSampler`のschedule構築とCFG分岐をそのまま継承し、`ddim_alphas`と
`ddim_alphas_prev`が定めるnoise levelを同じ順序で辿る。したがって同じstep数なら network
評価回数も終端SNRも`ddim`と一致し、違いは更新式だけになる。1次に落とすと`ddim_eta=0.0`の
DDIMと代数的に一致するため、実checkpointなしでも数値でsampler実装を検証できる。

### timestep spacingを選べるようにし、既定は変えない

`uniform`はstep数が学習schedule（1000）を割り切るとき、最ノイズtimestepへ到達しない。
20 stepならt=951止まりで、そこは`alphas_cumprod`が5.6e-3、つまり信号成分が√で7.5%残る点である。
sampling自体は純noiseから始まるので、この差はsamplerが説明できない誤差として残る。step数が
多ければ吸収されるが、少ないと残る。割り切れないstep数はlinspace分岐に落ちてt=999から始まるため、
同じ`uniform`指定でもstep数によって挙動が質的に変わっていた。

`trailing`はstep数によらずt=999から始める。20/21/25/26 stepを並べると割り切れるstep数だけが
悪化する交互パターンになり、原因がsamplerではなくspacingにあることが確認できる。

Radeon 760Mで、96 kHz/24bitのCC0ピアノ録音を4 kHzへ帯域制限したreferenceに対する実測
（判定帯域4.0–14.8 kHz、劣化入力のlsdは3.189）:

| 構成 | 秒 | lsd | band_energy |
| --- | ---: | ---: | ---: |
| `ddim` / 20 / `eta0` / `trailing` | 145 | **1.498** | 45.2 |
| `ddim` / 12 / `eta0` / `trailing` | 143 | 1.499 | 42.0 |
| `dpmpp2m` / 20 / `trailing` | 144 | 1.509 | 45.7 |
| `ddim` / 50 / `eta1` / `trailing` | 172 | 1.546 | 95.3 |
| `ddim` / 50 / `eta1` / `uniform`（既定） | 169 | 1.633 | 103.3 |
| `ddim` / 20 / `eta0` / `uniform` | 145 | 3.270 | 48.8 |

`trailing`は既定より良く、かつ速い。step数を50から12へ下げても品質は落ちない（1.499）ので、
**step削減は`trailing`とセットでのみ成立する**。逆に`uniform`の20 stepは劣化入力（3.189）にすら
負け、reference帯域より上へ撒くエネルギー比が1262に達する（`trailing`はほぼ0）。t=951の時点で
残っている7.5%の信号成分をノイズとして扱った結果が、そのまま可聴帯域外のゴミとして出ている。

既定を`uniform`のままにしたのは、**出荷済みの挙動を維持する不変条件があること一点**による。
品質面の根拠は無い。推奨構成は`--discretize trailing --ddim_steps 20`である。

なお`band_energy`が1.0ならreferenceと同エネルギーだが、既定は103、つまりlevel合わせ後で
+20 dB過剰である。この素材に対してmodelは「復元」ではなく「明るく」している。LSDが改善するのは
log距離が「何も無い」を「多すぎる」より重く罰するためで、step数を下げるほど過剰量も減る
（103→42）。

### 条件stageのVAEをfirst stageと共有する

AudioSRは低域melをVAEでencodeしたものを条件に使うため、設定上はVAEを2つ持つ。配布済み
checkpointではこの2つが完全に同じ506 tensor（各1.542 GiB）で、片方は生成にも復元にも寄与しない
まま構築・device転送・常駐だけしていた。1つのmoduleを両方の名前から参照させ、redundantな
checkpoint keyはmodel構築前に解放する。

共有の判断は**設定ではなくcheckpointの中身**で行う。設定が同型でも別々に学習されたcheckpointは
あり得るので、`torch.equal`で全tensorの一致を確認できたときだけ共有し、一致しなければ従来どおり
2つ構築する。この判断のためにcheckpoint読み込みをmodel構築より前へ移した。

なお`clap.*`（507 tensor、0.746 GiB）はcheckpointに含まれるが、SR経路のcond_stage_configに
CLAPが無いため`strict=False`で捨てられており、runtimeには元から載っていない。

### 入力のサンプルレートを保つ

modelは48 kHz固定で、24 kHzより上を生成する能力を持たない。96 kHz入力は入口で48 kHzへ落とされ、
出力も48 kHzになるため、hi-res素材を渡すと**形式そのものが失われる**。`restore_high_rate`は
48 kHzの復元結果を元のrateへ戻し、24 kHz超には元音源の中身をSTFT上で差し戻す。低域で既に
行っている置換を上端へ適用しただけで、生成は一切行わない。

実測では、96 kHz/24bitのCC0ピアノ録音でも24 kHz超は-91 dBのフラットなノイズフロアだけで、
楽音は入っていなかった（20 kHz以上が平坦、最小値は33.8 kHz）。したがってこの機能の価値は
音質向上ではなく**入力形式と元音源が実際に持っていた内容の保全**にある。

既存の`super_resolution`系の戻り値と48 kHz契約は変えず、opt-inの別関数と`--preserve_input_rate`
として足した。出荷済みの呼び出し側が黙って壊れることを避けるためである。

### 生成に使わないfirst stage encodeを実行しない

super-resolutionの生成はnoiseから始まり、networkへ入るのは条件latentだけである。対象
spectrogramのlatentはshapeの参照にしか使われていなかったため、encoder forwardを1回分削る。
shapeは条件latentから取る。条件側は同一構成のautoencoderが同じ寸法のspectrogramを符号化する
ので、batch数と時間長は一致する。step数を減らすほど固定costの比率が上がるため、この削減の
効果はstep数が小さい構成ほど大きい。

### 低域置換をbatchでdevice上に保つ

低域置換はitemごとにlibrosaのSTFT/ISTFTをCPUで回していた。生成波形をtensorのまま保持し、
batch全体を1回の`torch.stft`/`torch.istft`で処理する。cutoff探索はcumulative energyの
prefix長から求め、走査loopを持たない。crossover binとlevel合わせのgain、その上下限は
変更していない。CPUとGPUのどちらが速いかは機種構成に依存するため、採用値は
`tools/benchmark_samplers.py`の実測で決める。

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

samplerと低域置換は実checkpointなしでも数値で検証する。`dpmpp2m`は1次で`ddim_eta=0.0`の
DDIMと一致すること、batch化した低域置換がlibrosa実装と一致すること、mel置換が元の走査loopと
一致すること、`trailing`が全step数で最終timestepへ到達し`uniform`が割り切れるstep数で到達しない
ことを、stub modelと合成信号で確認する。step数と品質のtrade-off、および低域置換を
CPUとGPUのどちらで回すべきかは数値等価では決まらないため、`tools/benchmark_samplers.py`で
実機のGPUと実checkpointを使って測る。

品質測定には帯域の広いreferenceが要る。`tools/benchmark_samplers.py`はreferenceが実際に内容を
持つ帯域を検出してその範囲だけで採点し、reference帯域より上の生成量は誤差ではなく別項目として
報告する。この検出は**最大binからの相対**ではなく**その録音自身のノイズフロアからの相対**で行う。
音楽は基音から大きく下がった位置でも内容を持ち続けるため（96 kHzのピアノ録音は16 kHzで-88 dB）、
ピーク相対で切ると判定できる帯域のほとんどを捨ててしまう。実測では、既知の12.4 kHzブリック
ウォールを持つspeechで13.1 kHz、ピアノで14.8 kHzを返し、合成brick wallに実録音相当のノイズフロア
（-40〜-80 dB）を載せた場合は誤差+1〜+14 binに収まる。

**referenceの素材はLSDで判定できるものを選ぶ。** `example/speech.wav`では全構成が劣化入力に
勝てなかった（最良1.967896 対 基準1.968455、差0.03%）。speechの4–13 kHzは摩擦音、つまりnoiseで
あり、modelはエネルギー量を正しく戻せても（band_energy 1.09）細部は別物になる。LSDはbin毎
frame毎の差を測るので、これを無音と区別できない。一方ピアノの同帯域は倍音、つまり基音から
位置が決まる構造なので、復元の成否がそのまま差として出る（3.189→1.498、8%の構成間差）。
`ranking_is_unresolved`はこの取り違えを防ぐために、劣化入力に対する改善が2%未満なら順位を
未確定として警告する。referenceが帯域制限されていると、モデルが本来行うべき高域生成が減点対象になり、構成
同士を分離できなくなる。どの構成も劣化入力に勝てない場合は順位を未確定として警告する。

### 実測で分かった律速（Radeon 760M、10秒モノラル）

1回の`generate_batch`の内訳は、vocoderが129.8秒で82%、samplingが28.0秒で17%、VAE decodeが1.6秒、
条件encodeが1.0秒、低域置換が0.01秒だった。step数を100から20へ減らしても総時間は187秒から147秒に
しか下がらない。この構成ではsamplerの改善余地よりvocoderのほうが大きい。CPUとGPUの相対性能で比率は
変わるため、他機種では再測定が要る。
