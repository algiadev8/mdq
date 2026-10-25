# CoreML 速度検証

計測日: 2026-10-23。Apple M3、8 CPUコア、16 GiB RAM、macOS 26.4.1、Rust 1.98.1。
Releaseビルド。基準コミット `1f7aec0` に本セッションの未コミット変更を加えた作業ツリーで実行。
モデルは同じ `multilingual-e5-small`、出力384次元、上限512トークン。

## 2026-10-25: 自動選択の修正

実利用でも形状切り替えにより少量更新が長時間停止する性能劣化が報告されたため、
`MDQ_EMBED_BACKEND=auto` はCPUを選択するよう修正した。検索クエリも同様。
CoreMLの初期化・モデルハッシュ計算・コンパイルは既定経路では実行しない。
CoreML実装は `MDQ_EMBED_BACKEND=coreml` の明示指定で残す。
以下は修正前の計測記録であり、定常推論の高速化を理由に自動選択すべきではなかった。

## 結論（2026-10-23の計測時点）

- モデル読み込み後の定常推論では、CoreMLがCPUの **2.35〜4.84倍**。
- モデル初期化・形状ごとのセッション作成・保存まで含む64件の処理では、CPU **2.505秒**、CoreML **33.227秒**。この小規模な処理ではCoreMLが約13.3倍遅かった。
- 現状のボトルネックはCoreMLの初期化／形状切り替え。定常推論の倍率をそのままインデックス処理全体の高速化率とは解釈できない。
- 自動選択ポリシーは変更していない。短い処理を優先する場合は `MDQ_EMBED_BACKEND=cpu` でCPU固定が可能。

## 定常推論

CPUは4スレッド、両経路ともバッチ8。同じ40文書を各3試行、実行順を交互にして比較し、時間は中央値。
入力は同じ英単語の反復と文書番号を含む合成データ。モデル読み込み・CoreMLセッション初期化・初回呼び出し・CPU互換性チェックは定常時間から除外。

| 実効トークン数/文書 | CoreML固定長 | CPU/40件 | CoreML/40件 | CPU時間÷CoreML時間 |
| --- | --- | --- | --- | --- |
| 30 | 32 | 0.157214秒 | 0.066874秒 | 2.35倍 |
| 126 | 128 | 0.742272秒 | 0.189493秒 | 3.92倍 |
| 512 | 512 | 4.856715秒 | 1.003959秒 | 4.84倍 |

CPUモデル読み込み: 0.773秒。Appleバックエンド準備（モデルSHA-256確認など）: 0.987秒。
各形状の最初の呼び出しは、それぞれ35.087秒、30.333秒、8.979秒だった。初回呼び出しにはセッション作成・必要なコンパイル／ロード・推論・CPU互換性検査を含む。
ディスクキャッシュは削除していないため、これらを統一条件の完全コールド起動値とは扱わない。

## モデル初期化・保存込み

長さの異なる日本語と英語の合成64チャンクを、同じ既定バッチ8・トークン長順で処理。
`semantic::embed_missing` 呼び出し全体を計測。モデル初期化、トークン化、推論、CoreML形状切り替え、CPU互換性検査、SQLite保存を含む。
BM25構築、CLI引数解析、Cargoビルドは含めない。埋め込みキャッシュは空、モデルダウンロード用・CoreMLコンパイル済みディスクキャッシュは既存のものを利用。
それぞれ別プロセスで1回ずつ測定しており、上の3試行中央値とは異なる。

| 経路 | 処理全体 |
| --- | --- |
| `MDQ_EMBED_BACKEND=cpu` | 2.505秒 |
| `MDQ_EMBED_BACKEND=coreml` | 33.227秒 |

ベンチマーク出力の `original` は旧バッチ64の参照経路であり、上のCPU/CoreML比較には使っていない。
同じ既定バッチ8の実装を選択した `optimized` フィールド同士を比較した。

## 実行経路と数値互換性

- ORTの暗黙CPU EPフォールバックを禁止したCoreML経路で完走。プロファイルでもモデルの融合ノードが `CoreMLExecutionProvider` で実行されたことを確認。
- CoreMLの設定はMLProgram / ComputeUnits::All。GPUとNeural Engineのどちらに内部配置されたかはこの計測では区別していない。これはCoreML経由の検証であり、GPU単独の速度を意味しない。
- 定常推論用のCPU/CoreML出力の最大要素差は約 `1.40e-7`、最小コサイン類似度は `0.99999999999976`。
- 64件のCoreML保存ベクトルもCPU参照と比較し、最大差は約 `1.19e-7`。
- スループットは合成データに依存する。入力長分布、文書数、キャッシュ、熱状態、バックグラウンド負荷により結果は変わる。

## 再現

モデルがローカルキャッシュに必要。初回はモデルダウンロードとApple管理キャッシュへの書き込みが発生する可能性がある。
計測を同時に実行しないこと。

```sh
# 定常推論: 入力長別、CPU/CoreML交互、3試行中央値
MDQ_COREML_PROFILE=target/coreml-speed cargo test --release --lib coreml_speed_matrix -- --ignored --nocapture

# 同じ実装のCPU側（モデル初期化・SQLite保存込み）
MDQ_EMBED_BACKEND=cpu cargo test --release --test embedding_performance benchmark_embedding_pipeline -- --ignored --nocapture

# 同じ実装のCoreML側（CPUへ自動退避せず、失敗はエラー）
MDQ_EMBED_BACKEND=coreml MDQ_COREML_PROFILE=target/coreml-profile cargo test --release --test embedding_performance benchmark_embedding_pipeline -- --ignored --nocapture
```

定常推論の全試行値は `target/coreml-speed-results.json` に出力される。
ORTプロファイルは指定したプレフィックスと実行日時を含むJSONファイルとして保存される。
本検証ではアプリケーションの自動選択・フォールバック方針は変更せず、手動速度ベンチマークだけを追加した。
