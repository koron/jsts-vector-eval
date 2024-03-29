# ベクトルの量子化(圧縮)が最近傍探索に与える影響の調査

## TL;DR

Product Quantizationを始めとしたベクトルの圧縮を目的としたベクトル量子化が、
最近傍探索のコンテキストにおいてどのような影響を及ぼすのか調査した。
本調査ではまず量子化の誤差、量子化前後でベクトルがどれだけ異なるかを平均二乗誤差(小さいほど良い)で求めた。
次に最近傍探索(k-NN)におけるTop 10の再現率(recall@10, 大きいほど良い)を各量子化方式に対して求めた。

量子化の誤差は、
量子化に利用できる情報量がデータに対して相対的に大きいほど誤差が小さくなるという、
言われてみれば当たり前の結果が確認できた。
これはつまり圧縮率が高いほど誤差が大きくなることを示している。
量子化モジュールの学習に利用したベクトルに限定すると当然誤差は小さくなる。
特筆すべき点として後発の量子化方式ほど、この学習に利用したベクトルの誤差がより小さくなる傾向がみられた。

k-NNにおけるTop 10の再現率は、前述の量子化誤差が大きくなる条件ほど後発の量子化方式が優位であった。
加えて学習に利用したベクトルの再現率は、後発の量子化方式の成績が極めてよかった。

量子化を用いた高圧縮環境における最近傍探索の再現率は高いとは言えない。
近傍探索を必要とする用途に合わせて別の方式を組み合わせて再現率を高める工夫が必要と言えるだろう。

## 前提知識

### 量子化方式に共通のプロパティ

今回取り扱った複数の量子化方式に共通するプロパティは以下の通り。

*   `d` 次元数: 入出力となるベクトルの次元数
*   `M` モジュール数: いずれの量子化も複数のモジュールに分割して行う、その数
*   `nbits` 1モジュールあたりの量子化ビット数

1つのベクトルは量子化の結果 `M * nbits` ビットで表されることになる。
例: `d = 768` 768次元のベクトルは `768 * 32 = 24576` ビット。
これを `M = 16, nbits = 8` で量子化すると `16 * 8 = 128` ビット。
圧縮率で言うと 1/192 となる。

### 量子化方式

* L2 量子化無し。単なるL2ノルムによる距離計算
* PQ (Production Quantization) ベクトルを `d / M` 次元ずつに分解しそれぞれを量子化する
* OPQ PQの前に回転を施すもの
* RQ (Residential Quantization) 1モジュールあたり $`2^{nbits}`$ 個のベクトルから選択し、全モジュールのベクトルを足し合わせることで量子化とする。PQの一般化
* LSQ (Local Search Quantization) RQの発展形

掲載順は、より単純なものを先に提示した。

## 検証手順

[qerr.py](pycmd/qerr.py)を実行して、量子化方式の量子化誤差をプロパティを変えつつ計算した。
出力結果は[data/qerr\_out.tsv](data/qerr_out.tsv)。
それを可視化したのが[Qerror.ipylib](./Qerror.ipynb)で、
以下のことを検証できた。

* `M`が大きくなるほど量子化誤差が小さくなる
* `d`が大きくなるほど量子化誤差が大きくなる
* `nbits`が大きくなるほど量子化誤差が小さくなる
* 評価に用いるベクトル数を増やしても、量子化誤差は変わらない
    * 学習に使ったベクトル数及び分布が十分であれば、量子化器は充分な性能を示す

[qerr\_d.py](pycmd/qerr_d.py)を実行して、`d`のみに注目して量子化誤差を計算した。
出力結果は[data/qerr\_d\_out.tsv](data/qerr_d_out.tsv)。
それを可視化したのが[Qerror\_d.ipylib](./Qerror_d.ipynb)で、
以下のことを検証できた。

*   評価用ベクトルに対しては量子化方式の優劣はつけ難い
    *   僅かではあるがPQがもっとも良さそうだ
*   学習用ベクトルに対しては、LSQがもっとも誤差が小さく、次いでRQ、PQとなった

[recall.py](./pycmd/recall.py)を実行して、各量子化方式における recall@10 を計算した。
結果は [recall.tsv](./data/recall.tsv) 。
LSQのようなより高度な量子化方式のほうが、高圧縮条件において良い recall@10 を記録している。

[recall-word.py](./pycmd/recall-word.py)を実行して、JSTSのデータを元に量子化方式における recall@10 を計算した。
結果は [recall-word.tsv](./data/recall-word.tsv) 。
JSTSのデータは高圧縮条件を満たしており、LSQなどの高度な量子化が高い recall@10 を記録しており、良く機能していることがわかる。
ただし `nbits` を6に落とすとLSQ等においても recall@10 の低下がみられ、なんらかの限界があることがわかる。

[recall-simulation.py](./pycmd/recall-simulation.py)を実行して、JSTSデータの規模感をシミュレーションし乱数により量子化方式における recall@10 を計算した。
JSTSデータの偏りが及ぼした影響を検証するため。
結果は [recall-simulation.tsv](./data/recall-simulation.tsv) 。
入力データの特徴としては次元は768と大きいのに対し、学習に用いたベクトル数は2731と少ない。
PQ, RQにおいては recall@10 が極端に小さくなった一方で、LSQにおいては高い数値を維持している。
ベクトル数が充分に少ないために、LSQが局所解(量子化用のベクトル)を上手く探索できたと推測される。

## 結論

現時点におけるベクトル量子化とk-NN及び近似最近傍探索は、
用途や要件に応じて各種方式を適宜組み合わせてより良いものを選ぶ必要がある。

## 参考資料

* [faiss:The index factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)

    今回、調査に利用したのは Encodings と Vector transforms の一部

* [直積量子化を用いた近似最近傍探索 松井勇佑 国立情報学研究所 2016(PDF)](https://yusukematsui.me/project/survey_pq/doc/prmu2016.pdf)

    Product Quantization の詳細な解説がされている

* [ショートコードによる大規模近似最近傍探索 松井勇佑 国立情報学研究所 2016(PDF)](https://yusukematsui.me/project/survey_pq/doc/ann_lecture_20161202.pdf)

    PQの発展形(OPQ, RQ, LSQ)への言及有り

* [近似最近傍探索の最前線](https://speakerdeck.com/matsui_528/jin-si-zui-jin-bang-tan-suo-falsezui-qian-xian)

    [158ページ: ANN手法選択フローチャート](https://speakerdeck.com/matsui_528/jin-si-zui-jin-bang-tan-suo-falsezui-qian-xian?slide=158)

* [koron/techdocs: Product Quantization (PQ)を用いた際の精度・誤差についての検討](https://github.com/koron/techdocs/tree/main/product-quantization-errors)

    Quantization としての量子化誤差の影響の検討プロジェクト。
    Nearest Neighborで利用した際の影響も見ないと、となって本プロジェクトに繋がった。
