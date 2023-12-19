<https://github.com/yahoojapan/JGLUE>

## Memo

OpenAIのembeddingsは、同一の文に対して微妙に異なるベクトルになっていることがある。
Googleのでは、そんなことはない。同一の文に対しては同じベクトルになっている。

ID:20は前半後半共に `室内の机の上にノートパソコンが置かれています。` で、どちらに対しても同じベクトルになっている。
ID:21の後半も同じ文なのに、若干異なるベクトルになっている。
以下にはID:20のものとID:21のものの冒頭7次元を抜粋したもの。

```
[-0.0015773575, -0.012670631, 0.003507396,  -0.0019979863, 0.0049698898, 0.01683809,  -0.024422348,
[-0.0013762178, -0.01236897,  0.0034336674, -0.0020307505, 0.0049417624, 0.016893255, -0.024479039,
```

今回はFaissのインデックスの性質を比較する目的であるため、このベクトルの不安定さはそぐわない。
よってOpenAIのほうはいったん忘れることにする。
