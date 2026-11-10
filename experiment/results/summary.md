# 实验汇总（自动生成，请人工核对后再写入论文）

## 评审模型校准（1.5B，9 项子检查 → 3 个量表）

- 已评审 24 棵；评审分数取值个数 = 1，方差 = {'judge_consistency': 0.0, 'judge_persona_stability': 0.0, 'judge_choice_meaningfulness': 0.0}
- 评审合成分与确定性量表分的相关系数 = 无定义（评审分零方差）
- 确定性量表分跨度 = 2.78~5.00（同期评审仅 5.00~5.00）
- 结论：1.5B 评审对全部样本给出相同分数，不具备区分度，不能作为主评测，仅作对照证据。

## Base (prompt only)

 depth  consistency_mean  consistency_std  persona_stability_mean  persona_stability_std  choice_meaningfulness_mean  choice_meaningfulness_std
     2               5.0              0.0                   3.667                  0.713                       3.167                      0.690
     4               5.0              0.0                   2.833                  0.690                       2.333                      1.008
     6               5.0              0.0                   2.666                  0.618                       1.333                      0.617

## LoRA-SFT

 depth  consistency_mean  consistency_std  persona_stability_mean  persona_stability_std  choice_meaningfulness_mean  choice_meaningfulness_std
     2               5.0              0.0                   4.500                   0.69                       4.167                      0.690
     4               5.0              0.0                   2.833                   0.69                       2.500                      0.472
     6               5.0              0.0                   3.167                   0.69                       2.333                      0.000


## 显著性检验（Wilcoxon signed-rank，配对 world×seed）

说明：delta = mean(条件 a) − mean(条件 b)；“depth2 vs depth4”即 a=depth2、b=depth4；“ft-vs-base”行中 a=base、b=ft（delta 为负表示 ft 更高）。

    method                   metric             test stat      p   delta  n_pairs
      base        persona_stability depth2 vs depth4  0.0 0.0339  0.8336        8
      base            dup_line_rate depth2 vs depth4  0.0 0.0180 -0.4167        8
      base                distinct2 depth2 vs depth4  0.0 0.0078  0.4622        8
      base    choice_meaningfulness depth4 vs depth6  0.0 0.0394  1.0000        8
      base            dup_line_rate depth4 vs depth6  0.0 0.0078 -0.2202        8
      base                distinct2 depth4 vs depth6  0.0 0.0078  0.2635        8
        ft        persona_stability depth2 vs depth4  0.0 0.0078  1.6669        8
        ft    choice_meaningfulness depth2 vs depth4  0.0 0.0078  1.6671        8
        ft            dup_line_rate depth2 vs depth4  0.0 0.0178 -0.3000        8
        ft                distinct2 depth2 vs depth4  0.0 0.0078  0.3461        8
        ft     intent_distinct_rate depth2 vs depth4  0.0 0.0231  0.1429        8
        ft         persona_hit_rate depth2 vs depth4  0.0 0.0078  0.2750        8
        ft            mean_line_len depth2 vs depth4  1.0 0.0156  1.2775        8
        ft            dup_line_rate depth4 vs depth6  0.0 0.0078 -0.2298        8
        ft                distinct2 depth4 vs depth6  0.0 0.0078  0.2591        8
ft-vs-base    choice_meaningfulness           depth2  0.0 0.0412 -1.0001        8
ft-vs-base     intent_distinct_rate           depth2  0.0 0.0143 -0.7500        8
ft-vs-base         persona_hit_rate           depth2  2.0 0.0234 -0.5000        8
ft-vs-base option_semantic_distance           depth4  0.0 0.0431 -0.0122        8
ft-vs-base     intent_distinct_rate           depth4  0.0 0.0078 -0.3571        8
ft-vs-base    choice_meaningfulness           depth6  0.0 0.0143 -0.9998        8
ft-vs-base                distinct2           depth6  1.0 0.0156 -0.0544        8
ft-vs-base option_semantic_distance           depth6  1.0 0.0156 -0.0261        8
ft-vs-base     intent_distinct_rate           depth6  0.0 0.0078 -0.3750        8

## 干预对照 v1 vs v2（A1 节拍表 + A2 去重重采样 + A4 秘密门控）
### 总体
               metric    v1    v2  delta  n_pairs     p
            composite 3.704 3.907  0.204       24 0.048
          consistency 5.000 5.000  0.000       24   NaN
    persona_stability 3.333 3.778  0.445       24 0.007
choice_meaningfulness 2.778 2.944  0.167       24 0.666
        dup_line_rate 0.365 0.312 -0.053       24 0.069
      option_dup_rate 0.203 0.175 -0.029       24 0.289
 intent_distinct_rate 0.653 0.795  0.142       24 0.171
     persona_hit_rate 0.239 0.477  0.238       24 0.000
        numeric_drift 3.125 2.208 -0.917       24 0.058
            distinct2 0.581 0.552 -0.028       24 0.274

### 按深度
              metric  depth    v1    v2  delta
           composite      2 4.389 4.500  0.111
           composite      4 3.500 3.722  0.222
           composite      6 3.222 3.500  0.278
       dup_line_rate      2 0.000 0.000  0.000
       dup_line_rate      4 0.442 0.308 -0.133
       dup_line_rate      6 0.653 0.627 -0.026
     option_dup_rate      2 0.000 0.000  0.000
     option_dup_rate      4 0.161 0.125 -0.036
     option_dup_rate      6 0.450 0.399 -0.050
intent_distinct_rate      2 0.625 1.000  0.375
intent_distinct_rate      4 0.696 0.732  0.036
intent_distinct_rate      6 0.637 0.653  0.016

### 干预执行统计（v2 每棵树均值）
```
{
  "calls": 45.86,
  "dedupe_calls": 13.44,
  "dedupe_fixed": 2.42,
  "dedupe_gaveup": 17.11,
  "reveal_retries": 5.42,
  "beat_leak_lines": 0.0,
  "leaf_reveal": 1.75,
  "leaf_reveal_rate": 0.146,
  "trees": 36
}
```

图：plots/fig5_improvement_v2.{png,pdf}（配对 n=24，探索性对照）
