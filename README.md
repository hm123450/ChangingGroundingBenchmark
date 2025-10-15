<div align="center">

# ChangingGrounding: 3D Visual Grounding in Changing Scenes

</div>

<p align="center">
  <!-- <b>Authors</b><br> -->
  <a href="https://github.com/hm123450" target="_blank">Miao Hu<sup>1</sup></a>,
  <a href="https://github.com/huang583824382" target="_blank">Zhiwei Huang<sup>2</sup></a>,
  <a href="https://tai-wang.github.io" target="_blank">Tai Wang<sup>4</sup></a>,
  <a href="https://oceanpang.github.io" target="_blank">Jiangmiao Pang<sup>4</sup></a>,
  <a href="http://dahua.site" target="_blank">Dahua Lin<sup>3,4</sup></a>,
  <a href="http://www.aiar.xjtu.edu.cn/info/1046/1229.htm" target="_blank">Nanning Zheng<sup>1*</sup></a>,
  <a href="https://runsenxu.com" target="_blank">Runsen Xu<sup>3,4*</sup></a>
</p>

<p align="center">
  <sup>1</sup>Xi’an Jiaotong University,
  <sup>2</sup>Zhejiang University,
  <sup>3</sup>The Chinese University of Hong Kong,
  <sup>4</sup>Shanghai AI Laboratory
</p>

<p align="center">
  <sup>*</sup>Corresponding Author
</p>

<!-- <a href="">📑 Paper</a>  |
  <a href="">📖 arXiv</a> -->



<p align="center">
  <a href="https://hm123450.github.io/CGB/">🌐 Homepage</a>
</p>


## 🔔News
🔥[2025-06-28]: MMSI-Bench is now officially supported by [OpenCompass Spatial Leaderboard](https://huggingface.co/spaces/opencompass/openlmm_spatial_leaderboard) as a key benchmark for spatial understanding. It includes a *circular* testing protocol that effectively reduces the impact of random guessing. The best-performing non-thinking model so far, [Seed-VL 1.5](https://www.google.com.hk/search?q=seed-vl-1.5), achieves **20.3%** accuracy.

🔥[2025-06-18]: MMSI-Bench has been supported in the [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) repository.

✨[2025-06-11]: MMSI-Bench was used for evaluation in the experiments of [VILASR](https://arxiv.org/abs/2506.09965).

🔥[2025-06-9]: MMSI-Bench has been supported in the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) repository. Note: The VLMEvalKit repository uses its default answer extraction method, while this repository uses MMSI-Bench-specific post-prompts and regular expressions for answer extraction.

🔥[2025-05-30]: We released our paper, benchmark, and evaluation codes.



## Introduction
We introduce MMSI-Bench, a VQA benchmark dedicated to multi-image spatial intelligence. Six 3D-vision researchers spent more than 300 hours crafting 1,000 challenging, unambiguous multiple-choice questions, each paired with a step-by-step reasoning process. We conduct extensive experiments and evaluate 34 MLLMs, observing a wide gap: the strongest open-source model attains roughly 30% accuracy and OpenAI’s o3 reasoning model reaches 40%, while humans score 97%. These results underscore the challenging nature of MMSI-Bench and the substantial headroom for future research.

![Alt text](assets/teaser.jpg)

## 📄 License

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## Acknowledgment
MMSI-Bench makes use of data from existing image datasets: [ScanNet](http://www.scan-net.org/), [nuScenes](https://www.nuscenes.org/), [Matterport3D](https://niessner.github.io/Matterport/), [Ego4D](https://ego4d-data.org/), [AgiBot-World](https://agibot-world.cn/), [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36), [DAVIS-2017](https://davischallenge.org/) ,and [Waymo](https://waymo.com/open/). We thank these teams for their open-source contributions.

## Contact
- Sihan Yang: sihany077@gmail.com
- Runsen Xu:  runsxu@gmail.com
