
<div align="center">
<h1>Towards Text-Image Interleaved Retrieval</h1> 
</div>

<p align="center">
<a href="https://arxiv.org/abs/2502.12799">
  <img src="https://img.shields.io/badge/Arxiv-2502.12799-orange.svg"></a> 
<a href="https://opensource.org/license/mit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg"></a> 
<a href="https://github.com/vec-ai/wikiHow-TIIR/pulls">
    <img src="https://img.shields.io/badge/Contributions-welcome-blue.svg?style=flat"></a>
</p>


## Introduction
Current multimodal information retrieval studies mainly focus on single-image inputs, which limits real-world applications involving multiple images and text-image interleaved content.
In this work, we introduce the text-image interleaved retrieval (TIIR) task, where the query and document are interleaved text-image sequences, and the model is required to understand the semantics from the interleaved context for effective retrieval.

<img src="./assets/idea.jpg"  height="320px"></a>

We construct a TIIR benchmark based on naturally interleaved wikiHow tutorials, where a specific pipeline is designed to generate interleaved queries.

| Part        | #Examples | Avg./Min/Max #Images | Avg. Text #Tokens | #Positives |
|-------------|-----------|----------------------|-------------------|-------------|
| Corpus      | 155,262   | 4.97 / 2 / 64        | 85.62             |             |
| Train Query | 73,084    | 2.88 / 2 / 4         | 105.15            | 1           |
| Test Query  | 7,654     | 2.81 / 2 / 4         | 105.59            | 1           |


To explore the task, we adapt several off-the-shelf retrievers and build a dense baseline by interleaved multimodal large language model (MLLM). We then propose a novel Matryoshka Multimodal Embedder (MME), which compresses the number of visual tokens at different granularity, to address the challenge of excessive visual tokens in MLLM-based TIIR models.

Experiments demonstrate that simple adaption of existing models does not consistently yield effective results. Our MME achieves significant improvements over the baseline by substantially fewer visual tokens.

<img src="./assets/results.jpg"  height="320px"></a>

## Todo

- [ ] Release code for model training and evaluation.
- [ ] Release code for data curation.
- [ ] Release the `wikiHow-TIIR` dataset.

## Acknowledgments

Our retrieval corpus is built opon [wikiHow-VGSI](https://github.com/YueYANG1996/wikiHow-VGSI).
