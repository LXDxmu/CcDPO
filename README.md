<p align="center">
  <h1 align="center">🌟 [NeurIPS 2025] Zooming from Context to Cue: Hierarchical Preference Optimization for Multi-Image MLLMs
</h1>
    <p align="center">
    <a href="https://github.com/LXDxmu"><strong>Xudong Li</strong></a>
    ·
    <a href=""><strong>Mengdan Zhang</strong></a>
    ·
    <a href=""><strong>Peixian Chen</strong></a>
    ·
    <a href=""><strong>Xiawu Zheng</strong></a>
    ·
    <a href=""><strong>Yan Zhang</strong></a>
    ·
    <a href=""><strong>Jingyuan Zheng</strong></a>
    <br>
    <a href=""><strong>Yunhang Shen</strong></a>
    ·
    <a href=""><strong>Ke Li</strong></a>
    ·
    <a href=""><strong>Chaoyou Fu</strong></a>
    ·
    <a href=""><strong>Xing Sun</strong></a>
    ·
    <a href=""><strong>Rongrong Ji</strong></a>
  </p>
  📖<a href="https://arxiv.org/pdf/2505.22396">Paper</a> |🏠<a href="https://github.com/LXDxmu/CcDPO">Homepage</a></h3>
  |🤗<a href="">Huggingface</a></h3>
<div align="center"></div>
<p align="center">
  <p>

🔥 **Motivation:** Multi-modal Large Language Models (MLLMs) excel at single-image tasks but struggle with multi-image understanding due to cross-modal misalignment, leading to hallucinations such as context omission, conflation, and misinterpretation. Existing methods using Direct Preference Optimization (DPO) constrain optimization to a solitary image reference within the input sequence, neglecting holistic context modeling.
    
🌈 **Proposed:** We propose Context-to-Cue Direct Preference Optimization (CcDPO), a multi-level preference optimization framework that enhances per-image perception in multi-image settings by zooming into visual clues—from sequential context to local details. CcDPO features: (i) Context-Level Optimization: Re-evaluates cognitive biases underlying MLLMs' multi-image context comprehension and integrates low-cost global sequence preferences for bias mitigation. (ii) Needle-Level Optimization: Directs attention to fine-grained visual details through region-targeted visual prompts and multimodal preference supervision. To support scalable optimization, we construct MultiScope-42k, an automatically generated dataset with high-quality hierarchical preference pairs. Experiments show that CcDPO significantly reduces hallucinations and yields consistent performance gains across general single- and multi-image tasks.

## 📢 News
- 🚀 [09/19/2025] Our paper is accepted by NeurIPS 2025!
- 🚀 [05/28/2025] We upload our paper to arxiv.

## 💡 Highlights
- 🔥 **Hierarchical Preference Optimization:** We propose CcDPO, a two-level preference optimization framework that enhances MLLMs' capability to accurately perceive visual information across hierarchical levels—from sequential multi-image contexts to individual fine-grained details.
- 🔥 **Comprehensive Hallucination Mitigation:** CcDPO addresses three fundamental hallucination types in multi-image understanding: context omission, context conflation, and detail misinterpretation.
- 🔥 **Scalable Dataset Construction:** We build MultiScope-42k, a large-scale automatically generated dataset with high-quality multi-level preference pairs, enabling cost-effective and scalable optimization.
- 🔥 **Superior Performance:** CcDPO consistently outperforms existing methods on seven multi-image benchmarks while maintaining strong single-image capabilities.

## 🔍 Multi-Image Hallucinations in MLLMs
We identify three fundamental hallucination types that critically degrade MLLMs' performance in multi-image scenarios:

(1) **Context Omission:** The model selectively ignores subsets of input images, generating responses based on incomplete sequences.

(2) **Context Conflation:** The model erroneously attributes visual elements across images.

(3) **Detail Misinterpretation:** Critical visual details in a certain image are either missed or misinterpreted.


## 🏗️ CcDPO Framework
CcDPO is a hierarchical preference alignment framework that refines MLLMs at two levels:

**(i) Context-Level Optimization:** By contrasting complete and disrupted multi-image captions using language-based preference optimization, we enhance MLLMs' contextual understanding by ensuring comprehensive integration of all relevant visual information across image sequences. We introduce two perturbation techniques:
- **Sequence Truncation**: Simulates context omission by removing or shortening captions
- **Content Swapping**: Simulates conflation by mismatching image indices and descriptions

**(ii) Needle-Level Optimization:** A hybrid preference optimization framework integrates two complementary objectives:
- **Language-based Preference**: Contrasts captions that align with or mismatch visually prompted regions
- **Vision Contrastive Preference**: Discriminates between images semantically matching or contradicting given captions

## 📊 MultiScope-42k Dataset
We construct **MultiScope-42k**, a large-scale preference dataset with automatically generated positive and perturbed response pairs at both context and needle levels. The dataset features:

- **27.3K** context-level pairs from COCO-2014
- **10.8K** needle-level language pairs from COCO-2017  
- **3.7K** needle-level vision contrastive pairs from Flickr30k
- Fully automated generation pipeline for cost-effective scalability

## 🚀 Training
The training code will be available soon.


## 🛠️ Evaluation
Evaluate the processed models using <a href="https://github.com/open-compass/VLMEvalKit"><strong>VLMEvalKit</strong></a>.


## 📚Citation
If our work is helpful for your research, please consider citing:
```
@article{li2025zooming,
  title={Zooming from Context to Cue: Hierarchical Preference Optimization for Multi-Image MLLMs},
  author={Li, Xudong and Zhang, Mengdan and Chen, Peixian and Zheng, Xiawu and Zhang, Yan and Zheng, Jingyuan and Shen, Yunhang and Li, Ke and Fu, Chaoyou and Sun, Xing and others},
  journal={arXiv preprint arXiv:2505.22396},
  year={2025}
}
```

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
License: Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use

---

<div align="center">

**⭐ If this project helps you, please give us a Star! ⭐**
</div>
