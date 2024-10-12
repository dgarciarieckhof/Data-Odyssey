# VLMs Cookbooks

## Introduction

This repo will serve as a cookbook to explore VLM use cases and demonstrate how to apply them through fairly simple examples.

|                  | Notebook                                                     | Description                                                  |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| V-RAG & Interpretability | [ColPali & VLMs](https://github.com/dgarciarieckhof/Data-Odyssey/blob/main/VLMs/tunnel_vision/notebook/ColPali%20%2B%20VLMs.ipynb) | Avoid tedious parsing and leverage the power of Visual Language Models and ColPali, along with an interpretability module.|                                               |
| Fine-tuning ColPali | [Dataset creation](https://github.com/dgarciarieckhof/Data-Odyssey/blob/main/VLMs/tunnel_vision/notebook/ColPali%20FineTuning%20Dataset.ipynb) | Generating a dataset of queries for training and fine-tuning ColPali models on custom datasets.|                                               |
| Fine-tuning ColPali | [Fine-tuning](https://github.com/dgarciarieckhof/Data-Odyssey/blob/main/VLMs/tunnel_vision/notebook/ColPali%20FineTuning%20Training.ipynb) | Training and fine-tuning ColPali models on custom datasets.|                                               |
| Showcasing ColPali & Qdrant | [Colpali & Qdrant](https://github.com/dgarciarieckhof/Data-Odyssey/blob/main/VLMs/tunnel_vision/notebook/ColPali%20%2B%20Qdrant.ipynb) | Using ColPali + Qdrant to index and search earnings reports, along with an interpretability module.|                                               |
| Zero-Shot Object Detection | TBD | TBD|                                               |

## Instructions

### Run locally

To run the notebooks locally, you can clone the repository and open the notebooks in Jupyter Notebook or in your IDE.

## Sources

```BibTeX
@article{Qwen2VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}

@misc{faysse2024colpaliefficientdocumentretrieval,
      title={ColPali: Efficient Document Retrieval with Vision Language Models}, 
      author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2407.01449},
      eprint={2407.01449},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.01449}, 
}
```
