## Transformer from scratch, implemented for mps and cuda with wandb for tracking ##

Implementation of the Transformer architecture, widely used across tasks in natural language processing (NLP), computer vision, and audio. The version in this repository is designed for language translation, aiming to translate from any language to any other, consistent with the goals of the original implementation.

The core idea is to use multi-head attention, which splits the input into several heads to enable parallel computation and to allow the model to learn different types of relationships between words or tokens in different subspaces. During attention computation, queries, keys, and values are used in a specific formula to determine the attention distribution. In the encoder, this mechanism captures relationships within the source text; in the decoder, it captures relationships between the source and target sequences. The query-key interaction identifies relevant relationships (e.g., between a word to translate and the rest of the sentence, or between source and target words), while the value provides the actual information that is passed forward based on this relevance.

<img width="267" alt="GENAI-1 151ded5440b4c997bac0642ec669a00acff2cca1" src="https://github.com/user-attachments/assets/4a566b02-ffc8-4734-b101-2568dcc207e8" />

## Acknowledgments

This project was inspired by [Umar Jamil]’s excellent tutorial on building Transformers from scratch:  
🔗 [YouTube Video]([https://youtu.be/example](https://www.youtube.com/watch?v=ISNdQcPhsts&t=10180s))  
🔗 [GitHub Repository]([https://github.com/username/repo-name](https://github.com/hkproj/pytorch-transformer))
