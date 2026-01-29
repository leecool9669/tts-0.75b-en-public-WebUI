---
license: cc-by-4.0
language:
- en
library_name: moshi
pipeline_tag: text-to-speech
tags:
- audio
---
# Model Card for Kyutai TTS (public data)

See also the [pre-print research paper](https://arxiv.org/abs/2509.08753),
the [project page](https://kyutai.org/next/tts), 
the [GitHub repository](https://github.com/kyutai-labs/delayed-streams-modeling/),
and the [evaluation pipeline](https://github.com/kyutai-labs/tts_longeval).


This is a model for streaming text-to-speech (TTS).
Unlike offline text-to-speech, where the model needs the entire text to produce the audio,
our model starts to output audio as soon as the first few words from the text have been given as input.
This model was trained on a mixed of public TTS datasets, allowing for a fair comparisons with other
methods.

## Model Details

The model architecture is a hierarchical Transformer that consumes tokenized text and generateds audio tokenized by Mimi,
see [the Moshi paper](https://arxiv.org/abs/2410.00037).
The frame rate is 12.5 Hz and each audio frame is represented by 16 audio tokens. You cannot use less tokens at inference.
The backbone model is 300M parameters, and the depth transformer is 450M parameters and uses partial weight sharing similar to [Hibiki](https://arxiv.org/abs/2502.03382).
The audio is shifted by 16 steps (1.28 sec.) with respect to the text, and the model uses an acoustic/semantic delay of 2.

## Model Description

Kyutai TTS is a decoder-only model for streaming speech-to-text.
It leverages the multistream architecture of [Moshi](https://moshi.chat/) to model text stream based on the speech stream.
The text stream is shifted w.r.t. the audio stream to allow the model to predict text tokens based on the input audio.

* Developed by: Kyutai
* Model type: Streaming Text-To-Speech.
* Language(s) (NLP): English
* License: Model weights are licensed under CC-BY 4.0
* Repository: [GitHub](https://github.com/kyutai-labs/delayed-streams-modeling/)

## Uses

### Direct Use

This model is able to perform streaming text-to-speech generation.
This model allows for voice cloning through prefixing, 
although when compared with our [main model](https://huggingface.co/kyutai/tts-1.6b-en_fr),
it achieves a speaker similarity score of 74.9%, against 80.9% for our main model. 
This level of speaker similarity is in line with some of the existing baselines like CSM 
and it thus seems safe to open source it.

This model does not perform watermarking for two reasons:
- watermarking can easily be deactivated for open source models,
- our early experiments show that all watermark systems used by existing TTS are removed by simply encodeding and decoding the audio with Mimi.


## How to Get Started with the Model


This model is provided primarily for the purpose of scientific comparisons on public benchmarks.
In particular, please check our pipeline for running TTS model evaluations on a number of benchmarks: [tts_longeval](https://github.com/kyutai-labs/tts_longeval).

Here is an example, first install `moshi`, for instance with
```bash
pip install -U "git+https://git@github.com/kyutai-labs/moshi.git#egg=moshi&subdirectory=moshi
```

```python
import torch
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_VOICE_REPO, TTSModel

text = "Hey there! How are you? I had the craziest day today."
voice = "expresso/ex03-ex01_happy_001_channel1_334s.wav"

checkpoint_info = CheckpointInfo.from_hf_repo('kyutai/tts-0.75b-en-public')
tts_model = TTSModel.from_checkpoint_info(
    checkpoint_info, n_q=16, temp=0.6, cfg_coef=3, device=torch.device("cuda")
)
entries = tts_model.prepare_script([text], padding_between=1)
# `voice` could also be a local wav file.
voice_path = tts_model.get_voice_path(voice)
prefix = tts_model.get_prefix(voice_path)

print("Generating audio...")
pcms = []
def _on_frame(frame):
    print("Step", len(pcms), end="\r")
    if (frame[:, 1:] != -1).all():
        pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu()
        pcms.append(pcm.clip(-1, 1))

# You could also generate multiple audios at once by extending the following lists.
all_entries = [entries]
prefixes = [prefix]
with tts_model.mimi.streaming(len(all_entries)):
    result = tts_model.generate(all_entries, [], on_frame=_on_frame, prefixes=prefixes)

print("Done generating.")
audios = torch.cat(pcms, dim=-1)

for audio, prefix in zip(audios, prefixes):
    # We need to skip the audio prefix.
    skip = int((tts_model.mimi.sample_rate * prefix.shape[-1]) / tts_model.mimi.frame_rate)
    audio = audio[..., skip:]
    # Now do something with this audio!
```


## Model Card Authors

Neil Zeghidour, Eugene Kharitonov, Manu Orsini, Václav Volhejn, Gabriel de Marmiesse, Edouard Grave, Patrick Perez, Laurent Mazaré, Alexandre Défossez

## License & Citation

License is CC-BY 4.0. For citations please use the following.
```
@techreport{kyutai2025streaming,
      title={Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling}, 
      author={Neil Zeghidour and Eugene Kharitonov and Manu Orsini and Václav Volhejn and Gabriel de Marmiesse and Edouard Grave and Patrick Pérez and Laurent Mazaré and Alexandre Défossez},
      year={2025},
      eprint={2509.08753},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.08753}, 
}
```
