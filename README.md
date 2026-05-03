# Audio2Audio Realtime Conversational Pipeline
This setup allows for flexible - swappable provider components for having either a completely offline system or some components from cloud providers.

in-progress

#### about this repository
for context, i made this repo in 2023 while i was playing with ASR and NLP while preparing for SIH 2023. The POC which we made for the hackathon worked quite well for trascription but was just an API integration - which was probably the reason why we were grand finalist and not the winners. This is the repo where i made and played with PoCs and a stable offline transcription system back then.

Now in 2026, open source models are way far ahead and can even run on constraint systems over CPU inference. And that is the reason why i came back to this repository to build a realtime streaming audio to audio pipeline to finish what i started.
  
<hr>

## Components

### Streaming TTS model
Choosing a TTS from a cloud provider is a no brainer but with hardware constraints and with concern for Indic support, i chose the following models. Pocket TTS performed the best so far on CPU in terms of TTFB.

Tested with: [Pocket TTS](), [Indic Parler TTS](https://huggingface.co/ai4bharat/indic-parler-tts)

### Noise Suppresion
For testing in browsers, the WebAudio API office noise suppresion which performs really well. But if the system is tested on actual telephony channel, it has to have it's filter against interference and this component is considered in the pipeline.

Tested with: [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)

### Smart Turn Detection
This model does not seem to work. Needs attention. 

Tested with: [Smart Turn V2](https://huggingface.co/pipecat-ai/smart-turn-v2)

### LLM
The brain. Does it's job text to text. I could use an AudioLM but i appreciate the accuraccy and control in the pipeline for now. We can use openrouter provider models or just a good model with edge inference capability like Gemma 4 by Google or even Phi 4 by Microsoft

Tested with: [Gemma 4](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main)

### Voice Activity Detection
Plays a key role in making the system feel truly conversational.

Tested with: [Silero](https://github.com/snakers4/silero-vad)

### Speech to Text
SOTA after Vosk which i had tested back in 2023. This just works but is not truly streaming i suppose considering it's delay in output.

Tested with: [Faster Whisper Base](https://huggingface.co/Systran/faster-whisper-base)
