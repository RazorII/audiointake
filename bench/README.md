# bench

The measurements behind audiointake's defaults and the numbers its README quotes. Everything here fetches public
data on demand into a work folder you name, and nothing it fetches is ever committed: no audio, no parquet, no
reference transcript. Install the extras first: `pip install "audiointake[bench]"` (pyarrow and pythainlp).

## FLEURS: word error rate per language and route

```
python bench/fleurs.py prep WORK                       # 50 fixed test clips per language, about 0.7 GB per language while it runs
python bench/fleurs_routes.py asr WORK de_de es_419 hu_hu th_th
python bench/fleurs_routes.py whisper WORK de_de es_419 hu_hu
python bench/fleurs_routes.py routes WORK de_de es_419 hu_hu
python bench/fleurs_routes.py e2e WORK de_de es_419 hu_hu th_th
python bench/fleurs.py score WORK ultra v3 whisper core vote vote3 e2e --json WORK/score.json
python bench/stats.py diff WORK/score.json es_419:core de_de:core
python bench/stats.py diff WORK/score.json es_419:vote es_419:core --subset no_digit
```

`core` is the route every language without a pack takes (Parakeet Ultra, with Parakeet v3 filling the gaps where
Ultra heard nothing); `vote` is the Whisper vote route as the package runs it; `e2e` is the whole transcription,
language screen included. The models are the revisions audiointake pins, read from the local snapshot folders
`audiointake setup` fills (`asr` and `whisper` refuse when a pinned revision isn't there); `--model-ref` passes the
repository name instead, for measuring a caller that resolves models by name. `whisper --variant plain` and
`routes --whisper-variant plain` give Whisper's plain call (no loudness normalisation, no language, no compression
guard) and both votes on it, to see how much of a vote's number is the Whisper call's. `stats.py` gives paired bootstrap intervals (10,000 resamples of clip ids,
`random.Random(42)`): the Spanish and German clips are translations of the same 50 sentences, so they pair by id.
Scoring keeps accents (á, ñ, ü, ő) and drops punctuation, ¿ and ¡; Thai is scored by character in spoken form.

## AMI: speakers in room audio

```
python bench/ami.py fetch AMI --b-sessions --annotations
python bench/ami.py run AMI OUT                         # word-by-word speaker attribution, speaker by time, WER
python bench/ami.py enroll AMI VOICES --model CAMPP.onnx
python bench/ami.py run AMI OUT --voices VOICES         # also: how many labels were named, and named right
python bench/enroll_seconds.py --ami AMI --ref AMI/ref --hyp OUT --model CAMPP.onnx
```

Four far-field test meetings (ES2004a, IS1009a, TS3003a, EN2002a; 92 minutes, one distant microphone, four people
each). `ami_attrib.py` scores each transcript word's speaker against the reference RTTM; `timeline_attrib.py` scores
speaker by time; `ami_wer.py` scores words against the AMI manual transcripts. `enroll_seconds.py` measures how many
seconds of speech a voiceprint needs (3, 5, 8, 15, 30 and 60 s), and how often a stranger is named instead.

## Path fixtures and the language-screen check

```
python bench/fixtures.py build FIX --fleurs WORK --ami AMI
python bench/uncovered.py OUT/x.asr.json OUT/x.diar.rttm
```

`fixtures.py` builds short recordings from public audio, each named by the route it exercises: `vote`, `turns`,
`gapfill-de`, `gapfill-es`, `silence`, `aac-m4a` (AAC in an .m4a) and `spatial-qta` (two audio streams, one with four
channels, in a QuickTime file). `uncovered.py` measures how much diarized speech produced no words, the measure behind
the `speech_not_transcribed` warning.

## Data and credits

- **FLEURS** (google/fleurs), CC BY 4.0. Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal
  Representations of Speech", 2022. Fetched at a fixed dataset commit, each file checked against its SHA-256.
- **AMI Meeting Corpus**, CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/. Carletta et al., "The AMI Meeting
  Corpus", 2005. Audio from the AMI mirror, checked against its SHA-256.
- **AMI reference RTTMs** from BUTSpeechFIT/AMI-diarization-setup (`only_words`), Apache-2.0, at a fixed commit.
