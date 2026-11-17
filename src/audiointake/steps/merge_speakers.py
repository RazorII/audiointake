"""Put the diarizer's speaker labels on the speech recogniser's words: the speaker-labelled transcript.

Usage::

    python -m audiointake.steps.merge_speakers <asr.json> <diar.json|diar.rttm|""> <out.txt> [--names speakers.json] [--sentences]
    python -m audiointake.steps.merge_speakers --count-speakers <diar.json|diar.rttm>   # prints the distinct speaker count

- ``--names``: the speaker-ID worker's sidecar. A label it named by voiceprint prints as the name ("Ana: ..."); the
  others keep "Speaker N", numbered among themselves. An unreadable sidecar leaves every label numbered.
- Each WORD goes to the diarizer turn it overlaps most. When people talk without a pause, the recogniser writes one
  run-on sentence across the change of speaker; a rule that gives the whole sentence to whoever spoke most of it
  then puts the other person's words in the wrong mouth. On four AMI far-field meetings (room audio, 4 speakers,
  human reference labels) the word rule labels 97.7 % of words right, against 91.0 % for one speaker per sentence.
  Word timings come from parakeet-mlx's tokens or the slim sidecar's "w" lists; a sentence with neither (Whisper's
  sentences, the Thai turns) or whose words don't rejoin to its text exactly keeps the sentence rule.
- ``--sentences``: the whole-sentence rule, for comparisons.

- If diarization is present and parseable, writes a speaker-labelled transcript ("Speaker 1: ...") to out.txt,
  overwriting the plain version.
- If diarization is missing or empty, writes a plain (unlabelled) transcript.
- If the ASR JSON can't be parsed, leaves out.txt as it is (the plain .txt is already there) and exits 0 when it
  exists, so the caller keeps the plain transcript.

Defensive about both JSON schemas: accepts several common key names.

The library form is ``merge(asr, rttm_text, names)``, which returns the same lines plus each line's speaker turn
with its start and end (``contracts.Merged``), and ``count_speakers(rttm_text)``.
"""
from __future__ import annotations

import bisect
import json
import os
import sys
from collections.abc import Mapping

from ..contracts import Merged, Turn
from . import rttm


def load(p):
    if not p or not os.path.exists(p):
        return None
    if p.endswith(".rttm"):  # the diarizer's RTTM output
        return {"segments": [{"start": s["start"], "end": s["start"] + s["dur"], "speaker": s["spk"]}
                             for s in rttm.segments_from_path(p)]}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _rttm_doc(text):
    return {"segments": [{"start": s["start"], "end": s["start"] + s["dur"], "speaker": s["spk"]}
                         for s in rttm.segments_from_text(text)]}


def _seglist(d, extra_keys=()):
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        for k in ("segments", "chunks", "sentences", "result", "results") + tuple(extra_keys):
            if isinstance(d.get(k), list):
                return d[k]
    return None


def _num(*vals):
    for v in vals:
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def seg_words(s):
    """[[start, end, word], ...] for one ASR sentence, or None.

    From parakeet-mlx's "tokens" (a token that starts with a space begins a word; punctuation rides on the word
    before it) or from the slim sidecar's "w" list. None unless the words, joined by single spaces, give back the
    sentence text exactly, so a split can never change what the transcript says. slim_asr imports this, so the
    sidecar and the merge always agree on what a word is."""
    out = []
    toks = s.get("tokens")
    if isinstance(toks, list) and toks:
        for t in toks:
            txt = t.get("text") if isinstance(t, dict) else None
            st, en = (_num(t.get("start")), _num(t.get("end"))) if isinstance(t, dict) else (None, None)
            if not isinstance(txt, str) or st is None or en is None:
                return None
            if not out or txt.startswith(" "):
                out.append([st, en, txt.strip()])
            else:
                out[-1][1] = en
                out[-1][2] += txt
    elif isinstance(s.get("w"), list) and s["w"]:
        for w in s["w"]:
            if not (isinstance(w, (list, tuple)) and len(w) == 3 and isinstance(w[2], str)):
                return None
            st, en = _num(w[0]), _num(w[1])
            if st is None or en is None:
                return None
            out.append([st, en, w[2]])
    else:
        return None
    out = [w for w in out if w[2]]
    text = (s.get("text") or "").strip()
    return out if out and " ".join(w[2] for w in out) == text else None


def asr_segments(d):
    segs = _seglist(d)
    if segs is None:
        return None
    out = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        text = (s.get("text") or s.get("word") or "").strip()
        if not text:
            continue
        st = _num(s.get("start"), s.get("start_time"), s.get("startTime"))
        en = _num(s.get("end"), s.get("end_time"), s.get("endTime"))
        if st is None:
            ts = s.get("timestamp") or s.get("offsets")
            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                st, en = _num(ts[0]), _num(ts[1])
        out.append({"start": st, "end": en, "text": text, "words": seg_words(s)})
    return out or None


def diar_segments(d):
    segs = _seglist(d)
    if segs is None:
        return None
    out = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        st = _num(s.get("startTimeSeconds"), s.get("start"), s.get("startTime"), s.get("start_time"))
        en = _num(s.get("endTimeSeconds"), s.get("end"), s.get("endTime"), s.get("end_time"))
        if st is None or en is None:
            continue
        spk = s.get("speakerId") or s.get("speaker") or s.get("label") or s.get("speaker_id") or "?"
        out.append({"start": st, "end": en, "spk": str(spk)})
    return out or None


def speaker_for(seg, diars):
    if not diars or seg["start"] is None:
        return None
    end = seg["end"] if seg["end"] is not None else seg["start"]
    best, best_ov = None, 0.0
    for d in diars:
        ov = min(end, d["end"]) - max(seg["start"], d["start"])
        if ov > best_ov:
            best_ov, best = ov, d["spk"]
    return best


class Turns:
    """The diarizer's turns, indexed by start, so each word's lookup is a bisect, not a scan."""

    def __init__(self, diars):
        self.d = sorted(diars, key=lambda d: d["start"])
        self.starts = [d["start"] for d in self.d]
        self.maxdur = max((d["end"] - d["start"] for d in self.d), default=0.0)

    def speaker(self, st, en):
        """The label whose turn overlaps [st, en] most, or None (a word in a gap between turns)."""
        best, best_ov = None, 0.0
        i = bisect.bisect_left(self.starts, en) - 1
        while i >= 0 and self.starts[i] >= st - self.maxdur:
            d = self.d[i]
            ov = min(en, d["end"]) - max(st, d["start"])
            if ov > best_ov:
                best_ov, best = ov, d["spk"]
            i -= 1
        return best


def timed_units(asr, diars, split):
    """(diarizer label or None, text, start, end) per word, or per sentence when split is off or it has no word
    timings. A word in a gap between turns takes the label of the word before it in its sentence, else the word
    after it, else the sentence's own label. Start and end are the word's (or the sentence's) times, None where the
    recogniser gave none."""
    turns = Turns(diars) if split else None
    out = []
    for s in asr:
        ws = s.get("words") if split else None
        if not ws:
            out.append((speaker_for(s, diars), s["text"], s["start"], s["end"]))
            continue
        labs = [turns.speaker(a, b) for a, b, _ in ws]
        for i in range(1, len(labs)):
            if labs[i] is None:
                labs[i] = labs[i - 1]
        for i in range(len(labs) - 2, -1, -1):
            if labs[i] is None:
                labs[i] = labs[i + 1]
        if labs[0] is None:
            labs = [speaker_for(s, diars)] * len(ws)
        out.extend((lab, w[2], w[0], w[1]) for lab, w in zip(labs, ws))
    return out


def units(asr, diars, split):
    """(diarizer label or None, text) per word, or per sentence (``timed_units`` without the times)."""
    return [(lab, text) for lab, text, _, _ in timed_units(asr, diars, split)]


def _labelled(asr, diars, names, split):
    """The labelled lines, and one Turn per line, by the one numbering rule."""
    lines: list[str] = []
    turns: list[Turn] = []
    mapping, cur, buf = {}, None, []
    t0 = t1 = None
    first_label = None
    last_end = 0.0

    def close():
        start = t0 if t0 is not None else last_end
        end = t1 if t1 is not None else start
        line_text = " ".join(buf).strip()
        lines.append(f"{cur}: {line_text}")
        turns.append(Turn(label=first_label, display=cur, start=float(start), end=float(end), text=line_text))

    for spk, text, st, en in timed_units(asr, diars, split):
        if spk is not None and spk not in mapping:
            mapping[spk] = names.get(spk) or f"Speaker {sum(1 for k in mapping if k not in names) + 1}"
        lab = mapping.get(spk, "Speaker ?")
        if lab != cur:
            if buf:
                close()
                last_end = turns[-1].end
            cur, buf = lab, [text]
            t0, t1, first_label = st, en, spk
        else:
            buf.append(text)
            if t0 is None:
                t0 = st
            if en is not None:
                t1 = en
    if buf:
        close()
    return lines, turns


class NoSentences(ValueError):
    """The ASR JSON holds no sentence with text: there is nothing to label, and the plain text stands as it is."""


def merge(asr: dict, rttm_text: str | None, names: Mapping[str, str] | None, split: bool = True) -> Merged:
    """The transcript lines for one recording, as the CLI writes them, and one Turn per line.

    ``asr`` is the recogniser's JSON (parsed); ``rttm_text`` the diarizer's RTTM text, or None / "" for none;
    ``names`` maps a diarizer label to the display it gets (a name). A label without one (or with an empty or None
    display, as the command line drops them) is "Speaker N", numbered among the labels without a name in order of
    first appearance; a word no turn covers is "Speaker ?".

    Without diarization (no RTTM, or one with no turns) there is one line, the plain text, and one Turn per
    sentence with label None and display "", so a renderer can still break the text where people paused.
    Raises ``NoSentences`` when ``asr`` has no sentence with text: the CLI then leaves the plain .txt as it is and
    succeeds, so that is not a failed merge. A broken RTTM line raises ValueError.
    """
    a = asr_segments(asr)
    if not a:
        raise NoSentences("the ASR JSON has no sentence with text")
    diars = diar_segments(_rttm_doc(rttm_text)) if rttm_text else None
    if diars:
        lines, turns = _labelled(a, diars, {k: v for k, v in (names or {}).items() if v}, split)
    else:
        lines = [" ".join(s["text"] for s in a).strip()]
        turns = []
        last = 0.0
        for s in a:
            st = s["start"] if s["start"] is not None else last
            en = s["end"] if s["end"] is not None else st
            turns.append(Turn(label=None, display="", start=float(st), end=float(en), text=s["text"]))
            last = en
    return Merged(lines=lines, text="\n".join(lines) + "\n", turns=turns)


def count_speakers(rttm_text: str | None) -> int:
    """The number of distinct diarizer labels in RTTM text (0 for none)."""
    segs = diar_segments(_rttm_doc(rttm_text)) if rttm_text else None
    return len({s["spk"] for s in segs}) if segs else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = "--sentences" not in argv
    argv = [a for a in argv if a != "--sentences"]
    if len(argv) == 2 and argv[0] == "--count-speakers":
        segs = diar_segments(load(argv[1]))
        print(len({s["spk"] for s in segs}) if segs else 0)
        return 0
    names = {}
    if len(argv) == 5 and argv[3] == "--names":
        # the speaker-ID sidecar: a label it named by voiceprint prints as that name; the rest stay "Speaker N"
        try:
            with open(argv[4], encoding="utf-8") as f:
                names = {k: v["name"] for k, v in json.load(f).get("labels", {}).items() if v.get("name")}
        except Exception as e:
            print(f"merge_speakers: names sidecar unreadable, labels stay numbered: {e}", file=sys.stderr)
        del argv[3:]
    if len(argv) != 3:
        print("usage: merge_speakers <asr.json> <diar.json|''> <out.txt> [--names speakers.json] [--sentences]",
              file=sys.stderr)
        return 1
    asr_p, diar_p, out_p = argv

    asr = asr_segments(load(asr_p))
    if not asr:
        # Leave the plain .txt in place if it exists.
        return 0 if os.path.exists(out_p) else 1

    diars = diar_segments(load(diar_p))
    if diars:
        lines, _ = _labelled(asr, diars, names, split)
    else:
        lines = [" ".join(s["text"] for s in asr).strip()]

    with open(out_p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
