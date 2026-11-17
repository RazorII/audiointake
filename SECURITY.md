# Security

Please report a security problem privately, not in a public issue: open the repository's **Security** tab and choose **Report a vulnerability**. Only the maintainer sees the report, and you'll get a reply there.

audiointake runs on your own Mac and sends nothing anywhere (see [PRIVACY.md](PRIVACY.md)). It does read recordings from anywhere, it downloads models and builds a program at setup, and it keeps voiceprints, so these are the things worth reporting:

- **A crafted recording** that makes audiointake, ffmpeg or a model library run code, start a program audiointake didn't mean to start, read or write outside the input and the outputs it names, change the input, or hang past its time limits.
- **A network connection from `transcribe`**, or from any command other than `setup` and `apply --lang` with a pack missing; any Hugging Face token or other credential sent anywhere; a download followed to a host other than those PRIVACY.md lists.
- **A way past the pinned fingerprints**: a model file, the speaker program's source or a prebuilt library used without its pinned SHA-256 or revision matching, a manifest path that escapes its folder (`..` or an absolute path), or a file that runs as code from a model folder.
- **Transcript text where it shouldn't be**: in a record, a log, an error message, `--json` output or a process's command line.
- **Voiceprints**: a voiceprint written without an `enroll` the user ran, readable by other users, stored anywhere but audiointake's own folder (or the folder `--voices` names), or a voiceprint's name reaching a transcript, a record or `--json` output for a recording in which that voice wasn't heard.
- **A run that stops to ask**: anything in `audiointake` that prompts, reads standard input for an answer, or waits on a person, since a run is meant to work through a batch in the background.
- **An overwritten file**: audiointake replacing a file it didn't write, or one changed since it wrote it, without `--overwrite`.
- **A file name or record field that ends up inside a command** the Claude skill builds. Acting on an offer is meant to be built from record ids, label ids and the user's own words only.

Fixes go out as a new release. A published release is never edited: GitHub's immutable releases lock its tag, so a fix is always the next version, and the plugin's skill installs that version.
