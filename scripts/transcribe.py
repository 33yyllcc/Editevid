import json, sys
from faster_whisper import WhisperModel
src, out = sys.argv[1], sys.argv[2]
m = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8', cpu_threads=4)
segs, info = m.transcribe(src, word_timestamps=True, vad_filter=True, beam_size=5)
res = {'language': info.language, 'segments': []}
for s in segs:
    res['segments'].append({'start': s.start, 'end': s.end, 'text': s.text,
        'words': [{'s': w.start, 'e': w.end, 'w': w.word} for w in (s.words or [])]})
    print(f"[{s.start:7.1f}] {s.text}", flush=True)
json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
print('DONE')
