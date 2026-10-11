# Golden tracks

One folder per track: `tests/golden/<name>/audio.<ext>` (not committed) and `truth.json` (committed).

truth.json:

    {
      "bpm": 124.0,
      "downbeats": [0.484, 2.42, 4.355],
      "drops": [30.97],
      "section_starts": [15.48, 30.97, 61.0]
    }

Values come from human annotations (the Harmonix Set, MIT licence) or a DAW tempo map, never from /listen itself.
`source` records where the annotation came from and any alignment shift applied to match the downloaded audio.
`downbeats` needs at least 8 entries spread across the track.
