# Recap of the offset-tracking tests: pair 003193 / 003368

**Goal.** The software measures how far the ice moved between two radar images taken 12 days apart.
For each small window of the image it searches for the best match in the second image. The search is
faster and safer if it starts near the right answer. We tested whether a velocity map (the MEaSUREs
model) gives a better starting point than a single constant shift.

## What we compared

1. A constant starting shift (-21 pixels in range, -4 in azimuth) and a wide search (+-32 / +-8 pixels).
2. The same constant shift with a narrow search (+-8 / +-4 pixels).
3. A starting shift taken from the velocity model, with the wide search.

## Data used

About 70% of the scene is ocean or has no velocity data, and the match there is random noise. All
comparisons use only the ice windows (about 754,000, or 30% of the scene) whose match quality (SNR) is
above 20. That leaves about 320,000 windows.

## Findings

1. **The two images are offset by a nearly constant amount, about -4.7 pixels in azimuth and -20.9 in
   range.** This "bulk shift" is not ice motion. It comes from how the two images were acquired and
   aligned, and the velocity model does not contain it. Around it, the ice windows are tight: half of
   them are within 0.1 pixel (azimuth) and 1 pixel (range), and 99 out of 100 within about 0.4 and
   3 to 4 pixels.
2. **The three set-ups give the same answer.** Where all three found a good match, the offsets differ
   by at most 0.02 pixel for 99% of the windows. The narrow search reproduced the wide search on every
   window it kept.
3. **The narrow search is not worth it.** It saves a few seconds out of a run of about 2 minutes, and
   it flags 7% fewer windows as good quality. The likely reason is that the quality score depends on
   the size of the search area, but this is not verified. A wrong start in a narrow window can also
   miss the true match on fast ice. We keep the wide search.
4. **The velocity model did not help on this pair.**
   - Over the ice, the measured offsets are almost unrelated to the model's predicted shifts (a
     regression slope of 0.05 to 0.12, where 1 would mean a perfect match).
   - Where the model asked for large range shifts (+7 pixels typical, 1,174 windows), the measured
     motion was about +1.5 pixels. The sign agreed in 99.9% of them, so the model is not reversed,
     but it asks for much more than the data show.
   - The cause is not established. It could be the model (a long-term mean, not this pair), the
     conversion from velocity to pixels, or the edge of the model's valid area.

## What these tests do not show

- The pair has little fast ice, so we cannot say whether the model helps on fast glaciers. We do not
  conclude that it is useless in general.
- The bulk shift was estimated as the median over ice windows, so it includes some average ice motion.
  A value from stationary ground (rock) would be more reliable.
- The 80% of low-quality windows say nothing about the true offsets there.

## Decision and next steps

- Keep the wide search window, and keep the model as an optional starting shift, marked as not yet
  validated.
- Test the model on a scene with fast glaciers.
- For this pair, map where the large model shifts fall, and check the velocity-to-pixel conversion
  for one window by hand.

Script used for the comparison: `pycuampcor/scripts/compare_offset_runs.py`. Details of the gross-offset
workflow: `docs/grossOffsets.md`, section 5.1c.
