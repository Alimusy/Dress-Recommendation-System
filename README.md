# Weather Aware Dress Recommendation System

Recommends what to wear by combining today's weather with the visual content
of a clothing catalogue, rather than by category labels alone.

## How it works

1. **Weather** — pulls current conditions for a city from the OpenWeatherMap
   API, then engineers features from the raw response: temperature band,
   whether it is raining, humidity, and the season derived from the date.
2. **Visual features** — each catalogue image goes through VGG16 pretrained on
   ImageNet with the classification head removed, producing an embedding that
   captures how a garment actually looks rather than how it was tagged.
3. **Catalogue** — the clothing inventory is read live from a Google Sheet, so
   items can be added without touching the code.
4. **Recommendation** — items are scored by similarity in the embedding space,
   filtered by what the weather features make appropriate.

Using image embeddings instead of tags is the interesting part. Two items
labelled "jacket" can be completely different weights of garment, and the
embedding sees that difference where the label does not.

## Setup

```bash
pip install pandas numpy requests tensorflow scikit-learn
export OPENWEATHER_API_KEY=your_key_here
```

Get a free key at [openweathermap.org](https://openweathermap.org/api). The
notebook reads it from the environment, never from the source.

## Status

Ongoing. The recommendation scoring works end to end; what it still needs is a
real user preference signal, since right now the ranking is driven entirely by
weather fit and visual similarity with nothing personal in it.
