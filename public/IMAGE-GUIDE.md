# PashuMitra — Image placement guide

Place the following images in this `/public` directory to replace the
gradient placeholders in the UI.

## Hero image
- File path: `/public/hero/hero.jpg`
- Recommended size: 1920 x 1080 (16:9), JPG/WebP, optimised < 400 KB
- Content: A high-quality photograph of Indian cattle/buffalo in a field
  (e.g. a Gir cow, Murrah buffalo, or a herd at sunrise)
- Used on: Home page hero section

## Breed images (one per breed)
- File path: `/public/breeds/{breed-id}.jpg`
  - Example: `/public/breeds/gir.jpg`, `/public/breeds/sahiwal.jpg`,
    `/public/breeds/murrah.jpg`, etc.
- Recommended size: 800 x 600 (4:3), JPG/WebP, optimised < 150 KB each
- Content: Clear side-profile photo showing distinguishing features
  (horns, hump, coat pattern)

## Breed ID list (filename reference)
### Indigenous cattle (26)
gir, sahiwal, red-sindhi, tharparkar, kankrej, ongole, hallikar,
amritmahal, krishna-valley, deoni, hariana, mewati, nagori, malvi,
kenkatha, kherigarh, punganur, pulikulam, kangayam, bargur, alambadi,
umblachery, dangi, gaolao, khillari

### Buffalo (4)
murrah, mehsana, jaffarabadi, surti

### Exotic (6)
holstein-friesian, jersey, brown-swiss, guernsey, red-dane, ayrshire

## Optional section images
- `/public/hero/dashboard.jpg` — Dashboard hero (optional)
- `/public/hero/classifier.jpg` — Classifier hero (optional)

## Logo
- `/public/logo.svg` — replace with PashuMitra logo (currently the default Z.ai logo)

Once images are dropped into these paths, they will automatically replace
the gradient placeholders in the UI. No code changes needed.
