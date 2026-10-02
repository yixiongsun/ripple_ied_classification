# Portfolio presentation system

Status: shared visual system revised to match the live portfolio page.

## Presentation context

The primary target is the existing dark-gray, scroll-based portfolio case study at `yixiongsun.github.io`. Scientific figures sit in the site's centered content column and should remain understandable when read in sequence or viewed alone.

- **Desktop (>= 621 px):** the live site provides a 760 px content column. Figures may use two panels or three short summary columns, but not a five-column scientific flow.
- **Mobile (<= 620 px):** the live site provides 339 px at a 390 px viewport. Use one column, convert left-to-right flows into top-to-bottom flows, and move secondary detail into the caption.
- **Fallback width:** every figure must remain readable at 339 px without hover, horizontal scrolling, cropped labels, or text crossing a container boundary.

The recommended sequence remains: pipeline, representative events, model architecture, cross-validation, robustness/ablation, then a one-sentence limitation. Place short explanatory copy between figures so each visual answers one question and hands off to the next.

## Palette

| Role | Hex | Use |
| --- | --- | --- |
| Ripple | `#6FB1E3` | Ripple waveforms, points, and class labels |
| IED | `#E8A15A` | IED waveforms, points, and class labels |
| Noise / uncertain | `#A89CC2` | Rejected or ambiguous events; always pair with a label or distinct marker |
| Selected compact model | `#E8E8E3` | White solid line or label; never a new accent hue |
| Full baseline | `#92938E` | Reference model and inactive comparison |
| Limitation | `#B6B7B1` | Neutral body text with an explicit limitation label |
| Strong text | `#E8E8E3` | Titles and essential labels |
| Body text | `#B6B7B1` | Captions and supporting annotations |
| Muted text | `#92938E` | Tertiary labels |
| Rule | `#373936` | Axes, separators, and quiet structure |
| Page | `#171817` | Exact website background |
| Figure surface | `#111210` | Matches the site's media-preview surface |

Blue, orange, and muted purple are reserved for ripple, IED, and noise. All structure, model identity, headings, warnings, and connectors use white or gray. Color is never the only class identifier: ripple, IED, and noise also use direct labels and, where marks repeat, circle, diamond, and square shapes respectively. The compact model uses a white solid line; the full baseline uses a lighter gray dashed line.

## Typography

Match the website's existing font stack in the exported SVG:

```css
font-family: Arial, Helvetica, sans-serif;
```

Do not add a web-font download solely for the figures.

| Element | Desktop | Mobile | Weight |
| --- | ---: | ---: | ---: |
| Figure title | 26–30 px | 22–24 px | 400–600 |
| Declarative takeaway | 17–19 px | 16–18 px | 400–600 |
| Panel heading | 16–18 px | 15–17 px | 600 |
| Axis / node label | 13–15 px | 13–15 px | 400–600 |
| Supporting annotation | 12–13 px | 12–13 px | 400 |
| HTML caption | 15–17 px | 15–16 px | 400 |

Use sentence case. Keep on-figure text short and place methodological qualifications in adjacent HTML captions. Use tabular numerals for aligned metrics. Do not put essential text below 12 px at its rendered size.

## Figure grammar

- Start with one declarative title and, only when needed, a one-line takeaway.
- Use direct labels instead of legends when there are three or fewer series.
- Keep axes and connectors neutral; semantic color belongs to data and decisions.
- Use the page background as the figure background. Use the darker media surface sparingly for grouping, with thin rules, restrained corner radii, and no gradients, 3-D effects, or shadows.
- Use white and line weight for the selected approach; do not introduce a separate model color.
- Render limitations in normal gray prose with an explicit label; avoid alarm styling.
- Pair uncertainty or calibration ranges with explicit numbers and visible distributions rather than decorative error bars alone.

## Responsive figure behavior

Each scientific figure should have one SVG source with responsive groups when practical, or separate desktop and mobile SVG layouts when reflow changes the reading order.

| Figure | Desktop | Mobile |
| --- | --- | --- |
| Pipeline | Five stages stacked vertically with straight gray connectors | The same vertical order with mobile-specific internal wrapping |
| Representative events | Three class columns × matched rows | One class at a time in three vertical sections; axes stay identical |
| Architecture | Two input streams joining late | Stacked streams followed by merge and decision logic |
| Cross-validation | Two panels side by side | Paired runs first, class F1 second |
| Robustness / ablation | Two panels side by side | Efficiency first, threshold sensitivity second |

No essential information may depend on hover. The SVG title and description provide the short accessible summary; the HTML page should supply the longer caption and alt text.

## Overall tone

The tone is precise, calm, and candid: clinically informed without imitating a hospital dashboard. Figures should emphasize the signal path, subject-aware validation, calibrated abstention, and uncertainty. Avoid trophy-style metrics, exaggerated claims, and decorative neuroscience imagery.

## Approval checkpoint

Before finalizing scientific figures, approve or revise:

1. the exact `#171817` page background and `#111210` figure-surface treatment;
2. the ripple / IED / noise color assignments;
3. the system-sans typography and minimum text size;
4. the white compact-model versus gray full-baseline treatment;
5. the 760 px desktop and 339 px mobile reflow rules shown in the style tile.

## Layout QA rules

- Treat any text crossing a box edge, connector, or neighboring label as a release-blocking defect.
- Keep at least 12 px of visual padding between text and its container at the intended rendered width.
- Measure the figure at 760 px and 339 px, not only at the high-resolution export size.
- Prefer wrapping or increasing container height over reducing text below the minimum size.
- Keep arrows in dedicated gutters so they never pass through text or boxes.

