# Coding Manual: Propagandistic / High-Affect Motifs

The authoritative source is `codebook/motifs.yaml`. The tables at the bottom of this file are **generated** from it (`aibi codebook-doc`), and a test fails if the two drift apart.

## Unit of analysis

The unit is one image, meaning one `ImageRecord`. A multi-image post produces several records.

Code the **image** only, never the caption, with two exceptions:

1. Embedded text (VM15) is part of the image.
2. Text-modality persuasion techniques (PT05, PT10, PT13, PT17, PT19, PT20) are coded only from embedded text.

Coders may read the post caption for context **after** coding the image. A caption-driven revision must be recorded in `note`.

## Procedure

1. Open the image at full resolution, using `local_path` or `post_url` in the coding sheet (`aibi sheet --out sheet.csv`).
2. **Layer 1 – visual motifs.** For each VM code, enter `1` if the inclusion criterion is met and no exclusion applies, otherwise `0`. Several motifs can be present at once. Prefer `0` when in doubt, and add a note.
3. **Layer 2 – persuasion techniques.** Enter `1` where the image as a whole performs the technique.
4. **Layer 3 – affect.** Rate the response the image is *designed to elicit in a general adult viewer* on the SAM 9-point scales:
   - Valence: 1 = very unpleasant … 9 = very pleasant
   - Arousal: 1 = calm … 9 = agitated
   - Dominance (optional)
5. **Layer 4 – target emotion.** Choose one primary emotion from the list and optionally a secondary one.
6. **Identification of real people.** Code "real, identifiable" (VM02, VM03, VM19, VM20) only when the person is a public figure recognisable without searching, or is named in embedded text. Do not attempt to identify private individuals.
7. **Safety stop.** If an image appears to sexualize a minor, stop. Do not code, save, screenshot or share it, and follow `docs/ETHICS_AND_LEGAL.md` §4 immediately.

## Training and reliability

- **Training.** Coders train on a 50-image set until α ≥ 0.80 on Layer 1 is reached.
- **Production double-coding.** Independent coding (no discussion) of ≥ 10% of the corpus, minimum 200 images.
- **Statistic.** `aibi reliability coderA.csv coderB.csv` reports Krippendorff's α per variable, with the verdict thresholds from METHODOLOGY §4.2.
- **Adjudication.** Disagreements are resolved by a third coder or by discussion. The result is entered as `coder_type: consensus`.

## Changing the codebook

Any change to a definition, inclusion/exclusion rule, or the set of codes requires:

1. A `codebook_version` bump in `codebook/motifs.yaml`
2. `aibi codebook-doc` to regenerate the tables
3. An entry in `docs/AMENDMENTS.md`, saying whether images coded under the previous version must be re-coded

<!-- BEGIN GENERATED -->

## Code tables (generated from `codebook/motifs.yaml`, v0.1.0)

### Layer 1 - Visual motifs

| ID | Name | Definition | Include | Exclude |
|---|---|---|---|---|
| VM01 | `national_symbols` | National flags, emblems, coats of arms, national personifications, or national colour schemes used as a dominant compositional element. | Flag held, draped, or forming the background; emblem centred; figure clothed in flag. | Incidental small flags in a crowded street scene; sports kits without nationalist framing. |
| VM02 | `leader_glorification` | A real, identifiable political, military, or religious leader shown idealized - heroic, monumental, superhuman, saintly, or triumphant. | Low-angle heroic portrait, halo or divine light, muscular or warrior rendering, statue-like monumentality. | Neutral press-style portraits; obviously satirical depictions (code VM20 instead). |
| VM03 | `leader_degradation` | A real, identifiable political figure shown humiliated, arrested, imprisoned, injured, diseased, criminalized, or in degrading circumstances. | Arrest/perp-walk scenes, prison garb, crying/begging, physical decline, being attacked. | Abstract caricature without degrading situation (code VM20). |
| VM04 | `military_force` | Soldiers, weapons, armoured vehicles, warships, aircraft, or military formations as a central element. | Troops in formation, tanks, missiles, parades, combat scenes. | Toy or fantasy weapons in clearly non-political fantasy art (code only if real-world militaries are referenced). |
| VM05 | `outgroup_dehumanization` | An identifiable ethnic, national, religious, political, or social group depicted as animals, vermin, insects, disease, monsters, or otherwise subhuman. | Group-coded figures (clothing, symbols, captions) rendered as rats, cockroaches, apes, parasites, zombies, demons. | Generic fantasy monsters not linked to any real group. |
| VM06 | `vulnerable_victims` | Children, elderly people, pregnant women, animals, or injured civilians shown in distress, danger, or suffering. | Crying or injured child, child alone in disaster, elderly person in rubble, rescued pet in flood. | Children in neutral or happy scenes. |
| VM07 | `destruction_catastrophe` | Ruins, explosions, fire, floods, collapsing buildings, or apocalyptic landscapes as the main subject or backdrop. | Bombed city, burning landmark, smoke over a recognisable building, disaster aftermath. | Small controlled fires (campfire, candles) or abstract fantasy landscapes without catastrophe. |
| VM08 | `religious_iconography` | Religious figures, symbols, sacred sites, or sacral visual conventions (halos, divine rays, angels) used to sanctify a person, group, or cause. | Cross, crescent, star of David, icons, angels, saints, divine light on a political subject. | Neutral documentary images of places of worship. |
| VM09 | `crowd_mobilization` | Large unified crowds, rallies, marches, or masses acting in concert. | Sea of people with flags or signs, marching columns, mass rally. | Incidental crowds (shopping street, concert audience) without political framing. |
| VM10 | `martyrdom_sacrifice` | Death, dying, funerals, or sacrifice framed as noble, heroic, or redemptive for a cause. | Fallen hero in pietà pose, flag-draped coffins, heroic death in battle. | Gore without sacrificial framing (code VM18). |
| VM11 | `threat_invasion` | An encroaching, invading, or infiltrating threat to an in-group - hordes at borders, enemy forces approaching, contamination imagery. | Masses crossing a border framed as threatening, enemy shadow over a nation, boats overwhelming a coast. | Neutral documentary migration imagery without threat framing. |
| VM12 | `golden_age_nostalgia` | Idealized past or pastoral in-group life - traditional family, rural idyll, lost homeland, past greatness. | Retro-styled perfect family, wheat fields with traditional dress, glorious historical reconstruction. | Generic landscapes with no in-group/identity framing. |
| VM13 | `conspiratorial_symbolism` | Imagery suggesting hidden control - puppet masters, hidden hands, all-seeing eyes, shadowy elites, octopus/tentacle-over-globe tropes. | Puppet strings from a figure, tentacles over a globe, cabal around a table. | Neutral use of eye or pyramid as decoration without control framing. |
| VM14 | `hate_symbols` | A symbol listed in the ADL Hate Symbols Database (https://www.adl.org/resources/hate-symbols) or an equivalent authoritative reference. | Swastikas (non-religious use), SS runes, Celtic cross in white-supremacist styling, numeric codes as visual elements. | Historical documentary context clearly critical of the symbol; Hindu/Buddhist religious swastika in religious context. |
| VM15 | `text_slogan_overlay` | Embedded text forming part of the image - slogans, captions, meme text, poster headlines. | Any legible text added as a message (not incidental signage). | Watermarks, artist signatures, incidental shop signs. |
| VM16 | `documentary_realism` | Photorealistic rendering in a news/photojournalistic or eyewitness style such that the image could be read as a photograph of a real event. | Press-photo framing, smartphone-snapshot look, news-agency style, plausible real location. | Obviously painterly, cartoon, 3D-render, or fantasy styles. |
| VM17 | `heroic_ingroup` | Idealized ordinary in-group members - workers, mothers, soldiers, farmers - rendered as strong, pure, or exemplary (socialist-realist / poster style). | Muscular worker with tools, heroic mother with children, noble farmer at dawn. | Neutral everyday depictions. |
| VM18 | `gore_bodily_harm` | Visible blood, wounds, mutilation, or corpses. | Any visible blood or injuries beyond minor scrapes; dead bodies. | Stylized non-realistic fantasy gore unconnected to real events is still coded 1 but noted. |
| VM19 | `sexualized_real_person` | Sexualized or nude depiction of a real, identifiable adult person (non-consensual intimate imagery risk). | Any sexualized depiction of a real public or private adult figure. | Fully fictional characters. |
| VM20 | `caricature` | Exaggerated, grotesque, or mocking depiction of an identifiable person or group. | Distorted facial features, mocking costumes, satirical exaggeration. | Flattering stylization (code VM02 if idealized). |

### Layer 2 - Persuasion techniques (SemEval-2021 Task 6)

| ID | Name | Modality | Definition |
|---|---|---|---|
| PT01 | `appeal_to_authority` | both | A claim is presented as true because an authority, expert, or institution supports it. |
| PT02 | `appeal_to_fear_prejudice` | both | Builds support by instilling anxiety or panic about an alternative, possibly based on prejudice. |
| PT03 | `bandwagon` | both | Persuades by suggesting everyone is joining, so the viewer should too. |
| PT04 | `black_and_white_fallacy` | both | Presents two alternatives as the only options when more exist; includes dictatorship (telling the audience what to do). |
| PT05 | `causal_oversimplification` | text | Assumes a single cause for an issue that has multiple causes. |
| PT06 | `doubt` | both | Questions the credibility of someone or something. |
| PT07 | `exaggeration_minimisation` | both | Represents something as larger/better/worse or smaller/less important than it is. |
| PT08 | `flag_waving` | both | Plays on strong national or group feeling to justify an action or idea. |
| PT09 | `glittering_generalities` | both | Uses virtue words/images (freedom, peace, family) that evoke approval without argument. |
| PT10 | `loaded_language` | text | Uses words with strong emotional implications. |
| PT11 | `straw_man` | both | Substitutes an opponent's position with a distorted, easier-to-refute version. |
| PT12 | `name_calling_labeling` | both | Labels the target as something the audience fears, hates, or finds undesirable (visually: depicting target as such). |
| PT13 | `obfuscation_vagueness` | text | Uses deliberately unclear language so the audience can form their own interpretation. |
| PT14 | `red_herring` | both | Introduces irrelevant material to divert attention. |
| PT15 | `reductio_ad_hitlerum` | both | Persuades the audience to disapprove of something by associating it with groups they hate (e.g. Nazis). |
| PT16 | `repetition` | both | Repeats the same message or visual element over and over. |
| PT17 | `slogans` | text | A brief, striking phrase, possibly with labeling and stereotyping. |
| PT18 | `smears` | both | Damages someone's reputation via negative depictions or claims. |
| PT19 | `thought_terminating_cliche` | text | Short generic phrases that discourage critical thought. |
| PT20 | `whataboutism` | text | Discredits a position by charging hypocrisy without refuting the argument. |
| PT21 | `transfer` | image | Projects positive or negative qualities of one person, entity, or value onto another, often via visual juxtaposition. |
| PT22 | `appeal_to_strong_emotions` | image | Uses images with strong emotional content (e.g. suffering, cuteness, horror) to persuade. |

### Layer 3 - Affect (SAM, 1-9)

- **valence**: 1 = very unpleasant; 5 = neutral; 9 = very pleasant
- **arousal**: 1 = very calm; 5 = moderate; 9 = very excited / agitated
- **dominance** (optional): 1 = viewer feels controlled / powerless; 9 = viewer feels in control

### Layer 4 - Discrete target emotion

`anger`, `fear`, `disgust`, `contempt`, `sadness`, `pride`, `joy`, `awe`, `none`
