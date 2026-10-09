"""Prompts for the vision-language model.

The wording follows the Vienna general notes: elements are classified by their shape
regardless of material or purpose (note (a)), parts belong with the whole unless
expressly listed (note (b)), every distinct element gets its own code (note (e)).
The model is therefore asked for an inventory of what is visible, in the vocabulary
the classification uses, and not for an interpretation of the brand.
"""

DEFAULT_PROMPT = (
    "You are describing a trade mark image so that its figurative elements can be "
    "classified. List every visual element you see, in the order of their prominence: "
    "human beings (men, women, children; their clothing, activity or profession), "
    "animals (species, pose, whether only the head is shown), plants, celestial bodies, "
    "landscapes, buildings, objects, vehicles, tools, containers, food, clothing, "
    "heraldic elements (shields, crowns, flags, emblems), geometric figures (circles, "
    "triangles, squares, lines, bands, arrows), ornamental motifs, letters or numerals "
    "and how they are written, and colours. Describe each element by its shape and "
    "appearance, in neutral and concrete terms. Do not name the brand, do not guess "
    "what the mark stands for, and do not transcribe long texts. Answer in English in "
    "3 to 6 sentences."
)

#: Shorter variant; measured against the default in docs/evals.md.
TERSE_PROMPT = (
    "List the figurative elements of this trade mark image: every object, living "
    "being, plant, celestial body, geometric shape, heraldic element, letter, numeral "
    "and colour that is visible, each named by its shape and appearance. Neutral, "
    "concrete English, no brand names, no interpretation. One sentence per element."
)

#: Second iteration (docs/evals.md, 2026-10-09): the default prompt's enumeration made
#: the model list the kinds of elements that are *absent* ("there are no animals,
#: plants, celestial bodies ..."), and those sentences attract exactly the wrong
#: entries. This one names only what is present, one sentence per element, and asks
#: how letters are written, since most marks carry a word in a special form.
INVENTORY_PROMPT = (
    "Describe this trade mark image as an inventory of its figurative elements for "
    "classification. Write one sentence per element that is actually visible: a living "
    "being and what it wears or does, an animal, a plant, a celestial body, a landscape, "
    "a building, an object, a vehicle, a tool, a container, food, a heraldic element, a "
    "geometric figure, an ornamental motif. For letters, numerals or words, say how they "
    "are written (typeface style, upper or lower case, special or fanciful form, "
    "arrangement, dark on light or light on dark) and quote only the word itself. End "
    "with one sentence naming the colours. Name only what is present; never mention kinds "
    "of elements that are absent. Neutral, concrete English; no brand names, no "
    "interpretation."
)

#: For the light model (SmolVLM-256M). The long prompts make it answer with one word,
#: loop, or echo the prompt ("trade" from "trade mark image"); a plain request for a
#: detailed description gets concrete sentences out of it (docs/evals.md).
LIGHT_PROMPT = (
    "Describe this image in detail. Name every object, living being, plant, celestial "
    "body, geometric shape, letter, numeral and colour that is visible, each by its shape "
    "and appearance. Do not name brands and do not interpret."
)

PROMPTS = {
    "default": DEFAULT_PROMPT,
    "terse": TERSE_PROMPT,
    "inventory": INVENTORY_PROMPT,
    "light": LIGHT_PROMPT,
}
