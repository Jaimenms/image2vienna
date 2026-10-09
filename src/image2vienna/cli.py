"""``i2vienna`` command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import (
    DEFAULT_EDITION,
    DEFAULT_HF_REPO,
    EDITIONS,
    LEVELS,
    default_describer,
    default_model,
    home,
)
from .describe import PROMPTS
from .embeddings import get_embedder
from .index import (
    ViennaIndex,
    available_indexes,
    build_index,
    find_previous_index,
    index_path,
    scheme_table_path,
)
from .scheme import SchemeTable, fetch_scheme, parse_scheme

app = typer.Typer(
    help="Map a trade mark image to Vienna Classification codes.", no_args_is_help=True
)
console = Console()


@app.command()
def editions():
    """List the editions this package knows, with their year of entry into force."""
    for e, year in EDITIONS.items():
        console.print(f"{e:>3}  in force {year}-01-01")


@app.command()
def indexes():
    """List indexes built under the image2vienna home."""
    table = Table("edition", "lang", "model", "notes", "rows", "MB", "path")
    for ref in available_indexes():
        meta = ViennaIndex.read_meta(ref.path)
        table.add_row(
            ref.edition,
            ref.lang,
            meta.model,
            "yes" if meta.notes else "",
            str(meta.rows),
            f"{ref.path.stat().st_size / 1e6:.1f}",
            str(ref.path),
        )
    console.print(table)
    console.print(f"home: {home()}")


@app.command()
def build(
    edition: str = typer.Option(DEFAULT_EDITION, help=f"One of {', '.join(EDITIONS)}"),
    lang: str = typer.Option("EN", help="EN or FR"),
    model: str = typer.Option(None, help="Embedder spec, e.g. st:intfloat/multilingual-e5-base"),
    notes: bool = typer.Option(False, help="Append the 'Including ...' notes to the texts"),
    previous: str = typer.Option(
        "auto", help="'auto': newest older edition for the same lang+model; 'none'; or a path"
    ),
    out: Path = typer.Option(None, help="Output parquet; default is under the home"),
    batch_size: int = typer.Option(128),
    force_download: bool = typer.Option(False, help="Re-download the WIPO XML"),
):
    """Download the WIPO XML for an edition, write the scheme table and the index."""
    model = model or default_model()
    lang = lang.upper()
    scheme = _prepare_scheme(edition, lang, force_download)

    prev: tuple[Path, Path] | None = None
    found = None
    if previous == "auto":
        found = find_previous_index(edition, lang, model, notes=notes)
    elif previous != "none":
        found = Path(previous)
    if found is not None:
        prev = (found, scheme_table_path(ViennaIndex.read_meta(found).edition, lang))
        console.print(f"reusing vectors from {found.name}")

    embedder = get_embedder(model)
    with console.status("embedding...") as status:
        index, report = build_index(
            scheme,
            embedder,
            edition=edition,
            lang=lang,
            notes=notes,
            previous=prev,
            batch_size=batch_size,
            progress=lambda done, total: status.update(f"embedding {done}/{total}"),
        )
    target = out or index_path(edition, lang, model, notes=notes)
    index.write(target)
    console.print(report.summary())
    console.print(f"wrote {target}")


def _prepare_scheme(edition: str, lang: str, force: bool) -> SchemeTable:
    xml_path = fetch_scheme(edition, lang, force=force)
    scheme = SchemeTable.from_nodes(parse_scheme(xml_path))
    target = scheme_table_path(edition, lang)
    scheme.write(target)
    console.print(
        f"[bold]edition {edition} {lang}[/]: {len(scheme)} entries from {xml_path.name} "
        f"-> {target.name}"
    )
    return scheme


@app.command("scheme")
def scheme_cmd(
    edition: str = typer.Option(DEFAULT_EDITION),
    lang: str = typer.Option("EN"),
    force_download: bool = typer.Option(False),
):
    """Write only the scheme table (titles + hierarchy) for an edition and language."""
    _prepare_scheme(edition, lang.upper(), force_download)


@app.command()
def describe(
    image: Path = typer.Argument(..., help="Image file (JPG, PNG, ...)"),
    describer: str = typer.Option(None, help=f"Vision model spec (default {default_describer()})"),
    prompt: str = typer.Option("default", help=f"Prompt name: {', '.join(PROMPTS)}"),
):
    """Stage one alone: print the vision model's description of the figurative elements."""
    from .describe import get_describer

    d = get_describer(describer or default_describer())
    console.print(d.describe(image, prompt=PROMPTS[prompt]))


@app.command()
def classify(
    image: str = typer.Argument(..., help="Image file; or '-' to classify a text from stdin"),
    edition: str = typer.Option("latest", help="Edition among built indexes"),
    level: str = typer.Option("section", help=f"{', '.join(LEVELS)} or auto"),
    top_k: int = typer.Option(10),
    lang: str = typer.Option("EN"),
    model: str = typer.Option(None),
    describer: str = typer.Option(None, help=f"Vision model spec (default {default_describer()})"),
    notes: bool = typer.Option(False, help="Use the index built with notes"),
    prompt: str = typer.Option("default", help=f"Prompt name: {', '.join(PROMPTS)}"),
    gap: float = typer.Option(None, help="Drop results more than this below the best score"),
    principal_only: bool = typer.Option(False, help="Leave auxiliary (A) sections out"),
    exclude: list[str] = typer.Option([], help="Codes whose subtree is left out, e.g. 29"),
    chunking: str = typer.Option("whole", help="Description as one query: whole, mean or max"),
    padded: bool = typer.Option(False, help="Print codes as 01.01.02 (EUIPO style)"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Rank Vienna codes for an image (or for a description read from stdin)."""
    from .classifier import ViennaClassifier

    clf = ViennaClassifier(edition, lang=lang, model=model, describer=describer, notes=notes)
    params = {
        "level": level,
        "top_k": top_k,
        "gap": gap,
        "principal_only": principal_only,
        "exclude_codes": tuple(exclude),
        "chunking": chunking,
    }
    if image == "-":
        text = sys.stdin.read()
        matches = clf.classify_text(text, **params)
    else:
        matches = clf.classify(Path(image), prompt=PROMPTS[prompt], **params)
        text = clf.last_description or ""
    if as_json:
        print(json.dumps({"description": text, "matches": [m.__dict__ for m in matches]}, indent=2))
        return
    if image != "-":
        console.print(f"[dim]{text}[/]\n")
    table = Table(
        "#",
        "code",
        "score",
        "sim",
        "category > division > section",
        title=f"Vienna {clf.edition} {clf.lang}",
    )
    for i, m in enumerate(matches, 1):
        code = m.padded if padded else m.pretty
        table.add_row(
            str(i),
            f"{'A ' if m.auxiliary else ''}{code}",
            f"{m.score:.3f}",
            f"{m.similarity:.3f}",
            m.text,
        )
    console.print(table)


@app.command()
def show(
    code: str = typer.Argument(..., help="e.g. 1.1.2 or 01.01.02"),
    edition: str = typer.Option(DEFAULT_EDITION),
    lang: str = typer.Option("EN"),
):
    """Print an entry's path, the text it is embedded with, and its notes."""
    from .scheme import normalize_code

    scheme = SchemeTable.read(scheme_table_path(edition, lang.upper()))
    node = scheme.node(normalize_code(code))
    console.print(f"[bold]{'A ' if node.auxiliary else ''}{node.code}[/] ({node.level})")
    console.print(" | ".join(scheme.path_of(node.code)))
    console.print(scheme.text_of(node.code))
    for n in node.notes:
        console.print(f"  note: {n}")
    if node.associated:
        console.print(f"  associated with principal sections {', '.join(node.associated)}")


@app.command("l3d")
def l3d_cases(
    n: int = typer.Option(300, help="How many labelled images to keep"),
    seed: int = typer.Option(0),
    out: Path = typer.Option(None, help="JSONL path; default evals/l3d_<n>.jsonl"),
    force: bool = typer.Option(False, help="Re-download the labels"),
):
    """Eval cases from the Large Labelled Logo Dataset (EUIPO images, Vienna 8 codes)."""
    from .eval.l3d import build_l3d_cases

    with console.status("fetching L3D labels and images...") as status:
        cases = build_l3d_cases(
            n=n, seed=seed, force=force, progress=lambda i, t: status.update(f"images {i}/{t}")
        )
    target = out or Path("evals") / f"l3d_{n}.jsonl"
    from .eval import save_cases

    save_cases(cases, target)
    console.print(f"{len(cases)} cases -> {target}")


@app.command("euipo")
def euipo_cases(
    n: int = typer.Option(300, help="How many figurative marks to fetch"),
    query: str = typer.Option(
        "markFeature==FIGURATIVE", help="RSQL filter for the Trademark Search API"
    ),
    out: Path = typer.Option(None, help="JSONL path; default evals/euipo_<n>.jsonl"),
):
    """Eval cases from EUIPO's Trademark Search API (needs EUIPO_CLIENT_ID / _SECRET)."""
    from .eval import save_cases
    from .eval.euipo import build_euipo_cases

    with console.status("fetching from api.euipo.europa.eu...") as status:
        cases = build_euipo_cases(
            n=n, query=query, progress=lambda i, t: status.update(f"marks {i}/{t}")
        )
    target = out or Path("evals") / f"euipo_{n}.jsonl"
    save_cases(cases, target)
    console.print(f"{len(cases)} cases -> {target}")


@app.command("describe-cases")
def describe_cases(
    cases: Path = typer.Argument(..., help="JSONL of eval cases; descriptions are written back"),
    describer: str = typer.Option(None, help=f"Vision model spec (default {default_describer()})"),
    prompt: str = typer.Option("default", help=f"Prompt name: {', '.join(PROMPTS)}"),
    limit: int = typer.Option(None, help="Only the first N cases"),
    force: bool = typer.Option(False, help="Describe again even when cached"),
):
    """Run the vision model once over the cases and cache the descriptions in the file."""
    from .eval.describe import describe_cases as run

    spec = describer or default_describer()
    with console.status("describing...") as status:
        done, total = run(
            cases,
            spec,
            prompt=prompt,
            limit=limit,
            force=force,
            progress=lambda i, t, s: status.update(f"describing {i}/{t} ({s:.1f} s/img)"),
        )
    console.print(f"{done} described ({total} cases) with {spec}, prompt {prompt} -> {cases}")


@app.command("eval")
def eval_cmd(
    cases: Path = typer.Argument(..., help="JSONL with cached descriptions"),
    edition: str = typer.Option("latest"),
    level: str = typer.Option("section"),
    top_k: int = typer.Option(10),
    lang: str = typer.Option("EN"),
    model: str = typer.Option(None),
    describer: str = typer.Option(None, help="Which cached descriptions to score"),
    prompt: str = typer.Option("default"),
    notes: bool = typer.Option(False),
    principal_only: bool = typer.Option(False),
    exclude: list[str] = typer.Option(
        [], help="Codes whose subtree is left out of candidates and gold, e.g. 29"
    ),
    chunking: str = typer.Option("whole", help="whole, mean or max over sentences"),
    path_weight: float = typer.Option(0.0, help="Weight of path support"),
    subtree_weight: float = typer.Option(0.0, help="Weight of subtree support"),
    limit: int = typer.Option(None, help="Only the first N cases"),
    show_misses: int = typer.Option(0, help="Print this many misses at the target level"),
):
    """Hit-rate at each level against office-assigned codes, from cached descriptions."""
    from .classifier import ViennaClassifier
    from .eval import load_cases
    from .eval.describe import description_key
    from .eval.harness import evaluate, exclude_gold
    from .search import Weights

    spec = describer or default_describer()
    key = description_key(spec, prompt)
    items = [c for c in load_cases(cases) if key in c.descriptions][:limit]
    if exclude:
        items = exclude_gold(items, tuple(exclude))
    if not items:
        raise typer.BadParameter(f"no case in {cases} has a description under {key!r}")
    clf = ViennaClassifier(edition, lang=lang, model=model, notes=notes)
    weights = Weights(own=1.0, path=path_weight, subtree=subtree_weight)
    with console.status("evaluating...") as status:
        result = evaluate(
            items,
            lambda c: clf.classify_text(
                c.descriptions[key],
                level=level,
                top_k=top_k,
                weights=weights,
                principal_only=principal_only,
                exclude_codes=tuple(exclude),
                chunking=chunking,
            ),
            level=level,
            top_k=top_k,
            progress=lambda i, n: status.update(f"evaluating {i}/{n}"),
        )
    console.print(
        f"[bold]{cases.name}[/] n={result.n} edition={clf.edition} model={clf.index.meta.model} "
        f"notes={clf.notes} describer={spec} prompt={prompt} weights={weights} "
        f"chunking={chunking} exclude={list(exclude)}"
    )
    console.print(result.table())
    for case_id, gold, preds in result.misses[:show_misses]:
        console.print(f"  miss {case_id}: gold {gold} | got {' '.join(preds)}")


@app.command()
def download(
    repo: str = typer.Argument(DEFAULT_HF_REPO, help="Hugging Face model repo made by hf-export"),
    revision: str = typer.Option(None, help="Branch, tag or commit"),
):
    """Fetch a prebuilt index and its scheme table from the Hugging Face Hub."""
    from .index.publish import download_from_hf

    with console.status(f"downloading from huggingface.co/{repo}..."):
        path, cfg = download_from_hf(repo, revision=revision)
    console.print(f"ready: {path}")
    console.print(
        f"try: i2vienna classify logo.png --edition {cfg['edition']} --lang {cfg['lang']}"
    )


@app.command("hf-export")
def hf_export(
    out: Path = typer.Argument(..., help="Directory to create (becomes the HF model repo)"),
    edition: str = typer.Option("latest"),
    lang: str = typer.Option("EN"),
    model: str = typer.Option(None),
    repo_id: str = typer.Option(None, help="Hub repo id written into the model card"),
):
    """Assemble a Hugging Face model repository (handler.py + index + scheme + package)."""
    from .classifier import resolve_built_edition
    from .hf import export_hf_repo

    model = model or default_model()
    edition = resolve_built_edition(edition, lang.upper(), model, False, home())
    path = export_hf_repo(
        out,
        edition=edition,
        lang=lang,
        model=model,
        repo_id=repo_id or f"<user>/image2vienna-{lang.lower()}",
    )
    console.print(f"HF repo assembled at {path}")
    console.print(
        f"publish with: scripts/publish_hf.sh (or hf upload <user>/<repo> {path} --repo-type model)"
    )


@app.command("web-export")
def web_export(
    out: Path = typer.Argument(..., help="Directory to create (becomes a static HF Space)"),
    edition: str = typer.Option("latest", help="Edition among built indexes"),
    lang: str = typer.Option("EN"),
    model: str = typer.Option(None, help="Embedder the index was built with"),
    notes: bool = typer.Option(False, help="Ship the index built with notes"),
    web_model: str = typer.Option(None, help="transformers.js model id (default: Xenova twin)"),
    web_dtype: str = typer.Option("q8", help="ONNX weights the browser loads: q8, fp16, fp32"),
    vision_model: str = typer.Option(
        None, help="transformers.js vision model for uploads; '' for none"
    ),
    repo_id: str = typer.Option(None, help="Space id written into the README"),
    examples: Path = typer.Option(
        None, help="JSONL of demo examples (default evals/demo_examples.jsonl when present)"
    ),
):
    """Assemble a static Hugging Face Space that classifies images in the browser."""
    from .web import WEB_DEFAULT_MODEL, WEB_DEFAULT_VISION, export_web_demo

    if examples is None and Path("evals/demo_examples.jsonl").is_file():
        examples = Path("evals/demo_examples.jsonl")
    path = export_web_demo(
        out,
        model=model or WEB_DEFAULT_MODEL,
        edition=edition,
        lang=lang.upper(),
        notes=notes,
        repo_id=repo_id or "<user>/image2vienna",
        web_model=web_model,
        web_dtype=web_dtype,
        examples=examples,
        vision_model=WEB_DEFAULT_VISION if vision_model is None else (vision_model or None),
    )
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    console.print(f"static Space assembled at {path} ({total / 1e6:.1f} MB)")
    console.print(
        "publish with: scripts/publish_space.sh (or hf upload <user>/<space> ... --repo-type space)"
    )


if __name__ == "__main__":
    app()
