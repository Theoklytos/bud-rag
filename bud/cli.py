"""Command-line interface for Bud RAG Pipeline."""

import json
import os
from pathlib import Path

import click

from bud import __version__
from bud.config import (
    CONFIG_FILE,
    get_config_dir,
    get_data_dir,
    get_output_dir,
    load_config,
    save_config,
    validate_config,
)


@click.group()
@click.version_option(version=__version__, prog_name="bud")
def main():
    """Bud RAG Pipeline - A personal conversation RAG system."""
    pass


@main.command()
def configure():
    """Interactive configuration shell."""
    from rich.console import Console
    from rich.prompt import Prompt, Confirm

    console = Console()

    console.print("\n[bold cyan]Bud RAG Pipeline Configuration[/bold cyan]\n")
    console.print("This will guide you through setting up your configuration.\n")

    config = load_config()

    # Data directory
    current_data = config.get("data_dir", "")
    default_data_dir = str(Path.home() / "data" / "conversations")
    data_dir = Prompt.ask(
        "[green]Data directory[/green] (where your conversation data is stored)",
        default=current_data or default_data_dir,
    )
    config["data_dir"] = data_dir

    # Output directory
    current_output = config.get("output_dir", "")
    default_output_dir = str(Path.home() / "data" / "bud_output")
    output_dir = Prompt.ask(
        "[green]Output directory[/green] (where pipeline outputs will be stored)",
        default=current_output or default_output_dir,
    )
    config["output_dir"] = output_dir

    # LLM configuration
    console.print("\n[bold]LLM Configuration[/bold]\n")

    current_llm_provider = config.get("llm", {}).get("provider", "ollama")
    llm_provider = Prompt.ask(
        "[blue]LLM Provider[/blue]",
        choices=["ollama", "openai", "anthropic"],
        default=current_llm_provider,
    )
    config.setdefault("llm", {})["provider"] = llm_provider

    current_llm_url = config.get("llm", {}).get("base_url", "http://localhost:11434")
    llm_base_url = Prompt.ask(
        "[blue]LLM Base URL[/blue]", default=current_llm_url
    )
    config.setdefault("llm", {})["base_url"] = llm_base_url

    current_llm_model = config.get("llm", {}).get("model", "gemini-3-flash-preview:latest")
    llm_model = Prompt.ask(
        "[blue]LLM Model[/blue]", default=current_llm_model
    )
    config.setdefault("llm", {})["model"] = llm_model

    # Embeddings configuration
    console.print("\n[bold]Embeddings Configuration[/bold]\n")

    current_emb_provider = config.get("embeddings", {}).get("provider", "ollama")
    emb_provider = Prompt.ask(
        "[blue]Embeddings Provider[/blue]",
        choices=["ollama", "openai"],
        default=current_emb_provider,
    )
    config.setdefault("embeddings", {})["provider"] = emb_provider

    current_emb_url = config.get("embeddings", {}).get("base_url", "http://localhost:11434")
    emb_base_url = Prompt.ask(
        "[blue]Embeddings Base URL[/blue]", default=current_emb_url
    )
    config.setdefault("embeddings", {})["base_url"] = emb_base_url

    current_emb_model = config.get("embeddings", {}).get("model", "mxbai-embed-large:335m")
    emb_model = Prompt.ask(
        "[blue]Embeddings Model[/blue]", default=current_emb_model
    )
    config.setdefault("embeddings", {})["model"] = emb_model

    # Save and validate
    console.print("\n[bold]Saving configuration...[/bold]")
    save_config(config)

    is_valid, errors = validate_config(config)

    if is_valid:
        console.print(
            f"\n[green]Configuration saved successfully![/green]"
        )
        console.print(f"[dim]Location: {CONFIG_FILE}[/dim]")
    else:
        console.print("\n[red]Configuration has errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        console.print("\n[dim]Configuration still saved. Review and fix errors.[/dim]")


@main.command()
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, resolve_path=True),
    help="Path to output directory (overrides config)",
)
@click.option(
    "--samples", "-s",
    type=int,
    default=5,
    help="Conversations to sample per iteration (default: 5)",
)
@click.option(
    "--iterations", "-n",
    type=int,
    default=10,
    help="Maximum discovery iterations (default: 10)",
)
@click.option(
    "--stability", "-t",
    type=float,
    default=0.75,
    help="Stability threshold to stop early (default: 0.75)",
)
@click.option(
    "--resume/--no-resume", "-r",
    default=False,
    help="Resume from existing concept map",
)
@click.option(
    "--blend/--no-blend",
    default=False,
    help=(
        "Use cross-boundary blending instead of whole-conversation sampling. "
        "Slices randomly across all turns in the archive, crossing conversation "
        "boundaries, to expose structural patterns invisible to per-conversation sampling."
    ),
)
@click.option(
    "--blend-slices", "-S",
    type=int,
    default=6,
    help="Number of cross-boundary slices per blended sample (default: 6)",
)
@click.option(
    "--blend-width", "-W",
    type=int,
    default=8,
    help="Turns per blend slice (default: 8)",
)
@click.option(
    "--progressive/--no-progressive",
    default=False,
    help=(
        "Progressive cursor-based blending: takes one slice per file per iteration, "
        "advancing a saved cursor so the entire archive is covered exhaustively. "
        "Cursor persists across runs; use --reset-cursor to start over."
    ),
)
@click.option(
    "--reset-cursor",
    is_flag=True,
    default=False,
    help="Reset the progressive blend cursor to the beginning of all files.",
)
def discover(output_dir, samples, iterations, stability, resume, blend, blend_slices, blend_width, progressive, reset_cursor):
    """Run the iterative pattern discovery phase.

    Samples conversations and asks the LLM to notice structural, geometric,
    and topological patterns. Accumulates a concept map that can be injected
    into the chunking stage via 'bud process --with-discovery'.

    Sampling modes (pick one):
      default      -- random whole-conversation sampling
      --blend      -- random cross-boundary slices (no memory)
      --progressive -- cursor-based: one slice per file, exhaustive coverage
    """
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
    )

    console = Console()

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = get_output_dir()

    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    config.setdefault("pipeline", {
        "chunk_min_tokens": 10,
        "chunk_max_tokens": 800,
        "schema_evolution_confidence_threshold": 5,
    })

    console.print("\n[bold cyan]Bud RAG Pipeline — Discovery Phase[/bold cyan]\n")
    console.print(f"[dim]Output directory: {output_dir}[/dim]")
    if progressive:
        console.print(f"[dim]Mode: progressive ({blend_width} turns/file/iteration)[/dim]")
    elif blend:
        console.print(f"[dim]Mode: blend ({blend_slices} slices × {blend_width} turns)[/dim]")
    else:
        console.print(f"[dim]Samples per iteration: {samples}[/dim]")
    console.print(f"[dim]Max iterations: {iterations}[/dim]")
    console.print(f"[dim]Stability threshold: {stability}[/dim]\n")

    from bud.stages.index import IndexManager
    index_mgr = IndexManager(output_dir, config)
    index_mgr.ensure_directories()

    parsed_dir = output_dir / "parsed"
    if not parsed_dir.exists() or not any(parsed_dir.glob("*.jsonl")):
        console.print(
            "[red]No parsed conversations found.[/red]\n"
            "[dim]Run 'bud process' first (or at least the parse stage) to generate "
            "parsed/*.jsonl files.[/dim]"
        )
        return

    from bud.lib.llm import LLMClient
    from bud.stages.discover import DiscoveryMap, run_discovery

    llm = LLMClient(config)

    concept_map = DiscoveryMap(index_mgr.discovery_map_path)
    if resume:
        concept_map.load()
        console.print(
            f"[green]✓ Resuming from existing map "
            f"({concept_map.iterations_completed} iterations done, "
            f"stability={concept_map.stability_score:.2f})[/green]\n"
        )

    # Set up progressive cursor if requested
    cursor = None
    file_totals: dict = {}
    if progressive:
        from bud.stages.blend import BlendCursor
        cursor = BlendCursor(index_mgr.blend_cursor_path)
        if reset_cursor:
            cursor.reset()
            console.print("[yellow]⚠  Blend cursor reset — starting from the beginning[/yellow]\n")
        else:
            cursor.load()
            if not cursor.is_empty():
                console.print(
                    f"[green]✓ Resuming blend cursor "
                    f"({len(cursor.data)} file(s) tracked)[/green]\n"
                )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=8,
    ) as progress:
        task = progress.add_task("Sampling...", total=None)

        def on_sampling(iteration_num, max_iter):
            if progressive:
                mode_tag = "progressive"
            elif blend:
                mode_tag = "blend"
            else:
                mode_tag = "sample"
            progress.update(
                task,
                description=(
                    f"[dim]iter {iteration_num}/{max_iter}  "
                    f"{mode_tag}  "
                    f"waiting for LLM...[/dim]"
                ),
            )

        def on_iteration(iteration_num, score, cmap):
            signals   = len(cmap.data.get("boundary_signals", []))
            archetypes = len(cmap.data.get("chunk_archetypes", []))
            anchors   = len(cmap.data.get("coherence_anchors", []))
            stable_bar = "▓" * int(score * 10) + "░" * (10 - int(score * 10))
            progress.update(
                task,
                description=(
                    f"iter {iteration_num}/{iterations}  "
                    f"[{'green' if score >= stability else 'yellow'}]{stable_bar}[/{'green' if score >= stability else 'yellow'}] "
                    f"stability={score:.2f}  "
                    f"[dim]signals={signals}  archetypes={archetypes}  anchors={anchors}[/dim]"
                ),
            )

        concept_map = run_discovery(
            parsed_dir=str(parsed_dir),
            concept_map=concept_map,
            llm=llm,
            n_samples=samples,
            stability_threshold=stability,
            max_iterations=iterations,
            on_iteration=on_iteration,
            on_sampling=on_sampling,
            use_blend=blend,
            blend_slices=blend_slices,
            blend_width=blend_width,
            use_progressive=progressive,
            cursor=cursor,
        )

    console.print(f"\n[green]✓ Discovery complete![/green]")
    console.print(f"  Iterations: {concept_map.iterations_completed}")
    console.print(f"  Stability score: {concept_map.stability_score:.2f}")
    console.print(f"  Boundary signals: {len(concept_map.data.get('boundary_signals', []))}")
    console.print(f"  Coherence anchors: {len(concept_map.data.get('coherence_anchors', []))}")
    console.print(f"  Chunk archetypes: {len(concept_map.data.get('chunk_archetypes', []))}")
    console.print(f"  Anti-patterns: {len(concept_map.data.get('anti_patterns', []))}")
    console.print(f"\n[dim]Concept map saved to: {index_mgr.discovery_map_path}[/dim]")

    if progressive and cursor is not None:
        from bud.stages.blend import blend_progressive as _bp, _load_turns
        file_totals = {
            f.name: len(_load_turns(f))
            for f in sorted((output_dir / "parsed").glob("*.jsonl"))
        }
        coverage = cursor.coverage(file_totals)
        if coverage:
            console.print("\n  [dim]Blend cursor coverage:[/dim]")
            for fname, pct in sorted(coverage.items()):
                bar = "▓" * int(pct * 20) + "░" * (20 - int(pct * 20))
                console.print(f"    {bar} {pct*100:.0f}%  {fname}")
        console.print(f"  [dim]Cursor saved to: {index_mgr.blend_cursor_path}[/dim]")

    console.print("[dim]Run 'bud process --with-discovery' to use it for chunking.[/dim]\n")


@main.command()
@click.option(
    "--data-dir", "-d",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to input data directory (overrides config)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, resolve_path=True),
    help="Path to output directory (overrides config)",
)
@click.option(
    "--resume/--no-resume", "-r",
    default=False,
    help="Resume from last checkpoint",
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=50,
    help="Number of conversations per batch (default: 50)",
)
@click.option(
    "--prompt", "-p",
    type=click.Choice(["conversational", "factual", "mythic", "synthesis"]),
    default="conversational",
    help="Prompt preset to use for chunking",
)
@click.option(
    "--with-discovery/--no-discovery",
    default=False,
    help="Inject discovery concept map into chunking prompts",
)
def process(data_dir, output_dir, resume, batch_size, prompt, with_discovery):
    """Run the full RAG pipeline.

    Processes conversation data and builds the vector index.
    """
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn,
        BarColumn, MofNCompleteColumn, TimeElapsedColumn,
    )

    console = Console()

    # Use overrides or config
    if data_dir:
        data_dir = Path(data_dir)
    else:
        data_dir = get_data_dir()

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = get_output_dir()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Bud RAG Pipeline[/bold cyan]\n")

    # Load config for LLM/embedding settings
    config = load_config()

    # Add pipeline-specific config
    config.setdefault("pipeline", {
        "chunk_min_tokens": 10,
        "chunk_max_tokens": 800,
        "schema_evolution_confidence_threshold": 5,
    })

    console.print(f"[dim]Data directory: {data_dir}[/dim]")
    console.print(f"[dim]Output directory: {output_dir}[/dim]")
    console.print(f"[dim]Prompt preset: {prompt}[/dim]")
    console.print(f"[dim]Batch size: {batch_size}[/dim]")
    if with_discovery:
        console.print(f"[dim]Discovery: enabled[/dim]")
    console.print("")

    # Initialize index manager
    from bud.stages.index import IndexManager
    index_mgr = IndexManager(output_dir, config)

    # Create output directories
    index_mgr.ensure_directories()

    # Load or create schema
    from bud.lib.schema_manager import SchemaManager
    schema_mgr = SchemaManager(index_mgr.schema_path)
    schema = schema_mgr.load()
    if not schema_mgr.validate(schema):
        schema = schema_mgr.get_default_schema()
        schema_mgr.save(schema)

    console.print(f"[green]✓ Schema loaded (v{schema['version']})[/green]")

    # Find conversation files
    conv_files = sorted(data_dir.glob("conversations_*.json"))
    if not conv_files:
        console.print("[yellow]No conversation files found[/yellow]")
        return

    console.print(f"[green]✓ Found {len(conv_files)} conversation file(s)[/green]\n")

    # Initialize vector store
    index_path = str(index_mgr.index_dir / "chunks")
    from bud.lib.store import VectorStore
    store = VectorStore(index_path, dim=768)
    if resume and os.path.exists(f"{index_path}.faiss"):
        store.load()
        console.print(f"[green]✓ Loaded existing index ({store.count()} chunks)[/green]")

    # Build progress tracking
    from bud.lib.progress import ProgressTracker
    tracker = ProgressTracker(index_mgr.progress_path)

    # Load embedding queue if resuming
    failed_chunks = []
    if resume:
        from bud.stages.embed import load_embed_queue
        failed_chunks = load_embed_queue(index_mgr.embed_queue_path)
        if failed_chunks:
            console.print(f"[yellow]✓ Resuming {len(failed_chunks)} failed embeddings[/yellow]")

    # Initialize LLM and embedding clients
    from bud.lib.llm import LLMClient
    from bud.lib.embeddings import EmbeddingClient
    llm = LLMClient(config)
    embedding_client = EmbeddingClient(config)

    # Initialize prompt loader
    from bud.lib.prompt_loader import PromptLoader
    prompts_dir = str(Path(__file__).parent / "prompts")
    prompt_loader = PromptLoader(prompts_dir)
    system_prompt = prompt_loader.load(prompt, {
        "owner_name": "User",
        "schema": json.dumps(schema["dimensions"], indent=2),
        "file_context": f"{len(conv_files)} conversation files",
    })

    # Load discovery concept map if requested
    concept_map_summary = None
    if with_discovery:
        from bud.stages.discover import DiscoveryMap
        dm = DiscoveryMap(index_mgr.discovery_map_path).load()
        if dm.is_empty():
            console.print(
                "[yellow]⚠ No discovery map found. Run 'bud discover' first, "
                "or omit --with-discovery.[/yellow]\n"
            )
        else:
            concept_map_summary = dm.to_summary()
            console.print(
                f"[green]✓ Loaded discovery map "
                f"({dm.iterations_completed} iterations, "
                f"stability={dm.stability_score:.2f})[/green]"
            )

    # Parse conversations
    parsed_dir = output_dir / "parsed"
    from bud.stages.parse import parse_conversations_file, parse_all

    console.print("[cyan]→ Parsing conversations[/cyan]")
    total_conversations = parse_all(data_dir, parsed_dir)

    # Load parsed conversations
    parsed_files = sorted(parsed_dir.glob("conversations_*.jsonl"))
    all_conversations = []
    for pf in parsed_files:
        with open(pf) as f:
            for line in f:
                all_conversations.append(json.loads(line.strip()))

    console.print(f"[green]✓ Parsed {total_conversations} conversations[/green]\n")

    # Process in batches
    from bud.stages.chunk import chunk_conversation
    from bud.stages.embed import embed_chunks, clear_embed_queue

    total_chunks = 0
    errors = 0

    n_convs = len(all_conversations)
    n_batches = max(1, (n_convs + batch_size - 1) // batch_size)

    console.print(f"[cyan]→ Chunking and embedding {n_convs} conversations[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=8,
    ) as progress:
        conv_task  = progress.add_task("[cyan]Conversations[/cyan]", total=n_convs)
        op_task    = progress.add_task("", total=None)

        conv_idx = 0
        for i in range(0, n_convs, batch_size):
            batch = all_conversations[i:i + batch_size]
            batch_num = i // batch_size + 1
            filename = f"conversations_{batch_num}.jsonl"

            # Skip already-processed batches when resuming
            if resume and tracker.is_complete(filename, batch_num):
                progress.update(conv_task, advance=len(batch))
                conv_idx += len(batch)
                progress.update(
                    op_task,
                    description=f"[dim]skipped batch {batch_num}/{n_batches} (already done)[/dim]",
                )
                continue

            # --- Chunking ---
            batch_chunks = []
            for conv in batch:
                conv_idx += 1
                name = (conv.get("conversation_name") or conv["id"])[:52]
                progress.update(
                    op_task,
                    description=(
                        f"[yellow]chunk[/yellow]  "
                        f"batch {batch_num}/{n_batches}  "
                        f"[dim]{name}[/dim]"
                    ),
                )
                try:
                    chunks = chunk_conversation(
                        conv, schema, llm, config, system_prompt, prompt_preset=prompt,
                        schema_version=schema["version"],
                        concept_map_summary=concept_map_summary,
                    )
                    batch_chunks.extend(chunks)

                    for chunk in chunks:
                        for proposal in chunk.get("schema_proposals", []):
                            schema_mgr.propose_candidate(
                                proposal["dimension"],
                                proposal["value"],
                                proposal.get("rationale", "")
                            )
                except Exception as e:
                    errors += 1
                    progress.print(f"  [red]✗ chunk error  {conv['id']}: {e}[/red]")

                progress.update(conv_task, advance=1)

            total_chunks += len(batch_chunks)

            # --- Embedding ---
            all_chunks_for_embed = failed_chunks + batch_chunks
            n_embed = len(all_chunks_for_embed)

            def _on_chunk(done, total, _batch=batch_num):
                progress.update(
                    op_task,
                    description=(
                        f"[blue]embed[/blue]   "
                        f"batch {_batch}/{n_batches}  "
                        f"[dim]{done}/{total} chunks[/dim]"
                    ),
                )

            progress.update(
                op_task,
                description=(
                    f"[blue]embed[/blue]   "
                    f"batch {batch_num}/{n_batches}  "
                    f"[dim]0/{n_embed} chunks[/dim]"
                ),
            )
            batch_embed_errors: list[str] = []

            def _on_error(chunk, error_msg, _errs=batch_embed_errors):
                if len(_errs) < 1:
                    _errs.append(error_msg)

            failed = embed_chunks(
                all_chunks_for_embed, embedding_client, store,
                index_mgr.embed_queue_path,
                on_chunk=_on_chunk,
                on_error=_on_error,
            )

            if failed < len(all_chunks_for_embed):
                clear_embed_queue(index_mgr.embed_queue_path)
            failed_chunks = []

            embedded = n_embed - failed
            if failed > 0:
                first_err = batch_embed_errors[0] if batch_embed_errors else "unknown error"
                progress.print(
                    f"  [yellow]⚠  batch {batch_num}: {failed}/{n_embed} chunks failed to embed[/yellow]\n"
                    f"    [dim]{first_err}[/dim]"
                )
            progress.update(
                op_task,
                description=(
                    f"[green]✓ batch {batch_num}/{n_batches}[/green]  "
                    f"[dim]{len(batch_chunks)} chunks  "
                    f"{embedded} embedded  "
                    f"{total_chunks} total  "
                    f"{errors} errors[/dim]"
                ),
            )

            tracker.mark_complete(filename, batch_num)

        progress.update(
            op_task,
            description=(
                f"[bold green]done[/bold green]  "
                f"[dim]{total_chunks} chunks  "
                f"{store.count() if store else 0} in index  "
                f"{errors} errors[/dim]"
            ),
        )

    # Apply schema evolution
    promoted = schema_mgr.apply_promotions(config)
    if promoted:
        console.print(f"\n[yellow]Schema evolved! Promoted: {', '.join(promoted)}[/yellow]")
        schema = schema_mgr.load()

    # Final index save
    if store:
        store.save()
        console.print(f"\n[green]✓ Final index saved ({store.count()} total chunks)[/green]")

    # Warn about queued embed failures
    from bud.stages.embed import load_embed_queue
    queued = load_embed_queue(index_mgr.embed_queue_path)
    if queued:
        console.print(
            f"\n[yellow]⚠  {len(queued)} chunk(s) failed to embed and are queued for retry.[/yellow]\n"
            f"   Fix the embedding service, then run: [bold]bud process --resume[/bold]\n"
            f"   Queue file: [dim]{index_mgr.embed_queue_path}[/dim]"
        )

    # Print summary
    console.print(f"\n[bold cyan]Pipeline Complete![/bold cyan]")
    console.print(f"  Conversations: {total_conversations}")
    console.print(f"  Chunks: {total_chunks}")
    console.print(f"  Errors: {errors}")
    console.print(f"  Schema version: v{schema['version']}")

    if promoted:
        console.print(f"  Promoted: {', '.join(promoted)}")

    console.print(f"\n[dim]Output: {output_dir}[/dim]")


@main.command()
@click.argument("query_text")
@click.option(
    "--k",
    type=click.INT,
    default=5,
    help="Number of results to return (default: 5)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, resolve_path=True),
    help="Path to output directory (overrides config)",
)
def query(query_text, k, output_dir):
    """Query the vector index for relevant context.

    SEARCH_TEXT: The search query to find relevant conversation context
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Load config
    config = load_config()

    # Use override or config for output_dir
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = get_output_dir()

    console.print(f"\n[bold cyan]Bud RAG Pipeline - Query[/bold cyan]\n")
    console.print(f"[bold]Query:[/bold] {query_text}")
    console.print(f"[bold]Top-K:[/bold] {k}\n")

    # Build paths
    index_dir = output_dir / "index"
    index_path = str(index_dir / "chunks")
    metadata_path = index_dir / "chunks_metadata.jsonl"

    # Load FAISS index
    from bud.lib.store import VectorStore
    store = VectorStore(index_path, dim=0)  # dim=0 will be inferred from loaded index

    try:
        store.load()
    except Exception as e:
        console.print(f"[red]Error loading index: {e}[/red]")
        console.print(f"[dim]Make sure you've run 'bud process' first.[/dim]")
        return

    if store.count() == 0:
        console.print("[red]No chunks found in the index.[/red]")
        console.print(f"[dim]Index path: {index_path}[/dim]")
        return

    console.print(f"[green]✓ Loaded index with {store.count()} chunks[/green]\n")

    # Embed the user query
    from bud.lib.embeddings import EmbeddingClient
    embedding_client = EmbeddingClient(config)

    try:
        query_embedding = embedding_client.embed(query_text)
    except Exception as e:
        console.print(f"[red]Error embedding query: {e}[/red]")
        return

    # Search for top-k similar chunks
    search_results = store.search(query_embedding, k)

    if not search_results:
        console.print("[yellow]No matching chunks found.[/yellow]")
        return

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(search_results, 1):
        context_parts.append(f"[{i}] {chunk.get('text', '')}")

    context = "\n\n".join(context_parts)

    # Generate answer using LLM
    from bud.lib.llm import LLMClient
    llm = LLMClient(config)

    # Format prompt with context and query
    prompt = f"""You are a helpful assistant answering questions based on conversation context.

Context (from conversation history):
{context}

---
User Question: {query_text}

Instructions:
- Answer based ONLY on the context above
- Be concise and focused
- If context is unclear or incomplete, say so
- Cite source by rank number when relevant
"""

    try:
        answer = llm.complete("You are a helpful assistant.", prompt)
    except Exception as e:
        console.print(f"[red]Error generating answer: {e}[/red]")
        answer = "Unable to generate answer due to LLM error."

    # Display results as table
    table = Table(title="Search Results")
    table.add_column("Rank", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta", no_wrap=True)
    table.add_column("Source", style="green")

    for i, chunk in enumerate(search_results, 1):
        rank = str(i)
        score = chunk.get("score", "N/A")
        source = chunk.get("source_file", chunk.get("source", "conversations.jsonl"))
        table.add_row(rank, str(score), source)

    console.print(table)

    # Display answer
    console.print(f"\n[bold]Answer:[/bold]")
    console.print(answer)

    console.print(f"\n[dim]Query completed.[/dim]")


@main.command()
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, resolve_path=True),
    help="Path to output directory (overrides config)",
)
def status(output_dir):
    """Show pipeline status and configuration info."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Use override or config
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = get_output_dir()

    config = load_config()

    # Bud header
    console.print("\n[bold cyan]Bud RAG Pipeline[/bold cyan]")
    console.print(f"[dim]Version: {__version__}[/dim]\n")

    # Configuration panel
    config_panel = Panel(
        f"[bold]Config file:[/bold] {CONFIG_FILE}\n"
        f"[bold]Data directory:[/bold] {config.get('data_dir', 'Not set')}\n"
        f"[bold]Output directory:[/bold] {config.get('output_dir', 'Not set')}\n"
        f"\n[bold]LLM:[/bold]\n"
        f"  Provider: {config.get('llm', {}).get('provider', 'Not set')}\n"
        f"  Model: {config.get('llm', {}).get('model', 'Not set')}\n"
        f"  Base URL: {config.get('llm', {}).get('base_url', 'Not set')}\n"
        f"\n[bold]Embeddings:[/bold]\n"
        f"  Provider: {config.get('embeddings', {}).get('provider', 'Not set')}\n"
        f"  Model: {config.get('embeddings', {}).get('model', 'Not set')}\n"
        f"  Base URL: {config.get('embeddings', {}).get('base_url', 'Not set')}",
        title="Configuration",
        border_style="green",
    )
    console.print(config_panel)

    # Vector index status
    index_dir = output_dir / "index"
    index_path = str(index_dir / "chunks")
    faiss_path = f"{index_path}.faiss"
    metadata_path = f"{index_path}_metadata.jsonl"

    index_exists = os.path.exists(faiss_path)
    if index_exists:
        index_size = os.path.getsize(faiss_path)
        metadata_size = os.path.getsize(metadata_path) if os.path.exists(metadata_path) else 0
    else:
        index_size = 0
        metadata_size = 0

    # Get chunk count from index
    chunk_count = 0
    if index_exists:
        try:
            from bud.lib.store import VectorStore
            store = VectorStore(index_path, dim=0)
            store.load()
            chunk_count = store.count()
        except Exception:
            chunk_count = 0

    # Pipeline status table
    table = Table(title="Pipeline Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")

    # Vector Index status
    if index_exists:
        index_status = "[green]Ready[/green]"
        index_details = f"{index_dir}\n{chunk_count} chunks\n{index_size:,} bytes (index)"
        if metadata_size > 0:
            index_details += f", {metadata_size:,} bytes (metadata)"
    else:
        index_status = "[yellow]Not built[/yellow]"
        index_details = f"{index_dir}\nRun 'bud process' to build"

    table.add_row("Vector Index", index_status, index_details)

    # Knowledge Base status
    kb_dir = output_dir / "knowledge_base"
    kb_exists = kb_dir.exists() and any(kb_dir.iterdir())
    if kb_exists:
        kb_status = "[green]Ready[/green]"
        kb_details = str(kb_dir)
    else:
        kb_status = "[yellow]Empty[/yellow]"
        kb_details = str(kb_dir)

    table.add_row("Knowledge Base", kb_status, kb_details)

    # Last Processed - get from progress file
    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        try:
            with open(progress_path) as f:
                progress = json.load(f)
            total_files = len(progress)
            total_batches = sum(len(data.get("completed", [])) for data in progress.values())
            last_file = list(progress.keys())[-1] if progress else "N/A"
            last_status = "Complete" if total_batches == sum(
                len(data.get("completed", [])) for data in progress.values()
            ) and all(
                len(data.get("failed", {})) == 0 for data in progress.values()
            ) else "Partial"
            last_processed = f"{last_file} (batch {total_batches})\n{last_status}"
        except Exception:
            last_processed = "N/A"
    else:
        last_processed = "Never (no processing done)"

    table.add_row("Last Processed", "", last_processed)

    console.print(table)

    # Validation
    is_valid, errors = validate_config(config)

    if is_valid:
        console.print("\n[green]Configuration is valid.[/green]")
    else:
        console.print("\n[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  - {error}")


if __name__ == "__main__":
    main()
