"""Comprehensive CLI for the Virtual PR Firm application.

This module provides a full-featured command-line interface with multiple
operation modes, configuration management, and professional CLI patterns.
"""

import argparse
import sys
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import time
from contextlib import contextmanager

# Import application modules
from config import get_config, create_config_template, ConfigurationError
from validation import validate_shared_store, ValidationError
from logging_config import configure_logging, create_request_context, log_info, log_error
from caching import initialize_cache, get_cache_manager
from flow import create_main_flow

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Global console for rich output
console = Console() if RICH_AVAILABLE else None


class CLIError(Exception):
    """Raised when CLI operations fail."""
    pass


def print_error(message: str, error: Optional[Exception] = None):
    """
    Print an error message to the user, using the global Rich console when available.
    
    Parameters:
        message (str): Primary error message to display.
        error (Optional[Exception]): Optional exception whose string representation will be printed on the following line for additional context.
    """
    if console:
        console.print(f"[red]Error:[/red] {message}")
        if error:
            console.print(f"[dim]{str(error)}[/dim]")
    else:
        print(f"Error: {message}")
        if error:
            print(f"  {str(error)}")


def print_success(message: str):
    """
    Print a success message to the user, using colored output when Rich is available.
    
    Parameters:
        message (str): Message text to display; will be prefixed with a check mark. Uses the global Rich Console (if available) for colored output, otherwise prints plain text to stdout.
    """
    if console:
        console.print(f"[green]✓[/green] {message}")
    else:
        print(f"✓ {message}")


def print_info(message: str):
    """
    Print an informational message to the user.
    
    Uses the global Rich Console (if available) to render a colored info icon; otherwise falls back to plain stdout.
    
    Parameters:
        message (str): The message text to display.
    """
    if console:
        console.print(f"[blue]ℹ[/blue] {message}")
    else:
        print(f"ℹ {message}")


def print_warning(message: str):
    """
    Print a warning message to the user, using the global Rich console when available.
    
    Parameters:
        message (str): Warning text to display. Shown with a leading warning glyph and colored output when Rich is enabled.
    """
    if console:
        console.print(f"[yellow]⚠[/yellow] {message}")
    else:
        print(f"⚠ {message}")


@contextmanager
def progress_context(description: str, total: Optional[int] = None):
    """
    Context manager that provides a progress indicator for long-running operations.
    
    When a Rich Console is available (global `console` is set) this yields a Rich progress task object
    that can be updated with `progress.update(...)` inside the context. If a numeric `total` is
    provided the progress bar is determinate and reports percentage; otherwise an indeterminate
    spinner and description are shown.
    
    When no Rich Console is available this prints simple "Starting: ..." and "Completed: ..." lines
    and yields None.
    
    Parameters:
        description: Short description shown alongside the progress indicator.
        total: Optional total work units for a determinate progress bar. If omitted or None, an
            indeterminate spinner is used.
    
    Yields:
        A Rich progress task handle when Rich is available, otherwise None.
    """
    if console and total is not None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(description, total=total)
            yield task
    elif console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(description, total=None)
            yield task
    else:
        print(f"Starting: {description}")
        yield None
        print(f"Completed: {description}")


def create_parser() -> argparse.ArgumentParser:
    """
    Create and return the top-level argparse.ArgumentParser for the CLI.
    
    The parser configures global options and the following subcommands with their key options:
    - serve: start Gradio web UI (--port, --host, --share, --auth, --ssl-keyfile, --ssl-certfile)
    - run: execute a single content-generation task (--topic, --platforms, --brand-bible-file, --content-type, --target-audience, --output-format, --output-file, --dry-run)
    - config: manage configuration with subcommands:
        - generate (--output, --format)
        - validate (--file)
        - show (--format)
    - validate: validate an input data file (--input, --format)
    - health: run system health checks (--detailed)
    - info: display system and environment information
    
    Also provides global flags such as --config-file, --log-level, --log-format, --log-file, --no-color, --verbose, and --version.
    
    Returns:
        argparse.ArgumentParser: Configured parser ready for argument parsing.
    """
    parser = argparse.ArgumentParser(
        prog="virtual-pr-firm",
        description="Virtual PR Firm - AI-powered content generation for social media platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the web interface
  virtual-pr-firm serve --port 8080

  # Run a single content generation task
  virtual-pr-firm run --topic "Product launch announcement" --platforms twitter,linkedin

  # Generate configuration template
  virtual-pr-firm config generate --output config.yaml

  # Validate input data
  virtual-pr-firm validate --input data.json

  # Run with custom configuration
  virtual-pr-firm serve --config-file my-config.yaml --log-level DEBUG
        """
    )
    
    # Global options
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to configuration file (YAML/JSON)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-format',
        choices=['human', 'json'],
        default='human',
        help='Log format (default: human)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='Virtual PR Firm 1.0.0'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Serve command
    serve_parser = subparsers.add_parser(
        'serve',
        help='Start the Gradio web interface',
        description='Start the web-based user interface for interactive content generation.'
    )
    serve_parser.add_argument(
        '--port', '-p',
        type=int,
        default=7860,
        help='Port to serve on (default: 7860)'
    )
    serve_parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    serve_parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public link for the interface'
    )
    serve_parser.add_argument(
        '--auth',
        type=str,
        help='Authentication (username:password)'
    )
    serve_parser.add_argument(
        '--ssl-keyfile',
        type=str,
        help='SSL key file path'
    )
    serve_parser.add_argument(
        '--ssl-certfile',
        type=str,
        help='SSL certificate file path'
    )
    
    # Run command
    run_parser = subparsers.add_parser(
        'run',
        help='Run content generation task',
        description='Execute a single content generation task with specified parameters.'
    )
    run_parser.add_argument(
        '--topic', '-t',
        type=str,
        required=True,
        help='Topic or goal for content generation'
    )
    run_parser.add_argument(
        '--platforms', '-p',
        type=str,
        required=True,
        help='Comma-separated list of target platforms'
    )
    run_parser.add_argument(
        '--brand-bible-file',
        type=str,
        help='Path to brand bible file'
    )
    run_parser.add_argument(
        '--output-format',
        choices=['json', 'yaml', 'text'],
        default='json',
        help='Output format (default: json)'
    )
    run_parser.add_argument(
        '--output-file',
        type=str,
        help='Output file path (default: stdout)'
    )
    run_parser.add_argument(
        '--content-type',
        type=str,
        choices=['press_release', 'social_post', 'blog_post', 'email', 'newsletter'],
        default='social_post',
        help='Content type (default: social_post)'
    )
    run_parser.add_argument(
        '--target-audience',
        type=str,
        help='Target audience description'
    )
    run_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate inputs without executing flow'
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration management',
        description='Manage application configuration files and settings.'
    )
    config_subparsers = config_parser.add_subparsers(
        dest='config_command',
        help='Configuration subcommands',
        metavar='SUBCOMMAND'
    )
    
    # Config generate
    config_generate_parser = config_subparsers.add_parser(
        'generate',
        help='Generate configuration template',
        description='Create a new configuration template file.'
    )
    config_generate_parser.add_argument(
        '--output', '-o',
        type=str,
        default='config.yaml',
        help='Output file path (default: config.yaml)'
    )
    config_generate_parser.add_argument(
        '--format',
        choices=['yaml', 'json'],
        default='yaml',
        help='Configuration format (default: yaml)'
    )
    
    # Config validate
    config_validate_parser = config_subparsers.add_parser(
        'validate',
        help='Validate configuration file',
        description='Validate a configuration file for errors.'
    )
    config_validate_parser.add_argument(
        '--file', '-f',
        type=str,
        required=True,
        help='Configuration file to validate'
    )
    
    # Config show
    config_show_parser = config_subparsers.add_parser(
        'show',
        help='Show current configuration',
        description='Display the current configuration with all values.'
    )
    config_show_parser.add_argument(
        '--format',
        choices=['yaml', 'json'],
        default='yaml',
        help='Output format (default: yaml)'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate input data',
        description='Validate input data files and structures.'
    )
    validate_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input file to validate (JSON/YAML)'
    )
    validate_parser.add_argument(
        '--format',
        choices=['json', 'yaml'],
        help='Input format (auto-detected if not specified)'
    )
    
    # Health command
    health_parser = subparsers.add_parser(
        'health',
        help='System health check',
        description='Perform comprehensive system health checks.'
    )
    health_parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed health information'
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='System information',
        description='Display system information and environment details.'
    )
    
    return parser


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load application configuration from the given file path or the environment and return a normalized settings dictionary.
    
    Parameters:
        config_file (Optional[str]): Path to a configuration file. If None, uses the default discovery order (environment, default locations).
    
    Returns:
        Dict[str, Any]: A structured dict with top-level sections: 'debug', 'environment', 'logging', 'gradio', 'llm', 'flow', and 'security'. Each section contains the corresponding runtime and client settings (e.g., logging.level, gradio.port, llm.provider, flow.enable_caching, security.enable_auth).
    
    Raises:
        CLIError: If underlying configuration loading or validation fails (wraps ConfigurationError).
    """
    try:
        config = get_config(config_file)
        return {
            'debug': config.debug,
            'environment': config.environment,
            'logging': {
                'level': config.logging.level,
                'format': config.logging.format,
                'file': config.logging.file,
                'max_size': config.logging.max_size,
                'backup_count': config.logging.backup_count,
                'correlation_id': config.logging.correlation_id
            },
            'gradio': {
                'port': config.gradio.port,
                'host': config.gradio.host,
                'share': config.gradio.share,
                'auth': config.gradio.auth,
                'ssl_keyfile': config.gradio.ssl_keyfile,
                'ssl_certfile': config.gradio.ssl_certfile
            },
            'llm': {
                'provider': config.llm.provider,
                'model': config.llm.model,
                'temperature': config.llm.temperature,
                'max_tokens': config.llm.max_tokens,
                'timeout': config.llm.timeout,
                'retries': config.llm.retries,
                'retry_delay': config.llm.retry_delay,
                'api_key': config.llm.api_key,
                'base_url': config.llm.base_url
            },
            'flow': {
                'timeout': config.flow.timeout,
                'max_retries': config.flow.max_retries,
                'retry_delay': config.flow.retry_delay,
                'enable_streaming': config.flow.enable_streaming,
                'enable_caching': config.flow.enable_caching,
                'cache_ttl': config.flow.cache_ttl,
                'cache_size': config.flow.cache_size
            },
            'security': {
                'enable_auth': config.security.enable_auth,
                'enable_rate_limiting': config.security.enable_rate_limiting,
                'rate_limit_requests': config.security.rate_limit_requests,
                'rate_limit_window': config.security.rate_limit_window,
                'max_request_size': config.security.max_request_size,
                'allowed_origins': config.security.allowed_origins,
                'session_timeout': config.security.session_timeout
            }
        }
    except ConfigurationError as e:
        raise CLIError(f"Configuration error: {e}")


def setup_logging(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Configure application logging.
    
    CLI-provided values in `args` take precedence over values in `config['logging']`. Calls the project's
    `configure_logging` helper with resolved level, format, file, max size, backup count, and correlation-id inclusion.
    
    Parameters:
        args (argparse.Namespace): Parsed command-line arguments (expects `log_level`, `log_format`, `log_file`).
        config (dict): Loaded configuration dictionary containing a `logging` section with keys
            `level`, `format`, `file`, `max_size`, `backup_count`, and `correlation_id`.
    """
    log_level = args.log_level or config['logging']['level']
    log_format = args.log_format or config['logging']['format']
    log_file = args.log_file or config['logging']['file']
    
    configure_logging(
        level=log_level,
        format_type=log_format,
        log_file=log_file,
        max_size=config['logging']['max_size'],
        backup_count=config['logging']['backup_count'],
        include_correlation_id=config['logging']['correlation_id']
    )


def cmd_serve(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Start and launch the Gradio web interface for the application.
    
    Requires Gradio to be installed; constructs and passes launch options from the provided CLI namespace and then calls the app's launch method. The following fields are read from `args`: `port`, `host`, `share`, `auth` (expected as "user:pass"), `ssl_keyfile`, `ssl_certfile`, and `verbose`. Prints informational messages and will raise CLIError if Gradio is not available or if the interface fails to start.
    """
    if not GRADIO_AVAILABLE:
        raise CLIError("Gradio is not installed. Install with: pip install gradio")
    
    print_info("Starting Virtual PR Firm web interface...")
    
    # Import here to avoid circular imports
    from main import create_gradio_interface
    
    try:
        # Create Gradio interface
        app = create_gradio_interface()
        
        # Prepare launch arguments
        launch_kwargs = {
            'server_port': args.port,
            'server_name': args.host,
            'share': args.share,
            'auth': args.auth.split(':') if args.auth else None,
            'ssl_keyfile': args.ssl_keyfile,
            'ssl_certfile': args.ssl_certfile,
            'show_error': True,
            'quiet': not args.verbose
        }
        
        # Remove None values
        launch_kwargs = {k: v for k, v in launch_kwargs.items() if v is not None}
        
        print_success(f"Web interface starting on http://{args.host}:{args.port}")
        if args.share:
            print_info("Public link will be generated...")
        
        # Launch the interface
        app.launch(**launch_kwargs)
        
    except Exception as e:
        log_error(e, {'command': 'serve', 'args': vars(args)})
        raise CLIError(f"Failed to start web interface: {e}")


def cmd_run(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Run a single content-generation task from CLI arguments.
    
    Parses platforms and optional brand bible, validates the assembled task requirements, optionally performs a dry run, initializes caching, executes the main content-generation flow, and emits results to stdout or a file in JSON, YAML, or plain-text formats.
    
    Parameters:
        args (argparse.Namespace): Parsed CLI arguments. Expected attributes used by this command:
            - platforms (str): Comma-separated platform identifiers.
            - brand_bible_file (Optional[str]): Path to an XML brand bible file.
            - topic (str): Topic or goal for the content.
            - content_type (str): Desired content type.
            - target_audience (Optional[str]): Target audience description.
            - dry_run (bool): If True, validate inputs but do not execute the flow.
            - output_format (str): One of 'json', 'yaml', or 'text'.
            - output_file (Optional[str]): Path to write output; if omitted, prints to stdout.
        config (dict): Application configuration dictionary (used for flow/cache initialization).
    
    Side effects:
        - May read the brand bible file.
        - Initializes and uses the cache manager.
        - Runs the main content-generation flow which may produce side effects in shared state.
        - Writes output to disk if --output-file is specified, or prints to stdout.
        - Exits process with code 1 on validation failures.
    
    Raises:
        CLIError: On unexpected failures during execution (wraps the original exception).
    """
    print_info("Running content generation task...")
    
    try:
        # Parse platforms
        platforms = [p.strip() for p in args.platforms.split(',') if p.strip()]
        
        # Load brand bible if specified
        brand_bible = {"xml_raw": ""}
        if args.brand_bible_file:
            brand_bible_path = Path(args.brand_bible_file)
            if not brand_bible_path.exists():
                raise CLIError(f"Brand bible file not found: {args.brand_bible_file}")
            
            with open(brand_bible_path, 'r', encoding='utf-8') as f:
                brand_bible["xml_raw"] = f.read()
        
        # Prepare shared store
        shared = {
            "task_requirements": {
                "platforms": platforms,
                "topic_or_goal": args.topic,
                "content_type": args.content_type
            },
            "brand_bible": brand_bible,
            "stream": None
        }
        
        if args.target_audience:
            shared["task_requirements"]["target_audience"] = args.target_audience
        
        # Validate inputs
        print_info("Validating inputs...")
        validate_shared_store(shared)
        print_success("Input validation passed")
        
        if args.dry_run:
            print_success("Dry run completed - inputs are valid")
            return
        
        # Initialize cache
        initialize_cache(config['flow'])
        
        # Execute flow
        with progress_context("Generating content", len(platforms)):
            flow = create_main_flow()
            flow.run(shared)
        
        # Get results
        content_pieces = shared.get("content_pieces", {})
        
        if not content_pieces:
            print_warning("No content was generated")
            return
        
        # Format output
        if args.output_format == 'json':
            output_data = content_pieces
        elif args.output_format == 'yaml':
            output_data = yaml.dump(content_pieces, default_flow_style=False)
        else:  # text
            output_data = ""
            for platform, content in content_pieces.items():
                output_data += f"\n=== {platform.upper()} ===\n{content}\n"
        
        # Output results
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                if args.output_format == 'yaml':
                    f.write(output_data)
                elif args.output_format == 'json':
                    json.dump(output_data, f, indent=2)
                else:
                    f.write(output_data)
            print_success(f"Results saved to {args.output_file}")
        else:
            if args.output_format == 'json':
                print(json.dumps(output_data, indent=2))
            elif args.output_format == 'yaml':
                print(output_data)
            else:
                print(output_data)
        
        print_success(f"Generated content for {len(content_pieces)} platform(s)")
        
    except ValidationError as e:
        print_error("Input validation failed:")
        for error in e.errors:
            print_error(f"  {error.field}: {error.message}")
        sys.exit(1)
    except Exception as e:
        log_error(e, {'command': 'run', 'args': vars(args)})
        raise CLIError(f"Content generation failed: {e}")


def cmd_config_generate(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Create a configuration template file at the path specified by args.output.
    
    If a file already exists at that path, prompts the user to confirm overwrite (expects 'y' to proceed).
    On success prints a confirmation and a short guidance message. On failure logs context and raises CLIError.
    
    Parameters:
        args (argparse.Namespace): Command arguments; must include `output` (path to write the template).
        config (Dict[str, Any]): Current application configuration (not used to generate the template but provided for command consistency).
    
    Raises:
        CLIError: If template generation fails for any reason.
    """
    try:
        output_path = Path(args.output)
        
        if output_path.exists():
            response = input(f"File {args.output} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print_info("Configuration generation cancelled")
                return
        
        # Generate configuration template
        create_config_template(str(output_path))
        
        print_success(f"Configuration template created at {args.output}")
        print_info("Edit the file to customize your settings")
        
    except Exception as e:
        log_error(e, {'command': 'config_generate', 'args': vars(args)})
        raise CLIError(f"Failed to generate configuration: {e}")


def cmd_config_validate(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Validate a configuration file and display a brief summary.
    
    Validates the configuration located at the path specified by args.file using load_config().
    On success prints a confirmation and a short summary (number of sections) and renders
    a small table of example keys/values when Rich's console is available; otherwise lists section names.
    
    Parameters:
        args (argparse.Namespace): Command-line namespace; must provide `file` (path to config file).
        config (dict): Active application configuration (passed through command dispatch, not modified).
    
    Raises:
        CLIError: If the config file does not exist or validation fails.
    """
    try:
        config_file = Path(args.file)
        
        if not config_file.exists():
            raise CLIError(f"Configuration file not found: {args.file}")
        
        # Load and validate configuration
        test_config = load_config(str(config_file))
        
        print_success("Configuration file is valid")
        print_info(f"Loaded {len(test_config)} configuration sections")
        
        # Show some key settings
        if console:
            table = Table(title="Configuration Summary")
            table.add_column("Section", style="cyan")
            table.add_column("Key", style="magenta")
            table.add_column("Value", style="green")
            
            for section, values in test_config.items():
                if isinstance(values, dict):
                    for key, value in list(values.items())[:3]:  # Show first 3 items
                        table.add_row(section, key, str(value))
                else:
                    table.add_row(section, "", str(values))
            
            console.print(table)
        else:
            print("Configuration sections:")
            for section in test_config.keys():
                print(f"  - {section}")
        
    except Exception as e:
        log_error(e, {'command': 'config_validate', 'args': vars(args)})
        raise CLIError(f"Configuration validation failed: {e}")


def cmd_config_show(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Print the current application configuration to stdout in JSON or YAML format.
    
    If args.format == 'json', the configuration dictionary is serialized with json.dumps(indent=2, default=str).
    Otherwise the configuration is serialized as YAML via yaml.dump. Writes the resulting text to stdout.
    
    Parameters:
        args: argparse.Namespace with at least a `format` attribute ('json' for JSON output; any other value yields YAML).
        config: Mapping of configuration values to be serialized and displayed.
    
    Raises:
        CLIError: If serialization or printing fails.
    """
    try:
        if args.format == 'json':
            output = json.dumps(config, indent=2, default=str)
        else:
            output = yaml.dump(config, default_flow_style=False, default_style='')
        
        print(output)
        
    except Exception as e:
        log_error(e, {'command': 'config_show', 'args': vars(args)})
        raise CLIError(f"Failed to show configuration: {e}")


def cmd_validate(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Validate an input JSON or YAML file against the shared store schema and print a brief summary.
    
    This command handler:
    - Ensures the input file exists.
    - Determines format from --format or the file extension ('.yaml' / '.yml' -> YAML, '.json' -> JSON).
    - Loads the file and validates its contents with validate_shared_store().
    - Prints a short summary of discovered task requirements (platform count and topic).
    
    Behavior:
    - On schema validation failures (ValidationError) it prints per-field errors and exits the process with code 1.
    - On other unexpected failures it logs context and raises CLIError.
    """
    try:
        input_file = Path(args.input)
        
        if not input_file.exists():
            raise CLIError(f"Input file not found: {args.input}")
        
        # Determine format
        if args.format:
            file_format = args.format
        else:
            if input_file.suffix.lower() in ['.yaml', '.yml']:
                file_format = 'yaml'
            elif input_file.suffix.lower() == '.json':
                file_format = 'json'
            else:
                raise CLIError("Could not determine file format. Use --format to specify.")
        
        # Load file
        with open(input_file, 'r', encoding='utf-8') as f:
            if file_format == 'yaml':
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Validate data
        print_info("Validating data structure...")
        validate_shared_store(data)
        
        print_success("Data validation passed")
        
        # Show summary
        task_req = data.get('task_requirements', {})
        platforms = task_req.get('platforms', [])
        topic = task_req.get('topic_or_goal', '')
        
        print_info(f"Found {len(platforms)} platform(s): {', '.join(platforms)}")
        print_info(f"Topic: {topic[:50]}{'...' if len(topic) > 50 else ''}")
        
    except ValidationError as e:
        print_error("Data validation failed:")
        for error in e.errors:
            print_error(f"  {error.field}: {error.message}")
        sys.exit(1)
    except Exception as e:
        log_error(e, {'command': 'validate', 'args': vars(args)})
        raise CLIError(f"Validation failed: {e}")


def cmd_health(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Run a system health check and report status for key components.
    
    Performs lightweight checks for optional dependencies (Gradio, Rich), configuration loading, data validation, and cache initialization, then prints a summary table (uses Rich when available). If all checks pass prints a success message; if any check fails prints a warning and exits the process with status code 1.
    
    Parameters:
        args (argparse.Namespace): Parsed CLI arguments (used only for logging context on errors).
        config (dict): Application configuration; used for testing components such as cache initialization.
    
    Raises:
        CLIError: If an unexpected error occurs while performing the health check.
    """
    try:
        print_info("Performing system health check...")
        
        health_status = {
            'gradio': GRADIO_AVAILABLE,
            'rich': RICH_AVAILABLE,
            'config': True,
            'validation': True,
            'logging': True,
            'cache': True
        }
        
        # Test configuration
        try:
            test_config = load_config()
            health_status['config'] = True
        except Exception as e:
            health_status['config'] = False
            print_warning(f"Configuration: {e}")
        
        # Test validation
        try:
            test_data = {
                "task_requirements": {
                    "platforms": ["twitter"],
                    "topic_or_goal": "Test topic"
                },
                "brand_bible": {"xml_raw": ""}
            }
            validate_shared_store(test_data)
            health_status['validation'] = True
        except Exception as e:
            health_status['validation'] = False
            print_warning(f"Validation: {e}")
        
        # Test cache
        try:
            initialize_cache(config['flow'])
            cache_manager = get_cache_manager()
            health_status['cache'] = cache_manager is not None
        except Exception as e:
            health_status['cache'] = False
            print_warning(f"Cache: {e}")
        
        # Display results
        if console:
            table = Table(title="System Health Check")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            
            for component, status in health_status.items():
                status_text = "✓ OK" if status else "✗ FAILED"
                table.add_row(component.title(), status_text)
            
            console.print(table)
        else:
            print("System Health Check:")
            for component, status in health_status.items():
                status_text = "✓ OK" if status else "✗ FAILED"
                print(f"  {component.title()}: {status_text}")
        
        # Overall status
        all_healthy = all(health_status.values())
        if all_healthy:
            print_success("All systems are healthy")
        else:
            print_warning("Some systems have issues")
            sys.exit(1)
        
    except Exception as e:
        log_error(e, {'command': 'health', 'args': vars(args)})
        raise CLIError(f"Health check failed: {e}")


def cmd_info(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Print system and environment information to the user.
    
    Displays application version, Python version, platform, architecture, processor, and availability of optional dependencies. When Rich is available it renders a formatted table to the global console; otherwise it writes a plain-text listing to stdout.
    
    Parameters:
        args (argparse.Namespace): Parsed CLI arguments (used only for logging context on error).
        config (dict): Application configuration (not used by this command).
    
    Raises:
        CLIError: If gathering or displaying system information fails.
    """
    try:
        import platform
        import sys
        
        info = {
            'version': '1.0.0',
            'python_version': sys.version,
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'dependencies': {
                'gradio': GRADIO_AVAILABLE,
                'rich': RICH_AVAILABLE,
                'yaml': True,
                'json': True
            }
        }
        
        if console:
            table = Table(title="System Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in info.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        table.add_row(f"{key}.{subkey}", str(subvalue))
                else:
                    table.add_row(key, str(value))
            
            console.print(table)
        else:
            print("System Information:")
            for key, value in info.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
        
    except Exception as e:
        log_error(e, {'command': 'info', 'args': vars(args)})
        raise CLIError(f"Failed to get system info: {e}")


def main():
    """
    CLI entry point — parse arguments, load config, set up logging, and dispatch subcommands.
    
    This function:
    - Builds and parses the command-line arguments.
    - Loads application configuration and initializes logging.
    - Creates a request logging context and dispatches to the selected subcommand handler
      (serve, run, config [generate|validate|show], validate, health, info).
    - Honors the --no-color flag by disabling Rich console colors if available.
    
    Behavior and exit codes:
    - If no subcommand is provided or an invalid subcommand is used, prints help and exits with code 1.
    - On a handled CLIError, prints a user-friendly error message and exits with code 1.
    - On KeyboardInterrupt, prints a cancellation message and exits with code 130.
    - On any other unexpected exception, logs the error, prints an error message, optionally prints a traceback when --verbose is set, and exits with code 1.
    
    Side effects:
    - May write to stdout/stderr (help, informational and error messages).
    - Terminates the process via sys.exit(...) on error conditions or when help is shown.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle no command
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Disable colors if requested
    if args.no_color and console:
        console.no_color = True
    
    try:
        # Load configuration
        config = load_config(args.config_file)
        
        # Setup logging
        setup_logging(args, config)
        
        # Create request context for logging
        with create_request_context():
            log_info("CLI command started", {
                'command': args.command,
                'args': vars(args)
            })
            
            # Execute command
            if args.command == 'serve':
                cmd_serve(args, config)
            elif args.command == 'run':
                cmd_run(args, config)
            elif args.command == 'config':
                if args.config_command == 'generate':
                    cmd_config_generate(args, config)
                elif args.config_command == 'validate':
                    cmd_config_validate(args, config)
                elif args.config_command == 'show':
                    cmd_config_show(args, config)
                else:
                    parser.print_help()
                    sys.exit(1)
            elif args.command == 'validate':
                cmd_validate(args, config)
            elif args.command == 'health':
                cmd_health(args, config)
            elif args.command == 'info':
                cmd_info(args, config)
            else:
                parser.print_help()
                sys.exit(1)
            
            log_info("CLI command completed successfully")
    
    except CLIError as e:
        print_error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print_info("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        log_error(e, {'command': args.command if hasattr(args, 'command') else 'unknown'})
        print_error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()