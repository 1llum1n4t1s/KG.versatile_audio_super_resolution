#!/usr/bin/env python3
import argparse
import logging
import os

from audiosr.sampling import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_SAMPLER,
    SUPPORTED_DISCRETIZATIONS,
    SUPPORTED_SAMPLERS,
)


def _ddim_eta(value):
    try:
        eta = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("DDIM eta must be a number") from exc
    if not eta == eta or eta in (float("inf"), float("-inf")):
        raise argparse.ArgumentTypeError("DDIM eta must be finite")
    if eta < 0:
        raise argparse.ArgumentTypeError("DDIM eta must not be negative")
    return eta


def _ddim_steps(value):
    try:
        steps = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("DDIM steps must be an integer") from exc
    if not 1 <= steps <= 1000:
        raise argparse.ArgumentTypeError("DDIM steps must be between 1 and 1000")
    return steps


DEFAULT_SUFFIX = "_AudioSR_Processed_48K"


def rate_suffix(sample_rate):
    """Return the default suffix for a result written at ``sample_rate``."""
    return f"_AudioSR_Processed_{f'{sample_rate / 1000:g}'.replace('.', '_')}K"


def build_parser():
    """Build the command-line parser without importing the model stack."""
    parser = argparse.ArgumentParser(
        description="Run AudioSR super resolution on one or more audio files."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i",
        "--input_audio_file",
        type=str,
        help="Input audio file for audio super resolution",
    )
    input_group.add_argument(
        "-il",
        "--input_file_list",
        type=str,
        help="A file that contains all audio files that need to perform audio super resolution",
    )

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        help="The path to save model output",
        default="./output",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The checkpoint you gonna use",
        default="basic",
        choices=["basic", "speech"],
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="The device for computation. If not specified, the script will automatically choose the device based on your environment.",
        default="auto",
    )
    parser.add_argument(
        "--ddim_steps",
        type=_ddim_steps,
        default=50,
        help="The sampling step for DDIM (1-1000)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=DEFAULT_SAMPLER,
        choices=list(SUPPORTED_SAMPLERS),
        help=(
            "Sampling algorithm. 'ddim' keeps the established schedule; "
            "'dpmpp2m' solves the probability-flow ODE and needs far fewer steps"
        ),
    )
    parser.add_argument(
        "--ddim_eta",
        type=_ddim_eta,
        default=1.0,
        help=(
            "How stochastic each DDIM update is: 0.0 is deterministic and 1.0 "
            "is fully ancestral. Only the ddim sampler uses it"
        ),
    )
    parser.add_argument(
        "--discretize",
        type=str,
        default=DEFAULT_DISCRETIZATION,
        choices=list(SUPPORTED_DISCRETIZATIONS),
        help=(
            "Timestep spacing. 'uniform' keeps the established schedule but "
            "stops short of the noisiest timestep when the step count divides "
            "1000; 'trailing' always reaches it"
        ),
    )
    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Strength of the low-pass audio conditioning (typically 2.5 to 5.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Change this value (any integer number) will lead to a different generation result.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        help="Suffix for the output file",
        default=DEFAULT_SUFFIX,
    )
    parser.add_argument(
        "--chunking",
        action="store_true",
        help="Enable chunking for long audio files.",
    )
    parser.add_argument(
        "--chunk_duration",
        type=int,
        default=15,
        help="Chunk duration in seconds for long audio processing.",
    )
    parser.add_argument(
        "--overlap_duration",
        type=int,
        default=2,
        help="Overlap duration in seconds for long audio processing.",
    )
    parser.add_argument(
        "--preserve_input_rate",
        action="store_true",
        help=(
            "Write the result at the input's own sample rate instead of 48 kHz, "
            "keeping whatever the input carried above 24 kHz. The model works at "
            "48 kHz and generates nothing above 24 kHz either way."
        ),
    )

    return parser


def main(argv=None):
    """Run AudioSR and return a process exit status."""
    args = build_parser().parse_args(argv)

    # Keep model imports out of module import and parser/help paths.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    import torch
    from audiosr import (
        build_model,
        get_time,
        read_list,
        restore_high_rate,
        save_wave,
        super_resolution,
        super_resolution_long_audio,
    )

    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(logging.WARNING)

    torch.set_float32_matmul_precision("high")
    save_path = os.path.join(args.save_path, get_time())
    os.makedirs(save_path, exist_ok=True)

    random_seed = args.seed
    sample_rate = 48000
    latent_t_per_second = 12.8
    guidance_scale = args.guidance_scale

    if args.input_file_list is not None:
        print("Generate audio based on the text prompts in %s" % args.input_file_list)
        # Ignore blank/whitespace-only lines so they cannot become input paths.
        files_todo = [path.strip() for path in read_list(args.input_file_list) if path.strip()]
    else:
        files_todo = [args.input_audio_file]

    if not files_todo:
        build_parser().error("input file list does not contain any paths")

    audiosr = build_model(model_name=args.model_name, device=args.device)

    for input_file in files_todo:
        if args.chunking:
            waveform = super_resolution_long_audio(
                audiosr,
                input_file,
                seed=random_seed,
                guidance_scale=guidance_scale,
                ddim_steps=args.ddim_steps,
                sampler=args.sampler,
                ddim_eta=args.ddim_eta,
                discretize=args.discretize,
                chunk_duration_s=args.chunk_duration,
                overlap_duration_s=args.overlap_duration,
            )
        else:
            waveform = super_resolution(
                audiosr,
                input_file,
                seed=random_seed,
                guidance_scale=guidance_scale,
                ddim_steps=args.ddim_steps,
                sampler=args.sampler,
                ddim_eta=args.ddim_eta,
                discretize=args.discretize,
                latent_t_per_second=latent_t_per_second,
            )
        output_rate = sample_rate
        if args.preserve_input_rate:
            waveform, output_rate = restore_high_rate(waveform, input_file)

        # The stock suffix names the model's own rate, which stops being the
        # output's rate as soon as the input's is preserved. An explicit
        # --suffix is the caller's choice and is left alone.
        suffix = args.suffix
        if suffix == DEFAULT_SUFFIX and output_rate != sample_rate:
            suffix = rate_suffix(output_rate)
        name = os.path.splitext(os.path.basename(input_file))[0] + suffix

        save_wave(
            waveform,
            inputpath=input_file,
            savepath=save_path,
            name=name,
            samplerate=output_rate,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
