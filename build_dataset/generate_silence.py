import os
from argparse import ArgumentParser

import numpy as np
import soundfile as sf
from tqdm import tqdm


def parse_args():
    """
    Parses command-line arguments for generating silent audio files.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - output_dir (str): Directory to save the generated silence audio files.
            - num_silence (int): Number of silent audio files to generate.
            - sampling_rate (int): Sampling rate of the audio files (in Hz).
            - duration (int): Duration of each audio file (in seconds).
    """
    parser = ArgumentParser(description="Generate Silence audio files.")

    parser.add_argument("--output_dir", type=str,
                        default="output", help="Directory to save output files.")
    parser.add_argument("--num_silence", type=int,
                        default=100000, help="Number of silence samples to generate.")
    parser.add_argument("--sampling_rate", type=int,
                        default=16000, help="Sampling rate in Hz.")
    parser.add_argument("--duration", type=int,
                        default=5, help="Duration of each silence sample in seconds.")

    return parser.parse_args()


def generate_silence(output_dir, num_silence, sampling_rate, duration):
    """
    Parses command-line arguments for generating silent audio files.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - output_dir (str): Directory to save the generated silence audio files.
            - num_silence (int): Number of silent audio files to generate.
            - sampling_rate (int): Sampling rate of the audio files (in Hz).
            - duration (int): Duration of each audio file (in seconds).
    """
    total_samples = sampling_rate * duration
    silence = np.zeros(total_samples)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(num_silence), desc="Generating Silence"):
        # Save each silent audio file
        output_filename = os.path.join(output_dir, f"silence_{i}.wav")
        sf.write(output_filename, silence, sampling_rate)


def main(args):
    """
    Main function to initiate silence generation using parsed arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments from `parse_args()`.
    """
    generate_silence(
        args.output_dir,
        args.num_silence,
        args.sampling_rate,
        args.duration,
    )

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
