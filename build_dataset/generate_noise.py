import os
from argparse import ArgumentParser

import numpy as np
import soundfile as sf
from tqdm import tqdm


def parse_args():
    """
    Parses command-line arguments for configuring Gaussian noise generation.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - output_dir (str): Directory to save the generated audio files.
            - num_noise (int): Number of noise audio files to generate.
            - sampling_rate (int): Audio sampling rate in Hz.
            - duration (int): Duration of each audio file in seconds.
            - seed (int): Random seed for reproducibility.
            - sigma (float): Standard deviation (RMS) of the Gaussian noise.
    """
    parser = ArgumentParser(description="Generate Gaussian noise audio files.")

    parser.add_argument("--output_dir", type=str,
                        default="output", help="Directory to save output files.")
    parser.add_argument("--num_noise", type=int,
                        default=100000, help="Number of noise samples to generate.")
    parser.add_argument("--sampling_rate", type=int,
                        default=16000, help="Sampling rate in Hz.")
    parser.add_argument("--duration", type=int,
                        default=5, help="Duration of each noise sample in seconds.")
    parser.add_argument("--seed", type=int,
                        default=0, help="Random seed for reproducibility.")
    parser.add_argument("--sigma", type=float,
                        default=0.01, help="RMS")

    return parser.parse_args()


def generate_gaussian_noise(output_dir, num_noise, sampling_rate, duration, sigma, seed):
    """
    Parses command-line arguments for configuring Gaussian noise generation.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - output_dir (str): Directory to save the generated audio files.
            - num_noise (int): Number of noise audio files to generate.
            - sampling_rate (int): Audio sampling rate in Hz.
            - duration (int): Duration of each audio file in seconds.
            - seed (int): Random seed for reproducibility.
            - sigma (float): Standard deviation (RMS) of the Gaussian noise.
    """
    total_samples = sampling_rate * duration
    np.random.seed(seed)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(num_noise), desc="Generating Noise"):
        # Generate Gaussian noise
        gaussian_noise = np.random.randn(total_samples) * sigma
        
        # Save each noise sample
        output_filename = os.path.join(output_dir, f"gaussian_noise_{i}.wav")
        sf.write(output_filename, gaussian_noise, sampling_rate)

        
def main(args):
    """
    Main execution function that runs the noise generation with parsed arguments.

    Args:
        args (argparse.Namespace): Command-line arguments returned by `parse_args()`.
    """
    generate_gaussian_noise(
        args.output_dir,
        args.num_noise,
        args.sampling_rate,
        args.duration,
        args.sigma,
        args.seed
    )

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
