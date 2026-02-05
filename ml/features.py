import torch
import torchaudio
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80

class FeatureExtractor:
    def __init__(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
    def get_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Returns Log-Mel Spectrogram.
        Shape: (Channels, Mel_Bands, Time)
        """
        mel = self.mel_transform(waveform)
        log_mel = self.amplitude_to_db(mel)
        return log_mel

    def get_phase_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract Phase-aware features using Modified Group Delay or similar.
        For simplicity and stablity, we use the Instantaneous Frequency (diff of phase).
        
        Ref: 'Phase-aware speech enhancement' techniques often use cos(phase), sin(phase).
        Here we extract the Phase via STFT and return the Delta Phase (Instantaneous Frequency).
        """
        # STFT
        window = torch.hann_window(N_FFT).to(waveform.device)
        stft = torch.stft(
            waveform, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH, 
            window=window, 
            return_complex=True
        ) # Shape: (Batch, Freq, Time)
        
        # Angle
        phase = torch.angle(stft)
        
        # Univariate Phase Unwrapping (along time axis)
        # unwrap in torch is not always available in older versions, checking manually
        # If not, we just use raw phase or simple diff.
        # Group Delay is often defined as -d(phi)/d(omega) (freq axis)
        
        # Feature 1: Group Delay (derivative along Frequency axis)
        # We wrap diffs to range [-pi, pi]
        gd = phase[:, 1:, :] - phase[:, :-1, :]
        gd = (gd + np.pi) % (2 * np.pi) - np.pi
        
        # Resize to match Mel dimensions if we want to stack, 
        # OR we can just keep it as a separate head input.
        # For CNN, it's easier to Resize Group Delay to same Time dim (it is already) 
        # but Frequency dim is N_FFT/2 + 1 (513).
        # We can average bins to get 80 bands or just interpolate.
        
        gd_img = gd.unsqueeze(1) # Add channel dim: (Batch, 1, Freq, Time)
        
        # Resize frequency axis to match Mel (N_MELS)
        # Using bilinear interpolation
        gd_resized = torch.nn.functional.interpolate(
            gd_img, 
            size=(N_MELS, gd_img.shape[-1]), 
            mode='bilinear', 
            align_corners=False
        )
        
        return gd_resized.squeeze(1)

    def extract_all(self, waveform: torch.Tensor) -> dict:
        """
        Returns dictionary of features.
        """
        return {
            "mel": self.get_mel_spectrogram(waveform),
            "phase": self.get_phase_features(waveform)
        }

# Global instance
extractor = FeatureExtractor()
