"""
MIDI Generator Inference Pipeline

This module provides inference capabilities for the MIDI Generator model,
converting audio stems to MIDI transcriptions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import pretty_midi
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass
import torchaudio
from ..audio.preprocessing import load_waveform, build_mel_transforms

# Resilient imports to support both package and module execution
try:
    # When run as a package (recommended): uv run python -m src.cli ...
    from ..models.interpretable_transcription.interpretable_transcription import TranscriptionModel
    from ..utils.logging import MIDIGeneratorLogger
except Exception as _rel_imp_err:  # pragma: no cover - fallback path
    try:
        # When PYTHONPATH includes project root or executed from repo root
        from src.models.interpretable_transcription.interpretable_transcription import TranscriptionModel
        from src.utils.logging import MIDIGeneratorLogger
    except Exception as _abs_imp_err:
        # Give a clear, actionable error message
        _msg = (
            "💥 attempted relative import with no known parent package\n\n"
            "How to fix:\n"
            "  1) Run commands as a module from project root:\n"
            "     uv run python -m src.cli transcribe <audio.wav> <model.pth>\n"
            "     uv run python -m src.cli api --port 8000\n"
            "  2) Or set PYTHONPATH to project root, e.g.:\n"
            "     export PYTHONPATH=.$PYTHONPATH\n\n"
            "If this persists, ensure the repository layout is intact and src/ is a package."
        )
        raise ImportError(_msg) from _abs_imp_err


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    model_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Match training dataset parameters
    sample_rate: int = 22050
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    win_length: Optional[int] = None  # Not used by torchaudio MelSpectrogram by default
    f_min: float = 30.0
    f_max: float = 11000.0
    top_db: float = 80.0
    clip_length: float = 4.0  # seconds
    overlap: float = 0.5  # overlap between clips
    onset_threshold: float = 0.03
    frame_threshold: float = 0.08
    # Hysteresis for frame continuation
    frame_threshold_off: float = 0.04
    velocity_scale: float = 127.0
    min_note_duration: float = 0.02  # minimum note duration in seconds
    # Evaluation-like decoding controls
    smooth_window: int = 5  # odd window for mean smoothing over frames (per pitch)
    min_ioi: float = 0.05   # minimum inter-onset interval in seconds (per pitch)
    merge_gap: float = 0.03 # merge same-pitch notes if gap < this (seconds)
    pitch_min: Optional[int] = None
    pitch_max: Optional[int] = None
    enable_fallback_segmentation: bool = False
    # Output metadata
    instrument_name: str = "Piano"


class MIDIInferenceEngine:
    """
    Main inference engine for converting audio stems to MIDI.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = MIDIGeneratorLogger("MIDIInferenceEngine")
        self.model = None
        self.device = torch.device(config.device)
        # Initialize transforms via shared factory to match training exactly
        self._mel_transform, self._amp_to_db = build_mel_transforms(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            top_db=self.config.top_db,
        )
        
        self.logger.info(f" Initializing MIDI Inference Engine on device: {self.device}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model from checkpoint."""
        try:
            self.logger.info(f" Loading model from: {self.config.model_path}")
            
            # Initialize model architecture
            self.model = TranscriptionModel(
                n_mels=self.config.n_mels,
                cnn_channels=64,
                lstm_hidden=256,
                num_layers=2
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.success(" Model loaded successfully!")
            
        except Exception as e:
            self.logger.error(f" Failed to load model: {str(e)}")
            raise
    
    def _preprocess_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Preprocess audio file to mel spectrogram.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Mel spectrogram as numpy array [n_mels, time_frames]
        """
        try:
            self.logger.info(f" Processing audio: {audio_path}")
            
            # Load waveform using shared utility (mono + resample)
            wav = load_waveform(Path(audio_path), self.config.sample_rate)  # [1, T]
            # Mel spectrogram then dB
            spec = self._mel_transform(wav).squeeze(0)  # [n_mels, time]
            spec_db = self._amp_to_db(spec)
            mel_np = spec_db.numpy()
            
            self.logger.info(f" Mel spectrogram shape: {mel_np.shape}")
            return mel_np
            
        except Exception as e:
            self.logger.error(f" Audio preprocessing failed: {str(e)}")
            raise
    
    def _split_into_clips(self, mel_spec: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Split mel spectrogram into overlapping clips for inference.
        
        Args:
            mel_spec: Full mel spectrogram [n_mels, time_frames]
            
        Returns:
            List of (clip, start_time) tuples
        """
        clips = []
        
        # Calculate clip parameters
        frames_per_second = self.config.sample_rate / self.config.hop_length
        clip_frames = int(self.config.clip_length * frames_per_second)
        hop_frames = int(clip_frames * (1 - self.config.overlap))
        
        total_frames = mel_spec.shape[1]
        
        # Extract clips
        start_frame = 0
        while start_frame < total_frames:
            end_frame = min(start_frame + clip_frames, total_frames)
            
            # Pad if necessary
            clip = mel_spec[:, start_frame:end_frame]
            if clip.shape[1] < clip_frames:
                padding = clip_frames - clip.shape[1]
                clip = np.pad(clip, ((0, 0), (0, padding)), mode='constant')
            
            start_time = start_frame / frames_per_second
            clips.append((clip, start_time))
            
            if end_frame >= total_frames:
                break
                
            start_frame += hop_frames
        
        self.logger.info(f" Split into {len(clips)} clips of {self.config.clip_length}s each")
        return clips
    
    def _run_inference(self, mel_clip: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run model inference on a single mel spectrogram clip.
        
        Args:
            mel_clip: Mel spectrogram clip [n_mels, time_frames]
            
        Returns:
            Dictionary with 'onset', 'frame', 'velocity' predictions
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            mel_tensor = torch.from_numpy(mel_clip).float().unsqueeze(0).to(self.device)
            
            # Run inference
            outputs = self.model(mel_tensor)
            
            # Apply sigmoid to get probabilities (model outputs logits)
            onset_probs = torch.sigmoid(outputs['onset']).cpu().numpy()[0]  # [time, 128]
            frame_probs = torch.sigmoid(outputs['frame']).cpu().numpy()[0]  # [time, 128]
            # Velocity head trained on 0..1 targets; clamp into [0,1]
            velocity = outputs['velocity'].cpu().numpy()[0]
            velocity = np.clip(velocity, 0.0, 1.0)
            
            return {
                'onset': onset_probs,
                'frame': frame_probs,
                'velocity': velocity
            }
    
    def _postprocess_predictions(self, predictions: List[Tuple[Dict, float]]) -> Dict[str, np.ndarray]:
        """
        Combine predictions from multiple clips into full-length arrays.
        
        Args:
            predictions: List of (prediction_dict, start_time) tuples
            
        Returns:
            Combined predictions dictionary
        """
        if not predictions:
            raise ValueError("No predictions to combine")
        
        # Calculate total length
        frames_per_second = self.config.sample_rate / self.config.hop_length
        last_pred, last_start = predictions[-1]
        total_frames = int((last_start + self.config.clip_length) * frames_per_second)
        
        # Initialize combined arrays
        combined_onset = np.zeros((total_frames, 128))
        combined_frame = np.zeros((total_frames, 128))
        combined_velocity = np.zeros((total_frames, 128))
        weights = np.zeros((total_frames, 128))
        
        # Combine predictions with overlap handling
        for pred_dict, start_time in predictions:
            start_frame = int(start_time * frames_per_second)
            clip_frames = pred_dict['onset'].shape[0]
            end_frame = min(start_frame + clip_frames, total_frames)
            
            # Add predictions with weights
            frame_slice = slice(start_frame, end_frame)
            pred_slice = slice(0, end_frame - start_frame)
            
            combined_onset[frame_slice] += pred_dict['onset'][pred_slice]
            combined_frame[frame_slice] += pred_dict['frame'][pred_slice]
            combined_velocity[frame_slice] += pred_dict['velocity'][pred_slice]
            weights[frame_slice] += 1
        
        # Average overlapping regions
        mask = weights > 0
        combined_onset[mask] /= weights[mask]
        combined_frame[mask] /= weights[mask]
        combined_velocity[mask] /= weights[mask]
        
        return {
            'onset': combined_onset,
            'frame': combined_frame,
            'velocity': combined_velocity
        }
    
    def _predictions_to_midi(self, predictions: Dict[str, np.ndarray]) -> pretty_midi.PrettyMIDI:
        """
        Convert model predictions to MIDI file.
        
        Args:
            predictions: Dictionary with onset, frame, velocity predictions
            
        Returns:
            PrettyMIDI object
        """
        # Use a common resolution DAWs expect (ticks per beat)
        midi = pretty_midi.PrettyMIDI(resolution=480)
        # Name the instrument track from config for DAW friendliness
        instrument = pretty_midi.Instrument(program=0, name=self.config.instrument_name, is_drum=False)
        
        # Time resolution
        frames_per_second = self.config.sample_rate / self.config.hop_length
        time_per_frame = 1.0 / frames_per_second
        
        # Extract notes for each pitch
        # Optional pitch range constraint
        pitch_lo = self.config.pitch_min if self.config.pitch_min is not None else 0
        pitch_hi = self.config.pitch_max if self.config.pitch_max is not None else 127

        # Simple mean smoothing over frames to reduce chatter
        def _smooth(x: np.ndarray, k: int) -> np.ndarray:
            if k <= 1:
                return x
            k = int(k)
            if k % 2 == 0:
                k += 1
            pad = k // 2
            xpad = np.pad(x, (pad, pad), mode='edge')
            kernel = np.ones(k, dtype=np.float32) / float(k)
            y = np.convolve(xpad, kernel, mode='valid')
            return y.astype(np.float32)

        for pitch in range(pitch_lo, pitch_hi + 1):
            onset_probs = predictions['onset'][:, pitch]
            frame_probs = _smooth(predictions['frame'][:, pitch], self.config.smooth_window)
            velocity_vals = predictions['velocity'][:, pitch]
            
            # Find onset peaks
            onsets = self._find_onsets(onset_probs, self.config.onset_threshold)
            
            # Find note boundaries
            notes = self._extract_notes(
                onsets, frame_probs, velocity_vals, 
                pitch, time_per_frame, self.config.frame_threshold
            )

            # Fallback segmentation disabled by default; enable only if explicitly requested
            if not notes and self.config.enable_fallback_segmentation:
                th_on = self.config.frame_threshold
                th_off = getattr(self.config, 'frame_threshold_off', max(0.0, th_on * 0.5))
                n = len(frame_probs)
                i = 0
                while i < n:
                    if frame_probs[i] >= th_on:
                        start = i
                        below = 0
                        j = i + 1
                        while j < n:
                            if frame_probs[j] >= th_off:
                                below = 0
                            else:
                                below += 1
                                if below >= 2:
                                    break
                            j += 1
                        end = j
                        start_time = start * time_per_frame
                        end_time = max(start_time + self.config.min_note_duration, end * time_per_frame)
                        seg_vel = float(np.max(velocity_vals[start:max(start+1, end)]))
                        velocity = int(np.clip(seg_vel * self.config.velocity_scale, 1, 127))
                        notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time))
                        i = max(i + 1, end)
                        continue
                    i += 1
            
            instrument.notes.extend(notes)

        midi.instruments.append(instrument)

        # Remove very short notes
        instrument.notes = [note for note in instrument.notes 
                       if note.end - note.start >= self.config.min_note_duration]
        
        # Merge near-adjacent same-pitch notes if gap < merge_gap
        if self.config.merge_gap > 0 and instrument.notes:
            instrument.notes.sort(key=lambda n: (n.pitch, n.start))
            merged = []
            prev = instrument.notes[0]
            for cur in instrument.notes[1:]:
                if cur.pitch == prev.pitch and cur.start - prev.end <= self.config.merge_gap:
                    # extend previous
                    prev.end = max(prev.end, cur.end)
                    prev.velocity = max(prev.velocity, cur.velocity)
                else:
                    merged.append(prev)
                    prev = cur
            merged.append(prev)
            instrument.notes = merged
        
        # Sort notes by start time
        instrument.notes.sort(key=lambda x: x.start)
        
        # Ensure DAW-friendly track: if no notes, add a harmless control change
        if len(instrument.notes) == 0:
            # Add a pan control change (center) at time 0.0 to materialize the track
            instrument.control_changes.append(
                pretty_midi.ControlChange(number=10, value=64, time=0.0)
            )
            self.logger.warning("No notes detected; writing an empty track with a control change for DAW compatibility")
        self.logger.info(f" Generated MIDI with {len(instrument.notes)} notes")
        return midi
    
    def _find_onsets(self, onset_probs: np.ndarray, threshold: float) -> List[int]:
        """Find onset threshold crossings (rising edges) above threshold."""
        onsets: List[int] = []
        above = onset_probs >= threshold
        prev = False
        last_onset = -10**9
        # time per frame
        tpf = self.config.hop_length / self.config.sample_rate
        min_frames_apart = max(1, int(self.config.min_ioi / tpf))
        for i, cur in enumerate(above):
            if cur and not prev:
                if i - last_onset >= min_frames_apart:
                    onsets.append(i)
                    last_onset = i
            prev = cur
        return onsets
    
    def _extract_notes(self, onsets: List[int], frame_probs: np.ndarray, 
                      velocity_vals: np.ndarray, pitch: int, 
                      time_per_frame: float, frame_threshold: float) -> List[pretty_midi.Note]:
        """Extract notes from onsets and frame predictions using hysteresis and min-duration enforcement."""
        notes: List[pretty_midi.Note] = []
        n = len(frame_probs)
        # thresholds
        th_on = frame_threshold
        th_off = getattr(self.config, 'frame_threshold_off', max(0.0, frame_threshold * 0.5))
        min_frames = max(1, int(self.config.min_note_duration / time_per_frame))

        for onset_frame in onsets:
            # If the frame probability at onset is very low, skip
            if frame_probs[onset_frame] < th_on and frame_probs[onset_frame] < th_off:
                continue

            # Walk forward until it drops below off-threshold for a couple frames
            end_frame = onset_frame + 1
            below_count = 0
            while end_frame < n:
                if frame_probs[end_frame] >= th_off:
                    below_count = 0
                else:
                    below_count += 1
                    # require two consecutive below-threshold frames to end the note
                    if below_count >= 2:
                        break
                end_frame += 1

            # Enforce minimum duration by extending end_frame if needed
            if end_frame - onset_frame < min_frames:
                end_frame = min(n - 1, onset_frame + min_frames)

            # Calculate timing
            start_time = onset_frame * time_per_frame
            end_time = max(start_time + self.config.min_note_duration, end_frame * time_per_frame)

            # Calculate velocity from onset frame
            vel_val = float(velocity_vals[onset_frame])
            velocity = int(np.clip(vel_val * self.config.velocity_scale, 1, 127))

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            notes.append(note)

        return notes
    
    def transcribe_audio(self, audio_path: Union[str, Path], 
                        output_path: Optional[Union[str, Path]] = None) -> pretty_midi.PrettyMIDI:
        """
        Main transcription function: convert audio to MIDI.
        
        Args:
            audio_path: Path to input audio file
            output_path: Optional path to save MIDI file
            
        Returns:
            PrettyMIDI object
        """
        try:
            self.logger.info(f"🎵 Starting transcription: {audio_path}")
            
            # Preprocess audio
            mel_spec = self._preprocess_audio(audio_path)
            
            # Split into clips
            clips = self._split_into_clips(mel_spec)
            
            # Run inference on each clip
            predictions = []
            for i, (clip, start_time) in enumerate(clips):
                self.logger.info(f"🔄 Processing clip {i+1}/{len(clips)}")
                pred = self._run_inference(clip)
                predictions.append((pred, start_time))
            
            # Combine predictions
            combined_predictions = self._postprocess_predictions(predictions)
            
            # Convert to MIDI
            midi = self._predictions_to_midi(combined_predictions)
            
            # Save if output path provided
            if output_path:
                midi.write(str(output_path))
                self.logger.success(f"💾 MIDI saved to: {output_path}")
            
            self.logger.success("✅ Transcription completed successfully!")
            return midi
            
        except Exception as e:
            self.logger.error(f"❌ Transcription failed: {str(e)}")
            raise


def create_inference_engine(model_path: str, **kwargs) -> MIDIInferenceEngine:
    """
    Factory function to create inference engine.
    
    Args:
        model_path: Path to trained model checkpoint
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured MIDIInferenceEngine
    """
    config = InferenceConfig(model_path=model_path, **kwargs)
    return MIDIInferenceEngine(config)
