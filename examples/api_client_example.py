"""
Example API Client for MIDI Generator

This script demonstrates how to use the MIDI Generator API to transcribe audio files to MIDI.
"""

import requests
import time
import json
from pathlib import Path
import argparse


class MIDIGeneratorClient:
    """Client for interacting with the MIDI Generator API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self):
        """Check if the API server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def transcribe_audio(self, audio_path: str, **params):
        """
        Transcribe an audio file to MIDI.
        
        Args:
            audio_path: Path to audio file
            **params: Additional transcription parameters
            
        Returns:
            Job ID for tracking the transcription
        """
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio_file': f}
                response = requests.post(
                    f"{self.base_url}/transcribe",
                    files=files,
                    params=params
                )
                response.raise_for_status()
                return response.json()
        except requests.RequestException as e:
            print(f"Transcription request failed: {e}")
            return None
    
    def get_job_status(self, job_id: str):
        """Get the status of a transcription job."""
        try:
            response = requests.get(f"{self.base_url}/status/{job_id}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Status check failed: {e}")
            return None
    
    def download_midi(self, job_id: str, output_path: str):
        """Download the generated MIDI file."""
        try:
            response = requests.get(f"{self.base_url}/download/{job_id}")
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"MIDI file downloaded to: {output_path}")
            return True
        except requests.RequestException as e:
            print(f"Download failed: {e}")
            return False
    
    def transcribe_and_wait(self, audio_path: str, output_path: str = None, **params):
        """
        Transcribe audio and wait for completion.
        
        Args:
            audio_path: Path to audio file
            output_path: Path to save MIDI file (optional)
            **params: Additional transcription parameters
        """
        # Start transcription
        print(f"🎵 Starting transcription of: {audio_path}")
        result = self.transcribe_audio(audio_path, **params)
        
        if not result:
            print("❌ Failed to start transcription")
            return False
        
        job_id = result['job_id']
        print(f"📋 Job ID: {job_id}")
        print(f"📊 Status: {result['status']}")
        
        # Poll for completion
        while True:
            status = self.get_job_status(job_id)
            if not status:
                print("❌ Failed to get job status")
                return False
            
            print(f"📊 Status: {status['status']}")
            
            if status['status'] == 'completed':
                print(f"✅ Transcription completed!")
                print(f"🎼 Notes generated: {status.get('num_notes', 'unknown')}")
                print(f"⏱️ Processing time: {status.get('processing_time', 'unknown')}s")
                
                # Download MIDI file
                if not output_path:
                    audio_file = Path(audio_path)
                    output_path = audio_file.with_suffix('.mid')
                
                if self.download_midi(job_id, output_path):
                    print(f"💾 MIDI saved to: {output_path}")
                    return True
                else:
                    return False
            
            elif status['status'] == 'failed':
                print(f"❌ Transcription failed: {status.get('error', 'Unknown error')}")
                return False
            
            elif status['status'] in ['pending', 'processing']:
                progress = status.get('progress', 0) * 100
                print(f"🔄 Processing... {progress:.1f}%")
                time.sleep(2)  # Wait 2 seconds before checking again
            
            else:
                print(f"⚠️ Unknown status: {status['status']}")
                time.sleep(2)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="MIDI Generator API Client")
    parser.add_argument("audio_file", help="Path to audio file to transcribe")
    parser.add_argument("--output", "-o", help="Output MIDI file path")
    parser.add_argument("--server", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--onset-threshold", type=float, default=0.5, help="Onset threshold")
    parser.add_argument("--frame-threshold", type=float, default=0.5, help="Frame threshold")
    parser.add_argument("--velocity-scale", type=float, default=127.0, help="Velocity scale")
    parser.add_argument("--clip-length", type=float, default=4.0, help="Clip length")
    parser.add_argument("--profile", type=str, default=None, help="Decoding profile (e.g., bass, piano)")
    
    args = parser.parse_args()
    
    # Create client
    client = MIDIGeneratorClient(args.server)
    
    # Health check
    print("🔍 Checking API server health...")
    health = client.health_check()
    if not health:
        print("❌ API server is not responding")
        return 1
    
    print(f"✅ API server is healthy: {health['status']}")
    
    # Transcription parameters
    params = {
        'onset_threshold': args.onset_threshold,
        'frame_threshold': args.frame_threshold,
        'velocity_scale': args.velocity_scale,
        'clip_length': args.clip_length
    }
    
    # Transcribe
    success = client.transcribe_and_wait(args.audio_file, args.output, **params)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
