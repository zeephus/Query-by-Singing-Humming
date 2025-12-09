import os
import librosa
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# prase files from the dataset
def parse_dataset(corpus_full_path, midi_folder_name, wav_folder_name, query_list_name):
    """
    Parses the query list to find valid wav/MIDI pairs
    Returns a list of dictionaries: [{'wav': path, 'mid': path, 'id': id}, ...]
    """

    query_path = os.path.join(corpus_full_path, query_list_name)
    
    if not os.path.exists(query_path):
        print(f"ERROR: Could not find list at {query_path}")
        return []

    print(f"Parsing list: {query_list_name}...")
    dataset = []
    
    with open(query_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                
                # clean wav paths
                raw_wav_path = parts[0].replace('\\', '/')
                wav_filename = raw_wav_path.split('/')[-1] 
                
                # clean midi id/filenmame
                midi_id = parts[1]
                if not midi_id.endswith('.mid'):
                    midi_filename = f"{midi_id}.mid"
                else:
                    midi_filename = midi_id
                    midi_id = midi_id.replace('.mid', '')

                # make full paths
                full_wav = os.path.join(corpus_full_path, wav_folder_name, wav_filename)
                full_mid = os.path.join(corpus_full_path, midi_folder_name, midi_filename)
                
                # only add if both paths exist
                if os.path.exists(full_wav) and os.path.exists(full_mid):
                    dataset.append({
                        'wav': full_wav,
                        'mid': full_mid,
                        'id': midi_id
                    })
    
    print(f"Found {len(dataset)} valid pairs.")
    return dataset

# audio feature extraction 
def extract_audio_f0(audio_path):
    print(f"Loading Audio: {os.path.basename(audio_path)}...", flush=True)
    # Using sr=None (uses 8k)
    y, sr = librosa.load(audio_path, sr=None)
    
    print(f"Extracting Audio Pitch (pYIN)...", flush=True)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr,
                                                 fmin=librosa.note_to_hz('C2'), 
                                                 fmax=librosa.note_to_hz('C5'))
    if f0 is not None:
        f0 = scipy.signal.medfilt(f0, kernel_size=5)

    return f0, sr, y
    return f0, sr, y

# MIDI extraction
def extract_midi_f0(midi_path, target_sr, hop_length=512):
    """
    Extracts pitch from MIDI but formats it to look exactly like the Audio F0.
    target_sr: Must match the sample rate of the audio file for alignment.
    """
    print(f"Loading MIDI: {os.path.basename(midi_path)}...", flush=True)
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Failed to load MIDI: {e}")
        return None, None

    # calculate length in frames
    total_time = pm.get_end_time()
    n_frames = int(total_time * target_sr / hop_length) + 1
    
    # construct arrays 
    times = np.arange(n_frames) * hop_length / target_sr
    f0_midi = np.full(n_frames, np.nan)
    
    # extraxt notes
    if len(pm.instruments) > 0:
        instrument = pm.instruments[0]
        for note in instrument.notes:
            # map time to frames array
            start_frame = int(note.start * target_sr / hop_length)
            end_frame = int(note.end * target_sr / hop_length)
            
            # pitch to frequency
            freq = librosa.midi_to_hz(note.pitch)
            
            # Fpopulate array
            end_frame = min(end_frame, n_frames)
            f0_midi[start_frame:end_frame] = freq
            
    return f0_midi, times


# MAIN
# set directory
base_dir = os.path.abspath(os.path.join(os.getcwd(), 'QBSH_Final', 'IOACAS_QBH_Corpus', 'IOACAS_pt1'))

# parse data
pairs = parse_dataset(base_dir, 'midfile', 'wavfile', 'query.list')

if len(pairs) > 0:
    # pick the 1st pair to analyze
    target_pair = pairs[0] 
    
    # run analysis
    a_f0, a_sr, a_y = extract_audio_f0(target_pair['wav'])
    
    # B. get midi data (with matching sr)
    m_f0, m_times = extract_midi_f0(target_pair['mid'], target_sr=a_sr)

    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot Audio
    plt.subplot(2, 1, 1)
    a_times = librosa.times_like(a_f0, sr=a_sr)
    plt.plot(a_times, a_f0, color='red', label='Humming (Audio)')
    plt.title(f"Query: {os.path.basename(target_pair['wav'])}")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    
    # Plot MIDI
    plt.subplot(2, 1, 2)
    plt.plot(m_times, m_f0, color='green', linewidth=2, label='Ground Truth (MIDI)')
    plt.title(f"Target: {os.path.basename(target_pair['mid'])}")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.xlim(0, a_times[-1] + 1)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
else:
    print("No valid pairs found to process.")
