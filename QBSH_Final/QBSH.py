import os
import librosa
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
import time
import json 

# features and signal processing

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scipy.signal.lfilter(b, a, data)
    return y

def prep_hybrid_features(f0_array, source_type='audio', sr=22050, audio_signal=None):
    """
    Returns THREE feature vectors:
    1. feat_abs: absolute pitch 
    2. feat_rel: change in pitch (delta) 
    3. feat_rhythm: sibilance for onset information
    """
    s = pd.Series(f0_array)
    f0_clean = s.interpolate(limit_direction='both').to_numpy()
    if pd.isna(f0_clean).all() or np.all(f0_clean == 0): return None, None, None
    f0_clean[f0_clean == 0] = 1e-6 # avoid log(0)
    midi_notes = librosa.hz_to_midi(f0_clean)
    
    # absolute pitch 
    singing_range = midi_notes[midi_notes > librosa.note_to_midi('C2')]
    center = np.median(singing_range) if len(singing_range) > 0 else np.mean(midi_notes)
    feat_abs = midi_notes - center
    
    # change in pitch 
    feat_rel = np.diff(midi_notes, prepend=midi_notes[0])

    # sibilance
    feat_rhythm = np.zeros_like(midi_notes)

    if source_type == 'audio' and audio_signal is not None:
        # Extracts sibilance using ZCR
        zcr = librosa.feature.zero_crossing_rate(audio_signal, frame_length=1024, hop_length=512)[0]
        
        # match f0 length
        if len(zcr) > len(midi_notes): zcr = zcr[:len(midi_notes)]
        else: zcr = np.pad(zcr, (0, len(midi_notes) - len(zcr)))
        
        # normalize
        feat_rhythm = zcr / (np.max(zcr) + 1e-6)

    elif source_type == 'midi':
        # guess/synthesize onsets from pitch jumps
        pitch_jumps = np.abs(np.diff(midi_notes, prepend=midi_notes[0]))
        
        # if pitch jumpts > 1 semitone, mark as onset
        feat_rhythm[pitch_jumps > 1] = 1.0
        
        # slight smoothing
        feat_rhythm = scipy.ndimage.gaussian_filter1d(feat_rhythm, sigma=1)

    return feat_abs, feat_rel, feat_rhythm

#dynamic time warping cost calculation

def calculate_dtw_cost(query_f0, target_f0, query_audio=None, tolerance=1.0):
    # pass both 'audio' and 'midi' to the prep function
    q_abs, q_rel, q_rhy = prep_hybrid_features(query_f0, 'audio', audio_signal=query_audio)
    
    # prep target midi features
    t_abs, t_rel, t_rhy = prep_hybrid_features(target_f0, 'midi')
    
    if q_abs is None or t_abs is None: return float('inf')

    # run dtw on absolute pitch
    X = q_abs.reshape(1, -1)
    Y = t_abs.reshape(1, -1)
    try:
        D, wp = librosa.sequence.dtw(X, Y, subseq=True, metric='euclidean')
    except: return float('inf')

    # cost for pitch height
    w_q_abs, w_t_abs = q_abs[wp[:, 0]], t_abs[wp[:, 1]]
    cost_abs = np.mean(np.abs(w_q_abs - w_t_abs))

    # cost for pitch shape
    w_q_rel, w_t_rel = q_rel[wp[:, 0]], t_rel[wp[:, 1]]
    cost_rel = np.mean(np.abs(w_q_rel - w_t_rel))

    # cost for sibilance/onsets
    w_q_rhy, w_t_rhy = q_rhy[wp[:, 0]], t_rhy[wp[:, 1]]
    cost_rhy = np.mean(np.abs(w_q_rhy - w_t_rhy))

    # weighted final cost
    final_cost = (0.4 * cost_abs) + (0.4 * cost_rel) + (0.2 * cost_rhy)
    
    return final_cost

# extracting functions

def extract_audio_f0(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    except Exception as e:
        return None, None, None
    
    y_filtered = butter_lowpass_filter(y, cutoff=1200, fs=sr, order=4)
    
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_filtered, 
        sr=sr,
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C5'),
        frame_length=1024 
    )
    
    if f0 is not None: 
        midi_f0 = librosa.hz_to_midi(f0)
        delta = np.diff(midi_f0, prepend=midi_f0[0])
        f0[np.abs(delta) > 10.0] = np.nan 
        f0 = scipy.signal.medfilt(f0, kernel_size=3)
           
    return f0, sr, y   

def extract_midi_f0(midi_path, target_sr, hop_length=512):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        return None, None

    total_time = pm.get_end_time()
    n_frames = int(total_time * target_sr / hop_length) + 1
    times = np.arange(n_frames) * hop_length / target_sr
    f0_midi = np.full(n_frames, np.nan)
    
    if len(pm.instruments) > 0:
        instrument = pm.instruments[0]
        for note in instrument.notes:
            start_frame = int(note.start * target_sr / hop_length)
            end_frame = int(note.end * target_sr / hop_length)
            freq = librosa.midi_to_hz(note.pitch)
            end_frame = min(end_frame, n_frames)
            f0_midi[start_frame:end_frame] = freq
            
    return f0_midi, times

# parallel processing (to run faster)

def process_single_query(query_info, midi_database):

    q_f0, _, q_y = extract_audio_f0(query_info['wav'])

    if q_f0 is None or np.all(np.isnan(q_f0)):
        return query_info['id'], 9999, []

    scores = []
    for target in midi_database:
        if target['f0'] is None:
            scores.append({'id': target['id'], 'cost': float('inf')})
            continue
            
        cost = calculate_dtw_cost(q_f0, target['f0'], query_audio=q_y, tolerance=1.0)
        scores.append({'id': target['id'], 'cost': cost})

    scores.sort(key=lambda x: x['cost'])
    
    ground_truth = query_info['id']
    all_ids = [s['id'] for s in scores]
    
    try:
        rank = all_ids.index(ground_truth) + 1
    except ValueError:
        rank = 9999 

    # Only return top 5 scores to save space
    return ground_truth, rank, scores[:5] 

# caching and evaluation

def save_insights(ranks, total_queries, output_dir="results_graphs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_ranks = np.array([r for r in ranks if r < 9000])
    
    # MRR (MIREX standard) 
    recip_ranks = [1.0/r for r in valid_ranks]
    mrr = np.mean(recip_ranks) if recip_ranks else 0
    
    print(f"Mean reciprocal rank: {mrr:.4f}")
    print(f"Median rank: {np.median(valid_ranks) if len(valid_ranks)>0 else 'N/A'}")
    print(f"Saved graphs to folder: {output_dir}")

    # plot CMC curve and histograms
    plt.figure(figsize=(10, 6))
    max_rank = 200
    accuracies = []
    for r in range(1, max_rank + 1):
        acc = np.sum(valid_ranks <= r) / total_queries * 100
        accuracies.append(acc)
        

    # CMC Curve
    plt.plot(range(1, max_rank + 1), accuracies, linewidth=3, color='#2ca02c')
    plt.title(f'Cumulative Match Characteristic (Top-{max_rank})', fontsize=14)
    plt.xlabel('Rank', fontsize=12)
    plt.xlim(1, max_rank)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, ('CMC_Curve_Top_'+str(max_rank)+'.png')))
    plt.close()

    # rank distribution (first 50 ranks)
    plt.figure(figsize=(10, 6))
    sns.histplot([r for r in valid_ranks if r <= 50], bins=50, binrange=(1, 50), color='#1f77b4')
    plt.title('Rank Distribution (First 50 Ranks)', fontsize=14)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'Rank_Distribution.png'))
    plt.close()

    # log scale for failures
    plt.figure(figsize=(10, 6))
    plt.hist(valid_ranks, bins=50, color='salmon', log=True)
    plt.title('Log-Scale Distribution (Mapping Failures)', fontsize=14)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Count (Log Scale)', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'Failure_Analysis_Log.png'))
    plt.close()

def evaluate_system_smart(query_list, midi_directory, n_jobs=4):
    cache_file = "./QBSH_Final/results_cache.json"

    # check cache file 
    if os.path.exists(cache_file):
        print(f"\nLoading previous results from {cache_file}")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        results = cached_data['results']
        total_queries = len(results)
    else:
        # run processing
        print("Loading MIDI database")
        midi_database = []
        files = [f for f in os.listdir(midi_directory) if f.endswith('.mid')]
        for file in files:
            m_f0, _ = extract_midi_f0(os.path.join(midi_directory, file), target_sr=22050)
            if m_f0 is not None:
                midi_database.append({'id': file.replace('.mid', ''), 'f0': m_f0})

        print(f"\nProcessing {len(query_list)} queries (n_jobs={n_jobs})")
        start_time = time.time()
        
        # parallel run
        raw_results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(process_single_query)(q, midi_database) for q in query_list
        )
        
        # format for JSON
        results = []
        for truth, rank, scores in raw_results:
            serializable_scores = [{'id': s['id'], 'cost': float(s['cost'])} for s in scores]
            results.append([truth, int(rank), serializable_scores])

        # save cache
        with open(cache_file, 'w') as f:
            json.dump({'results': results}, f)
        print(f"\nResults saved to {cache_file}")
        
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        total_queries = len(results)

    # print report
    ranks = [r for (_, r, _) in results]
    
    top1 = sum(1 for r in ranks if r == 1)
    top10 = sum(1 for r in ranks if r <= 10)
    top50 = sum(1 for r in ranks if r <= 50)
    top100 = sum(1 for r in ranks if r <= 100)
    top200 = sum(1 for r in ranks if r<=200)
    
    print(f"\nFINAL RESULTS ({total_queries} Queries)")
    print(f"Top-1:   {top1/total_queries*100:.2f}%")
    print(f"Top-10:  {top10/total_queries*100:.2f}%")
    print(f"Top-50:  {top50/total_queries*100:.2f}%")
    print(f"Top-100: {top100/total_queries*100:.2f}%")
    print(f"Top-200: {top200/total_queries*100:.2f}%")
    
    save_insights(ranks, total_queries, output_dir="./QBSH_Final/results_graphs")

# parse dataset

def parse_dataset(corpus_full_path, midi_folder_name, wav_folder_name, query_list_name):
    query_path = os.path.join(corpus_full_path, query_list_name)
    dataset = []
    if not os.path.exists(query_path): return []
    with open(query_path, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                wav_filename = parts[0].replace('\\', '/').split('/')[-1] 
                midi_id = parts[1].replace('.mid', '')
                full_wav = os.path.join(corpus_full_path, wav_folder_name, wav_filename)
                full_mid = os.path.join(corpus_full_path, midi_folder_name, f"{midi_id}.mid")
                if os.path.exists(full_wav) and os.path.exists(full_mid):
                    dataset.append({'wav': full_wav, 'mid': full_mid, 'id': midi_id})
    return dataset

# main execution

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.getcwd(), 'QBSH_Final', 'IOACAS_QBH_Corpus', 'IOACAS_pt1'))
    midi_dir_path = os.path.join(base_dir, 'midfile')

    pairs = parse_dataset(base_dir, 'midfile', 'wavfile', 'query.list')

    if len(pairs) > 0:
        evaluate_system_smart(pairs, midi_dir_path, n_jobs=4)
    else:
        print("Error: No pairs found.")
