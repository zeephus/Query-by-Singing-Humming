import os

def parse_corpus(corpus_path, wav_subfolder='wavfile', midi_subfolder='midfile', list_filename='query.list'):
    """
    Parses a QBSH corpus folder with flexible subfolder names.
    
    Args:
        corpus_path (str): The root path to the corpus.
        wav_subfolder (str): Name of the folder containing wav files (default: 'wavfile').
        midi_subfolder (str): Name of the folder containing midi files (default: 'midfile').
        list_filename (str): Name of the text file mapping queries to MIDI (default: 'query.list').
                           
    Returns:
        list: A list of tuples, where each tuple is (wav_full_path, midi_full_path)
    """
    
    # 1. Construct paths using the arguments
    wav_dir = os.path.join(corpus_path, wav_subfolder)
    midi_dir = os.path.join(corpus_path, midi_subfolder)
    query_list_path = os.path.join(corpus_path, list_filename)
    
    dataset = []

    # 2. Check if the list file exists
    if not os.path.exists(query_list_path):
        print(f"Error: Could not find query list at {query_list_path}")
        return []

    print(f"Parsing corpus at: {corpus_path}...")

    # 3. Read the mapping file
    with open(query_list_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue 
            
            # Split the line. (Assumes format: "wav_filename   midi_id")
            parts = line.split() 
            
            if len(parts) >= 2:
                wav_name = parts[0]
                midi_id = parts[1]
                
                # 4. Handle Extensions logic (Robustness check)
                if not wav_name.lower().endswith('.wav'):
                    wav_name += '.wav'
                
                if not midi_id.lower().endswith('.mid'):
                    midi_file_name = midi_id + '.mid'
                else:
                    midi_file_name = midi_id

                # 5. Construct absolute paths
                full_wav_path = os.path.join(wav_dir, wav_name)
                full_midi_path = os.path.join(midi_dir, midi_file_name)

                # 6. Verify files exist
                if os.path.exists(full_wav_path) and os.path.exists(full_midi_path):
                    dataset.append((full_wav_path, full_midi_path))
                # else:
                #     print(f"Warning: File not found for pair: {wav_name} <-> {midi_file_name}")

    print(f"Successfully loaded {len(dataset)} pairs from {corpus_path}")
    return dataset

# --- Main Execution ---

if __name__ == "__main__":
    # --- 1. Parse IOACAS Corpus (Uses the defaults) ---
    path_ioacas = os.path.join("IOACAS_QBH_Corpus", "IOACAS_pt1")
    dataset_ioacas = parse_corpus(path_ioacas) 
    # Defaults used: wav_subfolder='wavfile', midi_subfolder='midfile'

    
    # --- 2. Parse ThinkIT Corpus (Overrides the defaults) ---
    # Assuming the folder is named 'ThinkIT_Corpus' inside your 'Final' folder
    path_thinkit = "ThinkIT_Corpus" 
    
    dataset_thinkit = parse_corpus(
        path_thinkit, 
        wav_subfolder='Query',      # Matches ThinkIT structure
        midi_subfolder='REFMIDI',   # Matches ThinkIT structure
        list_filename='query.list'  # Matches ThinkIT structure
    )

    # --- 3. Combine them for your experiment ---
    full_dataset = dataset_ioacas + dataset_thinkit
    
    print(f"\nTotal items in combined dataset: {len(full_dataset)}")
    
    if len(full_dataset) > 0:
        print("Example pair from combined set:")
        print(full_dataset[-1]) # Prints an item from the ThinkIT set