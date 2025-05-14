from drasdic.util.raven_util import frames_to_st_dict
from drasdic.data.test import get_test_dataloader
from drasdic.evaluation.dcase_evaluation import evaluate
from tqdm import tqdm
import torch
import yaml
import os
import pandas as pd
from glob import glob

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    import warnings
    warnings.warn("Only using CPU! Check CUDA")

def fill_holes(m, max_hole):
    stops = (m[:-1] & ~m[1:]).nonzero(as_tuple=True)[0]  # indices where holes start

    for stop in stops:
        look_forward = m[stop + 1 : stop + 1 + max_hole]
        if torch.any(look_forward):
            next_start = (look_forward.nonzero(as_tuple=True)[0].min() + stop + 1).item()
            m[stop : next_start] = True

    return m

def delete_short(m, min_pos):
    starts = (m[1:] & ~m[:-1]).nonzero(as_tuple=True)[0] + 1

    clips = []

    for start in starts:
        look_forward = m[start:]
        ends = (~look_forward).nonzero(as_tuple=True)[0]
        if len(ends) > 0:
            clips.append((start.item(), start.item() + ends.min().item()))
            
    if m[0]:
        look_forward = m
        ends = (~look_forward).nonzero(as_tuple=True)[0]
        if len(ends) > 0:
            clips.append((0, ends.min().item()))

    # Create a new empty tensor of the same size
    m_new = torch.zeros_like(m, dtype=torch.bool)

    # Add back valid segments
    for clip in clips:
        if clip[1] - clip[0] >= min_pos:
            m_new[clip[0] : clip[1]] = True

    return m_new

def evaluate_model(model, args, modelid = "X"):
    model.eval()
    model = model.to(device)
    all_outputs = []
    
    for test_directory in args['test_directories']:
        name = os.path.basename(test_directory)
        output_dir = os.path.join(args['experiment_dir'], f"test_outputs_{modelid}", name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        audio_fps = sorted(glob(os.path.join(test_directory, "*.wav")))
        for audio_fp in audio_fps:
            print(f"Inference for {audio_fp}")
            anno_fp = audio_fp.replace(".wav", ".csv")
            
            ########
            if "cached_eval" in args and args["cached_eval"]:
                anno_st = pd.read_csv(anno_fp)
                audiofilename = list(anno_st["Audiofilename"])[0]
                raven_fp = os.path.join(output_dir, audiofilename.replace(".wav", ".txt"))
                if os.path.exists(raven_fp):
                    print(f"Loading cached predictions from {raven_fp}")
                    df = pd.read_csv(raven_fp, sep='\t')

                    # Make DCASE version
                    df_dcase = pd.DataFrame({})
                    df_dcase["Audiofilename"] = df["Begin Time (s)"].map(lambda x : audiofilename)
                    df_dcase["Starttime"] = df["Begin Time (s)"]
                    df_dcase["Endtime"] = df["End Time (s)"]

                    all_outputs.append(df_dcase)
                    continue
            
            ########
            
            ensemble_logits = []
            
            if args['support_selection_method'] == "random":
                n_prompts = args['n_ensemble_prompts']
            elif args['support_selection_method'] == "fixed":
                n_prompts = args['n_shots']
            
            for prompt_n in range(n_prompts):
            
                test_dataloader = get_test_dataloader(audio_fp, anno_fp, args, seed=args['seed']+prompt_n, index=prompt_n)
                audiofilename = test_dataloader.dataset.audiofilename
                
                with torch.no_grad():
                    data_item = test_dataloader.dataset.get_support()

#                     #
#                     out_dir = '/home/jupyter/demo_audio'
#                     if not os.path.exists(out_dir):
#                         os.makedirs(out_dir)

#                     import torchaudio
#                     torchaudio.save(os.path.join(out_dir, os.path.basename(audio_fp)), support_audio.unsqueeze(0), 16000)
#                     df = frames_to_st_dict(support_anno)
#                     df = pd.DataFrame(df)
#                     df.to_csv(os.path.join(out_dir, os.path.basename(audio_fp).replace('.wav', '.txt')), sep='\t', index= False)

#                     #
                    support_audio = data_item["audio"]
                    support_anno = data_item["labels"]
                    support_feats = data_item["features"]

                    support_encoded = model.encode_support(support_audio.unsqueeze(0).to(device), support_anno.unsqueeze(0).to(device), support_feats.unsqueeze(0).to(device))

                    query_logits_windowed = []
                    for batch in test_dataloader:
                        support_encoded_tiled = torch.tile(support_encoded, (batch['query_audio'].size(0),1,1))
                        logits, _ = model.forward_with_precomputed_support(support_encoded_tiled, batch['query_audio'].to(device))
                        query_logits_windowed.append(logits)

                    query_logits_windowed = torch.cat(query_logits_windowed)
                    output_window_dur_samples = query_logits_windowed.size(-1) #int(args['window_len_sec'] * args['sr'] // model.downsample_factor)

                    first_quarterwindow_end_sample = output_window_dur_samples // 4
                    last_quarterwindow_start_sample = output_window_dur_samples - first_quarterwindow_end_sample

                    logits_firstquarterwindow = query_logits_windowed[0,:first_quarterwindow_end_sample]
                    logits_middlewindows = query_logits_windowed[:,first_quarterwindow_end_sample:last_quarterwindow_start_sample]
                    logits_middlewindows = torch.flatten(logits_middlewindows)
                    logits_lastquarterwindow = query_logits_windowed[-1, last_quarterwindow_start_sample:]

                    logits = torch.cat([logits_firstquarterwindow,logits_middlewindows,logits_lastquarterwindow])
                    ensemble_logits.append(logits)
            
            with torch.no_grad():
                logits = sum(ensemble_logits) / args['n_ensemble_prompts']
                
                preds = (logits>=0)
                max_hole = min(test_dataloader.dataset.min_support_vox_dur * 0.5, 1)
                max_hole_samples = int((max_hole * args['sr']) // model.downsample_factor)
                preds = fill_holes(preds, max_hole_samples)

                min_pos = min(test_dataloader.dataset.min_support_vox_dur * 0.5, 0.5)
                min_pos_samples = int((min_pos * args['sr']) // model.downsample_factor)
                preds = delete_short(preds, min_pos_samples)
                
            preds_st_dict = frames_to_st_dict(preds.to(torch.int)*2, sr=args['sr'] // model.downsample_factor)
            
            # Time shift
            df = pd.DataFrame(preds_st_dict)
            df["Begin Time (s)"] += test_dataloader.dataset.query_starttime
            df["End Time (s)"] += test_dataloader.dataset.query_starttime
            
            # Save off Raven st
            fn = os.path.join(output_dir, audiofilename.replace(".wav", ".txt"))
            df.to_csv(fn, sep='\t', index=False)
            
            # Make DCASE version
            df_dcase = pd.DataFrame({})
            df_dcase["Audiofilename"] = df["Begin Time (s)"].map(lambda x : audiofilename)
            df_dcase["Starttime"] = df["Begin Time (s)"]
            df_dcase["Endtime"] = df["End Time (s)"]
            
            all_outputs.append(df_dcase)
    
    preds_fp = os.path.join(args["experiment_dir"], f"test_outputs_{modelid}", "all_preds.csv")
    all_outputs = pd.concat(all_outputs)
    all_outputs.to_csv(preds_fp, index=False)
        
    evaluate(preds_fp, args["test_data_parent_dir"], args["name"], f"model_{modelid}", args["experiment_dir"])
