import os
import sys
import shutil
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from moviepy.editor import VideoFileClip, AudioFileClip

# Importazioni specifiche di EchoMimicV2
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
from src.utils.util import save_videos_grid
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2
from src.models.dwpose.dwpose_detector import dwpose_detector as dwprocessor

# Configura ffmpeg
ffmpeg_path = './ffmpeg-4.4-amd64-static'
if ffmpeg_path not in os.getenv('PATH', ''):
    print("Aggiunta di ffmpeg al PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

# Imposta la configurazione
CONFIG_PATH = "./configs/prompts/infer.yaml"  # Usa infer_acc.yaml per la versione accelerata
MAX_SIZE = 768
WORKING_DIR = "./avatar_project"
os.makedirs(WORKING_DIR, exist_ok=True)

# ============= FUNZIONI DI UTILITÀ ================

def extract_frame_from_video(video_path, frame_number=30, output_path=None):
    """Estrae un frame specifico dal video e lo salva come immagine"""
    if output_path is None:
        output_path = f"{WORKING_DIR}/reference_frame.png"
    
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_number >= total_frames:
        frame_number = total_frames // 2  # Usa il frame a metà video
        print(f"Frame richiesto troppo grande. Uso il frame {frame_number} invece.")
    
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = video.read()
    
    if success:
        cv2.imwrite(output_path, frame)
        print(f"Frame salvato in {output_path}")
        return output_path
    else:
        print("Impossibile estrarre il frame")
        return None

def prepare_video_for_pose_extraction(video_path):
    """Prepara il video per l'estrazione delle pose"""
    video_dir = f"{WORKING_DIR}/video"
    os.makedirs(video_dir, exist_ok=True)
    
    # Copia il video nella directory appropriata
    video_filename = os.path.basename(video_path)
    destination_path = f"{video_dir}/{video_filename}"
    shutil.copy(video_path, destination_path)
    
    # Crea una directory per la versione a 24fps
    video_24fps_dir = f"{WORKING_DIR}/video_24fps"
    os.makedirs(video_24fps_dir, exist_ok=True)
    destination_24fps = f"{video_24fps_dir}/{video_filename}"
    
    # Converti il video a 24fps
    clip = VideoFileClip(destination_path)
    new_clip = clip.set_fps(24)
    audio = new_clip.audio
    if audio is not None:
        audio = audio.set_fps(16000)
        new_clip = new_clip.set_audio(audio)
    new_clip.write_videofile(destination_24fps, codec='libx264', audio_codec='aac')
    
    return destination_24fps

def resize_and_pad_param(imh, imw, max_size):
    """Calcola i parametri per ridimensionare e centrare l'immagine"""
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw/imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half-half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh/imw * imw_new))
        imh_new = max_size
        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half-half_h
        re = rb + imh_new
    
    return imh_new, imw_new, rb, re, cb, ce

def get_pose_params(detected_poses, height, width, max_size):
    """Calcola i parametri di posa dal video"""
    print('Elaborazione dei parametri di posa...')
    
    # Analisi preliminare delle pose
    w_min_all, w_max_all, h_min_all, h_max_all = [], [], [], []
    mid_all = []
    
    for num, detected_pose in enumerate(detected_poses):
        detected_poses[num]['num'] = num
        candidate_body = detected_pose['bodies']['candidate']
        score_body = detected_pose['bodies']['score']
        candidate_face = detected_pose['faces']
        score_face = detected_pose['faces_score']
        candidate_hand = detected_pose['hands']
        score_hand = detected_pose['hands_score']

        # Gestione del volto
        if candidate_face.shape[0] > 1:
            index = 0
            candidate_face = candidate_face[index]
            score_face = score_face[index]
            detected_poses[num]['faces'] = candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1])
            detected_poses[num]['faces_score'] = score_face.reshape(1, score_face.shape[0])
        else:
            candidate_face = candidate_face[0]
            score_face = score_face[0]

        # Gestione del corpo
        if score_body.shape[0] > 1:
            tmp_score = [score_body[k].mean() for k in range(0, score_body.shape[0])]
            index = np.argmax(tmp_score)
            candidate_body = candidate_body[index*18:(index+1)*18,:]
            score_body = score_body[index]
            score_hand = score_hand[(index*2):(index*2+2),:]
            candidate_hand = candidate_hand[(index*2):(index*2+2),:,:]
        else:
            score_body = score_body[0]
            
        all_pose = np.concatenate((candidate_body, candidate_face))
        all_score = np.concatenate((score_body, score_face))
        all_pose = all_pose[all_score>0.8]

        body_pose = np.concatenate((candidate_body,))
        mid_ = body_pose[1, 0]

        face_pose = candidate_face
        hand_pose = candidate_hand

        h_min, h_max = np.min(face_pose[:,1]), np.max(body_pose[:7,1])
        h_ = h_max - h_min
        
        mid_w = mid_
        w_min = mid_w - h_ // 2
        w_max = mid_w + h_ // 2
        
        w_min_all.append(w_min)
        w_max_all.append(w_max)
        h_min_all.append(h_min)
        h_max_all.append(h_max)
        mid_all.append(mid_w)

    # Calcolo dei parametri finali
    w_min = np.min(w_min_all)
    w_max = np.max(w_max_all)
    h_min = np.min(h_min_all)
    h_max = np.max(h_max_all)
    mid = np.mean(mid_all)

    margin_ratio = 0.25
    h_margin = (h_max-h_min)*margin_ratio
    
    h_min = max(h_min-h_margin*0.65, 0)
    h_max = min(h_max+h_margin*0.05, 1)

    h_new = h_max - h_min
    
    h_min_real = int(h_min*height)
    h_max_real = int(h_max*height)
    mid_real = int(mid*width)
    
    height_new = h_max_real-h_min_real+1
    width_new = height_new
    w_min_real = mid_real - width_new // 2
    if w_min_real < 0:
        w_min_real = 0
        width_new = mid_real * 2

    w_max_real = w_min_real + width_new
    w_min = w_min_real / width
    w_max = w_max_real / width

    imh_new, imw_new, rb, re, cb, ce = resize_and_pad_param(height_new, width_new, max_size)
    res = {
        'draw_pose_params': [imh_new, imw_new, rb, re, cb, ce], 
        'pose_params': [w_min, w_max, h_min, h_max],
        'video_params': [h_min_real, h_max_real, w_min_real, w_max_real],
    }
    return res

def align_reference_image(image_path, video_params, max_size):
    """Allinea l'immagine di riferimento in base ai parametri di posa"""
    img = cv2.imread(image_path)
    h_min_real, h_max_real, w_min_real, w_max_real = video_params
    
    # Crop dell'immagine
    img_cropped = img[h_min_real:h_max_real, w_min_real:w_max_real, :]
    
    # Ridimensiona e centra
    height, width = img_cropped.shape[:2]
    if height > width:
        new_height = max_size
        new_width = int(width * (new_height / height))
        img_resized = cv2.resize(img_cropped, (new_width, new_height))
        
        # Padding
        img_padded = np.zeros((max_size, max_size, 3), dtype=np.uint8)
        start_x = (max_size - new_width) // 2
        img_padded[:, start_x:start_x+new_width, :] = img_resized
    else:
        new_width = max_size
        new_height = int(height * (new_width / width))
        img_resized = cv2.resize(img_cropped, (new_width, new_height))
        
        # Padding
        img_padded = np.zeros((max_size, max_size, 3), dtype=np.uint8)
        start_y = (max_size - new_height) // 2
        img_padded[start_y:start_y+new_height, :, :] = img_resized
    
    aligned_path = f"{WORKING_DIR}/aligned_reference.png"
    cv2.imwrite(aligned_path, img_padded)
    print(f"Immagine allineata salvata in {aligned_path}")
    return aligned_path

def extract_poses_from_video(video_path, max_frames=120):
    """Estrae le pose dal video"""
    print(f"Estrazione delle pose da {video_path}")
    
    # Leggi il video
    vr = cv2.VideoCapture(video_path)
    fps = vr.get(cv2.CAP_PROP_FPS)
    total_frames = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    # Limitiamo il numero di frame se necessario
    num_frames = min(total_frames, max_frames)
    
    # Estrai i frame
    for i in range(num_frames):
        ret, frame = vr.read()
        if not ret:
            break
        frames.append(frame)
    
    height, width = frames[0].shape[:2]
    
    # Rileva le pose
    detected_poses = [dwprocessor(frame) for frame in frames]
    dwprocessor.release_memory()
    
    # Calcola i parametri delle pose
    pose_params = get_pose_params(detected_poses, height, width, MAX_SIZE)
    
    # Salva le pose in una directory
    pose_dir = f"{WORKING_DIR}/poses"
    os.makedirs(pose_dir, exist_ok=True)
    
    w_min, w_max, h_min, h_max = pose_params['pose_params']
    draw_pose_params = pose_params['draw_pose_params']
    
    for i, detected_pose in enumerate(detected_poses):
        if i >= max_frames:
            break
            
        # Normalizza le coordinate
        candidate_body = detected_pose['bodies']['candidate']
        candidate_face = detected_pose['faces'][0]
        candidate_hand = detected_pose['hands']
        
        candidate_body[:,0] = (candidate_body[:,0]-w_min)/(w_max-w_min)
        candidate_body[:,1] = (candidate_body[:,1]-h_min)/(h_max-h_min)
        candidate_face[:,0] = (candidate_face[:,0]-w_min)/(w_max-w_min)
        candidate_face[:,1] = (candidate_face[:,1]-h_min)/(h_max-h_min)
        candidate_hand[:,:,0] = (candidate_hand[:,:,0]-w_min)/(w_max-w_min)
        candidate_hand[:,:,1] = (candidate_hand[:,:,1]-h_min)/(h_max-h_min)
        
        detected_pose['bodies']['candidate'] = candidate_body
        detected_pose['faces'] = candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1])
        detected_pose['hands'] = candidate_hand
        detected_pose['draw_pose_params'] = draw_pose_params
        
        # Salva la posa come file npy
        np.save(f"{pose_dir}/{i}.npy", detected_pose)
    
    print(f"Pose estratte e salvate in {pose_dir}")
    return pose_dir, pose_params['video_params']

def initialize_echomimic(config_path, device="cuda", weight_dtype=torch.float16):
    """Inizializza i modelli di EchoMimicV2"""
    print("Inizializzazione di EchoMimicV2...")
    
    config = OmegaConf.load(config_path)
    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    
    # VAE
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path
    ).to(device, dtype=weight_dtype)
    
    # Reference UNet
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet"
    ).to(device, dtype=weight_dtype)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu")
    )
    
    # Denoising UNet
    denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs
    ).to(device, dtype=weight_dtype)
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False
    )
    
    # Pose Encoder
    pose_net = PoseEncoder(
        320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)
    ).to(device, dtype=weight_dtype)
    pose_net.load_state_dict(torch.load(config.pose_encoder_path))
    
    # Audio Processor
    audio_processor = load_audio_model(
        model_path=config.audio_model_path, device=device
    )
    
    # Scheduler
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)
    
    # Pipeline
    pipe = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_net,
        scheduler=scheduler
    ).to(device, dtype=weight_dtype)
    
    return pipe

def generate_avatar_video(
    pipe, 
    ref_image_path, 
    audio_path, 
    pose_dir,
    width=768, 
    height=768, 
    steps=30,  # Usa 6 per la versione accelerata
    cfg=2.5,
    fps=24,
    context_frames=12,
    context_overlap=3,
    seed=42
):
    """Genera il video finale dell'avatar"""
    print("Generazione del video dell'avatar...")
    
    # Configura il generatore di numeri casuali
    if seed is None:
        seed = np.random.randint(0, 1000000)
    generator = torch.manual_seed(seed)
    
    # Prepara la directory di output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"{WORKING_DIR}/outputs")
    save_dir.mkdir(exist_ok=True, parents=True)
    save_name = f"{save_dir}/{timestamp}"
    
    # Carica l'immagine di riferimento
    ref_image_pil = Image.open(ref_image_path).convert("RGB").resize((width, height))
    
    # Carica l'audio
    audio_clip = AudioFileClip(audio_path)
    
    # Determina la lunghezza del video
    max_length = min(
        int(audio_clip.duration * fps), 
        len(os.listdir(pose_dir))
    )
    
    # Carica le pose
    pose_list = []
    for index in range(max_length):
        tgt_musk = np.zeros((width, height, 3)).astype('uint8')
        tgt_musk_path = os.path.join(pose_dir, f"{index}.npy")
        
        if not os.path.exists(tgt_musk_path):
            print(f"File non trovato: {tgt_musk_path}")
            continue
            
        detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
        imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
        im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
        im = np.transpose(np.array(im), (1, 2, 0))
        tgt_musk[rb:re, cb:ce, :] = im

        tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
        pose_tensor = torch.Tensor(np.array(tgt_musk_pil)).to(
            dtype=torch.float16, device="cuda"
        ).permute(2, 0, 1) / 255.0
        pose_list.append(pose_tensor)
    
    # Stack delle pose
    poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
    
    # Taglia l'audio alla durata corretta
    audio_clip = audio_clip.set_duration(max_length / fps)
    
    # Genera il video
    video = pipe(
        ref_image_pil,
        audio_path,
        poses_tensor[:, :, :max_length, ...],
        width,
        height,
        max_length,
        steps,
        cfg,
        generator=generator,
        audio_sample_rate=16000,
        context_frames=context_frames,
        fps=fps,
        context_overlap=context_overlap,
        start_idx=0
    ).videos
    
    # Salva il video senza audio
    final_length = min(video.shape[2], poses_tensor.shape[2], max_length)
    video_sig = video[:, :, :final_length, :, :]
    
    save_videos_grid(
        video_sig,
        save_name + "_woa.mp4",
        n_rows=1,
        fps=fps
    )
    
    # Aggiungi l'audio
    video_clip = VideoFileClip(save_name + "_woa.mp4")
    video_clip = video_clip.set_audio(audio_clip)
    final_output = save_name + "_final.mp4"
    video_clip.write_videofile(final_output, codec="libx264", audio_codec="aac", threads=2)
    
    # Elimina il file intermedio
    os.remove(save_name + "_woa.mp4")
    
    print(f"Video dell'avatar generato con successo: {final_output}")
    return final_output

# ============ FUNZIONE PRINCIPALE ============

def create_avatar_from_video(
    video_path, 
    audio_path=None,  # Se None, usa l'audio del video
    frame_to_extract=30,
    use_accelerated=False,
    max_frames=120,
    steps=30,  # 30 per regolare, 6 per accelerato
    seed=42
):
    """Funzione principale per creare un avatar da un video"""
    print(f"Inizio creazione avatar dal video: {video_path}")
    
    # 1. Prepara il video per l'estrazione delle pose
    video_24fps = prepare_video_for_pose_extraction(video_path)
    
    # 2. Estrai un frame dal video per l'immagine di riferimento
    ref_frame_path = extract_frame_from_video(video_path, frame_number=frame_to_extract)
    
    # 3. Estrai le pose dal video
    pose_dir, video_params = extract_poses_from_video(video_24fps, max_frames=max_frames)
    
    # 4. Allinea l'immagine di riferimento
    aligned_ref_path = align_reference_image(ref_frame_path, video_params, MAX_SIZE)
    
    # 5. Usa l'audio del video se non è fornito uno specifico
    if audio_path is None:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        if audio_clip is None:
            print("Il video non contiene audio. Per favore fornisci un file audio separato.")
            return None
        audio_path = f"{WORKING_DIR}/extracted_audio.wav"
        audio_clip.write_audiofile(audio_path)
    
    # 6. Inizializza EchoMimicV2
    config_path = "./configs/prompts/infer_acc.yaml" if use_accelerated else "./configs/prompts/infer.yaml"
    pipe = initialize_echomimic(config_path)
    
    # 7. Genera il video finale
    steps = 6 if use_accelerated else 30
    output_video = generate_avatar_video(
        pipe,
        aligned_ref_path,
        audio_path,
        pose_dir,
        steps=steps,
        seed=seed
    )
    
    print(f"Avatar creato con successo! Video: {output_video}")
    return output_video

# Esempio di utilizzo della funzione
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python script.py percorso_al_video [percorso_audio] [frame_numero]")
        sys.exit(1)
        
    video_path = sys.argv[1]
    audio_path = sys.argv[2] if len(sys.argv) > 2 else None
    frame_numero = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    create_avatar_from_video(
        video_path=video_path,
        audio_path=audio_path,
        frame_to_extract=frame_numero,
        use_accelerated=True,  # Cambia a False per maggiore qualità (più lento)
        max_frames=120,
        seed=42
    )
