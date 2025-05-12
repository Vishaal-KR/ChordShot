from models.color_analysis import extract_dominant_colors
from models.Object_detection import detect_objects
from models.scene_recog_pretrained import predict_image_category

from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torchaudio
import torch
import os

print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Load model and processor ONCE
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small",
    torch_dtype=torch.float16,    # Use float16 for faster generation
).to(device)

def color_mood_descriptor(color):
    mapping = {
        "red": ("intense", ["electric guitar", "drums", "synth bass"]),
        "orange": ("vibrant", ["brass", "conga", "funk guitar"]),
        "yellow": ("bright", ["acoustic guitar", "xylophone", "ukulele"]),
        "green": ("fresh", ["flute", "bongo drums", "acoustic bass"]),
        "blue": ("cool", ["saxophone", "synth pad", "electric piano"]),
        "purple": ("bold", ["synth lead", "electronic drums", "bass guitar"]),
        "black": ("edgy", ["distorted guitar", "808 drums", "synth bass"]),
        "white": ("crisp", ["piano", "clean electric guitar", "hi-hats"]),
        "gray": ("gritty", ["lo-fi beats", "vinyl crackle", "sub-bass"]),
        "pink": ("playful", ["toy piano", "bells", "claps"]),
        "brown": ("earthy", ["djembe", "acoustic guitar", "woodwind"]),
        "gold": ("luxurious", ["harp", "string ensemble", "grand piano"]),
        "silver": ("futuristic", ["arpeggiator synth", "modular synth", "electronic percussion"]),
        "beige": ("warm", ["rhodes piano", "soft pads", "brush drums"]),
        "teal": ("calm", ["ambient synths", "kalimba", "slow strings"]),
        "cyan": ("refreshing", ["marimba", "synth plucks", "handpan"]),
        "magenta": ("exciting", ["lead synth", "snare drums", "bass synth"]),
    }
    return mapping.get(color.lower(), ("balanced", ["drums", "bass", "piano"]))


def build_dynamic_music_prompt(scene, colors, objects):
    primary_color = colors[0] if colors else "neutral"
    mood_adj, tempo_desc = color_mood_descriptor(primary_color)

    object_descriptions = {
        "waves": "flowing ambient pads and crashing reverb effects",
        "birds": "light melodic flutes and chirping arpeggios",
        "trees": "wooden percussion and rustling textures",
        "clouds": "airy synths and echoing chimes",
        "cars": "gritty industrial beats and rhythmic revs",
        "people": "vocal chops and human ambiance",
        "stars": "twinkling bell tones and celestial synths",
        "fire": "crackling textures and intense drums",
    }

    musical_elements = [
        object_descriptions.get(obj.lower(), f"musical textures inspired by {obj}")
        for obj in objects
    ]
    element_line = ", ".join(musical_elements)

    # Style tag by scene
    scene_styles = {
        "forest": "ambient acoustic",
        "city": "urban electronic",
        "beach": "tropical house",
        "space": "ethereal synth",
        "battlefield": "cinematic orchestral",
        "rainy street": "lo-fi chillhop",
    }
    style = scene_styles.get(scene.lower(), "experimental")

    prompt = (f"Create a {style} music track that reflects a {scene.lower()} setting. "
        f"The mood should be {mood_adj}, with {tempo_desc} and instrumentation such as {element_line}. "
        f"Avoid heavy drum beats unless necessary. Use natural, expressive instruments that match the scene's emotion and visuals."
    )
    return prompt

def generate_music_from_image(image_path, output_path="static/music/music_from_image.wav", max_new_tokens=768):
    # Step 1: Extract image features
    scene = predict_image_category(image_path)
    colors = extract_dominant_colors(image_path)
    objects = detect_objects(image_path)

    # Step 2: Build prompt
    prompt = build_dynamic_music_prompt(scene, colors, objects)
    print("🎵 Music Prompt:", prompt)

    # Step 3: Generate music
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Step 4: Move audio back to CPU and save
    audio_tensor = audio_values.cpu()

# Fix shape: (batch, channels, time) -> (channels, time)
    if audio_tensor.dim() == 3:
        audio_tensor = audio_tensor.squeeze(0)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

# 🔥 Important: Convert to float32
    audio_tensor = audio_tensor.to(torch.float32)

# Save WAV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, audio_tensor, 16000)
    print(f"✅ Music saved to: {output_path}")

    return output_path
