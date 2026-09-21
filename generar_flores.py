from diffusers import StableDiffusionPipeline
import torch
import os

model_id = "stabilityai/stable-diffusion-2-1"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Usando: {device} ({dtype})")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
).to(device)

os.makedirs("salida", exist_ok=True)

# ---------- 1. CAMPO GENERAL ----------
prompt_campo = """
masterpiece, best quality, ultra detailed,
vast wildflower meadow at sunset,
pink cosmos flowers,
white daisies,
yellow coreopsis flowers,
purple lavender,
millions of flowers covering the landscape,
golden sunlight,
dramatic sunset sky,
orange and pink clouds,
distant mountains,
soft depth of field,
photorealistic,
cinematic lighting,
HDR,
vibrant colors,
nature photography,
8k,
sharp focus,
beautiful composition
"""

negative_prompt = """
low quality,
worst quality,
blurry,
pixelated,
cartoon,
painting,
drawing,
cgi,
3d render,
watermark,
text,
logo,
people,
buildings,
cars,
oversaturated,
deformed flowers,
duplicates,
ugly
"""

print("Generando campo general...")
image = pipe(
    prompt=prompt_campo,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    guidance_scale=8.5,
    width=1536,
    height=1024,
).images[0]
image.save("salida/campo_flores.png")
print("OK: salida/campo_flores.png")

# ---------- 2. VARIANTE PANORAMICA ----------
prompt2 = """
A breathtaking endless flower field at golden sunset,
pink cosmos flowers in foreground,
white daisies,
yellow wildflowers,
purple lavender clusters,
warm cinematic sunlight,
dramatic colorful sky,
extreme realism,
professional landscape photography,
ultra detailed petals,
depth of field,
natural colors,
HDR,
8k,
award winning nature photograph
"""

print("Generando variante panoramica...")
image2 = pipe(
    prompt=prompt2,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    guidance_scale=8.5,
    width=1536,
    height=1024,
).images[0]
image2.save("salida/campo_flores_2.png")
print("OK: salida/campo_flores_2.png")

# ---------- 3. FLORES INDIVIDUALES (catalogo) ----------
flowers = [
    {
        "name": "cosmos_rosa",
        "prompt": """
        single pink cosmos flower,
        isolated flower,
        white background,
        botanical photography,
        highly detailed petals,
        professional studio lighting,
        realistic flower catalog photo,
        8k
        """,
    },
    {
        "name": "margarita_blanca",
        "prompt": """
        single white daisy flower with yellow center,
        isolated flower,
        white background,
        botanical photography,
        highly detailed petals with dew drops,
        professional studio lighting,
        realistic flower catalog photo,
        8k
        """,
    },
    {
        "name": "flor_amarilla",
        "prompt": """
        single yellow coreopsis wildflower,
        isolated flower,
        white background,
        botanical photography,
        glossy yellow petals,
        highly detailed,
        professional studio lighting,
        realistic flower catalog photo,
        8k
        """,
    },
    {
        "name": "lavanda",
        "prompt": """
        single purple lavender spike,
        isolated flower,
        white background,
        botanical photography,
        highly detailed buds,
        professional studio lighting,
        realistic flower catalog photo,
        8k
        """,
    },
]

for f in flowers:
    print(f"Generando {f['name']}...")
    img = pipe(
        prompt=f["prompt"],
        negative_prompt=negative_prompt,
        num_inference_steps=35,
        guidance_scale=7.5,
        width=768,
        height=768,
    ).images[0]
    out = f"salida/flor_{f['name']}.png"
    img.save(out)
    print(f"OK: {out}")

print("Listo. Revisa la carpeta salida/")
