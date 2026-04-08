"""
PNG → ICO converter
Pioche tous les PNG dans le dossier IMG_FOLDER et génère les .ico au même endroit
"""

from PIL import Image
import os

# ── Config ──────────────────────────────────────────────
IMG_FOLDER = r"E:\OneDrive\Documents\Programmation\Projets\Tools\tools\ICO\img"

ICO_SIZES = [
    (32, 32),
    # (48, 48),
    # (64, 64),
    # (128, 128),
    # (256, 256),
]
# ────────────────────────────────────────────────────────


def png_to_ico(png_path: str) -> None:
    output_path = os.path.splitext(png_path)[0] + ".ico"

    img = Image.open(png_path).convert("RGBA")
    img = Image.open(png_path).convert("RGBA")
    img = img.crop(img.getbbox())  # ← crop auto du padding transparent

    sizes = []
    for size in ICO_SIZES:
        resized = img.resize(size, Image.LANCZOS)
        sizes.append(resized)

    sizes[0].save(
        output_path,
        format="ICO",
        sizes=ICO_SIZES,
        append_images=sizes[1:]
    )

    print(f"  ✓ {os.path.basename(png_path)} → {os.path.basename(output_path)}")


def main():
    if not os.path.isdir(IMG_FOLDER):
        print(f"[ERREUR] Dossier introuvable : {IMG_FOLDER}")
        return

    png_files = [f for f in os.listdir(IMG_FOLDER) if f.lower().endswith(".png")]

    if not png_files:
        print("Aucun PNG trouvé dans le dossier.")
        return

    print(f"→ {len(png_files)} PNG trouvé(s), conversion en cours...\n")

    for filename in png_files:
        full_path = os.path.join(IMG_FOLDER, filename)
        try:
            png_to_ico(full_path)
        except Exception as e:
            print(f"  [ERREUR] {filename} : {e}")

    print("\nDone !")


if __name__ == "__main__":
    main()