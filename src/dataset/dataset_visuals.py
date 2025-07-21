from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import Normalize


def plot_character_surp_overlay(
    text_ds,
    input_file: Path | str,
    output_file: Path | str,
    division_factor_space: int,
    cmap_name="YlOrRd",
    alpha=0.8,
):
    """
    1) Make a white background
    2) Draw character‐level boxes (filled w/ semi‐transparent color)
    3) Paste the text image (which has transparent background) on top
    4) Add colorbar and save
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = text_ds.copy()
    if df["text_id"].nunique() != 1:
        raise ValueError("plot_character_surp_overlay expects exactly one text_id")

    # Convert normalized coords to pixels
    for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
        df[c] *= division_factor_space

    # Load text‐only PNG as RGBA (transparent behind black text)
    im_rgba = Image.open(input_path)
    if im_rgba.mode != "RGBA":
        im_rgba = im_rgba.convert("RGBA")
    width, height = im_rgba.size

    # Create a matplotlib figure matching the PNG size
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # Create a flat white background (H×W×3 for RGB)
    white_bg = np.full((height, width, 3), 255, dtype=np.uint8)

    # Show background first
    ax.imshow(white_bg, zorder=1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Prepare color mapping for char_level_surp
    vmin, vmax = df["char_level_surp"].min(), df["char_level_surp"].max()
    if vmax == vmin:
        vmin -= 1e-3
        vmax += 1e-3
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # Draw colored boxes behind the text
    for _, row in df.iterrows():
        x1, y1 = row["bbox_x1"], row["bbox_y1"]
        w = row["bbox_x2"] - x1
        h = row["bbox_y2"] - y1
        color = cmap(norm(row["char_level_surp"]))
        patch = patches.Rectangle(
            (x1, y1), w, h, linewidth=0, facecolor=color, alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    # Now place the text PNG on top
    ax.imshow(im_rgba, zorder=3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.01)
    cbar.set_label("char_level_surp")

    # Save
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    return str(output_path)


def plot_word_surp_overlay(
    text_ds,
    input_file: Path | str,
    output_file: Path | str,
    division_factor_space: int,
    cmap_name="YlOrRd",
    alpha=0.8,
):
    """
    1) Make a white background
    2) Draw word boxes (semi‐transparent color)
    3) Paste the original text PNG over them
    4) Add colorbar legend
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = text_ds.copy()
    if df["text_id"].nunique() != 1:
        raise ValueError("plot_word_surp_overlay expects exactly one text_id")

    # Convert normalized coords → pixels
    for c in ("word_bbox_x1", "word_bbox_y1", "word_bbox_x2", "word_bbox_y2"):
        df[c] *= division_factor_space

    # Load the text PNG in RGBA
    im_rgba = Image.open(input_path)
    if im_rgba.mode != "RGBA":
        im_rgba = im_rgba.convert("RGBA")
    width, height = im_rgba.size

    # Figure sized to match
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # White background array
    white_bg = np.full((height, width, 3), 255, dtype=np.uint8)
    ax.imshow(white_bg, zorder=1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Surprisal color mapping
    vmin, vmax = df["surp"].min(), df["surp"].max()
    if vmax == vmin:
        vmin -= 1e-3
        vmax += 1e-3
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # Draw boxes behind text
    for _, row in df.iterrows():
        x1, y1 = row["word_bbox_x1"], row["word_bbox_y1"]
        w = row["word_bbox_x2"] - x1
        h = row["word_bbox_y2"] - y1
        color = cmap(norm(row["surp"]))
        patch = patches.Rectangle(
            (x1, y1), w, h, linewidth=0, facecolor=color, alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    # Text image on top
    ax.imshow(im_rgba, zorder=3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.01)
    cbar.set_label("word_surprisal")

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    return str(output_path)


def plot_fixation_overlay(
    fix_ds: pd.DataFrame,
    text_ds: pd.DataFrame,
    input_file: str,
    output_file: str,
    division_factor_space: int,
) -> str:
    """
    Overlays word bounding boxes and fixation points on an image.

    Parameters:
    - fix_ds: DataFrame with columns ['fixid','x','y','text_id'].
    - text_ds: DataFrame with columns ['word_bbox_x1','word_bbox_y1',
               'word_bbox_x2','word_bbox_y2','text_id'].
    - input_file: path to the source image (e.g. Item_01.png).
    - output_dir: directory where the annotated image will be saved.

    Returns:
    - Path to the saved image.
    """

    # Expect a single text_id in fixations
    text_ids = fix_ds["text_id"].unique()
    if len(text_ids) != 1:
        raise ValueError(f"Expected one text_id, got {text_ids}")
    text_id = text_ids[0]

    # Subset word‐boxes and fixations
    sbs_text = text_ds[text_ds["text_id"] == text_id].copy()
    sbs_mc = fix_ds.copy()

    # Re‐scale to pixel coords
    for c in ["word_bbox_x1", "word_bbox_y1", "word_bbox_x2", "word_bbox_y2"]:
        sbs_text[c] *= division_factor_space
    sbs_mc["x"] *= division_factor_space
    sbs_mc["y"] *= division_factor_space

    # Unique valid boxes
    cols = ["word_bbox_x1", "word_bbox_y1", "word_bbox_x2", "word_bbox_y2"]
    valid_boxes = sbs_text[cols].dropna().drop_duplicates()

    # Load image & flatten alpha onto white
    im = Image.open(input_file)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, (0, 0), im)

    draw = ImageDraw.Draw(bg)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    # Draw word‐boxes
    for _, row in valid_boxes.iterrows():
        box = (
            row["word_bbox_x1"],
            row["word_bbox_y1"],
            row["word_bbox_x2"],
            row["word_bbox_y2"],
        )
        draw.rectangle(box, outline="red", width=2)

    # Draw fixations and IDs
    for _, row in sbs_mc[["fixid", "x", "y"]].iterrows():
        x, y = row["x"], row["y"]
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill="blue", outline="blue")
        draw.text((x + 5, y - 5), str(int(row["fixid"])), fill="blue", font=font)

    bg.save(output_file)
    return str(output_file)


def batch_plot_fixations(
    meco_df: pd.DataFrame,
    text_df: pd.DataFrame,
    images_dir: Path | str,
) -> None:
    """
    For each reader/text combo in meco_df, overlay fixations on the corresponding image.

    - meco_df: must contain columns ['reader','text','fixid','x','y','text_id'].
    - text_df: must contain columns ['text_id','word_bbox_x1','word_bbox_y1',
                                    'word_bbox_x2','word_bbox_y2',…].
    - images_dir: path to the folder containing Item_XX.png files.
    """
    from .dataset_visuals import plot_fixation_overlay

    images_dir = Path(images_dir)
    # create sibling output folder:
    out_dir = images_dir.parent / "texts_en_images_processed"
    out_dir.mkdir(exist_ok=True)

    readers = meco_df["reader"].drop_duplicates().tolist()
    texts = meco_df["text"].drop_duplicates().tolist()

    for rdr in readers:
        for txt in texts:
            # subset fixations & text‐boxes
            fix_sbs = meco_df[(meco_df["reader"] == rdr) & (meco_df["text"] == txt)]
            if fix_sbs.empty:
                continue
            text_sbs = text_df[text_df["text_id"] == txt]
            out_text_dir = out_dir / f"text_{txt}"
            out_text_dir.mkdir(exist_ok=True)
            out_file = out_text_dir / f"Item_{int(rdr):02d}_{int(txt):02d}_scanpath.png"
            img_file = images_dir / f"Item_{int(txt):02d}.png"
            # build image path (assumes files named Item_01.png, Item_02.png, …)
            if not img_file.is_file():
                print(
                    f"Warning: {img_file} not found, skipping reader={rdr}, text={txt}"
                )
                continue
            # call the overlay function
            saved = plot_fixation_overlay(
                fix_ds=fix_sbs.rename(
                    columns={"text": "text_id"}
                ),  # ensure column name
                text_ds=text_sbs,
                input_file=img_file,
                output_file=out_file,
            )
            print(f"Annotated image saved to {saved}")


def batch_plot_character_surp(
    text_df: pd.DataFrame,
    images_dir: Path | str,
    division_factor_space,
    out_suffix: str = "char_surp",
) -> None:
    """
    For each text_id, overlay character boxes colored by char_level_surp.
    """
    images_dir = Path(images_dir)
    out_dir = images_dir.parent / f"texts_en_images_{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tid in sorted(text_df["text_id"].unique()):
        subset = text_df[text_df["text_id"] == tid]
        img = images_dir / f"Item_{int(tid):02d}.png"
        if not img.exists():
            print(f"⚠️ {img} missing, skipping {tid}")
            continue
        outf = out_dir / f"Item_{int(tid):02d}_{out_suffix}.png"
        saved = plot_character_surp_overlay(subset, img, outf, division_factor_space)
        print(f"→ {saved}")


def batch_plot_word_surp(
    text_df: pd.DataFrame,
    images_dir: Path | str,
    division_factor_space: int,
    out_suffix: str = "word_surp",
) -> None:
    """
    For each text_id, overlay word boxes colored by word‐level 'surp'.
    """
    images_dir = Path(images_dir)
    out_dir = images_dir.parent / f"texts_en_images_{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tid in sorted(text_df["text_id"].unique()):
        subset = text_df[text_df["text_id"] == tid]
        img = images_dir / f"Item_{int(tid):02d}.png"
        if not img.exists():
            print(f"⚠️ {img} missing, skipping {tid}")
            continue
        outf = out_dir / f"Item_{int(tid):02d}_{out_suffix}.png"
        saved = plot_word_surp_overlay(subset, img, outf, division_factor_space)
        print(f"→ {saved}")
