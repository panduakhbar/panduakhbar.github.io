import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window, bounds
from shapely.geometry import box
from tqdm.auto import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

# Folder berisi seluruh orthomosaic
RASTER_DIR = Path("/content/orthomosaics")

# Satu GeoJSON gabungan
VECTOR_PATH = Path("/content/vector_dataset3.geojson")

# Folder output
OUTPUT_DIR = Path("/content/palmtree_coco")

# Field yang menyimpan nama kelas
CLASS_FIELD = "class"

# Band RGB GeoTIFF
RGB_BANDS = (1, 2, 3)

# Ukuran tile dalam piksel
TILE_SIZE = 640

# Overlap antar-tile:
# 0.20 berarti overlap 20%
OVERLAP = 0.20

# Minimum bagian objek yang harus terlihat di tile
MIN_VISIBLE_FRACTION = 0.30

# Minimum ukuran bounding box dalam piksel
MIN_BOX_SIZE = 3

# Simpan tile tanpa annotation atau tidak
KEEP_EMPTY_TILES = False

# Minimum bagian tile yang memiliki data raster
# Tile yang sebagian besar NoData akan dilewati
MIN_VALID_TILE_FRACTION = 0.20

# Kualitas JPG
JPEG_QUALITY = 95


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_rasters(folder: Path):
    """
    Mencari seluruh file .tif dan .tiff secara rekursif.
    """
    raster_paths = [
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
    ]

    return sorted(raster_paths)


def generate_starts(length: int, tile_size: int, stride: int):
    """
    Membuat posisi awal tile dan memastikan ujung raster tetap masuk.
    """
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


def choose_rgb_bands(src):
    """
    Menentukan band RGB yang dapat digunakan.
    """
    requested = [
        band
        for band in RGB_BANDS
        if 1 <= band <= src.count
    ]

    if len(requested) >= 3:
        return tuple(requested[:3])

    if src.count == 1:
        # Gandakan single band menjadi RGB
        return (1, 1, 1)

    raise ValueError(
        f"Raster memiliki {src.count} band. "
        "Sesuaikan nilai RGB_BANDS."
    )


def calculate_rgb_limits(src, rgb_bands):
    """
    Menghitung percentile untuk stretching raster non-uint8.
    Nilai dihitung per orthomosaic.
    """
    sample_height = min(src.height, 1000)
    sample_width = min(src.width, 1000)

    sample = src.read(
        indexes=rgb_bands,
        out_shape=(
            len(rgb_bands),
            sample_height,
            sample_width
        ),
        masked=True
    )

    limits = []

    for band in sample:
        values = band.compressed()

        if len(values) == 0:
            limits.append((0.0, 1.0))
            continue

        low, high = np.percentile(values, [2, 98])

        if high <= low:
            high = low + 1

        limits.append((float(low), float(high)))

    return limits


def convert_to_uint8(image, limits=None):
    """
    Mengubah raster array menjadi RGB uint8.
    Input : band, height, width
    Output: height, width, band
    """
    if np.ma.isMaskedArray(image):
        image = image.filled(0)

    if image.dtype == np.uint8:
        result = image
    else:
        result = np.zeros(image.shape, dtype=np.uint8)

        for band_index in range(image.shape[0]):
            low, high = limits[band_index]

            scaled = (
                image[band_index].astype(np.float32) - low
            ) / (high - low)

            scaled = scaled * 255

            result[band_index] = np.clip(
                scaled,
                0,
                255
            ).astype(np.uint8)

    return np.moveaxis(result, 0, -1)


def geometry_to_pixel_bbox(
    geometry,
    tile_transform,
    tile_width,
    tile_height
):
    """
    Mengubah bounding box koordinat peta menjadi
    bounding box koordinat piksel COCO: x, y, width, height.
    """
    min_x, min_y, max_x, max_y = geometry.bounds

    map_corners = [
        (min_x, min_y),
        (min_x, max_y),
        (max_x, min_y),
        (max_x, max_y),
    ]

    inverse_transform = ~tile_transform

    pixel_corners = [
        inverse_transform * coordinate
        for coordinate in map_corners
    ]

    columns = [coordinate[0] for coordinate in pixel_corners]
    rows = [coordinate[1] for coordinate in pixel_corners]

    x_min = max(0.0, min(columns))
    y_min = max(0.0, min(rows))

    x_max = min(float(tile_width), max(columns))
    y_max = min(float(tile_height), max(rows))

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    return x_min, y_min, bbox_width, bbox_height


# ============================================================
# VALIDATE INPUT
# ============================================================

raster_paths = find_rasters(RASTER_DIR)

if not raster_paths:
    raise FileNotFoundError(
        f"Tidak ditemukan file .tif atau .tiff di {RASTER_DIR}"
    )

if not VECTOR_PATH.exists():
    raise FileNotFoundError(
        f"GeoJSON tidak ditemukan: {VECTOR_PATH}"
    )

print(f"Orthomosaic ditemukan: {len(raster_paths)}")

for raster_path in raster_paths:
    print(" -", raster_path.name)


# ============================================================
# READ COMBINED GEOJSON
# ============================================================

gdf_original = gpd.read_file(VECTOR_PATH)

if gdf_original.empty:
    raise ValueError("GeoJSON tidak memiliki feature.")

if gdf_original.crs is None:
    raise ValueError(
        "GeoJSON tidak memiliki CRS. "
        "Tetapkan CRS yang benar di QGIS lalu export ulang."
    )

if CLASS_FIELD not in gdf_original.columns:
    print(
        f"Field '{CLASS_FIELD}' tidak ditemukan. "
        "Semua annotation diberi kelas 'object'."
    )
    gdf_original[CLASS_FIELD] = "object"

gdf_original[CLASS_FIELD] = (
    gdf_original[CLASS_FIELD]
    .fillna("object")
    .astype(str)
    .str.strip()
)

gdf_original = gdf_original[
    gdf_original.geometry.notna()
    & ~gdf_original.geometry.is_empty
].copy()

if gdf_original.empty:
    raise ValueError("Tidak ada geometry valid pada GeoJSON.")


# ============================================================
# CREATE GLOBAL COCO CATEGORIES
# ============================================================

class_names = sorted(
    gdf_original[CLASS_FIELD].unique()
)

category_ids = {
    class_name: category_id
    for category_id, class_name
    in enumerate(class_names, start=1)
}

coco = {
    "info": {
        "description": (
            "COCO dataset generated from multiple "
            "orthomosaics and one combined GeoJSON"
        )
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": category_id,
            "name": class_name,
            "supercategory": "object"
        }
        for class_name, category_id in category_ids.items()
    ]
}


# ============================================================
# PREPARE OUTPUT
# ============================================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_id = 1
annotation_id = 1

manifest_rows = []
raster_summary = []


# ============================================================
# PROCESS ALL ORTHOMOSAICS
# ============================================================

for raster_number, raster_path in enumerate(
    raster_paths,
    start=1
):
    print("\n" + "=" * 60)
    print(f"Processing: {raster_path.name}")

    raster_image_count = 0
    raster_annotation_count = 0

    with rasterio.open(raster_path) as src:

        if src.crs is None:
            print(
                f"SKIP: {raster_path.name} tidak memiliki CRS."
            )
            continue

        # Reproject GeoJSON ke CRS orthomosaic ini
        if gdf_original.crs != src.crs:
            raster_gdf = gdf_original.to_crs(src.crs)
        else:
            raster_gdf = gdf_original.copy()

        raster_extent = box(*src.bounds)

        # Ambil hanya rectangle yang menyentuh raster ini
        raster_gdf = raster_gdf[
            raster_gdf.intersects(raster_extent)
        ].copy()

        raster_gdf.reset_index(drop=True, inplace=True)

        print("CRS raster:", src.crs)
        print("Ukuran raster:", src.width, "x", src.height)
        print(
            "Annotation yang overlap:",
            len(raster_gdf)
        )

        if raster_gdf.empty:
            print("Tidak ada annotation. Raster dilewati.")
            continue

        spatial_index = raster_gdf.sindex

        rgb_bands = choose_rgb_bands(src)

        raster_is_uint8 = all(
            src.dtypes[band - 1] == "uint8"
            for band in set(rgb_bands)
        )

        rgb_limits = None

        if not raster_is_uint8:
            print("Menghitung RGB stretch...")
            rgb_limits = calculate_rgb_limits(
                src,
                rgb_bands
            )

        stride = max(
            1,
            int(TILE_SIZE * (1 - OVERLAP))
        )

        column_starts = generate_starts(
            src.width,
            TILE_SIZE,
            stride
        )

        row_starts = generate_starts(
            src.height,
            TILE_SIZE,
            stride
        )

        tile_positions = [
            (column_start, row_start)
            for row_start in row_starts
            for column_start in column_starts
        ]

        print("Jumlah kandidat tile:", len(tile_positions))

        # Prefix unik supaya nama file tidak bentrok
        raster_prefix = (
            f"{raster_number:03d}_{raster_path.stem}"
        )

        for column_start, row_start in tqdm(
            tile_positions,
            desc=raster_path.name
        ):
            window_width = min(
                TILE_SIZE,
                src.width - column_start
            )

            window_height = min(
                TILE_SIZE,
                src.height - row_start
            )

            window = Window(
                col_off=column_start,
                row_off=row_start,
                width=window_width,
                height=window_height
            )

            left, bottom, right, top = bounds(
                window,
                src.transform
            )

            tile_geometry = box(
                left,
                bottom,
                right,
                top
            )

            candidate_positions = spatial_index.query(
                tile_geometry,
                predicate="intersects"
            )

            candidates = raster_gdf.iloc[
                candidate_positions
            ]

            tile_transform = src.window_transform(window)
            tile_annotations = []

            for _, feature in candidates.iterrows():
                original_geometry = feature.geometry

                if (
                    original_geometry is None
                    or original_geometry.is_empty
                    or original_geometry.area <= 0
                ):
                    continue

                clipped_geometry = (
                    original_geometry.intersection(
                        tile_geometry
                    )
                )

                if clipped_geometry.is_empty:
                    continue

                visible_fraction = (
                    clipped_geometry.area
                    / original_geometry.area
                )

                if (
                    visible_fraction
                    < MIN_VISIBLE_FRACTION
                ):
                    continue

                x, y, bbox_width, bbox_height = (
                    geometry_to_pixel_bbox(
                        geometry=clipped_geometry,
                        tile_transform=tile_transform,
                        tile_width=window_width,
                        tile_height=window_height
                    )
                )

                if (
                    bbox_width < MIN_BOX_SIZE
                    or bbox_height < MIN_BOX_SIZE
                ):
                    continue

                class_name = feature[CLASS_FIELD]

                tile_annotations.append({
                    "category_id": category_ids[class_name],
                    "bbox": [
                        round(float(x), 2),
                        round(float(y), 2),
                        round(float(bbox_width), 2),
                        round(float(bbox_height), 2)
                    ],
                    "area": round(
                        float(
                            bbox_width
                            * bbox_height
                        ),
                        2
                    ),
                    "iscrowd": 0
                })

            if (
                not tile_annotations
                and not KEEP_EMPTY_TILES
            ):
                continue

            # Periksa berapa banyak area raster valid
            valid_mask = src.dataset_mask(
                window=window
            )

            valid_fraction = float(
                np.count_nonzero(valid_mask)
                / valid_mask.size
            )

            if (
                valid_fraction
                < MIN_VALID_TILE_FRACTION
            ):
                continue

            raster_data = src.read(
                indexes=rgb_bands,
                window=window,
                masked=True
            )

            rgb_image = convert_to_uint8(
                raster_data,
                limits=rgb_limits
            )

            filename = (
                f"{raster_prefix}"
                f"__r{row_start:07d}"
                f"_c{column_start:07d}.jpg"
            )

            image_path = OUTPUT_DIR / filename

            Image.fromarray(rgb_image).save(
                image_path,
                quality=JPEG_QUALITY
            )

            coco["images"].append({
                "id": image_id,
                "file_name": filename,
                "width": int(window_width),
                "height": int(window_height),

                # Field tambahan untuk tracking
                "source_raster": raster_path.name
            })

            for annotation in tile_annotations:
                annotation["id"] = annotation_id
                annotation["image_id"] = image_id

                coco["annotations"].append(
                    annotation
                )

                annotation_id += 1
                raster_annotation_count += 1

            manifest_rows.append({
                "file_name": filename,
                "source_raster": raster_path.name,
                "row_start": row_start,
                "column_start": column_start,
                "annotation_count": len(
                    tile_annotations
                ),
                "valid_fraction": round(
                    valid_fraction,
                    4
                )
            })

            image_id += 1
            raster_image_count += 1

    raster_summary.append({
        "source_raster": raster_path.name,
        "images": raster_image_count,
        "annotations": raster_annotation_count
    })

    print(
        f"Hasil {raster_path.name}: "
        f"{raster_image_count} images, "
        f"{raster_annotation_count} annotations"
    )


# ============================================================
# SAVE COCO JSON
# ============================================================

annotation_path = (
    OUTPUT_DIR / "_annotations.coco.json"
)

with open(
    annotation_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        coco,
        file,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SAVE MANIFEST AND SUMMARY
# ============================================================

manifest_path = OUTPUT_DIR / "manifest.csv"

pd.DataFrame(manifest_rows).to_csv(
    manifest_path,
    index=False
)

summary_path = OUTPUT_DIR / "raster_summary.csv"

pd.DataFrame(raster_summary).to_csv(
    summary_path,
    index=False
)


# ============================================================
# CREATE ZIP
# ============================================================

zip_path = shutil.make_archive(
    str(OUTPUT_DIR),
    "zip",
    root_dir=OUTPUT_DIR
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)

print("Orthomosaic diproses :", len(raster_paths))
print("Total images         :", len(coco["images"]))
print(
    "Total annotations    :",
    len(coco["annotations"])
)
print("Categories           :", coco["categories"])
print("COCO JSON            :", annotation_path)
print("Manifest             :", manifest_path)
print("Raster summary       :", summary_path)
print("ZIP                   :", zip_path)
