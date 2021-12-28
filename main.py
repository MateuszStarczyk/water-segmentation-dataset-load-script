import array
import os
import re
from typing import Tuple, List

import cv2
import numpy as np
import shapely
import shapely.affinity
import shapely.wkt
from geotiff import GeoTiff
from tifffile import tifffile


def convert_geojson_to_multipolygon(filename: str):
    file = open(filename)
    file_content = file.read()
    match = re.findall('\"coordinates\": .* } }', file_content, re.M)
    polygons = []
    for m in match:
        polygons.append(m.rstrip(' } }').lstrip('\"coordinates\": ').lstrip('[ ').rstrip(' ]').replace(',', '')
                        .replace(' ] [ ', ', ').replace('[ ', '(').replace(' ]', ')'))
    return f"MULTIPOLYGON ((({')), (('.join(polygons)})))"


def scale_coords(img_size, boundary_points):
    x_min, x_max, y_min, y_max = boundary_points
    h, w = img_size
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / (x_max - x_min), h_ / (y_min - y_max), x_min, y_max


def get_ground_truth_numpy_array(polygons, image_size):
    w, h = image_size

    img_mask = np.full((w, h), 0, np.uint8)

    exteriors = [np.array(poly.exterior.coords).round().astype(np.int32) for poly in polygons.geoms]
    cv2.fillPoly(img_mask, exteriors, 1)

    interiors = [np.array(pi.coords).round().astype(np.int32) for poly in polygons.geoms for pi in
                 poly.interiors]
    cv2.fillPoly(img_mask, interiors, 0)

    return img_mask


def read_geojson(image_size, filename: str, boundary_points: Tuple[float, float, float, float]):
    train_polygon = shapely.wkt.loads(convert_geojson_to_multipolygon(filename))

    x_scale, y_scale, x_offset, y_offset = scale_coords(image_size, boundary_points)

    train_polygon = shapely.affinity.scale(shapely.affinity.translate(train_polygon, -x_offset,
                                                                      -y_offset), xfact=x_scale,
                                           yfact=y_scale, origin=(0, 0, 0))
    return get_ground_truth_numpy_array(train_polygon, image_size)


def load():
    """
    Loads data from tif and geojson files
    :returns
    Dictionary of image name and loaded data from Sentinel-2.
    1st matrix is water ground truth of image.
    2nd matrix contains RGB and nir image data.
    3rd matrix contains Red Edge 1-4, SWIR 1 and SWIR 2.
    """
    loaded_data = dict()
    for name in os.listdir("./data/", ):
        if name.endswith(".geojson"):
            image_name = name.split('_json.geojson')[0]
            geo_file = GeoTiff(file="./data/{}_rgb_nir.tif".format(image_name))
            x_cords_min, y_cords_max = geo_file.tif_bBox[0]
            x_cords_max, y_cords_min = geo_file.tif_bBox[1]
            ground_truth_data = read_geojson(image_size=(geo_file.tif_shape[0], geo_file.tif_shape[0]),
                                             filename="./data/{}".format(name),
                                             boundary_points=(x_cords_min, x_cords_max, y_cords_min, y_cords_max))
            rgb_nir_data = tifffile.imread("./data/{}_rgb_nir.tif".format(image_name))
            swir_data = tifffile.imread("./data/{}_swir.tif".format(image_name))
            loaded_data[image_name] = ground_truth_data, rgb_nir_data, swir_data
    return loaded_data


if __name__ == '__main__':
    data = load()
