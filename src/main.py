from misc import get_img, save_img
from lpr import std_lpr, l2_norm
import os
import sys

if __name__ == "__main__":
    image_file_path = "../images/Barbara.tif"
    output_folder = "../output"
    if len(sys.argv) > 1:
        output_folder = "/".join(sys.argv[0].replace('\\', '/').split("/")[:-1])
        if len(output_folder) == 0:
            output_folder += './../output'
        elif output_folder.endswith("/"):
            output_folder += '../output'
        else:
            output_folder += "/../output"
        image_file_path = sys.argv[1].replace('\\', '/')
    input_image_name = image_file_path.split("/")[-1].split(".")[0]
    input_image = get_img(image_file_path)
    
    # structure-texture decomposition parameters
    patch_size = 5
    rho = 5.0   # ADMM adjustement parameter
    nb_iter = 400
    mu = 0.0111

    structure, texture = std_lpr(input_image,  patch_size, rho, mu, nb_iter)
    try:
        os.mkdir(output_folder + f"/{input_image_name}")
    except FileExistsError:
        pass

    save_img(structure, f"{output_folder}/{input_image_name}/structure_lpr.png")
    save_img(texture + 0.5, f"{output_folder}/{input_image_name}/texture_lpr.png")
