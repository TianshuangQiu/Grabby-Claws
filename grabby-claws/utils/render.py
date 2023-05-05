import imageio
import glob
import os
import pdb

if not os.path.exists("../../videos"):
    os.mkdir("../../videos")

for i in range(0, 1):
    writer = imageio.get_writer(f"../../videos/env{i}.mov", fps=30)
    file_names = [
        file
        for file in glob.glob(
            os.path.join("../../graphics_images", f"rgb_env{i}_cam*.png")
        )
    ]
    file_names.sort()
    # pdb.set_trace()
    for file in file_names:
        im = imageio.imread(file)
        writer.append_data(im)
    writer.close()
