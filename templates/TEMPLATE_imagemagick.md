# ImageMagick templates

### Make a mosaic
In matplotlib, we can remove the label of the axis to have a nive black box around the graph easy to trim with `ax.get_xaxis().set_ticks([])`
- tile: 16 colums and 25 lines
- geometry: resize input image to have a 100px length
- trim: remove the white space around the image
```
montage -mode concatenate -tile 16x25 -geometry 100x+0+0 -trim +repage -font Liberation-Sans-Regular output_recursive_snht/output_snht_[3210]* tmp/mosaic.png`
```

## Use a pipe to feed montage
First we remove border and resize a batch of images and put everything in the buffer `miff:-`, then use this buffer with `montage` in a pipe:
```
convert output_recursive_snht/output_snht_0[01234]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 4x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/test.png
```




### Example of python script using imagemagick
```python
import os,sys

"""
Usage:
python crop_merge.py
"""


def crop(img_in, crop_arg, img_out):
    cmd = ['/usr/bin/convert', img_in, '-crop', crop_arg, img_out]
    join_cmd = ' '.join(cmd)
    print(join_cmd)
    #subprocess.Popen(cmd) # make an error with temporary imagemagick file
    os.system(join_cmd)

def annotate(img_in, label, img_out):
    cmd = "convert -font Liberation-Sans-Regular -background '#0008' -fill white -gravity center -size 50x40 label:\"{}\" {} +swap -gravity north -composite {}".format(label, img_in, img_out)
    print(cmd)
    os.system(cmd)

def merge(im_list, tile_arg):
    #cmd = "montage {} -geometry +2+2 mosaic.png".format(' '.join(im_list), geom_arg)
    cmd = "montage -mode concatenate -tile {} -font Liberation-Sans-Regular {} tmp/mosaic.png".format(tile_arg, ' '.join(im_list))
    print(cmd)
    os.system(cmd)


def main():
    im_list = []
    for i in range(36):
        si = str(i).zfill(2)
        print('---', si)
        #img_in = 'output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_4KM/C3S_ALBB_DH_Global_4KM_2003{}_merged.png'.format(si)
        img_in = 'output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_4KM/C3S_ALBB_DH_Global_4KM_1984{}_merged.png'.format(si)
        crop_arg_v1 = '399x200+75+101'
        crop_arg_v2 = '399x200+75+364'
        #crop_arg_v2 = '399x200+1255+364'
        img_out_v1 = 'tmp/out_crop_v1.png'
        img_out_v2 = 'tmp/out_crop_v2.png'
        crop(img_in, crop_arg_v1, img_out_v1)
        crop(img_in, crop_arg_v2, img_out_v2)
        
        label_name1 = 'tmp/out_label_v1_{}.png'.format(si)
        label_name2 = 'tmp/out_label_v2_{}.png'.format(si)
        annotate(img_out_v1, si, label_name1)
        annotate(img_out_v2, si, label_name2)
        im_list.append(label_name2)

    merge(im_list, '6x6')

if __name__=='__main__':
    main()
```
