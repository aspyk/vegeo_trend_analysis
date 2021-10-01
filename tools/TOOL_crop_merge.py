import os,sys
import pathlib


"""
Usage:
python crop_merge.py
"""


def crop(img_in, crop_arg, img_out):
    fpath = pathlib.Path(img_in)
    if fpath.is_file():
        #cmd = ['/usr/bin/convert', img_in, '-crop', crop_arg, img_out]
        cmd = ['/usr/bin/convert', img_in, '-crop @1x2 -delete 0', '-fuzz 20% -trim +repage', img_out] # auto crop mosaic in half horizontally and remove white border until the black border
        #cmd = ['/usr/bin/convert', img_in, '-geometry 50%x', '-fuzz 20% -trim +repage', img_out] # resize and remove white border until the black border
    else:
        size = (557, 132)
        cmd = ['/usr/bin/convert', '-size', '{:d}x{:d}'.format(*size), 'xc:white', '-strokewidth 1 -stroke black -fill none -draw', '"rectangle 0,0 {2:d},{3:d}, line 0,0 {0:d},{1:d}, line 0,{1:d} {0:d},0"'.format(size[0], size[1], size[0]-1, size[1]-1), img_out]
    join_cmd = ' '.join(cmd)
    print(join_cmd)
    #subprocess.Popen(cmd) # make an error with temporary imagemagick file
    os.system(join_cmd)


def annotate(img_in, label, img_out):
    cmd = "convert -font Liberation-Sans-Regular -background '#000F' -fill white -gravity center -size 50x40 label:\"{}\" {} +swap -gravity north -composite {}".format(label, img_in, img_out)
    print(cmd)
    os.system(cmd)

def merge(im_list, tile_arg, **kwargs):
    #cmd = "montage {} -geometry +2+2 mosaic.png".format(' '.join(im_list), geom_arg)
    cmd = "montage -mode concatenate -tile {} -font Liberation-Sans-Regular {} tmp/mosaic_{}.png".format(tile_arg, ' '.join(im_list), kwargs['year'])
    print(cmd)
    os.system(cmd)

def overlay(im_list, **kwargs):
    cmd = 'convert {} -transparent white -background None -layers Flatten tmp/overlay_{}.png'.format(' '.join(im_list), kwargs['year']) 
    print(cmd)
    os.system(cmd)

def main():
    for year in [str(i) for i in range(1982,2006)]:
        im_list = []
        for i in range(36):
            si = str(i).zfill(2)
            print('---', si)
            #img_in = 'output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_4KM/C3S_ALBB_DH_Global_4KM_1984{}_merged.png'.format(si)
            #img_in = 'output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_4KM/C3S_ALBB_DH_Global_4KM_1984{}_withsnow.png'.format(si)
            #img_in = 'output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_4KM/C3S_ALBB_DH_Global_4KM_1984{}_snowmask.png'.format(si)
            #img_in = 'output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_4KM/C3S_ALBB_DH_Global_4KM_2003{}_merged.png'.format(si)
            #img_in = 'output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_4KM/C3S_ALBB_DH_Global_4KM_2003{}_withsnow.png'.format(si)
            img_in= 'output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_4KM/C3S_ALBB_DH_Global_4KM_{}{}_snowmask.png'.format(year, si)

            #img_in = 'output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_1KM/C3S_ALBB_DH_Global_1KM_2003{}_withsnow.png'.format(si)
            #img_in = 'output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_1KM/C3S_ALBB_DH_Global_1KM_2003{}_snowmask.png'.format(si)
            #img_in = 'output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_1KM/C3S_ALBB_DH_Global_1KM_2014{}_withsnow.png'.format(si)
            #img_in = 'output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_1KM/C3S_ALBB_DH_Global_1KM_2014{}_snowmask.png'.format(si)
            crop_arg_v1 = '410x200+62+101'
            crop_arg_v2 = '410x200+62+364' # Asia 4KM
            #crop_arg_v2 = '391x200+84+364' # Asia 1KM
            img_out_v1 = 'tmp/out_crop_v1.png'
            img_out_v2 = 'tmp/out_crop_v2.png'
            crop(img_in, crop_arg_v1, img_out_v1)
            crop(img_in, crop_arg_v2, img_out_v2)
            
            label_name1 = 'tmp/out_label_v1_{}.png'.format(si)
            label_name2 = 'tmp/out_label_v2_{}.png'.format(si)
            annotate(img_out_v1, si, label_name1)
            annotate(img_out_v2, si, label_name2)
            im_list.append(label_name2)

        merge(im_list, '4x9', year=year)
        overlay(im_list, year=year)

if __name__=='__main__':
    main()
