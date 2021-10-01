# image size is 410x199
# offset x: 62 for col1, 652 for col2
# offset y: 101 for row1, 364 for row2
#convert output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_4KM/C3S_ALBB_DH_Global_4KM_1984??_withsnow.png -crop 410x199+62+101 tmp/out_crop_v1.png
#convert output_comp_albedo_V1-V2_AVHRR/C3S_ALBB_DH_Global_4KM/C3S_ALBB_DH_Global_4KM_1984??_withsnow.png -crop 410x199+62+364 tmp/out_crop_v2.png
#convert +append tmp/out_crop_v1-*.png tmp/out_crop_v2-*.png tmp/out.png


 convert output_recursive_snht/output_snht_0[01234]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_01.png
 convert output_recursive_snht/output_snht_0[56789]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_02.png
 echo '100'
 convert output_recursive_snht/output_snht_1[01234]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_03.png
 convert output_recursive_snht/output_snht_1[56789]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_04.png
 echo '200'
 convert output_recursive_snht/output_snht_2[01234]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_05.png
 convert output_recursive_snht/output_snht_2[56789]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_06.png
 echo '300'
 convert output_recursive_snht/output_snht_3[01234]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_07.png
 convert output_recursive_snht/output_snht_3[56789]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_08.png
 echo '400'
 convert output_recursive_snht/output_snht_4[01234]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_09.png
 convert output_recursive_snht/output_snht_4[56789]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_10.png
 echo '500'
 convert output_recursive_snht/output_snht_5[01234]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_11.png
 convert output_recursive_snht/output_snht_5[56789]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_12.png
 echo '600'
 convert output_recursive_snht/output_snht_6[01234]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_13.png
 convert output_recursive_snht/output_snht_6[56789]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_14.png
 echo '700'
 convert output_recursive_snht/output_snht_7[01234]*.png -fuzz 20% -trim +repage -geometry 50%x miff:- | montage -mode concatenate -tile 5x -font Liberation-Sans-Regular -geometry +2+2 miff:- tmp/mosaic_snht_15.png

