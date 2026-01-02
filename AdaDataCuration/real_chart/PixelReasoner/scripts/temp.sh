while true; do

    bash /mnt/petrelfs/songmingyang/code/reasoning/data_construction/PixelReasoner/scripts/pixel_crop.sh
    bash /mnt/petrelfs/songmingyang/code/reasoning/data_construction/PixelReasoner/scripts/pixel_groundingcrop.sh
    bash /mnt/petrelfs/songmingyang/code/reasoning/data_construction/PixelReasoner/scripts/refocus_bar.sh
    bash /mnt/petrelfs/songmingyang/code/reasoning/data_construction/PixelReasoner/scripts/refocus_selfbar.sh
    sleep 10
done