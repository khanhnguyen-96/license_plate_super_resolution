curdir=$(pwd)

if [ -z "$1" ]; then
    folder=../experiment/test/results
else folder=$1
fi

dstype=simureal
testdir=/workspace/dataset/lpt_square/test/
#testdir=/workspace/dataset/lpt_square/test/LRrect/

#cd $folder
#rename 's/_x3_SR/_SR/' *.png
#cd $curdir
#python3.6 square_to_rect.py --image-dir $folder --save-dir $folder --o
python3.6 psnrssim.py --lrext .bmp --testdir $testdir --datasettype $dstype --save-dir $folder $folder/LR_*
