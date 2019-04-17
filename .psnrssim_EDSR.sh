dstype=realreal
folder=../experiment/origin_"$dstype"_resblk256/results
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LR_* > $folder/testlog.txt
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LRS_* >> $folder/testlog.txt

dstype=realsimu
folder=../experiment/origin_"$dstype"_resblk256/results
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LR_* > $folder/testlog.txt
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LRS_* >> $folder/testlog.txt

dstype=simureal
folder=../experiment/origin_"$dstype"_resblk256/results
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LR_* > $folder/testlog.txt
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LRS_* >> $folder/testlog.txt

dstype=simusimu
folder=../experiment/origin_"$dstype"_resblk256/results
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LR_* > $folder/testlog.txt
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LRS_* >> $folder/testlog.txt

