dstype=realreal
folder=../experiment/RDN_D16C8G64_"$dstype"/results
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LR_* > $folder/testlog.txt
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LRS_* >> $folder/testlog.txt

dstype=realsimu
folder=../experiment/RDN_D16C8G64_"$dstype"/results
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LR_* > $folder/testlog.txt
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LRS_* >> $folder/testlog.txt

dstype=simureal
folder=../experiment/RDN_D16C8G64_"$dstype"/results
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LR_* > $folder/testlog.txt
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LRS_* >> $folder/testlog.txt

dstype=simusimu
folder=../experiment/RDN_D16C8G64_"$dstype"/results
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LR_* > $folder/testlog.txt
python3.5 psnrssim.py --testdir test/ --datasettype $dstype $folder/LRS_* >> $folder/testlog.txt
