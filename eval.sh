testdir=/workspace/dataset/lpt_square/test/
declare -a arr=("EDSR_B32F256_48x48_all_rot_noblur" "ERDNv1_D16C8G64_69x69_all_rot_noblur" "ERDNv1_D16C8G64_81x81_all_rot_noblur" "ERDNv2_D16C8G64_69x69_all_rot_noblur" "ERDNv1b_D16C8G64_69x69_all_rot_noblur" "ERDNv3_D16C8G64_69x69_all_rot_noblur" "RDN_D16C8G64_48x48_all_rot_noblur" "RDN_D16C8G64_69x69_all_rot_noblur" "RDN_D16C8G64_scale1_69x69_all_rot_noblur")


#for model in "${arr[@]}"
#do    
#    python3.6 psnrssim.py --testdir $testdir --datasettype simureal --save-dir ../experiment/$model/results/ ../experiment/$model/results/LR_*
#done

zipcmd="zip -qy ../exp.zip"
for model in "${arr[@]}"
do
    zipcmd="${zipcmd} ../experiment/${model}/results/*.jpg ../experiment/${model}/results/*.txt"
done
$zipcmd
