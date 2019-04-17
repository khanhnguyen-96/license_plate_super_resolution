if [ -z "$3" ]; then
    testdir=/workspace/dataset/lpt_square/test/LR/
else testdir=$3
fi
if [ "$1" = "trainedsr" ]
then
    #python main.py --model EDSR --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 3 --save EDSR_B32F256_realreal --epochs 200 --datatype realreal --data_test LPT --data_train LPT 
    #python main.py --model EDSR --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 3 --save EDSR_B32F256_realsimu --epochs 200 --datatype realsimu --data_test LPT --data_train LPT 
    #python main.py --model EDSR --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 3 --save EDSR_B32F256_simureal --epochs 200 --datatype simureal --data_test LPT --data_train LPT 
    #python main.py --model EDSR --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 3 --save EDSR_B32F256_simusimu --epochs 200 --datatype simusimu --data_test LPT --data_train LPT 
    #python main.py --model EDSR --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_colors 3 --save EDSR_B32F256_LPTall --epochs 200 --datatype all --data_test LPT --data_train LPT 
    python main.py --model EDSR --scale 3 --n_resblocks 128 --n_feats 128 --res_scale 0.1 --n_colors 3 --save EDSR_B128F128_LPTall --epochs 300 --datatype all --data_test LPT --data_train LPT 
elif [ "$1" = "trainedsrsq" ]
then
    W=69
    H=69
    python main.py --model EDSR --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --load EDSR_B32F256_"$W"x"$H"_all_rot_noblur --epochs 220 --dir_data /workspace/dataset/lpt_square/images --datatype simureal --data_test LPTSQUARE --data_train LPTSQUARE --std-lr-width $W --std-lr-height $H --gauss-std -1
elif [ "$1" = "traineedsrsq" ]
then
    W=69
    H=69
    python3.6 main.py --model EEDSR --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --loss "1*ML1" --save EEDSR_B32F256_ML1_"$W"x"$H"_all_rot_noblur --epochs 300 --batch_size 24 --dir_data /workspace/dataset/lpt_square/images --datatype simureal --data_test LPTSQUARE --data_train LPTSQUARE --std-lr-width $W --std-lr-height $H --gauss-std -1 --interp nearest --freesize --reset
elif [ "$1" = "trainddbpn" ]
then
    python main.py --model DDBPN --scale 2 --load DDBPN_46x32_all_noblur --batch_size 24 --epochs 300 --dir_data /workspace/dataset/lpt_square/images_rect --datatype simureal --data_test LPT --data_train LPT --std-lr-width 46 --std-lr-height 32 --gauss-std -1 --interp nearest --reset 
elif [ "$1" = "trainddbpnsq" ]
then
    python main.py --model DDBPN --scale 4 --save DDBPN_testing --epochs 200 --dir_data /workspace/dataset/lpt_square/images --datatype simureal --data_test LPTSQUARE --data_train LPTSQUARE --std-lr-width 48 --std-lr-height 48 --gauss-std 1.6 
elif [ "$1" = "trainespcn" ]
then
    python main.py --model ESPCN --scale 3 --batch_size 24 --n_colors 1 --loss "1*MSE" --lr 0.001 --lr_decay 200 --save ESPCN_46x32_all_noblur --epochs 300 --dir_data /workspace/dataset/lpt_square/images_rect --datatype simureal --data_test LPT --data_train LPT --std-lr-width 46 --std-lr-height 32 --gauss-std -1 --interp nearest --reset
elif [ "$1" = "trainespcnsq" ]
then
    W=48
    H=48
    python main.py --model ESPCN --scale 3 --batch_size 24 --n_colors 1 --loss "1*MSE" --lr 0.01 --lr_decay 10 --save ESPCN_"$W"x"$H"_all_rot_noblur --epochs 300 --dir_data /workspace/dataset/lpt_square/images --datatype simureal --data_test LPTSQUARE --data_train LPTSQUARE --std-lr-width $W --std-lr-height $H --gauss-std -1 --interp nearest
elif [ "$1" = "trainrdn" ]
then
    python main.py --model RDN --scale 3 --RDNconfig B --batch_size 24 --save test --epochs 300 --dir_data /workspace/dataset/lpt_square/images_rect --datatype simureal --data_test LPT --data_train LPT --std-lr-width 23 --std-lr-height 16 --gauss-std -1 --interp nearest
elif [ "$1" = "trainrdnsq" ]
then
    W=48
    H=48
    python3.6 main.py --model RDN --scale 3 --RDNconfig B --batch_size 16 --model_best_loss 2 --save RDN_D16C8G64_"$W"x"$H"_all_rot_noblur_run2 --epochs 300 --dir_data /workspace/dataset/lpt_square/images --datatype simureal --data_test LPTSQUARE --data_train LPTSQUARE --std-lr-width $W --std-lr-height $H --gauss-std -1 --interp nearest --reset
elif [ "$1" = "trainerdn" ]
then
    W=69
    H=48
    python main.py --model ERDN1 --scale 3 --RDNconfig B --loss "1*ML1" --batch_size 24 --save ERDN1_D16C8G64_"$W"x"$H"_all_noblur --epochs 300 --dir_data /workspace/dataset/lpt_square/images_rect --datatype simureal --data_test LPT --data_train LPT --std-lr-width $W --std-lr-height $H --gauss-std -1 --interp nearest --freesize
elif [ "$1" = "trainerdnsq" ]
then
    W=69
    H=69
    python3.6 main.py --model ERDN1 --scale 3 --RDNconfig B --loss "1*ML1" --batch_size 24 --save ERDN1_D16C8G64_ML1d_"$W"x"$H"_all_rot_noblur --epochs 300 --dir_data /workspace/dataset/lpt_square/images --datatype simureal --data_test LPTSQUARE --data_train LPTSQUARE --std-lr-width $W --std-lr-height $H --gauss-std -1 --interp nearest --freesize --reset
elif [ "$1" = "trainerdnpsq" ]
then
    W=69
    H=69
    python main.py --model ERDNP1 --scale 3 --RDNconfig B --save ERDNP1_D16C8G64_"$W"x"$H"_all_rot_noblur --epochs 300 --dir_data /workspace/dataset/lpt_square/images --datatype simureal --data_test LPTSQUARE --data_train LPTSQUARE --std-lr-width $W --std-lr-height $H --gauss-std -1 --interp nearest --freesize
elif [ "$1" = "trainfn4lsrsq" ]
then
    W=69
    H=69
    python main.py --model FN4LSR --scale 3 --G0 30 --n_resblocks 5 --save FN4LSR_G27B5_"$W"x"$H"_all_rot_noblur --epochs 300 --lr 0.001 --lr_decay 80 --dir_data /workspace/dataset/lpt_square/images --datatype simureal --data_test LPTSQUARE --data_train LPTSQUARE --std-lr-width $W --std-lr-height $H --gauss-std -1 --interp nearest --freesize
elif [ "$1" = "trainfn4lsr" ]
then
    W=23
    H=16
    python main.py --model FN4LSR --scale 3 --G0 30 --n_resblocks 4 --save FN4LSR_G30B4_"$W"x"$H"_all_noblur --epochs 200 --lr 0.001 --lr_decay 80 --dir_data /workspace/dataset/lpt_square/images_rect --datatype simureal --data_test LPT --data_train LPT --std-lr-width $W --std-lr-height $H --gauss-std 1.6 --interp nearest
elif [ "$1" = "trainern4lsrsq" ]
then
    W=69
    H=69
    F=128
    C=64
    python main.py --model ERN4LSR --scale 3 --n_feats $F --n_convs $C --save ERN4LSR_F"$F"C"$C"_"$W"x"$H"_all_rot --epochs 220 --dir_data /workspace/dataset/lpt_square/images --datatype simureal --data_test LPTSQUARE --data_train LPTSQUARE --std-lr-width $W --std-lr-height $H --gauss-std -1 --interp nearest --freesize 
elif [ "$1" = "trainrcansq" ]
then
    python main.py --model RCAN --n_resblocks 20 --scale 3 --save RCAN_square48_augmentrot --epochs 300 --dir_data /workspace/dataset/lpt_square/images --datatype simureal --data_test LPTSQUARE --data_train LPTSQUARE --std-lr-width 48 --std-lr-height 48 --gauss-std 1.6
elif [ "$1" = "testedsr" ]
then
    python main.py --model EDSR --data_test Demo --data_train LPT --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train $2 --test_only --save_results --dir-testlpt $testdir --std-lr-width 46 --std-lr-height 32 --gauss-std -1 --interp nearest
elif [ "$1" = "testddbpn" ]
then
    python main.py --model DDBPN --data_test Demo --data_train LPT  --scale 2 --pre_train $2 --test_only --save_results --dir-testlpt $testdir --std-lr-width 46 --std-lr-height 32 --gauss-std -1 --interp nearest
elif [ "$1" = "testespcn" ]
then
    python main.py --model ESPCN --n_colors 1 --data_test Demo --data_train LPT  --scale 3 --pre_train $2 --test_only --save_results --dir-testlpt $testdir --std-lr-width 48 --std-lr-height 32 --gauss-std -1 --interp nearest
elif [ "$1" = "testespcnsq" ]
then
    python main.py --model ESPCN --data_test Demo --data_train LPTSQUARE  --scale 3 --n_colors 1 --pre_train $2 --test_only --save_results --dir-testlpt $testdir --std-lr-width 48 --std-lr-height 48 --gauss-std -1 --interp nearest
elif [ "$1" = "testrdn" ]
then
    python main.py --model RDN --RDNconfig B --data_test Demo --data_train LPT  --scale 3 --pre_train $2 --test_only --save_results --dir-testlpt $testdir --std-lr-width 46 --std-lr-height 32 --gauss-std -1 --interp nearest
elif [ "$1" = "testedsrsq" ]
then
    python main.py --model EDSR --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Demo --data_train LPTSQUARE --pre_train $2 --test_only --save_results --dir-testlpt $testdir --std-lr-width 69 --std-lr-height 69 --gauss-std -1 --interp nearest
elif [ "$1" = "testrdnsq" ]
then
    python3.6 main.py --model RDN --RDNconfig C --G0 31 --data_test Demo --data_train LPTSQUARE  --scale 3 --pre_train $2 --test_only --save_results --dir-testlpt $testdir --std-lr-width 48 --std-lr-height 48 --gauss-std -1 --interp nearest
elif [ "$1" = "testerdn" ]
then
    python main.py --model ERDN1 --RDNconfig B --data_test Demo --data_train LPT --scale 3 --pre_train $2 --test_only --save_results --dir-testlpt $testdir --std-lr-width 69 --std-lr-height 48 --gauss-std -1 --interp nearest --freesize
elif [ "$1" = "testerdnsq" ]
then
    python3.6 main.py --model ERDN1 --RDNconfig B --data_test Demo --data_train LPTSQUARE  --scale 3 --pre_train $2 --test_only --save_results --dir-testlpt $testdir --std-lr-width 69 --std-lr-height 69 --gauss-std -1 --interp nearest --freesize
elif [ "$1" = "testrcansq" ]
then
    python main.py --model RCAN --n_resblocks 20 --data_test Demo --data_train LPTSQUARE  --scale 3 --pre_train $2 --test_only --save_results --dir-testlpt $testdir --std-lr-width 48 --std-lr-height 48 --gauss-std 1.6
elif [ "$1" = "testddbpnsq" ]
then    
    python main.py --model DDBPN --scale 4 --data_test Demo --data_train LPTSQUARE --pre_train $2 --test_only --save_results --dir-testlpt $testdir --std-lr-width 48 --std-lr-height 48 --gauss-std -1 --interp nearest
fi
