Đây là thư mục chứa code.
Script để train và test: traintest.sh
Để train có thể dùng lệnh: nohup ./traintest.sh trainrdnsq &
Để test dùng lệnh: ./traintest.sh testrdnsq {path_to_model} [path_to_testset]. path_to_testset chứa 2 thư mục LR/ và HRS/, LR/ chứa ảnh để test, HRS/ chứa ảnh để so sánh. Ví dụ: ./traintest.sh testrdnsq ../experiment/RDN_D16C8G64_square23/model/model_best.pt
Ảnh khôi phục được lưu tại ../experiment/test/results/
