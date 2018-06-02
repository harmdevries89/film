

ngc batch run --name "Cloud-nv-us-west-2-268086" --image "mila1234/harm_film:3" --ace nv-us-west-2 --instance ngcv1 --commandline "git clone git@github.com:harmdevries89/film.git; cd film; export data_dir=/mnt/dataset1; export exp_dir=/results; export PYTHONPATH=/workspace/film; sh scripts/train/train_film.sh" --result /results --datasetid 9687:/mnt/dataset1