flag_merge=1

step=100
min_=0
max_=1999
# max_=10

pt_min=(300 500)
pt_max=(500 10000)

for file_min in {${min_}..${max_}..${step}}; do
   for index in 1 2; do
      for radius in AK8 AK15 CA15; do
         for bkg in Higgs QCD; do
            for name_variable in norm ; do
               echo $file_min $[$file_min+${step}] $pt_min[${index}] $pt_max[${index}] $name_variable $bkg $radius $flag_merge
               python code/preProcessing_Conv2D_variables.py $file_min $[$file_min+${step}] $pt_min[${index}] $pt_max[${index}] $name_variable $bkg $radius $flag_merge &
               sleep 1
            done
         done
      done
   done
done
