flag_merge=1

if (($flag_merge == 0)); then
  a=100
  step=0
else
  a=2000
  step=100
fi

pt_min=(300 500)
pt_max=(500 10000)

for file_min in {0..1999..$a}; do
   for index in 1 2; do
      for radius in AK8 AK15 CA15; do
         for bkg in Higgs QCD; do
            for name_variable in gen_ norm ; do
               echo $file_min $[$file_min+$a] $pt_min[${index}] $pt_max[${index}] $name_variable $bkg $radius $step $flag_merge
               python code/preProcessing_Sequential_variables.py $file_min $[$file_min+$a] $pt_min[${index}] $pt_max[${index}] $name_variable $bkg $radius $step $flag_merge &
               sleep 0.1
            done
         done
      done
   done
done
