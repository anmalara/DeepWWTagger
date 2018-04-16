a=100


for file_min in {0..1999..$a}; do
  for bkg in Higgs QCD; do
    for radius in AK8 AK15 CA15; do
      echo $file_min $[$file_min+$a] $bkg $radius
      python code/selectBranches.py $file_min $[$file_min+$a] $bkg $radius &
      sleep 0.1
    done
  done
done
