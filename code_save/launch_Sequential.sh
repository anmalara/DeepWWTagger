a=2000

pt_min=(300 500)
pt_max=(500 10000)


for file_min in {0..1999..$a}; do
  for index in 1 2; do
    for radius in AK8 AK15 CA15; do
      for name_variable in gen_ norm ; do
        echo $file_min $[$file_min+$a] $pt_min[${index}] $pt_max[${index}] $name_variable $radius
        python code/Sequential.py $file_min $[$file_min+$a] $pt_min[${index}] $pt_max[${index}] $name_variable $radius &
        sleep 0.1
      done
    done
  done
done
