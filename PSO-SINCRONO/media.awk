# media.awk
{ sum+=$1
  it+=1 }
#END { print " & " it " & " sum " & " sum/it "\\\\ \\hline"}
END { print sum/it }
