BEGIN {
  FS = OFS = ","
}

{
    if (NR > 1) {
        split($2,srcIP,".");

        for(i=0; i<upsampling; i++) {
            newIP = sprintf("%s.%s.%s.%s", srcIP[1]+1000*i, srcIP[2], srcIP[3], srcIP[4])
            print $1","newIP","$3
        }
        
    } else {
        print 
    }
    
}